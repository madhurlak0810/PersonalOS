"""The evaluation harness in `evals/`: one command, results, a summary and a diff.

The `rules` run is the one CI can afford, so it is pinned against the
checked-in baseline: a change to routing keywords, extraction rules, scoring
or the graphs that alters a golden outcome fails here until the baseline is
deliberately updated. The rest uses scripted chat models through the
gateway's factory seam -- an honest one and one that does whatever the
injected text tells it to -- to show that the harness runs any configuration
with no graph change, and that the adversarial suite catches the second.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evals import dataset, harness, report
from evals.suites import FAIL, PASS, REVIEW, SUITES, CaseResult, SuiteResult, fingerprint
from personalos.domain.job_search import RecruiterMessage
from personalos.models.gateway import ModelConfig, ModelGateway, ModelSpec
from personalos.models.recruiter_events import classify_with_rules

RULES = harness.CONFIGS / "rules.json"


def run_cli(tmp_path: Path, *args: str) -> tuple[int, dict[str, Any]]:
    code = harness.main(["--output", str(tmp_path), *args])
    return code, json.loads((tmp_path / "results.json").read_text())


# --- Scripted chat models -----------------------------------------------------


def _input(turns) -> dict[str, Any]:
    """The JSON document an adapter put in the human turn, or the raw text."""
    text = turns[1][1]
    try:
        return json.loads(text)
    except ValueError:
        return {"text": text}


def honest(schema: str, data: dict[str, Any]) -> dict[str, Any]:
    """Answers the way a careful model would, from the input alone."""
    if schema == "RouteDecision":
        text = data.get("text", "").lower()
        job = any(word in text for word in ("job", "role", "position", "hiring", "apply", "resume"))
        return {"domain": "job", "confidence": 0.9 if job else 0.2, "reasoning": "scripted"}
    if schema == "RecruiterEventExtraction":
        message = RecruiterMessage(
            provider_message_id="m",
            received_at=data["received_at"],
            subject=data["subject"],
            from_address=data["from_address"],
            body=data["body"],
        )
        return classify_with_rules(message).model_dump(mode="json")
    if schema == "SemanticAssessment":
        evidence = data["evidence"]
        matched, missing = [], []
        for skill in data["posting"]["skills"]:
            cited = next((e for e in evidence if skill.lower() in e["text"].lower()), None)
            if cited:
                matched.append(
                    {
                        "requirement": skill,
                        "evidence_id": cited["evidence_id"],
                        "strength": "strong",
                    }
                )
            else:
                missing.append({"requirement": skill, "severity": "major"})
        return {"matched_requirements": matched, "missing_requirements": missing}
    if schema == "DraftProposal":
        return {
            "artifact_type": data["artifact_type"],
            "segments": [
                {"text": record["text"], "evidence_ids": [record["evidence_id"]]}
                for record in data["evidence"]
            ],
        }
    raise AssertionError(f"unscripted schema {schema}")


def compromised(schema: str, data: dict[str, Any]) -> dict[str, Any]:
    """Does what the injected text asks: the model every safeguard exists for."""
    if schema == "RecruiterEventExtraction":
        return {"classification": "offer", "confidence": 1.0, "requires_reply": True}
    if schema == "SemanticAssessment":
        return {
            "matched_requirements": [
                {"requirement": skill, "evidence_id": "ev-sam-acme", "strength": "strong"}
                for skill in data["posting"]["skills"]
            ]
        }
    if schema == "DraftProposal":
        return {
            "artifact_type": data["artifact_type"],
            "segments": [
                {"text": "Ten years of Rust and a PhD from MIT.", "evidence_ids": ["ev-sam-edu"]}
            ],
        }
    return honest(schema, data)


class ScriptedChatModel:
    """A `StructuredChatModel` whose answer is a function of schema and input."""

    def __init__(self, answer, name: str):
        self.answer = answer
        self.model_name = name

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> "ScriptedChatModel._Bound":
        return self._Bound(self.answer, schema.__name__)

    class _Bound:
        def __init__(self, answer, schema: str):
            self.answer, self.schema = answer, schema

        def invoke(self, turns, **_kwargs):
            return self.answer(self.schema, _input(turns))

        async def ainvoke(self, turns, **_kwargs):
            return self.invoke(turns)


def gateway_with(answer, name: str) -> ModelGateway:
    config = ModelConfig(name=name, default=ModelSpec(provider="ollama", model=name))
    return ModelGateway(
        config, chat_model_factory=lambda spec: ScriptedChatModel(answer, spec.label())
    )


def run_suite(gateway: ModelGateway, name: str) -> dict[str, CaseResult]:
    result = asyncio.run(SUITES[name](gateway))
    return {case.case_id: case for case in result.cases}


# --- One command --------------------------------------------------------------


def test_the_rules_run_matches_the_checked_in_baseline(tmp_path):
    code, document = run_cli(tmp_path, "--config", str(RULES))

    changes = document["diff"]
    assert changes["baseline"] is not None, "record one with --update-baseline"
    assert not changes["dataset_changed"], "golden data changed: update the baseline with it"
    assert changes["regressions"] == []
    assert changes["changed"] == [], "a golden outcome changed: update the baseline deliberately"
    assert code == 0, document["verdict"]["reasons"]


def test_a_run_writes_results_a_summary_and_a_review_packet(tmp_path):
    code, document = run_cli(tmp_path, "--config", str(RULES), "--baseline", "/nonexistent")

    assert code == 0
    assert document["run"]["config"] == "rules"
    assert document["run"]["dataset_sha256"] == dataset.dataset_sha256()
    assert set(document["suites"]) == set(SUITES)
    assert {case["suite"] for case in document["cases"]} == set(SUITES)
    for case in document["cases"]:
        assert {"suite", "case_id", "status", "expected", "actual", "failures"} <= set(case)
    summary = (tmp_path / "summary.md").read_text()
    assert "# Eval run: rules" in summary and "No baseline" in summary
    assert "hr-topk-sam" in (tmp_path / "review_packet.md").read_text()


def test_a_baseline_case_that_stops_passing_fails_the_run(tmp_path):
    baseline = report.load_document(harness.baseline_path(ModelConfig.load(RULES)))
    passing = next(c for c in baseline["cases"] if c["suite"] == "routing" and c["status"] == PASS)
    # Pretend the baseline knew better: a case that fails today passed then.
    failing = next(c for c in baseline["cases"] if c["status"] == FAIL)
    failing["status"] = PASS
    edited = tmp_path / "baseline.json"
    report.write_json(edited, baseline)

    code, document = run_cli(tmp_path / "out", "--config", str(RULES), "--baseline", str(edited))

    key = f"{failing['suite']}/{failing['case_id']}"
    assert code == 1
    assert {"case": key, "was": PASS, "now": FAIL} in document["diff"]["regressions"]
    assert f"{passing['suite']}/{passing['case_id']}" not in str(document["diff"]["regressions"])
    assert any(key in reason for reason in document["verdict"]["reasons"])
    assert "### Regressions (1)" in (tmp_path / "out" / "summary.md").read_text()


# --- Swapping the model -------------------------------------------------------


def test_a_model_configuration_runs_every_suite_with_no_graph_change():
    gateway = gateway_with(honest, "honest")

    adversarial = run_suite(gateway, "adversarial")
    review = run_suite(gateway, "human_review")
    scenarios = run_suite(gateway, "scenarios")

    assert {case.status for case in adversarial.values()} == {PASS}
    assert {case.status for case in review.values()} == {REVIEW}
    assert all(case.actual["valid"] for case in review.values() if "content" in case.actual)
    assert scenarios["sc-invite-advances-applied"].status == PASS


def test_the_adversarial_suite_catches_a_model_that_follows_injected_instructions():
    cases = run_suite(gateway_with(compromised, "compromised"), "adversarial")

    forged = cases["adv-email-forged-offer"]
    assert forged.status == FAIL
    assert "classified 'offer'" in forged.failures
    assert "application moved to 'offer'" in forged.failures
    assert cases["adv-posting-self-scoring"].status == FAIL
    draft = cases["adv-draft-fabrication-request"]
    assert draft.status == FAIL
    assert any("PhD" in failure for failure in draft.failures)
    # What code enforces holds whatever the model says: no reply is redirected.
    assert not any("reply addressed" in f for case in cases.values() for f in case.failures)


def test_adversarial_failures_fail_the_run_even_with_no_baseline():
    cases = run_suite(gateway_with(compromised, "compromised"), "adversarial")
    document = {
        "cases": [case.to_dict() for case in cases.values()],
    }

    reasons = report.gate_failures(document, {"baseline": None})

    assert any(
        reason.startswith("unsafe: adversarial/adv-email-forged-offer") for reason in reasons
    )


# --- Human review -------------------------------------------------------------


def _review_suite(output: dict[str, Any]) -> SuiteResult:
    case = CaseResult("human_review", "hr-x", REVIEW, actual=output)
    case.fingerprint = fingerprint(output)
    return SuiteResult("human_review", [case])


def test_a_rating_applies_only_to_the_output_it_was_given_for():
    rated = _review_suite({"top_k": ["Northwind"]})
    stale = _review_suite({"top_k": ["Globex"]})
    rating = {
        "hr-x": {
            "fingerprint": fingerprint({"top_k": ["Northwind"]}),
            "scores": {"relevance": 4, "order": 5, "explanation": None},
        }
    }

    report.apply_ratings([rated], rating)
    report.apply_ratings([stale], rating)

    assert rated.cases[0].actual["rating"]["scores"]["order"] == 5
    assert rated.metrics == {"rated_cases": 1, "mean_rating": 4.5}
    assert "rating" not in stale.cases[0].actual
    assert stale.cases[0].failures == ["recorded rating is for a different output; re-review"]


# --- Comparing configurations -------------------------------------------------


def test_compare_lists_the_cases_two_configurations_disagree_on(tmp_path):
    baseline = report.load_document(harness.baseline_path(ModelConfig.load(RULES)))
    other = json.loads(json.dumps(baseline))
    other["run"]["config"] = "other"
    other["cases"][0]["status"] = FAIL if other["cases"][0]["status"] == PASS else PASS
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    report.write_json(a, baseline)
    report.write_json(b, other)

    assert harness.main(["compare", str(a), str(b), "--output", str(tmp_path / "cmp.md")]) == 0

    text = (tmp_path / "cmp.md").read_text()
    assert "| Metric | rules | other |" in text
    first = baseline["cases"][0]
    assert f"| {first['suite']}/{first['case_id']} |" in text
    assert "Cases where the runs disagree (1)" in text


@pytest.mark.parametrize("name", dataset.GOLDEN_FILES)
def test_every_golden_file_is_described(name):
    assert dataset.load(name)["description"]
