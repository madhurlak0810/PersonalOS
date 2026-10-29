"""Results files, baseline diffs and the human-readable summary.

A run produces one results document (`results.json`): which configuration
and dataset version ran, each suite's metrics and counts, and every case's
expected and actual outcome. Everything else -- the diff against a baseline,
`summary.md`, the reviewer's packet, a comparison across configurations -- is
computed from results documents, so any of them can be regenerated from files
already on disk.

A baseline is simply an earlier results document for the same configuration,
checked in under `evals/model_benchmarks/baselines/`.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personalos.models.gateway import ModelConfig, prompt_versions

from .metrics import INTERACTIVE_STEPS
from .suites import ERROR, FAIL, PASS, REVIEW, SKIPPED, SuiteResult, fingerprint

SCHEMA_VERSION = 1

#: Suites where any failure fails the run, baseline or not: an unsafe outcome
#: is never an accepted regression.
GATING_SUITES = ("adversarial",)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def apply_ratings(results: list[SuiteResult], ratings: dict[str, Any]) -> None:
    """Attach recorded human ratings to the review cases whose output they rated.

    A rating applies only while the output's fingerprint matches the one the
    reviewer saw; a changed output goes back to awaiting review.
    """
    for suite in results:
        for case in suite.cases:
            if case.status != REVIEW:
                continue
            rating = ratings.get(case.case_id)
            if rating and rating.get("fingerprint") == case.fingerprint:
                case.actual = {**case.actual, "rating": rating}
            elif rating:
                case.failures.append("recorded rating is for a different output; re-review")
        if suite.name == "human_review":
            scores = [
                value
                for case in suite.cases
                if isinstance(case.actual, dict) and "rating" in case.actual
                for value in case.actual["rating"].get("scores", {}).values()
                if isinstance(value, (int, float))
            ]
            rated = sum(
                isinstance(case.actual, dict) and "rating" in case.actual for case in suite.cases
            )
            suite.metrics["rated_cases"] = rated
            suite.metrics["mean_rating"] = round(sum(scores) / len(scores), 3) if scores else 0.0


def build_document(
    config: ModelConfig,
    results: list[SuiteResult],
    *,
    dataset_sha256: str,
    started_at: datetime,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "config": config.name,
            "description": config.description,
            "models": config.resolved(),
            "prompt_versions": prompt_versions(config),
            "config_sha256": fingerprint(config.model_dump(mode="json")),
            "dataset_sha256": dataset_sha256,
            "git_commit": _git_commit(),
            "started_at": started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 2),
        },
        "suites": {
            suite.name: {"metrics": suite.metrics, "counts": suite.counts()} for suite in results
        },
        "cases": [case.to_dict() for suite in results for case in suite.cases],
    }


def load_document(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Diff -------------------------------------------------------------------


def _key(case: dict[str, Any]) -> str:
    return f"{case['suite']}/{case['case_id']}"


def _status_flips(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list]:
    """Cases in both runs, sorted into regressions, fixes and other changes."""
    scored = (PASS, REVIEW)
    flips: dict[str, list] = {"regressions": [], "fixes": [], "changed": []}
    for key, case in after.items():
        old = before.get(key)
        if old is None:
            continue
        item = {"case": key, "was": old["status"], "now": case["status"]}
        if old["status"] in scored and case["status"] not in scored + (SKIPPED,):
            flips["regressions"].append(item)
        elif old["status"] not in scored and case["status"] in scored:
            flips["fixes"].append(item)
        elif old["status"] != case["status"] or (
            # A review case's output is judged by its reviewer, not diffed.
            old.get("actual") != case.get("actual")
            and case["suite"] != "human_review"
        ):
            flips["changed"].append(item)
    return flips


def _metric_deltas(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for suite, data in current.items():
        old_metrics = baseline.get(suite, {}).get("metrics", {})
        for name, value in data["metrics"].items():
            if name in old_metrics and isinstance(value, (int, float)):
                delta = round(value - old_metrics[name], 4)
                if delta:
                    deltas[f"{suite}.{name}"] = {
                        "was": old_metrics[name],
                        "now": value,
                        "delta": delta,
                    }
    return deltas


def diff(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """What changed since `baseline`: case status flips and metric deltas."""
    if baseline is None:
        return {"baseline": None}
    before = {_key(case): case for case in baseline["cases"]}
    after = {_key(case): case for case in current["cases"]}
    flips = _status_flips(before, after)
    metrics = _metric_deltas(current["suites"], baseline["suites"])

    return {
        "baseline": {
            "config": baseline["run"]["config"],
            "git_commit": baseline["run"].get("git_commit"),
            "started_at": baseline["run"].get("started_at"),
        },
        "dataset_changed": baseline["run"]["dataset_sha256"] != current["run"]["dataset_sha256"],
        "config_changed": baseline["run"]["config_sha256"] != current["run"]["config_sha256"],
        **flips,
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "metrics": metrics,
    }


def gate_failures(document: dict[str, Any], changes: dict[str, Any]) -> list[str]:
    """Why this run should fail a pre-change check, if it should."""
    reasons = [
        f"regression: {item['case']} ({item['was']} -> {item['now']})"
        for item in changes.get("regressions", ())
    ]
    reasons += [
        f"threshold: {breach['metric']} {breach['reason']}"
        for breach in (document.get("quality") or {}).get("breaches", ())
        if breach["critical"]
    ]
    for case in document["cases"]:
        if case["suite"] in GATING_SUITES and case["status"] in (FAIL, ERROR):
            reasons.append(f"unsafe: {_key(case)}: {'; '.join(case['failures'])}")
        elif case["status"] == ERROR:
            reasons.append(f"error: {_key(case)}: {'; '.join(case['failures'])}")
    return list(dict.fromkeys(reasons))


# --- Markdown ---------------------------------------------------------------

#: The metric shown first for each suite.
HEADLINE = {
    "routing": "route_accuracy",
    "recruiter_events": "event_classification_f1",
    "job_matching": "recommendation_accuracy",
    "scenarios": "pass_rate",
    "adversarial": "safe_rate",
    "human_review": "automatic_pass_rate",
    "approvals": "external_write_approval_recall",
    "recovery": "recovery_rate",
}


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _quality_lines(document: dict[str, Any]) -> list[str]:
    quality = document.get("quality")
    return ["", quality_markdown(quality, heading="##")] if quality else []


def summary_markdown(document: dict[str, Any], changes: dict[str, Any], gate: list[str]) -> str:
    run = document["run"]
    lines = [
        f"# Eval run: {run['config']}",
        "",
        f"- **Verdict:** {'FAIL' if gate else 'PASS'}",
        f"- **Commit:** {run.get('git_commit') or 'unknown'} · **started** {run['started_at']} · "
        f"**took** {run['duration_s']}s",
        f"- **Dataset:** `{run['dataset_sha256'][:12]}` · **config:** `{run['config_sha256'][:12]}`",
        "- **Models:** "
        + ", ".join(f"{role} = `{label}`" for role, label in run["models"].items()),
        "",
        "## Suites",
        "",
        "| Suite | Headline | Value | Δ vs baseline | Pass | Fail | Error | Skipped | Review |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    metric_changes = changes.get("metrics", {})
    for name, data in document["suites"].items():
        headline = HEADLINE.get(name, "pass_rate")
        delta = metric_changes.get(f"{name}.{headline}", {}).get("delta")
        counts = data["counts"]
        lines.append(
            f"| {name} | {headline} | {_fmt(data['metrics'].get(headline, 0.0))} | "
            f"{'' if delta is None else f'{delta:+.3f}'} | {counts[PASS]} | {counts[FAIL]} | "
            f"{counts[ERROR]} | {counts[SKIPPED]} | {counts[REVIEW]} |"
        )

    lines += _quality_lines(document)

    lines += ["", "## Against the baseline", ""]
    if changes.get("baseline") is None:
        lines.append("No baseline for this configuration. Record one with `--update-baseline`.")
    else:
        base = changes["baseline"]
        lines.append(
            f"Baseline from {base.get('started_at')} at commit {base.get('git_commit') or 'unknown'}."
        )
        if changes["dataset_changed"]:
            lines.append(
                "**The golden dataset changed since the baseline**: added or edited cases are not regressions."
            )
        if changes["config_changed"]:
            lines.append("**The model configuration changed since the baseline.**")
        for title, items in (("Regressions", changes["regressions"]), ("Fixes", changes["fixes"])):
            lines += ["", f"### {title} ({len(items)})", ""]
            lines += [f"- `{i['case']}`: {i['was']} → {i['now']}" for i in items] or ["None."]
        if changes["changed"]:
            lines += ["", f"### Other changed outcomes ({len(changes['changed'])})", ""]
            lines += [f"- `{i['case']}`: {i['was']} → {i['now']}" for i in changes["changed"]]
        if changes["added"] or changes["removed"]:
            lines += [
                "",
                f"Added cases: {len(changes['added'])} · removed: {len(changes['removed'])}",
            ]
        if metric_changes:
            lines += [
                "",
                "### Metric changes",
                "",
                "| Metric | Was | Now | Δ |",
                "| --- | --- | --- | --- |",
            ]
            lines += [
                f"| {name} | {_fmt(m['was'])} | {_fmt(m['now'])} | {m['delta']:+.4f} |"
                for name, m in metric_changes.items()
            ]

    if gate:
        lines += ["", "## Why the run failed", ""] + [f"- {reason}" for reason in gate]

    failing = [case for case in document["cases"] if case["status"] in (FAIL, ERROR)]
    lines += ["", f"## Failing cases ({len(failing)})", ""]
    lines += [
        f"- `{_key(case)}` ({case['status']}): {'; '.join(case['failures'])}" for case in failing
    ] or ["None."]

    review = [case for case in document["cases"] if case["status"] == REVIEW]
    awaiting = [case for case in review if "rating" not in (case.get("actual") or {})]
    lines += [
        "",
        "## Human review",
        "",
        f"{len(review)} case(s) for review, {len(awaiting)} awaiting a rating. "
        "Outputs and rubrics are in `review_packet.md`.",
        "",
    ]
    return "\n".join(lines)


#: Quality metrics in report order, in groups.
QUALITY_GROUPS = {
    "Routing and tools": (
        "routing_accuracy",
        "tool_call_validity_rate",
        "tool_selection_accuracy",
    ),
    "Approvals": (
        "approval_precision",
        "approval_recall",
        "external_write_approval_recall",
    ),
    "Workflows": (
        "workflow_completion_rate",
        "recovery_rate",
        "mean_recovery_time_ms",
        "duplicate_side_effect_rate",
    ),
    "Domain quality": (
        "job_match_precision_at_k",
        "event_classification_f1",
        "resume_evidence_grounding",
    ),
    "Latency": (
        "interactive_p50_ms",
        "interactive_p95_ms",
        "long_running_p50_ms",
        "long_running_p95_ms",
        "model_p50_ms",
        "model_p95_ms",
        "model_tokens_per_sec",
    ),
}


def _fmt_metric(value: Any) -> str:
    return "n/a" if value is None else _fmt(value)


def quality_markdown(quality: dict[str, Any], *, heading: str = "#") -> str:
    """The quality section: every metric, its threshold status, and latency per step."""
    metrics = quality["metrics"]
    breaches = {breach["metric"]: breach for breach in quality.get("breaches", ())}
    lines = [f"{heading} Quality metrics", ""]
    critical = [b for b in breaches.values() if b["critical"]]
    warnings = [b for b in breaches.values() if not b["critical"]]
    lines.append(
        f"{len(critical)} critical breach(es), {len(warnings)} warning(s); "
        f"{quality.get('spans', 0)} span(s) read."
    )
    lines += ["", "| Group | Metric | Value | Threshold |", "| --- | --- | --- | --- |"]
    for group, names in QUALITY_GROUPS.items():
        for name in names:
            breach = breaches.get(name)
            status = ""
            if breach:
                status = f"{'**FAIL**' if breach['critical'] else 'warn'}: {breach['reason']}"
            lines.append(f"| {group} | {name} | {_fmt_metric(metrics.get(name))} | {status} |")

    latency = quality.get("latency") or {}
    if latency.get("steps"):
        lines += [
            "",
            f"{heading}# Latency by step",
            "",
            "| Step | Kind | Runs | p50 ms | p95 ms |",
            "| --- | --- | --- | --- | --- |",
        ]
        for step, data in latency["steps"].items():
            kind = "interactive" if step in INTERACTIVE_STEPS else "long-running"
            lines.append(
                f"| {step} | {kind} | {data['count']} | {_fmt_metric(data['p50_ms'])} | "
                f"{_fmt_metric(data['p95_ms'])} |"
            )
        by_model = latency.get("model", {}).get("by_model") or {}
        if by_model:
            lines += [
                "",
                "| Model | Calls | p50 ms | p95 ms | Tokens/s |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| {name} | {data['count']} | {_fmt_metric(data['p50_ms'])} | "
                f"{_fmt_metric(data['p95_ms'])} | {_fmt_metric(data['tokens_per_sec'])} |"
                for name, data in by_model.items()
            ]

    approvals = quality.get("approvals") or {}
    findings = [
        f"- ungated consequential action: `{action}`"
        for action in approvals.get("ungated_consequential") or ()
    ] + [
        f"- approval asked for a harmless action: `{action}`"
        for action in approvals.get("unneeded_prompts") or ()
    ]
    if findings:
        lines += ["", f"{heading}# Approval findings", "", *findings]
    return "\n".join(lines) + "\n"


# --- History ----------------------------------------------------------------


def history_entry(document: dict[str, Any]) -> dict[str, Any]:
    """One line of the run history: what ran, on which models and prompts, and the metrics."""
    run = document["run"]
    return {
        "started_at": run.get("started_at"),
        "git_commit": run.get("git_commit"),
        "config": run.get("config"),
        "config_sha256": run.get("config_sha256"),
        "dataset_sha256": run.get("dataset_sha256"),
        "models": run.get("models"),
        "prompt_versions": run.get("prompt_versions"),
        "passed": (document.get("verdict") or {}).get("passed"),
        "metrics": (document.get("quality") or {}).get("metrics", {}),
    }


def append_history(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_entry(document), ensure_ascii=False) + "\n")


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def trend_markdown(entries: list[dict[str, Any]], metrics: list[str]) -> str:
    """One row per run, oldest first, with the versions that make two rows comparable."""
    lines = [
        "# Quality trend",
        "",
        "Rows are comparable only where config, dataset and prompt versions agree.",
        "",
        "| Started | Commit | Config | Dataset | Prompts | " + " | ".join(metrics) + " |",
        "| --- | --- | --- | --- | --- |" + " --- |" * len(metrics),
    ]
    for entry in entries:
        prompts = fingerprint(entry.get("prompt_versions") or {})[:8]
        values = [_fmt_metric(entry["metrics"].get(name)) for name in metrics]
        lines.append(
            f"| {entry.get('started_at')} | {entry.get('git_commit') or '-'} | "
            f"{entry.get('config')} | {(entry.get('dataset_sha256') or '')[:8]} | {prompts} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def review_packet_markdown(document: dict[str, Any]) -> str:
    """Each review case's output and rubric, plus the JSON to record a rating with."""
    lines = [
        f"# Review packet: {document['run']['config']}",
        "",
        "Rate each case 1-5 per rubric line and record it in "
        f"`evals/model_benchmarks/reviews/{document['run']['config']}.json` with the "
        "fingerprint shown. A rating is used only while the output keeps that fingerprint.",
        "",
    ]
    for case in document["cases"]:
        if case["suite"] != "human_review" or case["status"] == SKIPPED:
            continue
        actual = case.get("actual") or {}
        lines += [f"## {case['case_id']}", "", f"Status: {case['status']}"]
        if case["failures"]:
            lines.append(f"Automatic checks: {'; '.join(case['failures'])}")
        if "top_k" in actual:
            lines += [
                "",
                "| # | Company | Title | Score | Rec. | Matched | Missing |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for rank, item in enumerate(actual["top_k"], 1):
                lines.append(
                    f"| {rank} | {item['company']} | {item['title']} | {item['score']} | "
                    f"{item['recommendation']} | {', '.join(item['matched'])} | "
                    f"{', '.join(item['missing'])} |"
                )
        elif "content" in actual:
            lines += ["", "```text", actual["content"], "```"]
        rubric = (case.get("expected") or {}).get("rubric", [])
        lines += ["", "Rubric:", *[f"- {item}" for item in rubric], ""]
        template = {
            case["case_id"]: {
                "fingerprint": case.get("fingerprint"),
                "reviewer": "",
                "scores": {item.split(":", 1)[0]: None for item in rubric},
                "notes": "",
            }
        }
        lines += ["```json", json.dumps(template, indent=2), "```", ""]
    return "\n".join(lines)


def compare_markdown(documents: list[dict[str, Any]]) -> str:
    """Side by side: every suite metric and every case status, one column per run."""
    names = [doc["run"]["config"] for doc in documents]
    lines = ["# Model comparison", "", "| Metric | " + " | ".join(names) + " |"]
    lines.append("| --- |" + " --- |" * len(names))
    metric_names = dict.fromkeys(
        f"{suite}.{metric}"
        for doc in documents
        for suite, data in doc["suites"].items()
        for metric in data["metrics"]
    )
    for full in metric_names:
        suite, metric = full.split(".", 1)
        row = [
            _fmt(doc["suites"].get(suite, {}).get("metrics", {}).get(metric, ""))
            for doc in documents
        ]
        lines.append(f"| {full} | " + " | ".join(row) + " |")

    statuses = [{_key(case): case["status"] for case in doc["cases"]} for doc in documents]
    keys = list(dict.fromkeys(key for status in statuses for key in status))
    disagreements = [key for key in keys if len({status.get(key) for status in statuses}) > 1]
    lines += ["", f"## Cases where the runs disagree ({len(disagreements)})", ""]
    if disagreements:
        lines += ["| Case | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        lines += [
            f"| {key} | " + " | ".join(status.get(key, "-") for status in statuses) + " |"
            for key in disagreements
        ]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "SCHEMA_VERSION",
    "GATING_SUITES",
    "apply_ratings",
    "build_document",
    "load_document",
    "write_json",
    "diff",
    "gate_failures",
    "summary_markdown",
    "quality_markdown",
    "history_entry",
    "append_history",
    "load_history",
    "trend_markdown",
    "review_packet_markdown",
    "compare_markdown",
    "now",
]
