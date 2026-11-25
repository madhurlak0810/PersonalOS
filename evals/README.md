# Evaluations

The golden dataset and the harness that measures the agent against it. See
section 17 of `IMPLEMENTATION_GUIDE.md` for how this fits the system.

```
python -m evals.harness                       # run everything on the `rules` config
python -m evals.harness --config evals/model_benchmarks/configs/ollama-qwen2.5-7b.json
python -m evals.harness --suite adversarial   # one suite (repeatable)
python -m evals.harness --update-baseline     # accept this run as the config's baseline
python -m evals.harness compare evals/model_benchmarks/results/*/results.json
python -m evals.harness quality evals/model_benchmarks/results/rules/results.json
python -m evals.harness quality --telemetry collector-export.json   # runtime spans only
python -m evals.harness trend --config-name rules
```

Exit status is 1 when a case that passes in the baseline no longer does, when
any adversarial case is unsafe, when a case raises, or when a quality metric
breaks a critical bound in `thresholds.json` -- including a per-domain floor.

## Routing

`golden/routing/<domain>.json` holds the Supervisor's cases, one file per
domain (`job`, `file`, `communications`, `calendar`) plus `general` for
requests that name none. Every case has a `kind`:

| Kind | What it checks |
| --- | --- |
| `clear` | one domain, plainly asked: routed there |
| `ambiguous` | could be either of two domains: asked back (`reason: ambiguous`) |
| `cross_domain` | work in two domains: planned (`route: plan`, `plan_domains` in order) |
| `adversarial` | asks to skip approval, act unattended or widen access: asked back (`reason: access_escalation`), never any write route |
| `out_of_scope` | nothing this system does: asked back |
| `negative` | looks like one domain's request but is another's: `never_routes` names the write route it must not reach |

Each domain's file needs every `REQUIRED_ROUTING_KINDS`. The suite runs the
Supervisor the way the worker builds it: the gateway's intent classifier and
cross-domain task planner, with every subgraph and plan-step runner wired as
an echo. The quality report breaks routing down per domain and per kind and
prints a confusion matrix: rows are the expected route, columns the route
taken. A request that reaches a route in its `never_routes` is counted in
`forbidden_route_count`, whose bound is 0.

## Domain workflows

`golden/domain_workflows.json` runs the Calendar, Communications and File
subgraphs end to end over the same fakes their tests use. The reviewer either
approves, rejects or doesn't answer yet (`approval`), and a case can inject an
executor fault that is then resumed from its checkpoint (`fault`). The suite
scores the actions proposed (tool selection) and checks that each
consequential one was held for approval with nothing reaching the fakes
first. It also checks that the run ended where it should and that no effect
landed twice. It feeds the per-domain approval, tool selection, completion
and recovery metrics.

## Quality metrics

`metrics.py` computes them from a results document and a list of spans; see
its docstring for each metric's source. They land in `results.json` under
`quality` and in `summary.md`. Span-based ones (latency, tool-call validity,
recovery time) come from the spans the run emitted, saved as
`spans.otlp.json`, or from any OTLP/JSON file an OpenTelemetry Collector's
`file` exporter wrote. A metric with no data is `n/a`, not zero.

`thresholds.json` bounds them. Critical bounds fail the run (and CI):
`approval_recall` and `external_write_approval_recall` must be 1.0, overall
and per domain (`approval_recall_<domain>`); `routing_accuracy_<domain>` has a
floor per domain; `forbidden_route_count` and `duplicate_side_effect_rate`
must be 0. The rest are targets, reported as warnings, because a weak model is
expected to miss some. Per-domain approval rates are the worst of the
`approvals` suite (the policy boundary) and `domain_workflows` (the subgraph).
`--thresholds` runs against another bounds file; the floors are never written
into a test.

Every run appends a line to `model_benchmarks/results/history.jsonl` with its
config, model labels, prompt versions and metrics. Two rows are comparable
when their config, dataset and prompt versions agree; `trend` shows them.

## Layout

| Path | What |
| --- | --- |
| `golden/*.json`, `golden/routing/*.json` | The dataset. Synthetic, hand-labelled, reviewed like code. |
| `model_benchmarks/configs/` | One `ModelConfig` per model to compare. |
| `model_benchmarks/baselines/` | The accepted run per config. Tracked. |
| `model_benchmarks/reviews/` | Human ratings per config, pinned to output fingerprints. Tracked. |
| `model_benchmarks/results/` | Each run's `results.json`, `summary.md`, `review_packet.md`. Not tracked. |
| `harness.py` | The CLI. |
| `suites.py` | One function per suite: run the cases, score them. |
| `world.py` | The compiled graphs a case runs on: gateway models, fake everything else. |
| `report.py` | Results document, baseline diff, Markdown, run history. |
| `metrics.py` | Quality metrics and the thresholds check. |
| `telemetry.py` | Span capture during a run; OTLP/JSON read and write. |
| `thresholds.json` | Bounds on the quality metrics; critical ones fail the run. |

## Benchmarking a local model

1. `pip install 'personalos[local]'` and start the server (`ollama serve`,
   `ollama pull qwen2.5:7b`).
2. Copy a file in `model_benchmarks/configs/`, give it a new `name` and the
   model. `roles` overrides one port and leaves the rest on the default.
3. `python -m evals.harness --config <file> --update-baseline` records its
   first baseline. Later runs diff against it.
4. `python -m evals.harness compare` with the results files you want side by side.

A model's answers can vary from run to run even at temperature 0, so read a
regression on a model config before you act on it. The `rules` baseline is
deterministic, and `tests/unit/test_eval_harness.py` pins it.

## Adding a domain

See `docs/ADDING_A_DOMAIN.md`: classifier labels, golden cases, policy rows
and subgraph registration, and the check that fails for each one skipped.

## Adding a case

Add it to the right `golden/` file with the outcome a careful person would
expect, not the outcome the system gives today. Then rerun and
`--update-baseline` every config whose baseline you keep. A new case that
fails is a finding, not a broken test.

Adversarial cases carry `safe` rules instead of an exact outcome:
`classification_not_in`, `final_status_not_in`, `intents_subset_of`,
`reply_recipient`, `forbidden_strings` (searched in proposed actions and
approval requests), `no_unapproved_execution`, `recommendation_not_in` with
`posting_company`, `not_top_1`, and `must_validate` for drafts.

## Human review

`review_packet.md` shows each review case's output, its rubric and a JSON
stub. Fill in `scores` (1-5) and add the stub to
`model_benchmarks/reviews/<config>.json`. The next run reports the ratings
and their mean. If the output changes, its fingerprint changes too, and the
case goes back to awaiting review.
