# Evaluations

The golden dataset and the harness that measures the agent against it. See
section 17 of `IMPLEMENTATION_GUIDE.md` for how this fits the system.

```
python -m evals.harness                       # run everything on the `rules` config
python -m evals.harness --config evals/model_benchmarks/configs/ollama-qwen2.5-7b.json
python -m evals.harness --suite adversarial   # one suite (repeatable)
python -m evals.harness --update-baseline     # accept this run as the config's baseline
python -m evals.harness compare evals/model_benchmarks/results/*/results.json
```

Exit status is 1 when a case that passes in the baseline no longer does, when
any adversarial case is unsafe, or when a case raises.

## Layout

| Path | What |
| --- | --- |
| `golden/*.json` | The dataset. Synthetic, hand-labelled, reviewed like code. |
| `model_benchmarks/configs/` | One `ModelConfig` per model to compare. |
| `model_benchmarks/baselines/` | The accepted run per config. Tracked. |
| `model_benchmarks/reviews/` | Human ratings per config, pinned to output fingerprints. Tracked. |
| `model_benchmarks/results/` | Each run's `results.json`, `summary.md`, `review_packet.md`. Not tracked. |
| `harness.py` | The CLI. |
| `suites.py` | One function per suite: run the cases, score them. |
| `world.py` | The compiled graphs a case runs on: gateway models, fake everything else. |
| `report.py` | Results document, baseline diff, Markdown. |

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
