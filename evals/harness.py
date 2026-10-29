"""Run the golden dataset against one model configuration, and diff it against a baseline.

    python -m evals.harness                                   # the deterministic `rules` config
    python -m evals.harness --config evals/model_benchmarks/configs/ollama-qwen2.5-7b.json
    python -m evals.harness --suite routing --suite adversarial
    python -m evals.harness --update-baseline                 # accept this run as the baseline
    python -m evals.harness compare RESULTS.json RESULTS.json # side by side
    python -m evals.harness quality [RESULTS.json] [--telemetry OTLP.json ...]
    python -m evals.harness trend [--config-name rules]       # metrics over past runs

A run writes, under `evals/model_benchmarks/results/<config>/`:

- `results.json`: machine-readable; every case's expected and actual outcome,
  each suite's metrics, the quality metrics (`evals.metrics`), and which
  configuration, models, prompt versions and dataset version ran;
- `summary.md`: the same for a person, with the diff against the baseline;
- `review_packet.md`: the human-review outputs and the rubric to rate them on;
- `spans.otlp.json`: every span the run emitted, which the latency, tool-call
  validity and recovery-time metrics are computed from.

and appends one line per run to `evals/model_benchmarks/results/history.jsonl`,
which `trend` reads.

The baseline is `evals/model_benchmarks/baselines/<config>.json`. Run this
before changing a model, a prompt or a configuration: it exits 1 when a case
that passed in the baseline no longer does, when any adversarial case is
unsafe, when a case raised, or when a quality metric breaks a critical bound
in `evals/thresholds.json` (approval recall, duplicate side effects). A
failure the baseline already had is reported but does not fail the run, so a
weak model can be benchmarked without first being made perfect.

`quality` recomputes the quality report from files on disk: a results file,
OTLP/JSON exports of runtime telemetry (an OpenTelemetry Collector's `file`
exporter), or both. It exits 1 on a critical breach too.

The configuration is the only input that names a model. Every graph is built
by `evals.world` through `personalos.models.gateway.ModelGateway`, so comparing
two models is two runs with two config files.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from personalos.models.gateway import ModelConfig, ModelGateway

from . import dataset, metrics, report, telemetry
from .suites import SUITES

BENCHMARKS = Path(__file__).parent / "model_benchmarks"
CONFIGS = BENCHMARKS / "configs"
DEFAULT_CONFIG = CONFIGS / "rules.json"
HISTORY = BENCHMARKS / "results" / "history.jsonl"


def baseline_path(config: ModelConfig) -> Path:
    return BENCHMARKS / "baselines" / f"{config.name}.json"


def results_dir(config: ModelConfig) -> Path:
    return BENCHMARKS / "results" / config.name


def ratings_path(config: ModelConfig) -> Path:
    return BENCHMARKS / "reviews" / f"{config.name}.json"


def _as_baseline(document: dict) -> dict:
    """A results document without the timings and span metrics, which differ every run."""
    return {
        **{key: value for key, value in document.items() if key != "quality"},
        "cases": [
            {key: value for key, value in case.items() if key != "duration_ms"}
            for case in document["cases"]
        ],
    }


async def run_suites(gateway: ModelGateway, names: list[str]):
    return [await SUITES[name](gateway) for name in names]


def run(args: argparse.Namespace) -> int:
    config = ModelConfig.load(args.config)
    gateway = ModelGateway(config)
    names = args.suite or list(SUITES)
    unknown = sorted(set(names) - set(SUITES))
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; choose from {', '.join(SUITES)}")
        return 2

    started_at, clock = report.now(), time.perf_counter()
    with telemetry.capture() as captured:
        results = asyncio.run(run_suites(gateway, names))
    spans = captured.spans()
    ratings_file = ratings_path(config)
    if ratings_file.exists():
        report.apply_ratings(results, json.loads(ratings_file.read_text(encoding="utf-8")))
    document = report.build_document(
        config,
        results,
        dataset_sha256=dataset.dataset_sha256(),
        started_at=started_at,
        duration_s=time.perf_counter() - clock,
    )

    baseline_file = Path(args.baseline) if args.baseline else baseline_path(config)
    baseline = report.load_document(baseline_file) if baseline_file.exists() else None
    if baseline is not None and args.suite:
        # A partial run is compared with the same suites of the baseline only.
        baseline = {
            **baseline,
            "cases": [case for case in baseline["cases"] if case["suite"] in names],
        }
    document["quality"] = quality_section(document, spans, args.thresholds)
    changes = report.diff(document, baseline)
    document["diff"] = changes
    gate = report.gate_failures(document, changes)
    document["verdict"] = {"passed": not gate, "reasons": gate}

    out = Path(args.output) if args.output else results_dir(config)
    report.write_json(out / "results.json", document)
    telemetry.write_otlp_json(out / "spans.otlp.json", spans)
    if args.history:
        history = Path(args.history)
    else:
        # A run sent somewhere else keeps its history there too.
        history = out / "history.jsonl" if args.output else HISTORY
    report.append_history(history, document)
    summary = report.summary_markdown(document, changes, gate)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "review_packet.md").write_text(report.review_packet_markdown(document), encoding="utf-8")

    if args.update_baseline:
        if args.suite:
            print("--update-baseline needs a full run; drop --suite")
            return 2
        stored = {key: value for key, value in document.items() if key not in ("diff", "verdict")}
        report.write_json(baseline_file, _as_baseline(stored))
        print(f"baseline written: {baseline_file}")

    print(summary)
    print(f"results: {out / 'results.json'}")
    if gate and not args.update_baseline:
        return 1
    return 0


def quality_section(document: dict | None, spans, thresholds: str | None) -> dict:
    """The quality metrics for a run, and the thresholds they break."""
    section = metrics.compute(document, spans)
    bounds = metrics.load_thresholds(thresholds or metrics.THRESHOLDS)
    section["breaches"] = metrics.check_thresholds(section["metrics"], bounds)
    return section


def quality(args: argparse.Namespace) -> int:
    if not args.results and not args.telemetry:
        print("give a results file, --telemetry, or both")
        return 2
    document = report.load_document(args.results) if args.results else None
    spans = [span for path in args.telemetry for span in telemetry.load_otlp_json(path)]
    if args.results and not args.telemetry:
        # A harness run keeps its spans beside its results.
        saved = Path(args.results).with_name("spans.otlp.json")
        if saved.exists():
            spans = telemetry.load_otlp_json(saved)
    section = quality_section(document, spans, args.thresholds)
    text = report.quality_markdown(section)
    if args.output:
        report.write_json(Path(args.output), section)
    print(text)
    return 1 if any(breach["critical"] for breach in section["breaches"]) else 0


def trend(args: argparse.Namespace) -> int:
    entries = report.load_history(Path(args.history) if args.history else HISTORY)
    if args.config_name:
        entries = [entry for entry in entries if entry.get("config") == args.config_name]
    names = args.metric or [
        "routing_accuracy",
        "external_write_approval_recall",
        "approval_precision",
        "duplicate_side_effect_rate",
        "interactive_p95_ms",
        "long_running_p95_ms",
    ]
    print(report.trend_markdown(entries[-args.limit :], names))
    return 0


def compare(args: argparse.Namespace) -> int:
    documents = [report.load_document(path) for path in args.results]
    text = report.compare_markdown(documents)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        parser = argparse.ArgumentParser(
            prog="python -m evals.harness compare", description="Compare results files."
        )
        parser.add_argument("results", nargs="+", help="results.json or baseline files")
        parser.add_argument("--output", help="also write the comparison to this file")
        return compare(parser.parse_args(argv[1:]))
    if argv[:1] == ["quality"]:
        parser = argparse.ArgumentParser(
            prog="python -m evals.harness quality",
            description="Quality metrics from a results file and/or OTLP/JSON telemetry.",
        )
        parser.add_argument("results", nargs="?", help="results.json from a harness run")
        parser.add_argument(
            "--telemetry", action="append", default=[], help="OTLP/JSON span export (repeatable)"
        )
        parser.add_argument("--thresholds", help="thresholds file (default: evals/thresholds.json)")
        parser.add_argument("--output", help="also write the quality section as JSON here")
        return quality(parser.parse_args(argv[1:]))
    if argv[:1] == ["trend"]:
        parser = argparse.ArgumentParser(
            prog="python -m evals.harness trend", description="Quality metrics across past runs."
        )
        parser.add_argument("--config-name", help="only runs of this configuration")
        parser.add_argument("--metric", action="append", help="metric column (repeatable)")
        parser.add_argument("--limit", type=int, default=20, help="most recent runs to show")
        parser.add_argument("--history", help="history file (default: results/history.jsonl)")
        return trend(parser.parse_args(argv[1:]))

    parser = argparse.ArgumentParser(
        prog="python -m evals.harness", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="model configuration JSON")
    parser.add_argument(
        "--suite", action="append", choices=list(SUITES), help="run only this suite (repeatable)"
    )
    parser.add_argument("--baseline", help="baseline results file (default: per config)")
    parser.add_argument("--output", help="directory for results (default: per config)")
    parser.add_argument(
        "--update-baseline", action="store_true", help="write this run as the new baseline"
    )
    parser.add_argument("--thresholds", help="thresholds file (default: evals/thresholds.json)")
    parser.add_argument(
        "--history", help="history file to append to (default: results/history.jsonl)"
    )
    parser.add_argument("--verbose", action="store_true", help="show warnings from the system")
    args = parser.parse_args(argv)
    previous = logging.root.manager.disable
    if not args.verbose:
        # Fallbacks log a warning per case by design (a declining assessor,
        # an invalid model answer); the results file records each one.
        logging.disable(logging.WARNING)
    try:
        return run(args)
    finally:
        logging.disable(previous)


if __name__ == "__main__":
    sys.exit(main())
