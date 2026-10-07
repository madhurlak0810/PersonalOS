"""`event_classification_f1`: how well an extractor classifies recruiter emails.

Runs a `RecruiterEventExtractor` over the golden set in
`evals/golden/recruiter_emails.json` and reports per-class precision, recall
and F1, plus their unweighted mean -- macro F1, which is the headline number.
Macro rather than accuracy because the classes that matter most (an offer, an
interview invite) are the rare ones in a real inbox, and a metric weighted by
frequency would let a classifier that is only good at "general update" look fine.

The golden set is small and synthetic. It is a regression guard and a way to
compare extractors on the same messages, not an estimate of accuracy on real
mail.

    python -m evals.recruiter_event_classification          # deterministic rules
    python -m evals.recruiter_event_classification --llm    # Claude (needs the llm extra)
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from personalos.domain.job_search import RecruiterMessage
from personalos.domain.models import CommunicationEventClassification
from personalos.domain.recruiter_events import ExtractionOutcome
from personalos.models.recruiter_events import RuleBasedRecruiterEventExtractor

GOLDEN_PATH = Path(__file__).parent / "golden" / "recruiter_emails.json"

METRIC_NAME = "event_classification_f1"


class Extractor(Protocol):
    async def extract(self, message: RecruiterMessage) -> ExtractionOutcome: ...


@dataclass(frozen=True)
class GoldenEmail:
    message: RecruiterMessage
    expected: CommunicationEventClassification


@dataclass(frozen=True)
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    per_class: dict[CommunicationEventClassification, ClassScore]
    #: Macro F1 over every class in the vocabulary.
    event_classification_f1: float
    accuracy: float
    #: `(email id, expected, predicted)` for each miss.
    misses: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            METRIC_NAME: self.event_classification_f1,
            "accuracy": self.accuracy,
            "per_class": {label.value: vars(score) for label, score in self.per_class.items()},
            "misses": [list(miss) for miss in self.misses],
        }


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenEmail]:
    data = json.loads(path.read_text(encoding="utf-8"))
    received_at = datetime.fromisoformat(data["received_at"])
    return [
        GoldenEmail(
            message=RecruiterMessage(
                provider_message_id=email["id"],
                received_at=received_at,
                subject=email["subject"],
                from_address=email["from_address"],
                body=email["body"],
            ),
            expected=CommunicationEventClassification(email["expected"]),
        )
        for email in data["emails"]
    ]


def score(
    expected: list[CommunicationEventClassification],
    predicted: list[CommunicationEventClassification],
    ids: list[str] | None = None,
) -> ClassificationReport:
    """Per-class and macro F1 for paired labels. A class with no support scores 0."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must be the same length")
    per_class: dict[CommunicationEventClassification, ClassScore] = {}
    for label in CommunicationEventClassification:
        true_positive = sum(
            e is label and p is label for e, p in zip(expected, predicted, strict=True)
        )
        predicted_count = sum(p is label for p in predicted)
        support = sum(e is label for e in expected)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassScore(round(precision, 4), round(recall, 4), round(f1, 4), support)

    ids = ids or [str(index) for index in range(len(expected))]
    return ClassificationReport(
        per_class=per_class,
        event_classification_f1=round(
            sum(score.f1 for score in per_class.values()) / len(per_class), 4
        ),
        accuracy=round(
            (
                sum(e is p for e, p in zip(expected, predicted, strict=True)) / len(expected)
                if expected
                else 0.0
            ),
            4,
        ),
        misses=tuple(
            (email_id, e.value, p.value)
            for email_id, e, p in zip(ids, expected, predicted, strict=True)
            if e is not p
        ),
    )


async def evaluate(
    extractor: Extractor, golden: list[GoldenEmail] | None = None
) -> ClassificationReport:
    """Classify every golden email with `extractor` and score the result."""
    golden = golden if golden is not None else load_golden()
    predicted = [
        (await extractor.extract(email.message)).extraction.classification for email in golden
    ]
    return score(
        [email.expected for email in golden],
        predicted,
        [email.message.provider_message_id for email in golden],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="evaluate the Claude-backed extractor")
    args = parser.parse_args()

    extractor: Extractor
    if args.llm:
        from personalos.models.recruiter_events import anthropic_recruiter_event_extractor

        extractor = anthropic_recruiter_event_extractor()
    else:
        extractor = RuleBasedRecruiterEventExtractor()
    print(json.dumps(asyncio.run(evaluate(extractor)).to_dict(), indent=2))


if __name__ == "__main__":
    main()
