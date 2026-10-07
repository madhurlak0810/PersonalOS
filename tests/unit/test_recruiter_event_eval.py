"""`event_classification_f1` on the golden recruiter emails.

The fixture the acceptance criterion asks for is `evals/golden/recruiter_emails.json`;
the metric is `evals.recruiter_event_classification`. What is pinned here is the
deterministic fallback's score, because it is the one extractor that can run
in CI: it is the floor under the model, and a rule change that drops it should
fail a test rather than be noticed in an inbox.
"""

from collections import Counter

import pytest

from evals.recruiter_event_classification import METRIC_NAME, evaluate, load_golden, score
from personalos.domain.models import CommunicationEventClassification as C
from personalos.models.recruiter_events import RuleBasedRecruiterEventExtractor

#: Measured 0.8386 when the rules were written. The floor sits below that so a
#: reworded rule is not a failure, and well above what a broken one scores.
FALLBACK_F1_FLOOR = 0.8


def test_the_golden_set_covers_every_classification_evenly():
    golden = load_golden()

    support = Counter(email.expected for email in golden)

    assert set(support) == set(C)
    assert len(set(support.values())) == 1
    assert len({email.message.provider_message_id for email in golden}) == len(golden)


async def test_fallback_event_classification_f1_meets_the_floor():
    report = await evaluate(RuleBasedRecruiterEventExtractor())

    assert report.to_dict()[METRIC_NAME] == report.event_classification_f1
    assert report.event_classification_f1 >= FALLBACK_F1_FLOOR, report.misses


def test_macro_f1_is_the_unweighted_mean_of_per_class_f1():
    expected = [C.OFFER, C.OFFER, C.REJECTION, C.REJECTION]
    predicted = [C.OFFER, C.REJECTION, C.REJECTION, C.REJECTION]

    report = score(expected, predicted)

    # offer: P=1, R=.5 -> .6667; rejection: P=.6667, R=1 -> .8; five classes score 0.
    assert report.per_class[C.OFFER].f1 == pytest.approx(0.6667, abs=1e-4)
    assert report.per_class[C.REJECTION].f1 == pytest.approx(0.8, abs=1e-4)
    assert report.event_classification_f1 == pytest.approx((0.6667 + 0.8) / len(C), abs=1e-4)
    assert report.accuracy == 0.75
    assert report.misses == (("1", "offer", "rejection"),)
