"""Golden set: each job board adapter's normalization output, field for field.

`evals/golden/provider_normalization.json` holds, per provider, recorded-shape
API responses and the canonical `NormalizedPosting` JSON each listed posting
must become. A change to an adapter's mapping, to HTML flattening, or to the
dedupe identity shows up here as a diff against reviewed output.

To accept an intended change, regenerate and review the diff like code:

    python -m tests.unit.test_provider_golden --update
"""

import asyncio
import json
import sys
from uuid import UUID

import pytest

from personalos.domain.job_search import SearchProfile
from tests.fixtures.job_provider_contract import (
    GOLDEN_PATH,
    PROVIDERS,
    build,
    canonical,
    case,
    golden,
    normalize,
)

#: A profile with no roles or keywords: every listed posting is returned.
EVERYTHING = SearchProfile(user_id=UUID(int=0))

SHARED_TITLE = "Senior Backend Engineer"
INJECTED_TITLE = "Backend Engineer, Payments"


async def normalized_output(provider: str) -> list[dict]:
    raws = await build(provider).provider.search(EVERYTHING)
    return [canonical(posting) for posting in normalize(list(raws))]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_normalization_matches_the_golden_output(provider):
    expected = case(provider)["expected"]

    assert await normalized_output(provider) == expected


def test_the_same_opening_has_one_identity_on_every_board():
    """Cross-posted in four markups, the shared opening is one dedupe key."""
    providers = golden()["providers"]
    shared = {
        name: [p for p in entry["expected"] if p["title"] == SHARED_TITLE]
        for name, entry in providers.items()
    }

    assert all(len(found) == 1 for found in shared.values()), shared
    assert len({found[0]["dedupe_key"] for found in shared.values()}) == 1
    assert len({found[0]["description_hash"] for found in shared.values()}) == 1
    # ...while each board keeps its own provenance.
    assert len({(found[0]["source"], found[0]["source_job_id"]) for found in shared.values()}) == 4


def test_injected_instructions_are_normalized_as_plain_description_text():
    document = golden()
    for name, entry in document["providers"].items():
        (injected,) = [p for p in entry["expected"] if p["title"] == INJECTED_TITLE]
        assert document["injection"] in injected["description"], name
        assert "evil.test" not in (injected["url"] or ""), name
        assert "evil.test" not in injected["dedupe_key"], name


def test_dropped_postings_stay_dropped():
    """Malformed, unlisted and taken-down postings are not in any expected output."""
    titles = {
        name: {p["title"] for p in entry["expected"]}
        for name, entry in golden()["providers"].items()
    }

    assert "Secret Project Lead" not in titles["ashby"]  # isListed: false
    assert "Retired Role" not in titles["workday"]  # posted: false
    assert "Escape" not in titles["workday"]  # path outside job/
    assert len(titles["greenhouse"]) == 3  # the object-valued title is gone


async def _update() -> None:
    document = golden()
    for provider in PROVIDERS:
        document["providers"][provider]["expected"] = await normalized_output(provider)
    GOLDEN_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit("usage: python -m tests.unit.test_provider_golden --update")
    asyncio.run(_update())
    print(f"rewrote {GOLDEN_PATH}")
