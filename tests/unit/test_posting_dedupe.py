"""The posting dedupe identity: what collapses, what does not, and that it fits."""

import pytest

from personalos.domain.job_search import canonical_company, canonical_text
from tests.fixtures import job_search_fakes as fakes


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": "otherboard", "source_job_id": "zz-9", "url": "https://other.test/9"},
        {"company": "ACME, Inc."},
        {"company": "Acme Corp."},
        {"title": "  senior   backend engineer "},
        {"description": "We need Python and Postgres experience"},
        {"description": "We need python\n\n• and postgres   experience."},
        {"location": "New York", "salary_max": 1, "skills": ()},
    ],
    ids=["provider-ids", "inc", "corp", "title-case", "desc-case", "desc-format", "other-fields"],
)
def test_the_same_opening_keeps_its_identity_across_cosmetic_differences(overrides):
    assert fakes.posting(**overrides).dedupe_key == fakes.posting().dedupe_key
    assert fakes.posting(**overrides).description_hash == fakes.posting().description_hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"company": "Acme Robotics"},
        {"title": "Staff Backend Engineer"},
        {"description": "We need rust experience."},
    ],
    ids=["company", "title", "description"],
)
def test_a_different_opening_gets_a_different_key(overrides):
    assert fakes.posting(**overrides).dedupe_key != fakes.posting().dedupe_key


def test_the_key_is_stable_readable_and_fits_its_column():
    key = fakes.posting().dedupe_key

    assert key == fakes.posting().dedupe_key
    assert key.startswith("acme|senior backend engineer|")
    assert len(key.rsplit("|", 1)[1]) == 64


def test_long_titles_never_truncate_the_digest():
    """Two long titles differing only at the end must not collide."""
    first = fakes.posting(title="Engineer " * 40 + "I")
    second = fakes.posting(title="Engineer " * 40 + "II")

    assert len(first.dedupe_key) <= 150
    assert first.dedupe_key != second.dedupe_key


def test_canonical_forms():
    assert canonical_text("  Hello, W​orld—again ") == "hello world again"
    assert canonical_company("Acme Holdings Co., Ltd.") == "acme holdings"
    # A company that is nothing but a suffix keeps its name.
    assert canonical_company("Limited") == "limited"
