"""The candidate profile as a person edits it: one shape for the API and the CLI.

`CandidateProfileInput` is what a caller sends to create or update a profile,
and what reading one gives back under `profile`, so a profile read from either
surface can be sent back unchanged. Unknown fields are refused rather than
dropped: a misspelled `target_role` silently ignored would leave the search
running on criteria the person thinks they changed.

Profiles are versioned, never edited in place. `job_matches` record the
`profile_version` they were scored against, so an update writes version N+1
and leaves N exactly as it was.

**Scoping.** A profile belongs to the actor that wrote it. Over the API the
actor is the one the request's bearer token verified as (`apps.api.auth`); on
the CLI it is `--actor`. `actor_user_id` turns it into the candidate id the
profile and the job search are keyed by. This is scoping, not access control:
which actor may do what is the follow-up authorization phase.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from personalos.domain.job_search import (
    DEFAULT_MIN_SCORE,
    MAX_SHORTLIST_SIZE,
    EmploymentType,
    SeniorityLevel,
    WorkMode,
)

#: Namespace `actor_user_id` derives candidate ids in. Fixed forever: changing
#: it would detach every actor from the profiles they already wrote.
ACTOR_USER_NAMESPACE = UUID("e4fccdeb-51fc-4062-8642-ca67b3e7afb1")

#: Bounds on what one profile may hold.
MAX_PROFILE_ITEMS = 50
MAX_PROFILE_TEXT = 200

_Text = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROFILE_TEXT)
]
_Texts = Annotated[list[_Text], Field(default_factory=list, max_length=MAX_PROFILE_ITEMS)]


def actor_user_id(actor_id: str) -> UUID:
    """The candidate id an actor's profile and job searches are keyed by.

    An actor that is already a UUID is its own candidate id, so a caller that
    knows the candidate's id can name it directly; any other actor string maps
    to a stable id derived from it.
    """
    try:
        return UUID(actor_id)
    except ValueError:
        return uuid5(ACTOR_USER_NAMESPACE, actor_id)


class CandidateProfileInput(BaseModel):
    """What the candidate is looking for, as they state it."""

    model_config = ConfigDict(extra="forbid")

    target_roles: _Texts
    target_locations: _Texts
    #: Remote alone makes the search remote-only.
    work_modes: list[WorkMode] = Field(default_factory=list, max_length=len(WorkMode))
    seniority_levels: list[SeniorityLevel] = Field(
        default_factory=list, max_length=len(SeniorityLevel)
    )
    employment_types: list[EmploymentType] = Field(
        default_factory=list, max_length=len(EmploymentType)
    )
    requires_visa_sponsorship: bool = False
    #: Anything else that rules a role out ("no relocation", "no on-call").
    constraints: _Texts
    keywords: _Texts
    must_have_skills: _Texts
    excluded_companies: _Texts
    salary_min: int | None = Field(default=None, ge=0)
    salary_max: int | None = Field(default=None, ge=0)
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0.0, le=1.0)
    max_shortlist: int = Field(default=5, ge=1, le=MAX_SHORTLIST_SIZE)

    @model_validator(mode="after")
    def _salary_range_in_order(self) -> "CandidateProfileInput":
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_max < self.salary_min
        ):
            raise ValueError(
                f"salary_max ({self.salary_max}) cannot be below salary_min ({self.salary_min})"
            )
        return self


class CandidateProfileVersion(BaseModel):
    """One stored version of an actor's profile."""

    user_id: UUID
    profile_version: int
    created_at: datetime
    profile: CandidateProfileInput


class CandidateProfileVersions(BaseModel):
    """Every stored version of an actor's profile, newest first."""

    user_id: UUID
    versions: list[CandidateProfileVersion]


__all__ = [
    "ACTOR_USER_NAMESPACE",
    "MAX_PROFILE_ITEMS",
    "MAX_PROFILE_TEXT",
    "actor_user_id",
    "CandidateProfileInput",
    "CandidateProfileVersion",
    "CandidateProfileVersions",
]
