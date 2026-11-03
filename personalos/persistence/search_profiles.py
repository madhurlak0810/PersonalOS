"""The candidate's targeting profile, read from `candidate_profiles`.

`SqlSearchProfileStore` is what `JobSearchGraph`'s `SearchProfileStore` port is
bound to. Roles and locations have their own columns; everything else a
`SearchProfile` can say -- keywords, must-have skills, excluded companies,
salary bounds -- lives in `preferences` under the field's own name. A key that
is not a `SearchProfile` field is ignored rather than refused, so a profile
written by a newer version does not stop an older worker from searching.
"""

import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.job_search import SearchProfile
from personalos.persistence.models import CandidateProfileModel
from personalos.persistence.repositories import CandidateProfileRepository

logger = logging.getLogger(__name__)

#: Fields taken from their own columns, never from `preferences`.
_COLUMN_FIELDS = frozenset({"user_id", "profile_version", "target_roles", "target_locations"})


def search_profile_from_row(row: CandidateProfileModel) -> SearchProfile:
    """The `SearchProfile` one stored profile version describes."""
    preferences = {
        key: value
        for key, value in (row.preferences or {}).items()
        if key in SearchProfile.model_fields and key not in _COLUMN_FIELDS
    }
    return SearchProfile(
        user_id=row.user_id,
        profile_version=row.profile_version,
        target_roles=tuple(row.target_roles or ()),
        target_locations=tuple(row.target_locations or ()),
        **preferences,
    )


class SqlSearchProfileStore:
    """Binds the graph's `SearchProfileStore` port to the latest profile version.

    A candidate with no profile on file is searched for with an empty one --
    no criteria, so nothing is filtered out -- and the run says so in the log.
    Refusing would leave a first-time user unable to search at all; the
    inline query a request carries is folded into the profile by the graph
    either way.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    async def load(self, user_id: UUID) -> SearchProfile:
        """Return the candidate's highest-numbered profile version."""
        session = self.session_factory()
        try:
            row = CandidateProfileRepository(session).get_latest_by_user_id(user_id)
            if row is None:
                logger.info(
                    "no candidate profile on file for user %s; searching with an empty one",
                    user_id,
                )
                return SearchProfile(user_id=user_id)
            return search_profile_from_row(row)
        finally:
            session.close()


__all__ = ["SqlSearchProfileStore", "search_profile_from_row"]
