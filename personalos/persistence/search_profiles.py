"""The candidate's targeting profile, kept as versions in `candidate_profiles`.

`SqlSearchProfileStore` is what `JobSearchGraph`'s `SearchProfileStore` port is
bound to, and what the profile API and CLI write through. Roles and locations
have their own columns; everything else a `CandidateProfileInput` can say --
work modes, levels, visa, keywords, salary bounds -- lives in `preferences`
under the field's own name. A key that is not a `SearchProfile` field is
ignored rather than refused, so a profile written by a newer version does not
stop an older worker from searching.

Writes never touch an existing row: `create` writes version 1, `update` writes
the next version, and two updates racing for the same number are settled by
`uq_candidate_profiles_user_version` -- the loser is a `Conflict`, not a
silently overwritten version.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.errors import Conflict, NotFound
from personalos.domain.job_search import SearchProfile, WorkMode
from personalos.domain.search_profiles import (
    CandidateProfileInput,
    CandidateProfileVersion,
    CandidateProfileVersions,
)
from personalos.persistence.models import CandidateProfileModel, UserModel
from personalos.persistence.repositories import CandidateProfileRepository

logger = logging.getLogger(__name__)

#: Fields taken from their own columns, never from `preferences`.
_COLUMN_FIELDS = frozenset({"user_id", "profile_version", "target_roles", "target_locations"})


def _preferences(row: CandidateProfileModel) -> dict:
    """The row's `preferences`, with `remote_only` read off its work modes.

    Rows written before work modes existed say `remote_only` directly; rows
    written since say `work_modes`, and remote alone means remote-only.
    """
    preferences = dict(row.preferences or {})
    if preferences.get("work_modes") == [WorkMode.REMOTE.value]:
        preferences.setdefault("remote_only", True)
    elif preferences.get("remote_only") and "work_modes" not in preferences:
        preferences["work_modes"] = [WorkMode.REMOTE.value]
    return preferences


def search_profile_from_row(row: CandidateProfileModel) -> SearchProfile:
    """The `SearchProfile` one stored profile version describes."""
    preferences = {
        key: value
        for key, value in _preferences(row).items()
        if key in SearchProfile.model_fields and key not in _COLUMN_FIELDS
    }
    return SearchProfile(
        user_id=row.user_id,
        profile_version=row.profile_version,
        target_roles=tuple(row.target_roles or ()),
        target_locations=tuple(row.target_locations or ()),
        **preferences,
    )


def profile_version_from_row(row: CandidateProfileModel) -> CandidateProfileVersion:
    """The editable shape of one stored profile version."""
    preferences = {
        key: value
        for key, value in _preferences(row).items()
        if key in CandidateProfileInput.model_fields and key not in _COLUMN_FIELDS
    }
    return CandidateProfileVersion(
        user_id=row.user_id,
        profile_version=row.profile_version,
        created_at=row.created_at,
        profile=CandidateProfileInput(
            target_roles=list(row.target_roles or ()),
            target_locations=list(row.target_locations or ()),
            **preferences,
        ),
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

    # --- Editing ------------------------------------------------------------

    def ensure_user(self, user_id: UUID, display_name: str | None = None) -> None:
        """Make sure `users` has a row for this candidate.

        Profiles and workflow runs both reference `users.id`; a candidate id
        derived from an actor has no row until something writes one.
        """
        session = self.session_factory()
        try:
            self._ensure_user(session, user_id, display_name)
            session.commit()
        finally:
            session.close()

    def create(
        self, user_id: UUID, profile: CandidateProfileInput, *, display_name: str | None = None
    ) -> CandidateProfileVersion:
        """Write version 1 of a candidate's profile; a `Conflict` if one exists."""
        session = self.session_factory()
        try:
            if CandidateProfileRepository(session).get_latest_by_user_id(user_id) is not None:
                raise Conflict(
                    f"user {user_id} already has a profile; update it to write a new version",
                    details={"user_id": str(user_id)},
                )
            self._ensure_user(session, user_id, display_name)
            return self._write(session, user_id, 1, profile)
        finally:
            session.close()

    def update(self, user_id: UUID, profile: CandidateProfileInput) -> CandidateProfileVersion:
        """Write the next version of a candidate's profile, leaving earlier ones as they were."""
        session = self.session_factory()
        try:
            latest = CandidateProfileRepository(session).get_latest_by_user_id(user_id)
            if latest is None:
                raise NotFound(
                    f"user {user_id} has no profile to update; create one first",
                    details={"user_id": str(user_id)},
                )
            return self._write(session, user_id, latest.profile_version + 1, profile)
        finally:
            session.close()

    def get(self, user_id: UUID, profile_version: int | None = None) -> CandidateProfileVersion:
        """One version of a candidate's profile, the latest when none is named."""
        session = self.session_factory()
        try:
            repository = CandidateProfileRepository(session)
            row = (
                repository.get_latest_by_user_id(user_id)
                if profile_version is None
                else repository.get_version(user_id, profile_version)
            )
            if row is None:
                which = "a profile" if profile_version is None else f"profile v{profile_version}"
                raise NotFound(
                    f"user {user_id} has no {which}",
                    details={"user_id": str(user_id), "profile_version": profile_version},
                )
            return profile_version_from_row(row)
        finally:
            session.close()

    def versions(self, user_id: UUID) -> CandidateProfileVersions:
        """Every version of a candidate's profile, newest first; empty if none."""
        session = self.session_factory()
        try:
            rows = CandidateProfileRepository(session).list_by_user_id(user_id)
            return CandidateProfileVersions(
                user_id=user_id, versions=[profile_version_from_row(row) for row in rows]
            )
        finally:
            session.close()

    @staticmethod
    def _ensure_user(session: Session, user_id: UUID, display_name: str | None) -> None:
        if session.get(UserModel, user_id) is None:
            session.add(UserModel(id=user_id, display_name=display_name))
            session.flush()

    @staticmethod
    def _write(
        session: Session, user_id: UUID, profile_version: int, profile: CandidateProfileInput
    ) -> CandidateProfileVersion:
        stored = profile.model_dump(mode="json")
        now = datetime.utcnow()
        row = CandidateProfileModel(
            user_id=user_id,
            profile_version=profile_version,
            target_roles=stored.pop("target_roles"),
            target_locations=stored.pop("target_locations"),
            preferences=stored,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise Conflict(
                f"profile v{profile_version} for user {user_id} was written by another "
                f"request; read the latest version and try again",
                details={"user_id": str(user_id), "profile_version": profile_version},
            ) from exc
        logger.info("wrote profile v%s for user %s", profile_version, user_id)
        return profile_version_from_row(row)


__all__ = ["SqlSearchProfileStore", "profile_version_from_row", "search_profile_from_row"]
