"""The caller's job-search profile over HTTP: create, read, update, history. Version 1.

- `POST /v1/profile` writes version 1; a 409 if the actor already has one.
- `GET /v1/profile` is the latest version.
- `PUT /v1/profile` writes the next version. Earlier versions are never
  changed: the matches a job search scored record the version they were
  scored against.
- `GET /v1/profile/versions` lists every version, newest first, and
  `GET /v1/profile/versions/{n}` is one of them.

The body of a create or update is a `CandidateProfileInput`, and every read
returns one under `profile`, so what a read gives back can be sent straight
back as an update. Unknown fields are a 422.

**The actor is the authenticated caller.** A profile belongs to the actor the
request's bearer token verified as (`apps.api.auth.current_actor`), mapped to
a candidate id by `actor_user_id` -- the same id `POST /v1/chat` searches for
when no `user_id` is given. A caller cannot name another actor.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Path, Response

from apps.api.auth import current_actor
from personalos.bootstrap import build_search_profile_store
from personalos.domain.search_profiles import (
    CandidateProfileInput,
    CandidateProfileVersion,
    CandidateProfileVersions,
    actor_user_id,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.search_profiles import SqlSearchProfileStore

router = APIRouter()

_PROFILE_PATH = "/profile"


def get_profile_store() -> SqlSearchProfileStore:
    """FastAPI dependency: a profile store over the configured database."""
    return build_search_profile_store(SessionLocal)


def _actor(actor_id: str) -> tuple[str, UUID]:
    return actor_id, actor_user_id(actor_id)


@router.post(_PROFILE_PATH, response_model=CandidateProfileVersion, status_code=201)
async def create_profile(
    profile: CandidateProfileInput,
    http_response: Response,
    store: SqlSearchProfileStore = Depends(get_profile_store),
    actor_id: str = Depends(current_actor),
):
    """Write the first version of the actor's profile."""
    actor_id, user_id = _actor(actor_id)
    created = store.create(user_id, profile, display_name=actor_id)
    http_response.headers["Location"] = f"/v1/profile/versions/{created.profile_version}"
    return created


@router.get(_PROFILE_PATH, response_model=CandidateProfileVersion)
async def get_profile(
    store: SqlSearchProfileStore = Depends(get_profile_store),
    actor_id: str = Depends(current_actor),
):
    """The actor's latest profile version -- the one a job search loads."""
    _, user_id = _actor(actor_id)
    return store.get(user_id)


@router.put(_PROFILE_PATH, response_model=CandidateProfileVersion)
async def update_profile(
    profile: CandidateProfileInput,
    store: SqlSearchProfileStore = Depends(get_profile_store),
    actor_id: str = Depends(current_actor),
):
    """Write the next version of the actor's profile; earlier versions stay as they were."""
    _, user_id = _actor(actor_id)
    return store.update(user_id, profile)


@router.get(f"{_PROFILE_PATH}/versions", response_model=CandidateProfileVersions)
async def list_profile_versions(
    store: SqlSearchProfileStore = Depends(get_profile_store),
    actor_id: str = Depends(current_actor),
):
    """Every version of the actor's profile, newest first."""
    _, user_id = _actor(actor_id)
    return store.versions(user_id)


@router.get(f"{_PROFILE_PATH}/versions/{{profile_version}}", response_model=CandidateProfileVersion)
async def get_profile_version(
    profile_version: int = Path(ge=1),
    store: SqlSearchProfileStore = Depends(get_profile_store),
    actor_id: str = Depends(current_actor),
):
    """One numbered version of the actor's profile."""
    _, user_id = _actor(actor_id)
    return store.get(user_id, profile_version)


__all__ = ["router", "get_profile_store"]
