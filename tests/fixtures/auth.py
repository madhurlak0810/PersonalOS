"""Authentication for tests that drive the API.

Every route but `/health` needs a bearer token, and many tests act as more
than one actor (a candidate, then a reviewer). `install_test_auth` gives an
app a verifier that accepts `test-token:<actor>` as that actor, and
`bearer(actor)` builds the header -- so a test says who it is the same way a
real client does, through `Authorization`, and the route derives the actor
from the verified credential exactly as in production.

The verifier exists only here. Production verifies the local token
(`apps.api.auth.LocalTokenVerifier`); `tests/unit/test_api_auth.py` covers
that path end to end.
"""

from uuid import UUID

from fastapi import FastAPI

from apps.api.auth import ApiAuth, StreamTicketSigner
from personalos.domain.auth import AuthMethod, Principal, parse_scopes
from personalos.domain.search_profiles import actor_user_id

#: The actor a request is when a test does not say: the API's default actor.
DEFAULT_TEST_ACTOR = "api"

_PREFIX = "test-token:"
_TICKET_KEY = b"test stream ticket key, 32 bytes"


#: Separates an actor from the scopes its token carries: `test-token:alice|read,write`.
_SCOPES_SEPARATOR = "|"


class ActorTokenVerifier:
    """Accepts `test-token:<actor>[|<scopes>]` as `<actor>`; rejects everything else.

    Without `|<scopes>` the token carries the default for the actor's kind:
    every scope for a person, everything but `approve` for a system actor.
    """

    def verify(self, token: str) -> Principal | None:
        if not token.startswith(_PREFIX) or len(token) == len(_PREFIX):
            return None
        actor, separator, scopes = token[len(_PREFIX) :].partition(_SCOPES_SEPARATOR)
        return Principal(
            actor_id=actor,
            method=AuthMethod.LOCAL_TOKEN,
            scopes=parse_scopes(scopes) if separator else None,
        )


def test_api_auth() -> ApiAuth:
    """An `ApiAuth` over `ActorTokenVerifier`, with a fixed ticket key."""
    return ApiAuth(
        verifier=ActorTokenVerifier(),
        tickets=StreamTicketSigner(lambda: _TICKET_KEY, ttl_seconds=60),
    )


# Not a test, despite the name pytest would otherwise collect it by.
test_api_auth.__test__ = False  # type: ignore[attr-defined]


def install_test_auth(app: FastAPI) -> FastAPI:
    """Make `app` accept `bearer(actor)` headers. Returns the app."""
    app.state.api_auth = test_api_auth()
    return app


def bearer(actor: str = DEFAULT_TEST_ACTOR, *, scopes: str | None = None) -> dict[str, str]:
    """The `Authorization` header that authenticates as `actor`, with `scopes` if given."""
    suffix = f"{_SCOPES_SEPARATOR}{scopes}" if scopes is not None else ""
    return {"Authorization": f"Bearer {_PREFIX}{actor}{suffix}"}


def owner_id(actor: str = DEFAULT_TEST_ACTOR) -> UUID:
    """The `users.id` an actor's workflows and applications are keyed by."""
    return actor_user_id(actor)


__all__ = [
    "DEFAULT_TEST_ACTOR",
    "ActorTokenVerifier",
    "bearer",
    "install_test_auth",
    "owner_id",
    "test_api_auth",
]
