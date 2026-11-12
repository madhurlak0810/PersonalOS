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

from fastapi import FastAPI

from apps.api.auth import ApiAuth, StreamTicketSigner
from personalos.domain.auth import AuthMethod, Principal

#: The actor a request is when a test does not say: the API's default actor.
DEFAULT_TEST_ACTOR = "api"

_PREFIX = "test-token:"
_TICKET_KEY = b"test stream ticket key, 32 bytes"


class ActorTokenVerifier:
    """Accepts `test-token:<actor>` as `<actor>`; rejects everything else."""

    def verify(self, token: str) -> Principal | None:
        if not token.startswith(_PREFIX) or len(token) == len(_PREFIX):
            return None
        return Principal(actor_id=token[len(_PREFIX) :], method=AuthMethod.LOCAL_TOKEN)


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


def bearer(actor: str = DEFAULT_TEST_ACTOR) -> dict[str, str]:
    """The `Authorization` header that authenticates as `actor`."""
    return {"Authorization": f"Bearer {_PREFIX}{actor}"}


__all__ = [
    "DEFAULT_TEST_ACTOR",
    "ActorTokenVerifier",
    "bearer",
    "install_test_auth",
    "test_api_auth",
]
