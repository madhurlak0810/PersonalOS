"""The Google server's HTTP adapter and its transport boundary.

- `HttpxGoogleHttp` maps statuses onto the taxonomy, sends the brokered token
  as a bearer header, and never quotes a response body in an exception.
- The server and the stdio transport open no listening socket: stdio, or
  in-process, are the only ways in.
- A real child process speaks the stdio protocol end to end.
"""

import ast
import asyncio
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from mcp_servers.google.api import GoogleResponseInvalid, HttpxGoogleHttp
from personalos.domain.credentials import AccessToken, CredentialRef, SecretValue
from personalos.domain.errors import (
    NotFound,
    ProviderPermanentFailure,
    RetryableFailure,
    ValidationFailed,
)
from personalos.mcp.stdio import MCPStdioClient, RemoteMCPServer

ROOT = Path(__file__).resolve().parents[2]
SECRET = "ya29.secret-access-token"


def _token(expires_in: timedelta = timedelta(minutes=5)) -> AccessToken:
    return AccessToken(
        credential_ref=CredentialRef("google", "a@example.com"),
        secret=SecretValue(SECRET),
        expires_at=datetime.now(timezone.utc) + expires_in,
    )


def _http(handler) -> HttpxGoogleHttp:
    return HttpxGoogleHttp(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def test_the_token_is_sent_as_a_bearer_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"ok": True})

    assert await _http(handler).request("GET", "https://g.test/x", token=_token()) == {"ok": True}
    assert seen["auth"] == f"Bearer {SECRET}"


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (401, {}, ProviderPermanentFailure),
        (
            403,
            {"error": {"errors": [{"reason": "insufficientPermissions"}]}},
            ProviderPermanentFailure,
        ),
        (403, {"error": {"errors": [{"reason": "rateLimitExceeded"}]}}, RetryableFailure),
        (404, {}, NotFound),
        (429, {}, RetryableFailure),
        (503, {}, RetryableFailure),
        (400, {}, ValidationFailed),
    ],
)
async def test_statuses_map_onto_the_taxonomy_without_quoting_the_body(status, body, expected):
    body = {**body, "echo": f"Authorization: Bearer {SECRET}"}

    with pytest.raises(expected) as raised:
        await _http(lambda r: httpx.Response(status, json=body)).request(
            "GET", "https://g.test/x", token=_token()
        )
    assert SECRET not in str(raised.value) and SECRET not in repr(raised.value)


async def test_retry_after_is_carried():
    with pytest.raises(RetryableFailure) as raised:
        await _http(lambda r: httpx.Response(429, headers={"Retry-After": "7"})).request(
            "GET", "https://g.test/x", token=_token()
        )
    assert raised.value.retry_after == 7.0


async def test_a_non_json_success_is_invalid():
    with pytest.raises(GoogleResponseInvalid):
        await _http(lambda r: httpx.Response(200, text="<html>")).request(
            "GET", "https://g.test/x", token=_token()
        )


async def test_an_oversized_response_is_refused():
    http = HttpxGoogleHttp(
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 2048))
        ),
        max_response_bytes=1024,
    )
    with pytest.raises(GoogleResponseInvalid):
        await http.request("GET", "https://g.test/x", token=_token())


async def test_an_expired_token_is_not_presented():
    calls = []
    with pytest.raises(RetryableFailure):
        await _http(lambda r: calls.append(r) or httpx.Response(200)).request(
            "GET", "https://g.test/x", token=_token(timedelta(seconds=-1))
        )
    assert calls == []


# --- No listeners -------------------------------------------------------------

_LISTENING_MODULES = {"socket", "socketserver", "http.server", "uvicorn", "fastapi", "starlette"}
_LISTENING_CALLS = {"start_server", "start_unix_server", "create_server", "serve_forever"}


@pytest.mark.parametrize(
    "path",
    [
        *sorted((ROOT / "mcp_servers" / "google").glob("*.py")),
        ROOT / "personalos" / "mcp" / "stdio.py",
    ],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_the_google_server_and_its_transport_open_no_listener(path):
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not imported & _LISTENING_MODULES
    assert not called & _LISTENING_CALLS


# --- A real child process -----------------------------------------------------

_CHILD = textwrap.dedent("""
    import asyncio, logging, sys
    sys.path.insert(0, {root!r})
    logging.basicConfig(stream=sys.stderr)
    from mcp_servers.google.server import GoogleMCPServer
    from mcp_servers.google.tools import GoogleWorkspaceTools
    from personalos.mcp.stdio import serve_stdio
    from tests.fixtures.fake_google import FakeGoogleApi, FakeGoogleExchanger, gmail_message
    from personalos.executor.credentials import CredentialBroker
    from personalos.secrets.store import InMemorySecretStore
    from personalos.domain.credentials import CredentialRef, SecretValue

    store = InMemorySecretStore()
    store.put(CredentialRef("google", "a@example.com"), SecretValue("refresh"))
    exchanger = FakeGoogleExchanger()
    api = FakeGoogleApi(exchanger)
    api.add_message(gmail_message("m1", body="from the child"))
    tools = GoogleWorkspaceTools(CredentialBroker(store, {{"google": exchanger}}), api)
    asyncio.run(serve_stdio(GoogleMCPServer(tools)))
    """)


async def test_a_child_process_serves_the_tools_over_stdio(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(root=str(ROOT)))
    client = await MCPStdioClient.spawn([sys.executable, str(script)], timeout=20)
    try:
        remote = await RemoteMCPServer.connect("google", client)
        assert "gmail_send_message" in {tool.name for tool in remote.get_tools()}

        from personalos.domain.models import ActionTarget, ToolAuthorization, ToolCallRequest

        result = await remote.execute(
            ToolCallRequest(
                target=ActionTarget(server="google", tool="gmail_read_message"),
                params={"message_id": "m1"},
                authorization=ToolAuthorization(
                    credential_ref="cred://google/a@example.com", principal="test"
                ),
            )
        )
    finally:
        await asyncio.wait_for(client.close(), 10)
    assert result.ok, result.error
    assert result.result["body_text"] == "from the child"
