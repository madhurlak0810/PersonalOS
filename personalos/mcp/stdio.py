"""MCP over stdio: serve an `MCPServer` to its parent, and reach one from the gateway.

The wire format is MCP's stdio transport -- newline-delimited JSON-RPC 2.0 --
with the subset of the protocol PersonalOS uses: `initialize`,
`notifications/initialized`, `ping`, `tools/list` and `tools/call`. Written
against the protocol rather than an SDK's internals so the same bytes work
whichever `mcp` package version is installed, and so a stock MCP client can
still list and call the tools.

stdio is the only transport. A server served this way opens no socket: its
only peer is the process that spawned it -- the worker or API process that
owns the tool gateway -- and a browser has no route to it.

Two things ride in a `tools/call`'s `_meta`, never in its `arguments`:

``personalos/context``
    The `ExecutionContext` of the workflow run, for correlation.
``personalos/authorization``
    The `ToolAuthorization` the gateway side attached after policy. Because
    `arguments` are validated into the tool's `Intent` (`extra="forbid"`), a
    caller cannot smuggle one in as an argument either.

On the client side (`RemoteMCPServer`) the server's replies are treated as
untrusted: each is validated into a fixed shape, an unknown error code is
downgraded to a plain execution error, and the tool list is read once at
connect time. Notifications from the server -- `tools/list_changed`
included -- are ignored, so nothing a tool returns can change what tools the
gateway believes exist; that is policy's question in any case.
"""

import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO

from pydantic import BaseModel, ConfigDict, ValidationError

from personalos.domain.context import ExecutionContext
from personalos.domain.models import (
    ActionTarget,
    Tool,
    ToolAuthorization,
    ToolCallErrorCode,
    ToolCallRequest,
    ToolCallResult,
)
from personalos.mcp.base import MCPServer

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
META_CONTEXT = "personalos/context"
META_AUTHORIZATION = "personalos/authorization"
META_IDEMPOTENCY_KEY = "personalos/idempotency_key"
META_REPLAYED = "personalos/replayed"

#: Largest single JSON-RPC message either side will read.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
#: Longest error message carried back from a server.
MAX_ERROR_CHARS = 2000

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


class MCPTransportError(Exception):
    """The peer broke the protocol, went away, or did not answer in time."""


# --- Streams ------------------------------------------------------------------


async def open_pipe_streams(
    read_file: BinaryIO, write_file: BinaryIO, *, limit: int = MAX_MESSAGE_BYTES
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Wrap two pipe file objects as asyncio streams. stdio is two such pipes."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=limit)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), read_file)
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, write_file
    )
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    return reader, writer


async def _write(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    writer.write(json.dumps(message, separators=(",", ":"), default=str).encode("utf-8") + b"\n")
    await writer.drain()


# --- Server side --------------------------------------------------------------


def _request_id(line: bytes) -> Any:
    try:
        message = json.loads(line)
    except ValueError:
        return None
    return message.get("id") if isinstance(message, dict) else None


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class MCPStdioServer:
    """Serves one `MCPServer` over a reader/writer pair, one request at a time."""

    def __init__(self, server: MCPServer, *, version: str = "0.1.0"):
        """Take the server whose tools this exposes."""
        self.server = server
        self.version = version

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer requests until the reader hits EOF."""
        while True:
            try:
                line = await reader.readline()
            except (asyncio.LimitOverrunError, ValueError):
                await _write(writer, _error(None, _INVALID_REQUEST, "message too large"))
                return
            if not line:
                return
            if not line.strip():
                continue
            try:
                reply = await self.handle_line(line)
            except Exception:
                # One bad request must not take the server down with it.
                logger.exception("MCP stdio server failed to answer a request")
                reply = _error(_request_id(line), -32603, "internal error")
            if reply is not None:
                await _write(writer, reply)

    async def handle_line(self, line: bytes) -> dict[str, Any] | None:
        """One JSON-RPC message in, its response (or `None` for a notification) out."""
        try:
            message = json.loads(line)
        except ValueError:
            return _error(None, _PARSE_ERROR, "not valid JSON")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, _INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        method = message.get("method")
        request_id = message.get("id")
        if not isinstance(method, str):
            return None  # a response to something we never send; ignore it
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        is_notification = "id" not in message

        try:
            result = await self._dispatch(method, params)
        except _RPCError as error:
            return None if is_notification else _error(request_id, error.code, error.message)
        return None if is_notification or result is None else _response(request_id, result)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.server.name, "version": self.version},
                "instructions": self.server.description,
            }
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [self._describe(name) for name in self._tool_names()]}
        if method == "tools/call":
            return await self._call(params)
        raise _RPCError(_METHOD_NOT_FOUND, f"method '{method}' is not supported")

    def _tool_names(self) -> list[str]:
        return [tool.name for tool in self.server.get_tools()]

    def _describe(self, name: str) -> dict[str, Any]:
        schema = self.server.get_tool_schema(name)
        described: dict[str, Any] = {
            "name": name,
            "description": schema.description,
            "inputSchema": {
                **schema.parameters,
                "type": "object",
                "required": list(schema.required),
            },
        }
        if schema.response_schema:
            described["outputSchema"] = schema.response_schema
        return described

    async def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        meta = params.get("_meta") or {}
        if (
            not isinstance(name, str)
            or not isinstance(arguments, dict)
            or not isinstance(meta, dict)
        ):
            raise _RPCError(_INVALID_PARAMS, "tools/call needs a name and object arguments")

        try:
            context = (
                ExecutionContext.model_validate(meta[META_CONTEXT])
                if META_CONTEXT in meta
                else ExecutionContext.new()
            )
            authorization = (
                ToolAuthorization.model_validate(meta[META_AUTHORIZATION])
                if META_AUTHORIZATION in meta
                else None
            )
        except ValidationError:
            raise _RPCError(_INVALID_PARAMS, "malformed _meta context or authorization") from None

        result = await self.server.execute(
            ToolCallRequest(
                target=ActionTarget(server=self.server.name, tool=name),
                params=arguments,
                context=context,
                authorization=authorization,
            )
        )
        return encode_call_result(result)


class _RPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def encode_call_result(result: ToolCallResult) -> dict[str, Any]:
    """A `ToolCallResult` as an MCP `CallToolResult`."""
    if result.ok:
        structured: dict[str, Any] = result.result or {}
    else:
        # Code and message only: `details` can hold validator internals that
        # do not serialize, and echoes of the caller's input that need not
        # cross the process boundary.
        structured = {"error": {"code": result.error.code.value, "message": result.error.message}}
    return {
        "content": [{"type": "text", "text": json.dumps(structured, default=str)}],
        "structuredContent": structured,
        "isError": not result.ok,
        "_meta": {META_IDEMPOTENCY_KEY: result.idempotency_key, META_REPLAYED: result.replayed},
    }


async def serve_stdio(server: MCPServer) -> None:
    """Serve `server` on this process's stdin/stdout until stdin closes.

    stdout carries the protocol and nothing else, so logging must already be
    pointed at stderr.
    """
    reader, writer = await open_pipe_streams(
        os.fdopen(sys.stdin.fileno(), "rb", 0, closefd=False),
        os.fdopen(sys.stdout.fileno(), "wb", 0, closefd=False),
    )
    await MCPStdioServer(server).serve(reader, writer)


# --- Client side --------------------------------------------------------------


class _ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str
    message: str


class _CallToolResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    structuredContent: dict[str, Any] | None = None
    isError: bool = False


class _DescribedTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    inputSchema: dict[str, Any] = {}


class MCPStdioClient:
    """A JSON-RPC client over a reader/writer pair. One request in flight at a time."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        timeout: float = 60.0,
        process: asyncio.subprocess.Process | None = None,
    ):
        """Take the streams to the server (and the process, if this client spawned it)."""
        self._reader = reader
        self._writer = writer
        self._timeout = timeout
        self._process = process
        self._lock = asyncio.Lock()
        self._next_id = 0

    @classmethod
    async def spawn(
        cls,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float = 60.0,
    ) -> "MCPStdioClient":
        """Start `command` as a child process and speak to it over its stdin/stdout."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=dict(env) if env is not None else None,
            limit=MAX_MESSAGE_BYTES,
        )
        return cls(process.stdout, process.stdin, timeout=timeout, process=process)

    async def initialize(self) -> dict[str, Any]:
        """Run the MCP handshake; return the server's `initialize` result."""
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "personalos-gateway", "version": "0.1.0"},
            },
        )
        await _write(self._writer, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send one request and wait for its response, skipping anything else."""
        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            await _write(
                self._writer,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params or {}),
                },
            )
            try:
                return await asyncio.wait_for(self._await_response(request_id), self._timeout)
            except asyncio.TimeoutError:
                raise MCPTransportError(f"no response to '{method}' in {self._timeout}s") from None

    async def _await_response(self, request_id: int) -> Any:
        while True:
            try:
                line = await self._reader.readline()
            except (asyncio.LimitOverrunError, ValueError):
                raise MCPTransportError("server sent a message larger than the limit") from None
            if not line:
                raise MCPTransportError("server closed the connection")
            try:
                message = json.loads(line)
            except ValueError:
                raise MCPTransportError("server sent a line that is not JSON") from None
            if not isinstance(message, dict):
                raise MCPTransportError("server sent a message that is not an object")
            if "method" in message:
                # A notification or a server-initiated request. Neither can
                # change anything here: the tool list is not re-read, and
                # this client grants a server nothing it could ask for.
                if "id" in message:
                    await _write(
                        self._writer,
                        _error(message.get("id"), _METHOD_NOT_FOUND, "client accepts no requests"),
                    )
                continue
            if message.get("id") != request_id:
                continue
            if isinstance(message.get("error"), dict):
                error_message = str(message["error"].get("message", "error"))[:MAX_ERROR_CHARS]
                raise MCPTransportError(f"server error: {error_message}")
            return message.get("result")

    async def close(self) -> None:
        """Close the pipe to the server, and reap it if this client started it."""
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, NotImplementedError):
            # A pipe writer (`open_pipe_streams`) has no close waiter to await.
            pass
        if self._process is not None:
            try:
                await asyncio.wait_for(self._process.wait(), 5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()


class RemoteMCPServer(MCPServer):
    """An `MCPServer` whose tools live in another process, reached over stdio.

    Registered on the `MCPServerManager` like an in-process server, so the
    gateway, `MCPToolInvoker` and the policy engine are identical either way.
    Idempotency is enforced by the server that executes the call.
    """

    def __init__(self, name: str, description: str, client: MCPStdioClient, tools: list[Tool]):
        """Use `connect` instead; this takes an already-listed tool set."""
        super().__init__(name, description)
        self.client = client
        self._remote_tools = {tool.name: tool for tool in tools}

    @classmethod
    async def connect(cls, name: str, client: MCPStdioClient) -> "RemoteMCPServer":
        """Handshake, list the tools once, and return the proxy."""
        info = await client.initialize()
        listing = await client.request("tools/list")
        raw_tools = listing.get("tools") if isinstance(listing, dict) else None
        if not isinstance(raw_tools, list):
            raise MCPTransportError("server's tools/list result has no tool list")
        tools = []
        for item in raw_tools:
            try:
                described = _DescribedTool.model_validate(item)
            except ValidationError:
                raise MCPTransportError("server listed a malformed tool") from None
            tools.append(
                Tool(
                    name=described.name,
                    description=described.description,
                    parameters=described.inputSchema,
                    mcp_server=name,
                )
            )
        description = info.get("instructions", "") if isinstance(info, dict) else ""
        return cls(name, str(description)[:MAX_ERROR_CHARS], client, tools)

    def initialize(self):
        """Nothing to register: the tools were listed at connect time."""

    async def ping(self) -> None:
        """Ask the server process whether it is still answering. Raises if not."""
        await self.client.request("ping")

    def get_tools(self) -> list[Tool]:
        """The tool set read at connect time. Never refreshed by the server."""
        return list(self._remote_tools.values())

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """Forward the call, with context and authorization in `_meta`."""
        target = request.target
        if target.tool not in self._remote_tools:
            return ToolCallResult.failed(
                target,
                ToolCallErrorCode.TOOL_NOT_FOUND,
                f"Tool '{target.tool}' not found on server '{self.name}'",
            )
        meta: dict[str, Any] = {META_CONTEXT: request.context.model_dump(mode="json")}
        if request.authorization is not None:
            meta[META_AUTHORIZATION] = request.authorization.model_dump(mode="json")
        try:
            raw = await self.client.request(
                "tools/call", {"name": target.tool, "arguments": request.params, "_meta": meta}
            )
        except MCPTransportError as error:
            return ToolCallResult.failed(target, ToolCallErrorCode.EXECUTION_ERROR, str(error))
        return decode_call_result(target, raw)


def decode_call_result(target: ActionTarget, raw: Any) -> ToolCallResult:
    """An MCP `CallToolResult` from an untrusted peer, as a `ToolCallResult`."""
    try:
        parsed = _CallToolResult.model_validate(raw)
    except ValidationError:
        parsed = None
    if parsed is None or parsed.structuredContent is None:
        return ToolCallResult.failed(
            target, ToolCallErrorCode.EXECUTION_ERROR, "server returned a malformed tool result"
        )
    meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    if not parsed.isError:
        key = meta.get(META_IDEMPOTENCY_KEY)
        replayed = meta.get(META_REPLAYED)
        return ToolCallResult.succeeded(
            target,
            parsed.structuredContent,
            idempotency_key=key if isinstance(key, str) else None,
            replayed=replayed if isinstance(replayed, bool) else None,
        )
    try:
        error = _ErrorPayload.model_validate(parsed.structuredContent.get("error"))
        code = ToolCallErrorCode(error.code)
        message = error.message[:MAX_ERROR_CHARS]
    except (ValidationError, ValueError):
        code, message = ToolCallErrorCode.EXECUTION_ERROR, "server reported an unclassified error"
    return ToolCallResult.failed(target, code, message)


__all__ = [
    "PROTOCOL_VERSION",
    "META_CONTEXT",
    "META_AUTHORIZATION",
    "MAX_MESSAGE_BYTES",
    "MCPTransportError",
    "open_pipe_streams",
    "MCPStdioServer",
    "serve_stdio",
    "encode_call_result",
    "MCPStdioClient",
    "RemoteMCPServer",
    "decode_call_result",
]
