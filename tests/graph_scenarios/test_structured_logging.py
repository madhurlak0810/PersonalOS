"""Log lines written inside a graph node carry the run's ids without the node passing them.

The acceptance criterion this file exists for: **logs emitted inside a graph
node include the active `workflow_id` and `correlation_id` without the node
passing them explicitly** --
`test_a_node_log_line_carries_the_runs_workflow_and_correlation_ids`.

Each node below logs a bare message through an ordinary `logging.getLogger`.
The ids reach the line through the log scope: `traced_node` binds the run's
ids from its LangGraph config (how the worker runs every node), and
`workflow_span` binds them around the whole run (how the worker opens one).
Both are checked with tracing on and off, since logs must not depend on spans
being recorded.
"""

import io
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from personalos.observability import tracing
from personalos.observability.structured_logging import configure_logging
from personalos.observability.tracing import traced_node
from personalos.observability.tracing_setup import configure_tracing, reset_tracing

node_logger = logging.getLogger("personalos.graphs.test_node")


class _State(TypedDict, total=False):
    visited: list[str]


@pytest.fixture(params=["tracing_off", "tracing_on"])
def tracing_mode(request) -> Iterator[str]:
    if request.param == "tracing_on":
        configure_tracing(exporter=InMemorySpanExporter(), install_global=False)
    try:
        yield request.param
    finally:
        reset_tracing()


@pytest.fixture
def node_lines() -> Iterator[Callable[[], list[dict[str, Any]]]]:
    """Reads back what the test's nodes logged, as the JSON objects the handler wrote."""
    stream = io.StringIO()
    handler = configure_logging(level="DEBUG", stream=stream)

    def read() -> list[dict[str, Any]]:
        lines = (json.loads(line) for line in stream.getvalue().splitlines() if line)
        return [line for line in lines if line["logger"] == node_logger.name]

    try:
        yield read
    finally:
        logging.getLogger().removeHandler(handler)


def _graph(*, wrap: bool):
    """Two nodes, sync and async, that log and know nothing about workflows."""

    def plan(state: _State) -> dict[str, Any]:
        node_logger.info("planning")
        return {"visited": [*(state.get("visited") or []), "plan"]}

    async def act(state: _State) -> dict[str, Any]:
        node_logger.info("acting", extra={"event": "test.act"})
        return {"visited": [*(state.get("visited") or []), "act"]}

    nodes = {"plan": plan, "act": act}
    builder = StateGraph(_State)
    for name, fn in nodes.items():
        builder.add_node(name, traced_node(name, fn, graph="test") if wrap else fn)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "act")
    builder.add_edge("act", END)
    return builder.compile()


async def test_a_node_log_line_carries_the_runs_workflow_and_correlation_ids(
    tracing_mode, node_lines
):
    """The worker hands the ids to the graph in its config; the node never sees them."""
    workflow_id, correlation_id = uuid4(), uuid4()

    await _graph(wrap=True).ainvoke(
        {},
        config={
            "configurable": {
                "thread_id": f"job_search:{workflow_id}",
                "workflow_id": str(workflow_id),
                "correlation_id": str(correlation_id),
                "actor_id": "user-1",
            }
        },
    )

    lines = node_lines()
    assert [line["message"] for line in lines] == ["planning", "acting"]
    for line, node in zip(lines, ["plan", "act"], strict=True):
        assert line["workflow_id"] == str(workflow_id)
        assert line["correlation_id"] == str(correlation_id)
        assert line["thread_id"] == f"job_search:{workflow_id}"
        assert line["node"] == node
        assert line["component"] == "graphs"
    assert lines[0]["event"] == "log"
    assert lines[1]["event"] == "test.act"


async def test_ids_bound_around_a_run_reach_nodes_that_are_not_wrapped(tracing_mode, node_lines):
    """A run opened in a workflow span scopes its nodes' logs through the context alone."""
    workflow_id, correlation_id = uuid4(), uuid4()

    with tracing.workflow_span("graph.run", workflow_id=workflow_id, correlation_id=correlation_id):
        await _graph(wrap=False).ainvoke({})

    lines = node_lines()
    assert len(lines) == 2
    assert {line["workflow_id"] for line in lines} == {str(workflow_id)}
    assert {line["correlation_id"] for line in lines} == {str(correlation_id)}


async def test_the_scope_ends_with_the_run(tracing_mode, node_lines):
    """A line after the run names no workflow: the scope is not left behind."""
    workflow_id = uuid4()

    await _graph(wrap=True).ainvoke({}, config={"configurable": {"workflow_id": str(workflow_id)}})
    node_logger.info("after the run")

    lines = node_lines()
    after = lines[-1]
    assert after["message"] == "after the run"
    assert after["workflow_id"] is None
    assert after["correlation_id"] is None
    assert UUID(lines[0]["workflow_id"]) == workflow_id
