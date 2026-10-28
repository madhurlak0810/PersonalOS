"""Shared machinery for graph node contract tests.

Three pieces, used by every `tests/unit/test_*_nodes.py` module:

- `covers_node` marks a test class as the contract test for one registered
  node. `tests/contracts/test_node_contract_coverage.py` reads what it records
  and fails if a node in a compiled graph has no such class.
- `enforce_contracts` wraps a graph instance's node methods so every update
  they return is validated against its `NodeContract` output model. A test
  written to check one behaviour therefore also checks the node's key set and
  value types, without restating them.
- `ReplayChatModel` stands in for a provider chat model behind the
  `StructuredChatModel` port, replaying model outputs recorded in
  `tests/fixtures/recorded_model_outputs/`. No network, no clock, no seed:
  the same file replays the same answers in the same order on every run.
"""

import functools
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from personalos.graphs.contracts import NODE_CONTRACTS, NodeContract, contracts_for

RECORDED_OUTPUTS = Path(__file__).parent / "recorded_model_outputs"

#: `(graph, node)` -> the test class that covers it, filled by `covers_node`.
CONTRACT_TESTS: dict[tuple[str, str], type] = {}

_ClassT = TypeVar("_ClassT", bound=type)


def covers_node(graph: str, node: str) -> Callable[[_ClassT], _ClassT]:
    """Register the decorated class as the contract test for `graph`'s `node`."""
    key = (graph, node)
    if key not in NODE_CONTRACTS:
        raise LookupError(f"no NodeContract is registered for {graph}:{node}")

    def register(cls: _ClassT) -> _ClassT:
        existing = CONTRACT_TESTS.get(key)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(f"{graph}:{node} is already covered by {existing.__qualname__}")
        CONTRACT_TESTS[key] = cls
        cls.node_contract = NODE_CONTRACTS[key]
        return cls

    return register


def _checked(
    contract: NodeContract,
    method: Callable[..., Any],
    *,
    inputs: bool,
    visited: set[str] | None,
) -> Callable[..., Any]:
    """`method`, with its returned update (and optionally its state) validated."""

    def before(args: tuple[Any, ...]) -> None:
        if visited is not None:
            visited.add(contract.node)
        if inputs:
            contract.validate_input(args[0])

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            before(args)
            update = await method(*args, **kwargs)
            contract.validate_output(update)
            return update

        return run_async

    @functools.wraps(method)
    def run(*args: Any, **kwargs: Any) -> Any:
        before(args)
        update = method(*args, **kwargs)
        contract.validate_output(update)
        return update

    return run


def enforce_contracts(
    instance: Any, graph: str, *, inputs: bool = False, visited: set[str] | None = None
) -> Any:
    """Make every node method on `instance` check its contract. Returns `instance`.

    Outputs are always checked. Inputs only when asked: a node test that hands
    a node a deliberately malformed state to see it fail with a typed domain
    error must still see that error, not a validation error from this wrapper.
    A compiled-graph run passes `inputs=True`, since there every state a node
    receives was written by the nodes before it. `visited` collects the names
    of the nodes that ran.

    Call it before `build()`: the graph binds the node methods when it compiles.
    """
    for contract in contracts_for(graph).values():
        method = getattr(instance, contract.method)
        setattr(
            instance,
            contract.method,
            _checked(contract, method, inputs=inputs, visited=visited),
        )
    return instance


def recorded(name: str) -> Any:
    """Load one file of recorded model outputs."""
    return json.loads((RECORDED_OUTPUTS / f"{name}.json").read_text())


class _ReplayRunnable:
    def __init__(self, model: "ReplayChatModel"):
        self.model = model

    def _next(self, turns: Any) -> Any:
        self.model.calls.append(list(turns))
        if not self.model.replies:
            raise AssertionError("the model was called more times than outputs were recorded")
        reply = self.model.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def invoke(self, turns: Any, **_kwargs: Any) -> Any:
        return self._next(turns)

    async def ainvoke(self, turns: Any, **_kwargs: Any) -> Any:
        return self._next(turns)


class ReplayChatModel:
    """A `StructuredChatModel` that replays recorded outputs in order.

    A reply is returned as recorded -- usually the plain dict a provider's
    structured-output decoder hands back, which the adapter then validates --
    or raised, if it is an exception. Running out of replies is a test failure,
    so an unexpected extra model call cannot pass silently.
    """

    def __init__(self, *replies: Any, name: str = "replay"):
        self.replies = list(replies)
        self.model_name = name
        self.bound_schema: Any = None
        self.calls: list[list[Any]] = []

    @classmethod
    def from_recording(cls, name: str, case: str) -> "ReplayChatModel":
        """Replay the outputs recorded under `case` in `<name>.json`."""
        return cls(*recorded(name)[case]["outputs"], name=f"recorded:{name}")

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> _ReplayRunnable:
        self.bound_schema = schema
        return _ReplayRunnable(self)

    @property
    def exhausted(self) -> bool:
        return not self.replies


__all__ = [
    "RECORDED_OUTPUTS",
    "CONTRACT_TESTS",
    "covers_node",
    "enforce_contracts",
    "recorded",
    "ReplayChatModel",
]
