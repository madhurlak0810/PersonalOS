"""Enforce the intent-only boundary declared in ``intent_boundary.py``.

The executable half of the "Intent-only boundary" section of
``docs/ARCHITECTURE_BOUNDARIES.md``: no reasoning node imports a provider SDK,
a DB write path or a filesystem mutator; it emits an ``ActionIntent`` and the
executor acts. Most of this file plants deliberately non-compliant nodes and
checks they are caught, because a boundary check that cannot fail reads as
assurance it is not giving.
"""

from textwrap import dedent

import pytest

from tests.architecture.boundaries import LAYERS_BY_NAME, REPO_ROOT
from tests.architecture.intent_boundary import (
    EFFECT_LAYERS,
    EXECUTION_LAYERS,
    REASONING_LAYERS,
    SIDE_EFFECT_RULES,
    ModuleFacts,
    analyse_source,
    collect_facts,
    intent_boundary_violations,
)

BOUNDARY_DOC = REPO_ROOT / "docs" / "ARCHITECTURE_BOUNDARIES.md"

#: A node with the required shape: validate, compute, emit a typed intent,
#: return a state update. The planted violations below are this node with one
#: line of side effect added.
COMPLIANT_NODE = """
    from typing import Any, Protocol

    from personalos.domain.job_search import ActionIntent, ActionKind


    class Drafter(Protocol):
        async def draft(self, thread: str) -> str: ...


    class FollowUpNode:
        def __init__(self, drafter: Drafter) -> None:
            self._drafter = drafter

        async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
            thread = state["thread"]
            if not thread:
                return {"errors": ["no thread"]}
            body = await self._drafter.draft(thread)
            intent = ActionIntent(
                kind=ActionKind.SEND_RECRUITER_MESSAGE,
                target=thread,
                summary="follow up",
                payload={"body": body},
                idempotency_key="follow-up-" + thread,
            )
            with open("prompt.txt") as fh:  # reading is not a side effect
                fh.read()
            return {"pending_actions": [intent.model_dump(mode="json")]}
"""


@pytest.fixture(scope="module")
def facts() -> dict[str, ModuleFacts]:
    """Facts for the real tree, collected once."""
    return collect_facts()


def _plant(
    facts: dict[str, ModuleFacts],
    body: str,
    module: str = "personalos.graphs.planted_node",
) -> list[str]:
    """Violations for the real tree plus one synthetic module.

    The source goes through the same parser as the real tree; nothing is
    written to disk.
    """
    path = REPO_ROOT / (module.replace(".", "/") + ".py")
    planted = dict(facts)
    planted[module] = analyse_source(dedent(body), module, path)
    return [p for p in intent_boundary_violations(planted) if module in p]


def test_real_tree_respects_intent_boundary(facts: dict[str, ModuleFacts]):
    """No reasoning module in the tree performs or reaches a side effect."""
    problems = intent_boundary_violations(facts)
    assert not problems, "intent-only boundary violations:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


def test_compliant_node_is_not_flagged(facts: dict[str, ModuleFacts]):
    """The documented node shape passes, so the check stays usable."""
    assert _plant(facts, COMPLIANT_NODE) == []


# ----------------------------------------------------------------------
# Deliberately non-compliant nodes. Each one is a graph node that tries to
# act on the world itself instead of emitting an intent.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "offending_line,category",
    [
        # Provider SDKs: Gmail, Google Calendar, job boards, MCP client.
        ("from googleapiclient.discovery import build", "provider-sdk"),
        ("import googleapiclient.discovery as gdisc", "provider-sdk"),
        ("from google.oauth2.credentials import Credentials", "provider-sdk"),
        ("from google import auth", "provider-sdk"),
        ("from gcsa.google_calendar import GoogleCalendar", "provider-sdk"),
        ("from jobspy import scrape_jobs", "provider-sdk"),
        ("from linkedin_api import Linkedin", "provider-sdk"),
        ("from mcp import ClientSession", "provider-sdk"),
        # Raw DB write paths.
        ("from sqlalchemy.orm import Session", "database"),
        ("import psycopg2", "database"),
        ("import sqlite3", "database"),
        ("from langgraph.checkpoint.postgres import PostgresSaver", "database"),
        ("from celery import Celery", "database"),
        # Hand-rolled provider calls.
        ("import httpx", "network"),
        ("import requests", "network"),
        ("from urllib import request", "network"),
        ("import smtplib", "network"),
        # Processes and filesystem modules.
        ("import subprocess", "process"),
        ("import shutil", "filesystem"),
        # A raw model client outside personalos.models.
        ("from anthropic import Anthropic", "llm-sdk"),
    ],
)
def test_node_importing_side_effect_surface_is_caught(
    facts: dict[str, ModuleFacts], offending_line: str, category: str
):
    """A graph node importing a provider SDK, DB driver or the like fails."""
    problems = _plant(facts, offending_line + "\n" + dedent(COMPLIANT_NODE))
    assert problems, f"checker missed: {offending_line}"
    assert any(f"[{category}]" in p for p in problems), problems


def test_import_nested_in_a_node_body_is_caught(facts: dict[str, ModuleFacts]):
    """Hiding the import inside the node function does not help."""
    body = """
        async def send(state):
            from googleapiclient.discovery import build

            build("gmail", "v1").users().messages().send(userId="me", body={}).execute()
            return {}
    """
    assert _plant(facts, body)


@pytest.mark.parametrize(
    "call",
    [
        'open("out.txt", "w").write("x")',
        'open("out.txt", mode="a")',
        "open(path, mode)",  # unknowable mode: refused
        'Path("out.txt").write_text("x")',
        "state_path.write_bytes(b'x')",
        "Path('d').mkdir(parents=True)",
        "Path('f').unlink()",
        "Path('f').touch()",
        'Path("f").open("wb")',
        "os.remove('f')",
        "os.makedirs('d')",
        "os.replace('a', 'b')",
        "os.system('curl ...')",
        "rm('f')",  # from os import remove as rm
        "importlib.import_module('googleapiclient')",
        "__import__('sqlalchemy')",
    ],
)
def test_node_mutating_filesystem_is_caught(facts: dict[str, ModuleFacts], call: str):
    """Standard-library side effects need no suspicious import, so calls are checked."""
    body = f"""
        import importlib
        import os
        from os import remove as rm
        from pathlib import Path


        def node(state, path="p", mode="r", state_path=Path("s")):
            {call}
            return {{}}
    """
    assert _plant(facts, body), f"checker missed: {call}"


@pytest.mark.parametrize(
    "call",
    [
        'open("in.txt")',
        'open("in.txt", "rb")',
        'Path("in.txt").read_text()',
        'Path("in.txt").open()',
        "webbrowser.open(url)",
        'text.replace("a", "b")',
    ],
)
def test_read_only_calls_are_not_flagged(facts: dict[str, ModuleFacts], call: str):
    """Reads and look-alike method names do not trip the check."""
    body = f"""
        import webbrowser
        from pathlib import Path


        def node(state, url="u", text="t"):
            {call}
            return {{}}
    """
    assert _plant(facts, body) == []


def test_node_reaching_persistence_through_another_layer_is_caught(
    facts: dict[str, ModuleFacts],
):
    """A two-hop route the layer graph allows one hop at a time is still refused.

    ``graphs`` may import ``state`` and ``state`` may import ``persistence``, so
    a graph could reach a DB session through a state helper without any single
    import crossing a layer the wrong way.
    """
    planted = dict(facts)
    helper = "personalos.state.planted_helper"
    planted[helper] = analyse_source(
        "from personalos.persistence.repositories import JobRepository\n",
        helper,
        REPO_ROOT / "personalos/state/planted_helper.py",
    )
    node = "personalos.graphs.planted_node"
    planted[node] = analyse_source(
        f"from {helper} import JobRepository\n",
        node,
        REPO_ROOT / "personalos/graphs/planted_node.py",
    )
    problems = [p for p in intent_boundary_violations(planted) if node in p]
    assert problems
    assert "persistence" in problems[0] and helper in problems[0]


def test_node_reaching_a_helper_that_calls_an_sdk_is_caught(facts: dict[str, ModuleFacts]):
    """A non-reasoning helper that imports an SDK is charged to the node that reaches it."""
    planted = dict(facts)
    helper = "personalos.state.planted_gmail"
    planted[helper] = analyse_source(
        "import googleapiclient.discovery\n",
        helper,
        REPO_ROOT / "personalos/state/planted_gmail.py",
    )
    node = "personalos.graphs.planted_node"
    planted[node] = analyse_source(
        f"import {helper}\n", node, REPO_ROOT / "personalos/graphs/planted_node.py"
    )
    problems = [p for p in intent_boundary_violations(planted) if node in p]
    assert problems and "provider-sdk" in problems[0]


def test_executor_is_the_sanctioned_door(facts: dict[str, ModuleFacts]):
    """Reaching effects through the executor is the intended route, not a breach."""
    assert _plant(facts, "from personalos.executor.job_search import JobSearchExecutor\n") == []


def test_executor_may_hold_side_effects(facts: dict[str, ModuleFacts]):
    """The rules bind reasoning layers only; the executor is where effects live."""
    module = "personalos.executor.planted"
    assert _plant(facts, "import httpx\nfrom sqlalchemy.orm import Session\n", module) == []


def test_model_layer_may_import_its_llm_sdk(facts: dict[str, ModuleFacts]):
    """``personalos.models`` owns model clients; the llm-sdk ban exempts it."""
    body = "from langchain_anthropic import ChatAnthropic\n"
    assert _plant(facts, body, "personalos.models.planted") == []


@pytest.mark.parametrize("layer", ["domain", "policy", "models", "events"])
def test_every_reasoning_layer_is_covered(facts: dict[str, ModuleFacts], layer: str):
    """The ban is not specific to graphs: the values a node emits must be pure too."""
    module = f"{LAYERS_BY_NAME[layer].modules[0]}.planted"
    assert _plant(facts, "from googleapiclient.discovery import build\n", module)


# ----------------------------------------------------------------------
# Spec sanity and doc drift.
# ----------------------------------------------------------------------


def test_intent_boundary_layers_are_real():
    """Every layer the spec names exists in ``boundaries.LAYERS``."""
    named = set(REASONING_LAYERS) | set(EFFECT_LAYERS) | set(EXECUTION_LAYERS)
    for rule in SIDE_EFFECT_RULES:
        named |= set(rule.exempt_layers)
    assert named <= set(LAYERS_BY_NAME), named - set(LAYERS_BY_NAME)


def test_reasoning_and_effect_layers_are_disjoint():
    """A layer cannot be both forbidden side effects and made of them."""
    assert not set(REASONING_LAYERS) & (set(EFFECT_LAYERS) | set(EXECUTION_LAYERS))


def test_boundary_doc_describes_node_shape_and_the_check():
    """The doc states the node contract and points at the enforcing module."""
    text = BOUNDARY_DOC.read_text(encoding="utf-8")
    assert "tests/architecture/intent_boundary.py" in text
    for step in ("validate", "emit", "state update"):
        assert step in text, f"node shape step '{step}' missing from the doc"
