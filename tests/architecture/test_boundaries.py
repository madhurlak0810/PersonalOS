"""Enforce the architecture boundaries declared in ``boundaries.py``.

These tests are the executable half of ``docs/ARCHITECTURE_BOUNDARIES.md``. A
failure here means a dependency was added that crosses a layer the wrong way,
not that the check is being fussy: fix the import, or change the declared
boundary deliberately in both the spec and the doc.
"""

import pytest

from tests.architecture.boundaries import (
    APPROVAL_MINTING_MODULES,
    FORBIDDEN_IMPORTS,
    LAYERS,
    LAYERS_BY_NAME,
    REPO_ROOT,
    ImportEdge,
    Scan,
    iter_source_files,
    layer_for_module,
    module_name_for_path,
    scan_imports,
    violations,
)

BOUNDARY_DOC = REPO_ROOT / "docs" / "ARCHITECTURE_BOUNDARIES.md"


@pytest.fixture(scope="module")
def scan() -> Scan:
    """Scan the checked packages once for the whole module."""
    return scan_imports()


def test_no_boundary_violations(scan: Scan):
    """Every internal import respects the layer graph."""
    problems = violations(scan)
    assert not problems, "architecture boundary violations:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


def test_every_module_belongs_to_a_layer():
    """No module sits outside the declared architecture."""
    orphans = [
        module_name_for_path(path)
        for path in iter_source_files()
        if layer_for_module(module_name_for_path(path)) is None
    ]
    assert not orphans, f"modules with no declared layer: {orphans}"


def test_layer_allowances_reference_real_layers():
    """The spec cannot allow a layer that does not exist."""
    for layer in LAYERS:
        for allowed in layer.allows:
            assert allowed == "*" or allowed in LAYERS_BY_NAME, (
                f"layer '{layer.name}' allows unknown layer '{allowed}'"
            )


def test_forbidden_import_rules_reference_real_layers():
    """Fine-grained bans cannot name a layer that does not exist."""
    for ban in FORBIDDEN_IMPORTS:
        assert ban.layer in LAYERS_BY_NAME, f"unknown layer in ban: {ban.layer}"


# ----------------------------------------------------------------------
# The checker itself must be able to fail. A boundary test that cannot
# detect a violation is worse than no test, because it reads as assurance.
# ----------------------------------------------------------------------


def _planted(source_module: str, target_module: str) -> Scan:
    """A scan containing exactly one synthetic import edge."""
    return Scan(
        edges=[
            ImportEdge(
                source_module=source_module,
                target_module=target_module,
                path=REPO_ROOT / "personalos" / "planted.py",
                lineno=1,
            )
        ]
    )


@pytest.mark.parametrize(
    "source_module,target_module",
    [
        # The bypass this whole boundary exists to prevent.
        ("personalos.executor.job_search", "personalos.mcp.manager"),
        ("personalos.executor.job_search", "mcp_servers.jobs.server"),
        ("personalos.executor.job_search", "personalos.tools.registry"),
        ("personalos.graphs.job_search", "personalos.mcp.adapter"),
        ("personalos.graphs.job_search", "personalos.tools.gateway"),
        ("personalos.graphs.job_search", "personalos.persistence.repositories"),
        # Inverted dependencies.
        ("personalos.policy.engine", "personalos.persistence.repositories"),
        ("personalos.policy.engine", "personalos.tools.gateway"),
        ("personalos.domain.models", "personalos.persistence.models"),
        ("personalos.persistence.repositories", "personalos.executor.job_search"),
        ("personalos.tools.gateway", "personalos.mcp.manager"),
        ("personalos.mcp.manager", "mcp_servers.jobs.server"),
        ("personalos.executor.job_search", "personalos.bootstrap"),
    ],
)
def test_checker_rejects_planted_violation(source_module: str, target_module: str):
    """Each bypass we care about is actually detected."""
    problems = violations(_planted(source_module, target_module))
    assert problems, f"checker failed to flag {source_module} -> {target_module}"


@pytest.mark.parametrize(
    "source_module,target_module",
    [
        ("personalos.executor.job_search", "personalos.tools.gateway"),
        ("personalos.executor.job_search", "personalos.policy"),
        ("personalos.executor.job_search", "personalos.persistence.repositories"),
        ("personalos.graphs.job_search", "personalos.executor"),
        ("personalos.tools.gateway", "personalos.policy.intents"),
        ("personalos.mcp.adapter", "personalos.policy"),
        ("personalos.bootstrap", "mcp_servers.jobs.server"),
        ("apps.api.routes.jobs", "personalos.bootstrap"),
    ],
)
def test_checker_permits_intended_dependency(source_module: str, target_module: str):
    """The intended wiring is not flagged, so the check stays usable."""
    problems = violations(_planted(source_module, target_module))
    assert not problems, f"checker wrongly flagged {source_module} -> {target_module}"


# ----------------------------------------------------------------------
# Named guarantees, stated directly so a reader of the test names can see
# what the architecture promises.
# ----------------------------------------------------------------------


def _targets_from(scan: Scan, layer_name: str) -> set:
    return {
        edge.target_module
        for edge in scan.edges
        if (layer_for_module(edge.source_module) or LAYERS_BY_NAME["domain"]).name
        == layer_name
    }


def test_executor_never_imports_a_tool_adapter(scan: Scan):
    """Executors reach tools only through the gateway port."""
    banned_prefixes = ("personalos.mcp", "mcp_servers", "personalos.tools.registry")
    offenders = [
        target
        for target in _targets_from(scan, "executor")
        if target.startswith(banned_prefixes)
    ]
    assert not offenders, f"executor imports tool adapters directly: {offenders}"


def test_orchestration_never_imports_tools_or_storage(scan: Scan):
    """Graphs delegate side effects instead of performing them."""
    banned_prefixes = (
        "personalos.mcp",
        "mcp_servers",
        "personalos.tools",
        "personalos.persistence",
    )
    offenders = [
        target
        for target in _targets_from(scan, "graphs")
        if target.startswith(banned_prefixes)
    ]
    assert not offenders, f"graphs bypass the executor: {offenders}"


def test_policy_depends_only_on_domain(scan: Scan):
    """The policy layer stays pure so it can be tested exhaustively."""
    offenders = [
        target
        for target in _targets_from(scan, "policy")
        if not target.startswith(("personalos.policy", "personalos.domain"))
    ]
    assert not offenders, f"policy took on a dependency: {offenders}"


def test_domain_depends_on_nothing_internal(scan: Scan):
    """Domain models remain free of infrastructure."""
    offenders = [
        target
        for target in _targets_from(scan, "domain")
        if not target.startswith("personalos.domain")
    ]
    assert not offenders, f"domain took on a dependency: {offenders}"


def test_only_the_policy_layer_can_mint_approvals():
    """No layer outside policy touches the approval-minting internals.

    ``ApprovedIntent`` also refuses direct construction at runtime; this check
    catches the attempt at review time, and covers helper code that might reach
    for the mint token instead.
    """
    needles = ("_MINT_TOKEN", "mint_approved_intent")
    offenders = []
    for path in iter_source_files():
        module = module_name_for_path(path)
        if module in APPROVAL_MINTING_MODULES or module == "personalos.policy":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} ({needle})")
    assert not offenders, (
        "approvals may only be minted inside the policy layer; found references in: "
        f"{offenders}"
    )


# ----------------------------------------------------------------------
# Doc and spec must not drift apart.
# ----------------------------------------------------------------------


def test_boundary_doc_exists():
    """The prose boundary doc is part of the repo."""
    assert BOUNDARY_DOC.exists(), f"missing boundary doc at {BOUNDARY_DOC}"


def test_boundary_doc_documents_every_layer():
    """Every declared layer is described in the doc."""
    text = BOUNDARY_DOC.read_text(encoding="utf-8")
    missing = [layer.name for layer in LAYERS if layer.name not in text]
    assert not missing, f"layers missing from {BOUNDARY_DOC.name}: {missing}"


def test_boundary_doc_points_at_the_spec():
    """The doc tells readers where the enforced version lives."""
    text = BOUNDARY_DOC.read_text(encoding="utf-8")
    assert "tests/architecture/boundaries.py" in text
