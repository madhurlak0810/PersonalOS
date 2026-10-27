"""Machine-readable architecture boundaries.

This module is the single source of truth for who may import whom.
``docs/ARCHITECTURE_BOUNDARIES.md`` is the prose version of the same rules, and
``test_boundaries.py`` enforces them. If you need to change a boundary, change
it here and in the doc, in the same commit, with a reason.
"""

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root, derived from this file so the check works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Top-level packages that are part of the layered architecture. Everything
#: else (tests, evals, migrations) is outside it and is not checked.
CHECKED_PACKAGES = ("personalos", "apps", "mcp_servers")


@dataclass(frozen=True)
class Layer:
    """One architectural layer: what it is, and what it may depend on."""

    name: str
    #: Module prefixes that belong to this layer. Longest prefix wins.
    modules: tuple[str, ...]
    #: Names of layers this layer may import from. Importing within the same
    #: layer is always allowed and does not need to be listed.
    allows: tuple[str, ...] = ()
    #: What this layer is responsible for, quoted in failure messages.
    responsibility: str = ""


LAYERS: tuple[Layer, ...] = (
    Layer(
        name="composition",
        modules=("personalos", "personalos.bootstrap", "personalos.cli", "apps"),
        allows=("*",),
        responsibility=(
            "wires the layers together; the only place allowed to know about all of them"
        ),
    ),
    Layer(
        name="config",
        modules=("personalos.config",),
        allows=(),
        responsibility="reads settings from the environment",
    ),
    Layer(
        name="domain",
        modules=("personalos.domain",),
        allows=(),
        responsibility="defines entities and invariants; depends on nothing internal",
    ),
    Layer(
        name="policy",
        modules=("personalos.policy",),
        allows=("domain",),
        responsibility=("decides whether an intent may run; performs no I/O and never executes"),
    ),
    Layer(
        name="persistence",
        modules=("personalos.persistence",),
        allows=("domain", "config", "observability"),
        responsibility="stores and retrieves state; makes no decisions",
    ),
    Layer(
        name="secrets",
        modules=("personalos.secrets",),
        allows=("domain", "config"),
        responsibility=(
            "holds long-lived credentials in the OS secret store and exchanges them for "
            "short-lived tokens; reachable only from the executor's credential broker"
        ),
    ),
    Layer(
        name="events",
        modules=("personalos.events",),
        allows=("domain",),
        responsibility="carries domain events between layers",
    ),
    Layer(
        name="state",
        modules=("personalos.state",),
        allows=("domain", "persistence"),
        responsibility="tracks in-flight agent state",
    ),
    Layer(
        name="observability",
        modules=("personalos.observability",),
        allows=("domain", "config"),
        responsibility="logging, metrics and tracing",
    ),
    Layer(
        name="tools",
        modules=("personalos.tools",),
        allows=("domain", "policy", "observability"),
        responsibility=(
            "defines the tool boundary: the gateway that turns approved intents into adapter calls"
        ),
    ),
    Layer(
        name="mcp",
        modules=("personalos.mcp",),
        allows=("domain", "policy", "tools", "persistence", "config"),
        responsibility=(
            "speaks MCP; an adapter that executes approved intents and knows nothing "
            "about which servers exist"
        ),
    ),
    Layer(
        name="mcp_servers",
        modules=("mcp_servers",),
        allows=("domain", "mcp", "persistence", "config"),
        responsibility="concrete MCP tool implementations",
    ),
    Layer(
        name="providers",
        modules=("personalos.providers",),
        allows=("domain", "policy", "tools", "config"),
        responsibility=(
            "read-only job board adapters; executes approved job_providers intents and "
            "is reachable only through the tool gateway"
        ),
    ),
    Layer(
        name="retrieval",
        modules=("personalos.retrieval",),
        allows=("domain", "persistence", "config"),
        responsibility="retrieval and ranking over stored knowledge",
    ),
    Layer(
        name="models",
        modules=("personalos.models",),
        allows=("domain", "config", "observability"),
        responsibility="model clients and prompt plumbing",
    ),
    Layer(
        name="executor",
        modules=("personalos.executor",),
        allows=(
            "domain",
            "policy",
            "persistence",
            "tools",
            "events",
            "state",
            "config",
            "secrets",
            "observability",
        ),
        responsibility=("runs approved intents step by step; cannot reach a tool adapter directly"),
    ),
    Layer(
        name="graphs",
        modules=("personalos.graphs",),
        allows=(
            "domain",
            "policy",
            "executor",
            "events",
            "state",
            "models",
            "config",
            "observability",
        ),
        responsibility=(
            "orchestrates which step happens next; delegates all side effects to executors"
        ),
    ),
)

LAYERS_BY_NAME: dict[str, Layer] = {layer.name: layer for layer in LAYERS}


@dataclass(frozen=True)
class ForbiddenImport:
    """A ban that is finer-grained than the layer graph."""

    layer: str
    module_prefix: str
    reason: str


#: Bans on specific modules, for cases where a layer is allowed in general but
#: one of its modules is an adapter that must be reached through a port.
FORBIDDEN_IMPORTS: tuple[ForbiddenImport, ...] = (
    ForbiddenImport(
        layer="executor",
        module_prefix="personalos.tools.registry",
        reason=(
            "the registry takes raw arguments; executors must dispatch intents through "
            "personalos.tools.gateway so policy runs first"
        ),
    ),
    ForbiddenImport(
        layer="graphs",
        module_prefix="personalos.tools",
        reason=(
            "orchestration must not call tools at all; express the step as an intent "
            "and let an executor run it"
        ),
    ),
    ForbiddenImport(
        layer="executor",
        module_prefix="personalos.bootstrap",
        reason="the composition root wires executors, not the other way round",
    ),
    ForbiddenImport(
        layer="graphs",
        module_prefix="personalos.bootstrap",
        reason="the composition root wires graphs, not the other way round",
    ),
    # Tracing is cross-cutting, so the layers that open spans may import
    # `observability` -- but only its API (`observability.tracing`, which
    # depends on opentelemetry-api alone and is a no-op until a provider is
    # installed). Installing the SDK provider and its exporter is a process
    # decision, made once by the composition root.
    *(
        ForbiddenImport(
            layer=layer,
            module_prefix="personalos.observability.tracing_setup",
            reason=(
                "only the composition root installs the tracer provider and exporter; "
                "open spans through personalos.observability.tracing"
            ),
        )
        for layer in ("persistence", "tools", "models", "executor", "graphs")
    ),
)


#: The outbox's tables and the repository over them. Producers publish through
#: `EventPublisher` (or `stage_event`), workers consume through `EventConsumer`,
#: and both read and write `EventEnvelope`s; only the modules that *implement*
#: the outbox know it is a table, so the transport can be swapped
#: (`docs/EVENT_TRANSPORT.md`) without touching anyone else.
OUTBOX_INTERNALS: tuple[str, ...] = (
    "OutboxEventModel",
    "ConsumedEventModel",
    "OutboxEventRepository",
)

#: The modules that implement the outbox, and so may name its internals:
#: where they are defined, the transport itself, the consumer inbox, the
#: operator's view of parked events, and the workflow-command queue that
#: rides the same table under its own lease.
OUTBOX_OWNERS: tuple[str, ...] = (
    "personalos.persistence.models",
    "personalos.persistence.repositories",
    "personalos.persistence.event_outbox",
    "personalos.persistence.event_inbox",
    "personalos.persistence.event_failures",
    "personalos.persistence.workflow_commands",
)


#: Only these modules may mint an approval. Anything else that references the
#: minting internals is trying to fabricate authorization.
APPROVAL_MINTING_MODULES = ("personalos.policy.intents", "personalos.policy.engine")


def layer_for_module(module: str) -> Layer | None:
    """Return the layer owning a module, by longest matching prefix."""
    best: Layer | None = None
    best_len = -1
    for layer in LAYERS:
        for prefix in layer.modules:
            if module == prefix or module.startswith(prefix + "."):
                if len(prefix) > best_len:
                    best, best_len = layer, len(prefix)
    return best


def is_allowed(source: Layer, target: Layer) -> bool:
    """True when ``source`` may import from ``target``."""
    if source.name == target.name:
        return True
    if "*" in source.allows:
        return True
    return target.name in source.allows


@dataclass
class ImportEdge:
    """One import statement, resolved to absolute module names."""

    source_module: str
    target_module: str
    path: Path
    lineno: int

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.lineno} {self.source_module} -> {self.target_module}"


@dataclass
class SymbolReference:
    """A name a module imports or reaches as an attribute, for symbol-level bans."""

    source_module: str
    symbol: str
    path: Path
    lineno: int

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.lineno} {self.source_module} -> {self.symbol}"


@dataclass
class Scan:
    """Result of scanning the checked packages."""

    edges: list[ImportEdge] = field(default_factory=list)
    modules: set[str] = field(default_factory=set)
    files: list[Path] = field(default_factory=list)
    #: References to `OUTBOX_INTERNALS`, however they are spelled: imported by
    #: name, or reached as an attribute of an imported module.
    symbols: list[SymbolReference] = field(default_factory=list)


def module_name_for_path(path: Path) -> str:
    """Convert a file path to its dotted module name."""
    rel = path.resolve().relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(source_module: str, node: ast.ImportFrom, is_package: bool) -> str:
    """Resolve a relative import to an absolute module name."""
    parts = source_module.split(".")
    # Inside a package __init__, level 1 refers to the package itself; inside a
    # module it refers to the containing package.
    base = parts if is_package else parts[:-1]
    trimmed = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    target = ".".join(trimmed)
    return f"{target}.{node.module}" if node.module else target


def iter_source_files(packages: Iterable[str] = CHECKED_PACKAGES) -> list[Path]:
    """All Python files in the checked packages."""
    files: list[Path] = []
    for package in packages:
        files.extend(sorted((REPO_ROOT / package).rglob("*.py")))
    return files


def scan_imports(packages: Iterable[str] = CHECKED_PACKAGES) -> Scan:
    """Parse every checked file and collect its internal import edges.

    Uses the AST rather than importing anything, so the check is fast, has no
    side effects, and catches imports nested inside functions.
    """
    scan = Scan()
    internal_roots = tuple(CHECKED_PACKAGES)

    for path in iter_source_files(packages):
        source_module = module_name_for_path(path)
        is_package = path.name == "__init__.py"
        scan.modules.add(source_module)
        scan.files.append(path)

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scan.symbols.extend(outbox_references(tree, source_module, path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    targets = [_resolve_relative(source_module, node, is_package)]
                elif node.module:
                    targets = [node.module]

            for target in targets:
                if not target.startswith(internal_roots):
                    continue  # third-party or stdlib: not our concern
                scan.edges.append(
                    ImportEdge(
                        source_module=source_module,
                        target_module=target,
                        path=path,
                        lineno=node.lineno,
                    )
                )
    return scan


def outbox_references(tree: ast.AST, source_module: str, path: Path) -> list[SymbolReference]:
    """Every reference to `OUTBOX_INTERNALS` in `tree`: imported by name, or as an attribute."""
    references: list[SymbolReference] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        references.extend(
            SymbolReference(source_module, name, path, node.lineno)
            for name in names
            if name in OUTBOX_INTERNALS
        )
    return references


def violations(scan: Scan | None = None) -> list[str]:
    """Every boundary violation in the tree, as human-readable strings."""
    scan = scan or scan_imports()
    problems: list[str] = []

    for edge in scan.edges:
        source = layer_for_module(edge.source_module)
        target = layer_for_module(edge.target_module)

        if source is None:
            problems.append(
                f"{edge}: module '{edge.source_module}' belongs to no declared layer; "
                f"add it to LAYERS in tests/architecture/boundaries.py"
            )
            continue
        if target is None:
            problems.append(
                f"{edge}: imported module '{edge.target_module}' belongs to no declared "
                f"layer; add it to LAYERS in tests/architecture/boundaries.py"
            )
            continue

        if not is_allowed(source, target):
            problems.append(
                f"{edge}: layer '{source.name}' may not import layer '{target.name}'. "
                f"{source.name} {source.responsibility}; allowed: "
                f"{', '.join(source.allows) or 'nothing internal'}"
            )
            continue

        for ban in FORBIDDEN_IMPORTS:
            if source.name != ban.layer:
                continue
            if edge.target_module == ban.module_prefix or edge.target_module.startswith(
                ban.module_prefix + "."
            ):
                problems.append(
                    f"{edge}: '{ban.layer}' may not import '{ban.module_prefix}'. {ban.reason}"
                )

    for reference in scan.symbols:
        if reference.source_module not in OUTBOX_OWNERS:
            problems.append(
                f"{reference}: only the modules that implement the outbox may name "
                f"'{reference.symbol}'. Publish through EventPublisher / stage_event and "
                f"consume through EventConsumer (personalos.domain.event_transport); "
                f"see docs/EVENT_TRANSPORT.md"
            )

    return problems


__all__ = [
    "REPO_ROOT",
    "CHECKED_PACKAGES",
    "Layer",
    "LAYERS",
    "LAYERS_BY_NAME",
    "ForbiddenImport",
    "FORBIDDEN_IMPORTS",
    "APPROVAL_MINTING_MODULES",
    "OUTBOX_INTERNALS",
    "OUTBOX_OWNERS",
    "ImportEdge",
    "Scan",
    "SymbolReference",
    "layer_for_module",
    "is_allowed",
    "module_name_for_path",
    "iter_source_files",
    "outbox_references",
    "scan_imports",
    "violations",
]
