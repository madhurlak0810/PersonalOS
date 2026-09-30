"""The intent-only boundary: reasoning code proposes, it never acts.

``boundaries.py`` polices imports *between our own layers*. That cannot see the
failure this module exists for: a graph node that does ``import
googleapiclient``, opens a ``sqlalchemy`` session, or calls
``Path.write_text``. None of those are internal imports, so the layer graph
waves them through, and each one is a side effect that never passed through
``ActionIntent -> PolicyEngine -> executor``.

So this module states, for the *reasoning* layers -- the ones an LLM-backed
node lives in or calls into -- which outside-world surfaces are off limits,
and checks every file in those layers by AST:

1. **Imports.** No provider SDK, database driver, network or mail client,
   process spawner, filesystem-mutation module, or raw model SDK (outside
   ``personalos.models``). ``import x`` and ``from x import y`` both count,
   including imports nested inside functions.
2. **Calls.** No filesystem mutation through the standard library, which needs
   no suspicious import at all: ``open(path, "w")``, ``Path.write_text``,
   ``os.remove``, ``os.system`` and friends. Dynamic imports are refused too,
   because they cannot be checked.
3. **Reach.** A reasoning module may not *transitively* reach an effect layer
   (``persistence``, ``tools``, ``mcp``, ``mcp_servers``, composition) or a
   module that breaks rules 1-2, except through the executor layer, which is
   the sanctioned door. This closes routes the layer graph permits one hop at
   a time -- ``graphs -> state -> persistence``, for instance.

Like ``boundaries.py`` it imports nothing it checks, so it runs with nothing
installed (``scripts/check_boundaries.py``). The prose version is the
"Intent-only boundary" section of ``docs/ARCHITECTURE_BOUNDARIES.md``; change
both together.
"""

import ast
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from tests.architecture.boundaries import (
    CHECKED_PACKAGES,
    REPO_ROOT,
    _resolve_relative,
    iter_source_files,
    layer_for_module,
    module_name_for_path,
)

#: Layers whose code reasons, classifies, plans or decides. An LLM-backed node
#: lives in ``graphs``; the model client it calls lives in ``models``; the
#: values it emits live in ``domain``; ``policy`` and ``events`` are pure by
#: design. None of them may perform a side effect.
REASONING_LAYERS: tuple[str, ...] = ("graphs", "models", "domain", "policy", "events")

#: Layers that hold real I/O. A reasoning module reaching one of these, other
#: than through ``EXECUTION_LAYERS``, holds a side effect nobody approved.
EFFECT_LAYERS: tuple[str, ...] = ("persistence", "tools", "mcp", "mcp_servers", "composition")

#: The sanctioned door. Reach analysis does not expand past it: an executor
#: holds a ``ToolGateway``, which authorizes every intent before it runs.
EXECUTION_LAYERS: tuple[str, ...] = ("executor",)

_REMEDY = (
    "Emit a typed ActionIntent (or ToolIntent) and let the executor perform it; "
    "if the node needs data, declare a Protocol port and have the composition "
    "root bind an adapter."
)


@dataclass(frozen=True)
class SideEffectRule:
    """A family of outside-world surfaces reasoning code may not import."""

    category: str
    #: Module prefixes; ``x`` bans ``x`` and ``x.*``.
    modules: tuple[str, ...]
    reason: str
    #: Reasoning layers this particular rule does not apply to.
    exempt_layers: tuple[str, ...] = ()


SIDE_EFFECT_RULES: tuple[SideEffectRule, ...] = (
    SideEffectRule(
        category="provider-sdk",
        modules=(
            # Gmail / Google Calendar
            "googleapiclient",
            "google.oauth2",
            "google.auth",
            "google_auth_oauthlib",
            "google_auth_httplib2",
            "gmail",
            "simplegmail",
            "ezgmail",
            "gcsa",
            # Other mail / calendar providers
            "caldav",
            "msal",
            "msgraph",
            "O365",
            "exchangelib",
            # Job boards
            "linkedin_api",
            "jobspy",
            "indeed",
            "greenhouse",
            "lever",
            # An MCP client session can call any tool the server exposes.
            "mcp",
        ),
        reason="a provider client performs writes the policy engine never saw",
    ),
    SideEffectRule(
        category="database",
        modules=(
            "sqlalchemy",
            "psycopg2",
            "psycopg",
            "asyncpg",
            "sqlite3",
            "aiosqlite",
            "pgvector",
            "alembic",
            "redis",
            "pymongo",
            "motor",
            "celery",
            # LangGraph's own savers are fine as a port (``langgraph.checkpoint.base``);
            # the concrete database-backed ones are a raw DB write path.
            "langgraph.checkpoint.postgres",
            "langgraph.checkpoint.sqlite",
            "langgraph.store.postgres",
        ),
        reason="a raw DB or broker write is a state transition nothing journals or audits",
    ),
    SideEffectRule(
        category="network",
        modules=(
            "httpx",
            "requests",
            "aiohttp",
            "urllib.request",
            "urllib3",
            "http.client",
            "smtplib",
            "imaplib",
            "poplib",
            "ftplib",
            "socket",
            "websockets",
        ),
        reason="a bare HTTP or mail client is a hand-rolled provider SDK",
    ),
    SideEffectRule(
        category="process",
        modules=("subprocess", "pty"),
        reason="a spawned process can do anything, and none of it is an intent",
    ),
    SideEffectRule(
        category="filesystem",
        modules=("shutil", "tempfile"),
        reason="filesystem mutation belongs to an executor, behind an intent",
    ),
    SideEffectRule(
        category="llm-sdk",
        modules=("anthropic", "langchain_anthropic", "openai", "langchain_openai"),
        reason=(
            "model clients live in personalos.models behind a port; a node holding "
            "a raw client can hand the model tools that act outside policy"
        ),
        exempt_layers=("models",),
    ),
)

#: Fully qualified callables that mutate the filesystem or spawn a process.
FORBIDDEN_CALLS: dict[str, str] = {
    **{
        f"os.{name}": "filesystem"
        for name in (
            "remove",
            "unlink",
            "rmdir",
            "removedirs",
            "mkdir",
            "makedirs",
            "rename",
            "renames",
            "replace",
            "truncate",
            "ftruncate",
            "chmod",
            "chown",
            "link",
            "symlink",
            "write",
            "open",
            "mkfifo",
            "utime",
        )
    },
    **{
        f"os.{name}": "process"
        for name in (
            "system",
            "popen",
            "execv",
            "execve",
            "execl",
            "execlp",
            "execvp",
            "spawnl",
            "spawnv",
            "posix_spawn",
            "fork",
        )
    },
    "importlib.import_module": "dynamic-import",
    "__import__": "dynamic-import",
}

#: Method names that mutate the filesystem on ``pathlib.Path``. Matched on any
#: receiver, since the AST does not know types; these names are specific
#: enough that a false positive is rarer than a missed ``Path`` write.
#: (``rename``/``replace`` are deliberately absent: ``str.replace`` is
#: everywhere. The ``os.`` spellings above still catch them.)
FORBIDDEN_PATH_METHODS: frozenset[str] = frozenset(
    {"write_text", "write_bytes", "touch", "unlink", "mkdir", "rmdir", "symlink_to", "hardlink_to"}
)

#: Callables that open a file; flagged only in a writable or unknowable mode.
_OPENERS = frozenset({"open", "io.open", "codecs.open"})
_WRITE_MODE_CHARS = frozenset("wax+")


@dataclass
class Finding:
    """One side effect found directly in a module's source."""

    lineno: int
    category: str
    detail: str
    reason: str


@dataclass
class ModuleFacts:
    """What the checker needs to know about one module."""

    module: str
    path: Path
    #: Internal modules imported, as (module, lineno).
    internal_imports: list[tuple[str, int]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def rel_path(self) -> str:
        try:
            return self.path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _rule_for_import(module: str, layer_name: str) -> SideEffectRule | None:
    for rule in SIDE_EFFECT_RULES:
        if layer_name in rule.exempt_layers:
            continue
        if any(_matches(module, prefix) for prefix in rule.modules):
            return rule
    return None


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _open_mode(call: ast.Call, is_method: bool) -> ast.expr | None:
    """The mode argument of an open call, if one was passed."""
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    # ``open(file, mode)`` vs ``path.open(mode)``.
    index = 0 if is_method else 1
    if len(call.args) > index:
        return call.args[index]
    return None


def _writable(mode: ast.expr | None) -> bool | None:
    """True/False for a literal mode; None when it cannot be known statically."""
    if mode is None:
        return False  # default "r"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return bool(_WRITE_MODE_CHARS & set(mode.value))
    return None


def _record_import(facts: ModuleFacts, target: str, lineno: int, layer_name: str) -> None:
    """File one import as an internal edge or, if it is banned, a finding."""
    if target.startswith(CHECKED_PACKAGES):
        facts.internal_imports.append((target, lineno))
        return
    rule = _rule_for_import(target, layer_name)
    if rule:
        facts.findings.append(Finding(lineno, rule.category, f"imports '{target}'", rule.reason))


def _scan_imports(
    tree: ast.AST, facts: ModuleFacts, layer_name: str, is_package: bool
) -> dict[str, str]:
    """Record imports, and return local name -> fully qualified name.

    The alias map is what lets ``from os import remove as rm; rm(p)`` resolve
    to ``os.remove`` when calls are checked.
    """
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _record_import(facts, alias.name, node.lineno, layer_name)
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve_relative(facts.module, node, is_package)
                if node.level
                else node.module or ""
            )
            _record_import(facts, base, node.lineno, layer_name)
            base_banned = _rule_for_import(base, layer_name) is not None
            for alias in node.names:
                if alias.name == "*":
                    continue
                qualified = f"{base}.{alias.name}"
                aliases[alias.asname or alias.name] = qualified
                # ``from personalos.persistence import repositories`` names a
                # submodule, and ``from google import auth`` bans the same as
                # ``import google.auth``; either way, record the qualified name.
                if base.startswith(CHECKED_PACKAGES) or not base_banned:
                    _record_import(facts, qualified, node.lineno, layer_name)
    return aliases


def _call_finding(node: ast.Call, aliases: dict[str, str]) -> Finding | None:
    """The side effect a single call performs, if any."""
    dotted = _dotted(node.func)
    qualified = ""
    if dotted is not None:
        head, _, rest = dotted.partition(".")
        qualified = f"{aliases.get(head, head)}.{rest}" if rest else aliases.get(head, head)

    category = FORBIDDEN_CALLS.get(qualified)
    if category:
        reason = (
            "a dynamic import cannot be checked, so it is refused"
            if category == "dynamic-import"
            else "a node that mutates the world directly skips policy and the journal"
        )
        return Finding(node.lineno, category, f"calls '{qualified}'", reason)

    write_reason = "writing a file is a side effect; reading one should go through a port"
    # Methods are matched on any receiver, including ``Path(p).write_text()``
    # where the receiver is a call and there is no dotted name at all.
    method = node.func.attr if isinstance(node.func, ast.Attribute) else None
    if qualified in _OPENERS:
        # ``open(p, mode)``: a mode we cannot read is refused, since the check
        # cannot show it is read-only.
        writable = _writable(_open_mode(node, is_method=False))
        if writable is False:
            return None
        what = "a writable mode" if writable else "a mode that is not a literal"
        return Finding(node.lineno, "filesystem", f"calls '{qualified}' with {what}", write_reason)
    if method == "open":
        # ``something.open(...)`` is only a file when it has a file mode, so
        # flag a literal writable one and leave ``webbrowser.open(url)`` be.
        if _writable(_open_mode(node, is_method=True)):
            return Finding(
                node.lineno, "filesystem", "calls '.open()' with a writable mode", write_reason
            )
        return None
    if method in FORBIDDEN_PATH_METHODS:
        return Finding(
            node.lineno,
            "filesystem",
            f"calls '.{method}()'",
            "filesystem mutation belongs to an executor, behind an intent",
        )
    return None


def analyse_source(source: str, module: str, path: Path) -> ModuleFacts:
    """Parse one module and record its internal imports and direct side effects.

    ``module`` decides which layer's rules apply, so tests can plant a
    synthetic file under any layer without writing it into the tree.
    """
    facts = ModuleFacts(module=module, path=path)
    layer = layer_for_module(module)
    tree = ast.parse(source, filename=str(path))
    aliases = _scan_imports(
        tree, facts, layer.name if layer else "", is_package=path.name == "__init__.py"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            finding = _call_finding(node, aliases)
            if finding:
                facts.findings.append(finding)
    return facts


def collect_facts() -> dict[str, ModuleFacts]:
    """Facts for every module in the checked packages."""
    facts: dict[str, ModuleFacts] = {}
    for path in iter_source_files():
        module = module_name_for_path(path)
        facts[module] = analyse_source(path.read_text(encoding="utf-8"), module, path)
    return facts


def _known(target: str, facts: dict[str, ModuleFacts]) -> str | None:
    """The module an import target names, trimming attribute suffixes."""
    while target:
        if target in facts:
            return target
        target = target.rpartition(".")[0]
    return None


def _layer_name(module: str) -> str:
    layer = layer_for_module(module)
    return layer.name if layer else ""


def _reach_violations(root: ModuleFacts, facts: dict[str, ModuleFacts]) -> list[str]:
    """Effects a reasoning module reaches through other internal modules."""
    problems: list[str] = []
    # module -> the import chain that reached it, for the failure message.
    seen: dict[str, tuple[str, ...]] = {root.module: (root.module,)}
    queue = deque([root.module])

    while queue:
        current = queue.popleft()
        for target, _lineno in facts[current].internal_imports:
            module = _known(target, facts)
            if module is None or module in seen:
                continue
            chain = (*seen[current], module)
            seen[module] = chain
            layer = _layer_name(module)
            if layer in EXECUTION_LAYERS:
                continue  # the sanctioned door; its effects are gated
            if layer in EFFECT_LAYERS:
                problems.append(
                    f"{root.rel_path}: {root.module} reaches effect layer '{layer}' via "
                    f"{' -> '.join(chain)}. {_REMEDY}"
                )
                continue
            reached = facts[module]
            if layer not in REASONING_LAYERS and reached.findings:
                # A reasoning module checks itself; anything else it reaches is
                # checked here, on its behalf.
                first = reached.findings[0]
                problems.append(
                    f"{root.rel_path}: {root.module} reaches a side effect via "
                    f"{' -> '.join(chain)} ({reached.rel_path}:{first.lineno} "
                    f"[{first.category}] {first.detail}). {_REMEDY}"
                )
            queue.append(module)
    return problems


def intent_boundary_violations(facts: dict[str, ModuleFacts] | None = None) -> list[str]:
    """Every breach of the intent-only boundary, as human-readable strings."""
    facts = facts if facts is not None else collect_facts()
    problems: list[str] = []
    for module in sorted(facts):
        layer = _layer_name(module)
        if layer not in REASONING_LAYERS:
            continue
        mod = facts[module]
        for finding in mod.findings:
            problems.append(
                f"{mod.rel_path}:{finding.lineno} {module}: [{finding.category}] "
                f"{finding.detail} -- layer '{layer}' may only emit intents; "
                f"{finding.reason}. {_REMEDY}"
            )
        problems.extend(_reach_violations(mod, facts))
    return problems


__all__ = [
    "REASONING_LAYERS",
    "EFFECT_LAYERS",
    "EXECUTION_LAYERS",
    "SideEffectRule",
    "SIDE_EFFECT_RULES",
    "FORBIDDEN_CALLS",
    "FORBIDDEN_PATH_METHODS",
    "Finding",
    "ModuleFacts",
    "analyse_source",
    "collect_facts",
    "intent_boundary_violations",
]
