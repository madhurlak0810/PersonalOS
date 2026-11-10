"""Policy rules.

A rule inspects a :class:`~personalos.policy.intents.ToolIntent` and either
expresses an opinion (allow / deny / require approval) or abstains by returning
``None``. Rules never perform side effects and never touch storage or tool
adapters: they are pure functions of the intent, which is what makes the policy
layer cheap to test exhaustively.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from personalos.policy.intents import Decision, IntentOrigin, PolicyDecision, ToolIntent

IDEMPOTENCY_KEY_ARG = "idempotency_key"


class PolicyRule(ABC):
    """One check against an intent."""

    #: Stable identifier recorded on the decision, for audit and tests.
    name: str = "unnamed"

    @abstractmethod
    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Return a decision, or None to abstain."""

    # Helpers so rules do not each rebuild the decision payload.
    def _decide(self, intent: ToolIntent, decision: Decision, reason: str) -> PolicyDecision:
        return PolicyDecision(
            intent_id=intent.intent_id,
            tool_ref=intent.tool_ref,
            decision=decision,
            rule=self.name,
            reason=reason,
        )

    def allow(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build an allow decision attributed to this rule."""
        return self._decide(intent, Decision.ALLOW, reason)

    def deny(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build a deny decision attributed to this rule."""
        return self._decide(intent, Decision.DENY, reason)

    def require_approval(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build an approval-required decision attributed to this rule."""
        return self._decide(intent, Decision.REQUIRE_APPROVAL, reason)


class RequireProvenanceRule(PolicyRule):
    """Reject intents that do not say who asked for them.

    Provenance is what makes every other rule auditable, so an intent without
    it is treated as malformed rather than merely untidy.
    """

    name = "require_provenance"

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny when ``requested_by`` was left unset."""
        if not intent.requested_by or intent.requested_by == "unknown":
            return self.deny(intent, "intent has no requested_by provenance")
        return None


class ToolAllowlistRule(PolicyRule):
    """Only tools on the allowlist may run at all.

    This is the rule that makes the system default-deny in practice: a newly
    added MCP tool is unreachable until it is listed here.
    """

    name = "tool_allowlist"

    def __init__(self, allowed: Iterable[str]):
        """Take fully-qualified ``server.tool`` references."""
        self.allowed: set[str] = set(allowed)

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Allow listed tools, deny everything else."""
        if intent.tool_ref not in self.allowed:
            return self.deny(intent, f"tool '{intent.tool_ref}' is not on the allowlist")
        return self.allow(intent, f"tool '{intent.tool_ref}' is allowlisted")


class ArgumentAllowlistRule(PolicyRule):
    """Reject arguments a tool did not declare.

    Tool adapters splat arguments into handlers, so an unexpected key is a
    confused-deputy risk: it lets a caller (in particular a model) reach
    parameters the orchestration layer never meant to expose.
    """

    name = "argument_allowlist"

    def __init__(self, allowed_arguments: Mapping[str, Iterable[str]]):
        """Map ``server.tool`` to the argument names it accepts."""
        self.allowed_arguments = {
            tool_ref: set(names) | {IDEMPOTENCY_KEY_ARG}
            for tool_ref, names in allowed_arguments.items()
        }

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny unknown argument names; abstain for undeclared tools."""
        allowed = self.allowed_arguments.get(intent.tool_ref)
        if allowed is None:
            # No declaration for this tool: ToolAllowlistRule owns that call.
            return None
        unexpected = sorted(set(intent.arguments) - allowed)
        if unexpected:
            return self.deny(
                intent,
                f"unexpected arguments for '{intent.tool_ref}': {unexpected}",
            )
        return None


class MutatingToolRule(PolicyRule):
    """Mutating tools need an idempotency key, and normally need approval.

    The key requirement mirrors the guard in
    ``personalos.persistence.idempotency``: a side effect that cannot be
    deduplicated must not be attempted at all. Approval is required unless the
    tool is explicitly listed as safe to run unattended.
    """

    name = "mutating_tool"

    def __init__(self, auto_approved: Iterable[str] | None = None):
        """Take ``server.tool`` refs that may mutate without human sign-off."""
        self.auto_approved: set[str] = set(auto_approved or ())

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Abstain for read-only intents; gate mutating ones."""
        if not intent.mutating:
            return None
        key = intent.arguments.get(IDEMPOTENCY_KEY_ARG)
        if not key:
            return self.deny(
                intent,
                f"mutating tool '{intent.tool_ref}' requires an {IDEMPOTENCY_KEY_ARG}",
            )
        if intent.tool_ref in self.auto_approved:
            return self.allow(intent, f"mutating tool '{intent.tool_ref}' is auto-approved")
        return self.require_approval(
            intent, f"mutating tool '{intent.tool_ref}' needs human approval"
        )


class NodeToolAllowlistRule(PolicyRule):
    """Each graph node may reach only the tools listed for it.

    `ToolAllowlistRule` says which tools exist; this says which *node* may
    use which of them. A node is identified by the intent's `requested_by`,
    which the node's own code sets -- a model can propose a tool call, but it
    cannot choose which node it is proposing from -- so a model that asks a
    drafting node to send mail is denied here, before approval is even a
    question. A denial outranks any approval grant.

    Applies only to tools of `governed_servers`; everything else is left to
    the rest of the chain. Within those servers it is default-deny: a node
    with no entry reaches nothing.
    """

    name = "node_tool_allowlist"

    def __init__(
        self,
        allowlists: Mapping[str, Iterable[str]],
        governed_servers: Iterable[str] = ("google",),
    ):
        """Map a node's `requested_by` to the `server.tool` refs it may call."""
        self.allowlists: dict[str, frozenset[str]] = {
            node: frozenset(tools) for node, tools in allowlists.items()
        }
        self.governed_servers: frozenset[str] = frozenset(governed_servers)

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny a governed tool the requesting node is not listed for; else abstain."""
        if intent.server not in self.governed_servers:
            return None
        if intent.tool_ref not in self.allowlists.get(intent.requested_by, frozenset()):
            return self.deny(
                intent,
                f"node '{intent.requested_by}' is not allowed to call '{intent.tool_ref}'",
            )
        return None


class UntrustedOriginRule(PolicyRule):
    """Escalate anything a model proposed.

    A model-authored intent has already passed the allowlist by this point, so
    the remaining question is trust: reads proceed, side effects wait for a
    human regardless of any auto-approval granted to system code paths.
    """

    name = "untrusted_origin"

    def __init__(self, untrusted: Iterable[IntentOrigin] = (IntentOrigin.LLM,)):
        """Take the origins to treat as untrusted."""
        self.untrusted: set[IntentOrigin] = set(untrusted)

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Require approval for mutating intents from untrusted origins."""
        if intent.origin not in self.untrusted:
            return None
        if intent.mutating:
            return self.require_approval(
                intent,
                f"mutating intent from untrusted origin '{intent.origin.value}'",
            )
        return None


class DocumentOverwriteRule(PolicyRule):
    """Replacing an existing document needs its precondition and a human.

    Whether a write replaces something is read off the intent, not the disk
    (a rule performs no I/O): the files server refuses to replace a file
    unless the caller names the hash of the version it read, so an intent
    carrying that hash is an overwrite and one without it can only create.

    Independent of ``intent.mutating`` and of any auto-approval, so listing
    ``files.write_file`` as safe to run unattended -- reasonable for creating
    drafts -- can never extend to writing over a file that is already there.
    """

    name = "document_overwrite"

    def __init__(
        self,
        overwrite_tools: Iterable[str] = ("files.overwrite_document",),
        guarded_write_tools: Iterable[str] = ("files.write_file",),
        precondition_arg: str = "expected_sha256",
    ):
        """Take the tools that always overwrite and those that do when guarded."""
        self.overwrite_tools: set[str] = set(overwrite_tools)
        self.guarded_write_tools: set[str] = set(guarded_write_tools)
        self.precondition_arg = precondition_arg

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny an overwrite with no precondition; hold every other one for approval."""
        precondition = intent.arguments.get(self.precondition_arg)
        if intent.tool_ref in self.overwrite_tools:
            if not precondition:
                return self.deny(
                    intent,
                    f"'{intent.tool_ref}' requires {self.precondition_arg}: the hash of "
                    f"the version being replaced",
                )
            return self.require_approval(
                intent, f"'{intent.tool_ref}' replaces an existing document"
            )
        if intent.tool_ref in self.guarded_write_tools and precondition:
            return self.require_approval(
                intent,
                f"'{intent.tool_ref}' with {self.precondition_arg} replaces an existing document",
            )
        return None


#: Tools exposed by the jobs MCP server, with the argument surface each accepts.
#: Being listed here makes a tool reachable; whether it can run unattended is
#: still decided by MutatingToolRule and UntrustedOriginRule.
JOB_SEARCH_TOOL_ARGUMENTS = {
    "jobs.search_jobs": {"keywords", "locations", "job_type", "limit"},
    "jobs.scrape_job_details": {"job_id", "job_url"},
    "jobs.filter_jobs": {
        "jobs",
        "salary_min",
        "salary_max",
        "experience_level",
        "remote_only",
    },
    # Mutating: reachable, but MutatingToolRule holds it for approval.
    "jobs.save_favorite_job": {"job_id", "notes"},
}


#: Job provider adapters (`personalos.providers`). Both are reads against a
#: job board; there is deliberately no mutating entry, so a provider write is
#: unreachable rather than merely approval-gated.
JOB_PROVIDER_TOOL_ARGUMENTS = {
    "job_providers.search": {"provider", "profile"},
    "job_providers.get_job": {"provider", "source_job_id"},
}


#: Tools exposed by the files MCP server. Which paths they may touch is not a
#: policy question: the server confines them to its allowed roots itself.
FILE_TOOL_ARGUMENTS = {
    "files.read_file": {"path"},
    # Mutating: reachable, but MutatingToolRule holds it for approval.
    "files.write_file": {"path", "content", "encoding", "expected_sha256"},
    # Destructive: replaces an existing file, after backing it up. Held for
    # approval by its permission class and by DocumentOverwriteRule.
    "files.overwrite_document": {"path", "content", "encoding", "expected_sha256"},
}


_GMAIL_MESSAGE_ARGUMENTS = {
    "to",
    "cc",
    "subject",
    "body",
    "thread_id",
    "in_reply_to",
    "message_id",
}

#: Tools exposed by the Google MCP server (`mcp_servers.google`). None of them
#: takes an account or credential argument: whose mailbox and calendar a call
#: touches is the gateway-attached `ToolAuthorization`, not something a
#: caller names. Which node may call which tool is `GOOGLE_NODE_TOOL_ALLOWLISTS`.
GOOGLE_TOOL_ARGUMENTS = {
    "google.gmail_search": {"query", "max_results", "page_token"},
    "google.gmail_read_message": {"message_id", "missing_ok"},
    "google.gmail_list_history": {
        "start_history_id",
        "history_types",
        "max_results",
        "page_token",
    },
    "google.gmail_get_profile": set(),
    "google.gmail_list_labels": set(),
    "google.gmail_create_draft": _GMAIL_MESSAGE_ARGUMENTS,
    "google.gmail_send_message": _GMAIL_MESSAGE_ARGUMENTS,
    "google.calendar_list_events": {
        "calendar_id",
        "time_min",
        "time_max",
        "query",
        "max_results",
        "private_properties",
        "time_zone",
        "page_token",
    },
    "google.calendar_get_event": {"calendar_id", "event_id", "time_zone"},
    "google.calendar_freebusy": {"calendar_ids", "time_min", "time_max", "time_zone"},
    "google.calendar_create_event": {
        "calendar_id",
        "summary",
        "start",
        "end",
        "description",
        "location",
        "attendees",
        "private_properties",
        "time_zone",
    },
    "google.calendar_update_event": {
        "calendar_id",
        "event_id",
        "summary",
        "start",
        "end",
        "description",
        "location",
        "private_properties",
        "time_zone",
    },
    "google.calendar_delete_event": {"calendar_id", "event_id"},
}


#: The `requested_by` of the Gmail inbox poller's reads.
GMAIL_INBOX_POLLER = "worker:gmail_inbox_poller"

#: The `requested_by` of every Gmail send, and of the Sent-folder lookups made
#: around one (`personalos.executor.gmail_outbound`). Not a graph node: only
#: the send executor stamps it, and only after the approved action it is
#: carrying out has been checked against its approval.
GMAIL_SENDER = "executor:gmail_send"

#: Mutating tools that may run without a human's sign-off when *system* code
#: asks. A Gmail draft is `WRITE_REVERSIBLE`: it sits in the candidate's own
#: Drafts folder and reaches nobody. A model-originated intent for one is still
#: held by `UntrustedOriginRule`, and which node may ask is still
#: `GOOGLE_NODE_TOOL_ALLOWLISTS`.
AUTO_APPROVED_MUTATING_TOOLS: frozenset[str] = frozenset({"google.gmail_create_draft"})


def _job_search_node(node: str) -> str:
    return f"graph:job_search#{node}"


#: Which job-search node may call which Google tool, keyed by the
#: `requested_by` the node stamps on its intents. Deliberately narrow:
#: only the recruiter-response node and the send executor can send mail, and
#: only interview scheduling touches the calendar. That node may delete an
#: event: a cancellation is DESTRUCTIVE, so it is held for approval like every
#: write, and `CalendarActionExecutor` refuses to cancel an event this system
#: did not stamp.
GOOGLE_NODE_TOOL_ALLOWLISTS: Mapping[str, frozenset[str]] = {
    # Not a graph node: the worker sweep that watches the mailbox
    # (`personalos.executor.gmail_inbox`). It learns which messages arrived
    # and whether they are in scope -- ids, labels and search hits -- and
    # cannot read a message's content, let alone write anything.
    GMAIL_INBOX_POLLER: frozenset(
        {
            "google.gmail_list_history",
            "google.gmail_get_profile",
            "google.gmail_list_labels",
            "google.gmail_search",
        }
    ),
    _job_search_node("classify_recruiter_events"): frozenset(
        {"google.gmail_search", "google.gmail_read_message"}
    ),
    # Where a reply is drafted. It may save the draft to Gmail's Drafts
    # folder, and nothing else: the send is an approved action carried out
    # as `GMAIL_SENDER`, never by this node.
    _job_search_node("record_recruiter_events"): frozenset({"google.gmail_create_draft"}),
    # The send executor: the send itself, the Sent-folder lookup that decides
    # whether a send already landed, and reading the message a reply answers
    # (for its `Message-ID` and thread).
    GMAIL_SENDER: frozenset(
        {"google.gmail_send_message", "google.gmail_search", "google.gmail_read_message"}
    ),
    _job_search_node("handle_recruiter_response"): frozenset(
        {
            "google.gmail_search",
            "google.gmail_read_message",
            "google.gmail_create_draft",
            "google.gmail_send_message",
        }
    ),
    _job_search_node("draft_follow_up_for_triggered_checkpoint"): frozenset(
        {"google.gmail_search", "google.gmail_read_message", "google.gmail_create_draft"}
    ),
    _job_search_node("plan_interview_schedule"): frozenset(
        {
            "google.calendar_list_events",
            "google.calendar_get_event",
            "google.calendar_freebusy",
            "google.calendar_create_event",
            "google.calendar_update_event",
            "google.calendar_delete_event",
        }
    ),
}


def default_rules(
    node_tool_allowlists: Mapping[str, Iterable[str]] | None = None,
) -> list[PolicyRule]:
    """The rule chain the application boots with.

    Job search, job provider, file and Google tools are allowlisted; nothing
    else can run until it is added here deliberately. Google tools are further
    confined per node by `node_tool_allowlists` (default
    `GOOGLE_NODE_TOOL_ALLOWLISTS`).
    """
    tool_arguments = {
        **JOB_SEARCH_TOOL_ARGUMENTS,
        **JOB_PROVIDER_TOOL_ARGUMENTS,
        **FILE_TOOL_ARGUMENTS,
        **GOOGLE_TOOL_ARGUMENTS,
    }
    return [
        RequireProvenanceRule(),
        ToolAllowlistRule(tool_arguments.keys()),
        ArgumentAllowlistRule(tool_arguments),
        NodeToolAllowlistRule(
            GOOGLE_NODE_TOOL_ALLOWLISTS if node_tool_allowlists is None else node_tool_allowlists
        ),
        MutatingToolRule(AUTO_APPROVED_MUTATING_TOOLS),
        DocumentOverwriteRule(),
        UntrustedOriginRule(),
    ]


__all__ = [
    "PolicyRule",
    "RequireProvenanceRule",
    "ToolAllowlistRule",
    "ArgumentAllowlistRule",
    "MutatingToolRule",
    "NodeToolAllowlistRule",
    "DocumentOverwriteRule",
    "UntrustedOriginRule",
    "JOB_SEARCH_TOOL_ARGUMENTS",
    "JOB_PROVIDER_TOOL_ARGUMENTS",
    "FILE_TOOL_ARGUMENTS",
    "GOOGLE_TOOL_ARGUMENTS",
    "GOOGLE_NODE_TOOL_ALLOWLISTS",
    "GMAIL_INBOX_POLLER",
    "GMAIL_SENDER",
    "AUTO_APPROVED_MUTATING_TOOLS",
    "default_rules",
    "IDEMPOTENCY_KEY_ARG",
]
