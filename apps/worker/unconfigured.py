"""Stand-ins for the ports this build has no production adapter for yet.

The default worker graphs (`apps.worker.graphs`) wire every port that has a
real implementation. A few do not yet -- reading the candidate's mailbox,
their calendar, embedding text for evidence retrieval, drafting replies -- and
leaving them out would quietly switch whole branches of the job search off,
or make the graph refuse to compile. Each is bound to one of these instead.

Every method of every class here raises `PortNotConfigured` naming its port.
None of them returns an empty result: an inbox that answered "no messages"
or a calendar that answered "nothing booked" would be a silent wrong answer
the graph acted on. A step that reaches one fails, is checkpointed as failed,
and shows up on `GET /v1/workflows/{id}/failures` as retryable -- for once
the adapter is configured and the worker restarted.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from personalos.domain.errors import PortNotConfigured

#: Port names, as `PortNotConfigured.port` carries them and the worker logs them.
JOB_BOARDS_PORT = "job_boards"
MAILBOX_PORT = "mailbox"
CALENDAR_PORT = "calendar"
EMBEDDER_PORT = "embedder"
DRAFT_WRITER_PORT = "draft_writer"
REPLY_DRAFTER_PORT = "reply_drafter"
APPLICATION_STORE_PORT = "application_store"
APPLICATION_SUBMITTER_PORT = "application_submitter"


class _NotConfigured:
    """Base for every stand-in: knows its port, and how to refuse."""

    port: str = ""
    hint: str = ""

    def _refuse(self, operation: str) -> PortNotConfigured:
        message = f"the '{self.port}' port is not configured in this deployment ({operation})"
        if self.hint:
            message = f"{message}; {self.hint}"
        return PortNotConfigured(message, port=self.port)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(port={self.port!r})"


class NotConfiguredJobBoard(_NotConfigured):
    """`JobBoardProvider` for a deployment with no job board configured.

    The graph's fan-out records a provider that raises as a provider failure
    and carries on, by design, so a run with only this provider ends with an
    empty shortlist and a `provider_failures` entry saying why.
    """

    port = JOB_BOARDS_PORT
    hint = "set GREENHOUSE_BOARDS to search a job board"
    name = "not_configured"

    async def search(self, profile: Any) -> Sequence[Any]:
        raise self._refuse("search")


class NotConfiguredMailbox(_NotConfigured):
    """`RecruiterInbox`, and the executor for `send_recruiter_message`."""

    port = MAILBOX_PORT

    async def fetch(self, application_id: UUID) -> Sequence[Any]:
        raise self._refuse("fetch")

    async def execute(self, intent: Any, decision: Any) -> Any:
        raise self._refuse(f"execute {intent.kind.value}")


class NotConfiguredCalendar(_NotConfigured):
    """`CalendarReader` for the graph and `CalendarClient` for the executor."""

    port = CALENDAR_PORT

    async def events_between(self, user_id: UUID, start: datetime, end: datetime) -> Sequence[Any]:
        raise self._refuse("events_between")

    async def events_for_application(self, user_id: UUID, application_id: UUID) -> Sequence[Any]:
        raise self._refuse("events_for_application")

    async def find_by_property(self, name: str, value: str) -> Sequence[Any]:
        raise self._refuse("find_by_property")

    async def get_event(self, event_id: str) -> Any:
        raise self._refuse("get_event")

    async def create_event(self, **kwargs: Any) -> Any:
        raise self._refuse("create_event")

    async def update_event(self, event_id: str, **kwargs: Any) -> Any:
        raise self._refuse("update_event")


class NotConfiguredEmbedder(_NotConfigured):
    """`Embedder` for evidence retrieval, which every tailored packet starts with."""

    port = EMBEDDER_PORT
    model = "not_configured"

    async def embed(self, text: str) -> list[float]:
        raise self._refuse("embed")


class NotConfiguredDraftWriter(_NotConfigured):
    """`DraftWriter`, bound when the model configuration names none (`rules`)."""

    port = DRAFT_WRITER_PORT
    hint = "point LLM_GATEWAY_CONFIG at a configuration with a draft_writer model"

    async def write(self, artifact_type: Any, posting: Any, profile: Any, evidence: Any) -> Any:
        raise self._refuse("write")


class NotConfiguredReplyDrafter(_NotConfigured):
    """`RecruiterReplyDrafter`."""

    port = REPLY_DRAFTER_PORT

    async def draft(self, event: Any) -> Any:
        raise self._refuse("draft")


class NotConfiguredApplicationStore(_NotConfigured):
    """`ApplicationStore`: writing the application a reviewed packet became."""

    port = APPLICATION_STORE_PORT

    async def create(self, **kwargs: Any) -> Any:
        raise self._refuse("create")

    async def record_recruiter_response(self, application_id: UUID, response: Any) -> None:
        raise self._refuse("record_recruiter_response")

    async def record_follow_up(self, checkpoint: Any) -> None:
        raise self._refuse("record_follow_up")


class NotConfiguredApplicationSubmitter(_NotConfigured):
    """The executor for `submit_application`: sending a packet to the employer."""

    port = APPLICATION_SUBMITTER_PORT

    async def execute(self, intent: Any, decision: Any) -> Any:
        raise self._refuse(f"execute {intent.kind.value}")


__all__ = [
    "APPLICATION_STORE_PORT",
    "APPLICATION_SUBMITTER_PORT",
    "CALENDAR_PORT",
    "DRAFT_WRITER_PORT",
    "EMBEDDER_PORT",
    "JOB_BOARDS_PORT",
    "MAILBOX_PORT",
    "REPLY_DRAFTER_PORT",
    "NotConfiguredApplicationStore",
    "NotConfiguredApplicationSubmitter",
    "NotConfiguredCalendar",
    "NotConfiguredDraftWriter",
    "NotConfiguredEmbedder",
    "NotConfiguredJobBoard",
    "NotConfiguredMailbox",
    "NotConfiguredReplyDrafter",
]
