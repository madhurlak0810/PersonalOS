"""Execution context: the correlation identity threaded through one workflow run.

Some ids already exist in the system (`job_id`, `agent_id`), but nothing ties
together the *run* those ids belong to across the API, executor, and tool
boundaries. `ExecutionContext` is that missing thread: created once at the
point a workflow is accepted (an API request, a CLI invocation), it is carried
unchanged through every intent it produces and every event it emits, so a log
line, a persisted job, and an audited tool call can all be traced back to the
same run and the actor who started it.
"""

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ExecutionContext(BaseModel):
    """Correlation identity for one workflow run.

    `workflow_id` identifies the workflow being run (stable across retries of
    the same request), `run_id` identifies this particular execution attempt,
    `correlation_id` ties together every log line and event produced while
    handling one external request -- it is what a caller hands back to ask
    "what happened to my request" -- and `actor_id` records who or what
    triggered the run.

    Immutable: a context is not edited as it propagates, only carried.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID = Field(default_factory=uuid4)
    run_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID = Field(default_factory=uuid4)
    actor_id: str = "system"

    @classmethod
    def new(cls, actor_id: str = "system") -> "ExecutionContext":
        """Start a fresh context for a new workflow run."""
        return cls(actor_id=actor_id)

    def as_log_str(self) -> str:
        """Compact form for embedding in a log message."""
        return (
            f"workflow_id={self.workflow_id} run_id={self.run_id} "
            f"correlation_id={self.correlation_id} actor_id={self.actor_id}"
        )


__all__ = ["ExecutionContext"]
