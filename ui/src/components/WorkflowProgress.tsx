import type { ApiClient, WorkflowProgressEvent } from "../api/client";
import { formatDateTime } from "../format";
import type { FeedStatus } from "../progress/feed";
import { useWorkflowProgress, type FeedTimings } from "../progress/useWorkflowProgress";
import { href } from "../router";

interface Props {
  client: ApiClient;
  workflowId: string;
  onUnauthenticated?: () => void;
  /** Injected in tests. */
  timings?: FeedTimings;
  /** Show only the newest few events, for embedding in another view. */
  compact?: boolean;
}

const STATUS_TEXT: Record<FeedStatus, string> = {
  connecting: "Connecting…",
  live: "Live",
  reconnecting: "Reconnecting — resuming from the last event",
  polling: "Live stream unavailable — checking for updates every few seconds",
  failed: "Stopped",
  stopped: "Stopped",
};

function text(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

interface ApprovalRequestInfo {
  kind: string | null;
  risk: string | null;
}

function requestsOf(event: WorkflowProgressEvent): ApprovalRequestInfo[] {
  const requests = event.data.requests;
  if (!Array.isArray(requests)) return [];
  return requests.map((request) => {
    const fields = (typeof request === "object" && request !== null ? request : {}) as Record<
      string,
      unknown
    >;
    return { kind: text(fields.kind), risk: text(fields.risk) };
  });
}

/** What one event says, in a sentence. Every value is rendered as text. */
export function describeEvent(event: WorkflowProgressEvent): string {
  const step = text(event.data.step);
  switch (event.type) {
    case "step_started":
      return `Started ${step ?? "a step"}`;
    case "step_completed":
      return `Completed ${step ?? "a step"}`;
    case "approval_requested": {
      const kinds = requestsOf(event)
        .map((request) => request.kind)
        .filter(Boolean);
      return `Waiting for approval${step ? ` at ${step}` : ""}${kinds.length ? `: ${kinds.join(", ")}` : ""}`;
    }
    case "approval_resolved": {
      const by = text(event.data.resolved_by);
      const decisions = Array.isArray(event.data.decisions) ? event.data.decisions : [];
      const verdicts = decisions
        .map((item) =>
          typeof item === "object" && item !== null
            ? text((item as Record<string, unknown>).verdict)
            : null,
        )
        .filter(Boolean);
      return `Approval answered${by ? ` by ${by}` : ""}${verdicts.length ? `: ${verdicts.join(", ")}` : ""}`;
    }
    case "failure": {
      const message = text(event.data.message);
      const failureClass = text(event.data.failure_class);
      return `Failed${step ? ` at ${step}` : ""}${failureClass ? ` (${failureClass})` : ""}${message ? `: ${message}` : ""}`;
    }
    case "workflow_completed":
      return "Run finished";
    default:
      return String(event.type);
  }
}

/** `approval_requested` events no later `approval_resolved` has answered. */
export function openApprovalRequests(events: WorkflowProgressEvent[]): WorkflowProgressEvent[] {
  const open: WorkflowProgressEvent[] = [];
  for (const event of events) {
    if (event.type === "approval_requested") {
      open.push(event);
    } else if (event.type === "approval_resolved") {
      // Matched by interrupt id when the event names them; otherwise the
      // answer settles every request open on its thread.
      const answered = new Set(strings(event.data.interrupt_ids));
      for (let i = open.length - 1; i >= 0; i--) {
        const request = open[i]!;
        const interruptId = text(request.data.interrupt_id);
        const settled =
          answered.size > 0 && interruptId
            ? answered.has(interruptId)
            : request.thread_id === event.thread_id;
        if (settled) open.splice(i, 1);
      }
    }
  }
  return open;
}

/** A workflow's progress, live from `GET /v1/workflows/{id}/events`. */
export function WorkflowProgress({ client, workflowId, onUnauthenticated, timings, compact }: Props) {
  const { events, status, error } = useWorkflowProgress(client, workflowId, {
    timings,
    onUnauthenticated,
  });
  const pending = openApprovalRequests(events);
  const shown = compact ? events.slice(-5) : events;

  return (
    <section className="progress" aria-label="Workflow progress">
      <div className="progress-header">
        <h3>
          Workflow progress{" "}
          {!compact && (
            <code className="muted" data-testid="workflow-id">
              {workflowId}
            </code>
          )}
        </h3>
        <span className={`feed-status feed-${status}`} data-testid="feed-status" data-status={status}>
          {STATUS_TEXT[status]}
        </span>
      </div>
      {error && (
        <p className="callout error" role="alert">
          {error}
        </p>
      )}
      {pending.map((event) => {
        const requests = requestsOf(event);
        const step = text(event.data.step);
        return (
          <div
            key={event.event_id}
            className="callout warning approval-prompt"
            role="status"
            data-testid="approval-prompt"
          >
            <p>
              <strong>Approval needed</strong>
              {step ? ` at ${step}` : ""}
              {requests.length > 0 &&
                `: ${requests
                  .map((request) =>
                    [request.kind, request.risk ? `risk ${request.risk}` : null]
                      .filter(Boolean)
                      .join(", "),
                  )
                  .join("; ")}`}
              . Nothing has been sent; the run is paused until you decide.
            </p>
            <a href={href.approvals()}>Review it in the approvals inbox</a>
          </div>
        );
      })}
      {events.length === 0 && !error ? (
        <p className="empty">No progress yet.</p>
      ) : (
        <ol className="progress-events" data-testid="progress-events">
          {shown.map((event) => (
            <li
              key={event.event_id}
              className={`progress-event progress-${event.type}`}
              data-testid="progress-event"
              data-event-id={event.event_id}
              data-type={event.type}
            >
              <time dateTime={event.at} className="muted">
                {formatDateTime(event.at)}
              </time>{" "}
              <span>{describeEvent(event)}</span>
            </li>
          ))}
        </ol>
      )}
      {compact && (
        <a href={href.workflow(workflowId)} className="small">
          Full progress
        </a>
      )}
    </section>
  );
}
