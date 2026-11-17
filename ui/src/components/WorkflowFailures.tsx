/**
 * A workflow's failures: `GET /v1/workflows/{id}/failures`, with Retry
 * (`POST .../retry`) and Dismiss (`POST .../dismiss`).
 *
 * The rules this screen keeps:
 *
 * - **A policy denial is denied, not failed.** It is labelled `DENIED` and is
 *   never offered a retry, whatever else the response says: asking again gets
 *   the same answer.
 * - **A provider-permanent failure asks for a reconnect.** Revoked OAuth or a
 *   missing scope cannot be fixed by retrying, so the screen says how to
 *   reconnect and offers no retry; that would only loop.
 * - **Retry is offered only when the API says so** (`retryable` and a `retry`
 *   action), is sent once, and the card is updated from the API's answer.
 *   Nothing here retries on its own; a 409 is shown and never resent.
 * - **Errors are the API's sanitized text**, rendered inert, after the client's
 *   credential guard.
 */
import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  type ApiClient,
  type EventFailure,
  type FailedOperation,
  type RetryResponse,
  type WorkflowFailure,
} from "../api/client";
import { FAILURE_STATUS_TEXT, failureClassLabel, formatDateTime } from "../format";
import { href } from "../router";
import { describeError } from "./AuditTrail";
import { IntegrationsHealth, reconnectHint } from "./IntegrationsHealth";
import { WithheldNotice } from "./WithheldNotice";

interface Props {
  client: ApiClient;
  workflowId: string;
  onUnauthenticated?: () => void;
}

type Load =
  | { kind: "loading" }
  | {
      kind: "loaded";
      status: string;
      failures: WorkflowFailure[];
      events: EventFailure[];
      withheld: string[];
    }
  | { kind: "error"; message: string };

export function isPolicyDenial(failure: WorkflowFailure): boolean {
  return failure.status === "denied" || failure.failure_class === "policy_denial";
}

export function needsReconnect(failure: WorkflowFailure): boolean {
  return (
    failure.failure_class === "provider_permanent" ||
    (failure.actions ?? []).includes("reconnect") ||
    failure.integration?.reconnect_required === true
  );
}

/** Whether to show a Retry control: only what the API allows, never a denial or a reconnect. */
export function canRetry(failure: WorkflowFailure): boolean {
  return (
    !isPolicyDenial(failure) &&
    !needsReconnect(failure) &&
    failure.status === "failed" &&
    failure.retryable &&
    (failure.actions ?? []).includes("retry")
  );
}

export function canDismiss(failure: WorkflowFailure): boolean {
  return (
    (failure.actions ?? []).includes("dismiss") &&
    failure.status !== "dismissed" &&
    failure.status !== "retry_queued"
  );
}

/** The failure as the retry answer says it now stands. */
export function afterRetry(failure: WorkflowFailure, response: RetryResponse): WorkflowFailure {
  return {
    ...failure,
    status: response.status,
    attempts: response.attempt,
    step: response.step ?? failure.step,
    retryable: false,
    not_retryable_reason: "a retry is already queued",
    actions: [],
    operations: response.operations ?? failure.operations,
  };
}

const RECOVERY_TEXT: Record<string, string> = {
  replay_stored_receipt: "already done — its stored receipt is returned, nothing is sent again",
  reconcile_before_send: "outcome unknown — checked with the provider before anything is sent",
};

function Operations({ operations }: { operations: FailedOperation[] }) {
  if (operations.length === 0) return null;
  return (
    <>
      <h4>Side effects the step had started</h4>
      <ul className="operations" data-testid="operations">
        {operations.map((op) => (
          <li key={op.operation_id}>
            <code>{op.tool_name}</code> — {op.status}, {op.attempts} call{op.attempts === 1 ? "" : "s"}.
            On retry: {RECOVERY_TEXT[op.recovery] ?? op.recovery}.
            {op.error && <pre>{op.error}</pre>}
          </li>
        ))}
      </ul>
    </>
  );
}

type Outcome =
  | { kind: "retried"; response: RetryResponse }
  | { kind: "dismissed"; by: string }
  | { kind: "error"; message: string; conflict: boolean };

function FailureCard({
  client,
  initial,
  onUnauthenticated,
  onRefresh,
  onWithheld,
}: {
  client: ApiClient;
  initial: WorkflowFailure;
  onUnauthenticated?: () => void;
  onRefresh: () => void;
  onWithheld: (paths: string[]) => void;
}) {
  const [failure, setFailure] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  const denied = isPolicyDenial(failure);
  const reconnect = !denied && needsReconnect(failure);
  // A 409 locks the card: what is on screen is out of date.
  const locked = outcome?.kind === "error" && outcome.conflict;
  const status = denied ? "denied" : failure.status;
  const statusText = FAILURE_STATUS_TEXT[status] ?? { label: status, detail: "" };

  const fail = (error: unknown) => {
    if (error instanceof ApiError && error.isUnauthenticated) {
      onUnauthenticated?.();
      return;
    }
    setOutcome({
      kind: "error",
      message: describeError(error, "The workflow or failure no longer exists."),
      conflict: error instanceof ApiError && (error.isConflict || error.isNotFound),
    });
  };

  const retry = async () => {
    if (busy || !canRetry(failure)) return;
    setBusy(true);
    setOutcome(null);
    try {
      const { value, withheld } = await client.retryFailure(failure);
      onWithheld(withheld);
      setFailure(afterRetry(failure, value));
      setOutcome({ kind: "retried", response: value });
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  };

  const dismiss = async () => {
    if (busy || !canDismiss(failure)) return;
    setBusy(true);
    setOutcome(null);
    try {
      const response = await client.dismissFailure(failure);
      setFailure({
        ...failure,
        status: response.status,
        retryable: false,
        actions: [],
        dismissed_by: response.dismissed_by,
        dismissed_at: response.dismissed_at,
      });
      setOutcome({ kind: "dismissed", by: response.dismissed_by });
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <article
      className={`card failure failure-${status}`}
      data-testid="failure"
      data-failure-id={failure.failure_id}
      data-status={status}
    >
      <div className="card-header">
        <h3>{failure.step ? `Step ${failure.step}` : "A step"}</h3>
        <span className={`status status-${status === "denied" ? "denied" : "failed"}`} data-testid="failure-status">
          <span className="status-enum">{statusText.label}</span>
          <span className="status-detail">{statusText.detail}</span>
        </span>
      </div>
      <dl className="fields">
        <div className="field-row">
          <dt>Class</dt>
          <dd data-testid="failure-class">
            {failureClassLabel(failure.failure_class)} <code className="muted">{failure.failure_class}</code>
            {failure.cause_class && ` (caused by ${failureClassLabel(failure.cause_class)})`}
          </dd>
        </div>
        <div className="field-row">
          <dt>Failing step</dt>
          <dd>
            <code>{failure.step ?? "unknown"}</code> on thread <code>{failure.thread_id}</code>
          </dd>
        </div>
        <div className="field-row">
          <dt>Retryable</dt>
          <dd data-testid="failure-retryable">
            {failure.retryable ? "Yes" : `No — ${failure.not_retryable_reason ?? "the API does not allow it"}`}
          </dd>
        </div>
        <div className="field-row">
          <dt>Attempts</dt>
          <dd data-testid="failure-attempts">
            {failure.attempts} of {failure.max_attempts ?? 5}
          </dd>
        </div>
        {failure.failed_at && (
          <div className="field-row">
            <dt>Failed</dt>
            <dd>{formatDateTime(failure.failed_at)}</dd>
          </div>
        )}
        {failure.dismissed_by && (
          <div className="field-row">
            <dt>Dismissed</dt>
            <dd>
              by {failure.dismissed_by}
              {failure.dismissed_at ? `, ${formatDateTime(failure.dismissed_at)}` : ""}
            </dd>
          </div>
        )}
      </dl>
      <h4>Last error</h4>
      <pre className="error-text" data-testid="failure-error">
        {failure.last_error}
      </pre>
      <Operations operations={failure.operations ?? []} />

      {reconnect && (
        <div className="callout warning" role="status" data-testid="reconnect-prompt">
          <p>
            <strong>Reconnect {failure.integration?.provider ?? "the integration"}.</strong>{" "}
            {failure.integration?.reason ? `${failure.integration.reason}. ` : ""}
            Retrying would fail the same way, so no retry is offered. Run{" "}
            <code>{reconnectHint(failure.integration?.provider ?? "google")}</code>, then start the
            work again.
          </p>
          <a href={href.integrations()}>Integration health</a>
        </div>
      )}

      {outcome?.kind === "retried" && (
        <div className="callout success" role="status" data-testid="retry-queued">
          <p>
            Retry queued — attempt {outcome.response.attempt}
            {outcome.response.step ? ` of step ${outcome.response.step}` : ""}, from the checkpoint before it.
            Earlier steps are not re-run.
          </p>
        </div>
      )}
      {outcome?.kind === "dismissed" && (
        <p className="callout" role="status" data-testid="dismissed">
          Dismissed by {outcome.by}. Nothing was re-run or undone.
        </p>
      )}
      {outcome?.kind === "error" && (
        <div className="callout error" role="alert">
          <p>{outcome.message}</p>
          {outcome.conflict && (
            <button type="button" onClick={onRefresh}>
              Refresh failures
            </button>
          )}
        </div>
      )}

      {!locked && (canRetry(failure) || canDismiss(failure)) && (
        <div className="card-footer">
          <div className="actions">
            {canRetry(failure) && (
              <button type="button" className="approve" disabled={busy} onClick={() => void retry()}>
                Retry step
              </button>
            )}
            {canDismiss(failure) && (
              <button type="button" disabled={busy} onClick={() => void dismiss()}>
                Dismiss
              </button>
            )}
          </div>
        </div>
      )}
    </article>
  );
}

function EventFailureRow({ failure }: { failure: EventFailure }) {
  return (
    <li data-testid="event-failure">
      <code>{failure.event_type}</code> —{" "}
      {failure.disposition === "quarantined" ? "quarantined (malformed)" : "dead-lettered"} after{" "}
      {failure.attempts} attempt{failure.attempts === 1 ? "" : "s"}
      {failure.failed_at ? `, ${formatDateTime(failure.failed_at)}` : ""}
      <pre>{failure.last_error}</pre>
    </li>
  );
}

/** One workflow's failures and what may be done about them. */
export function WorkflowFailures({ client, workflowId, onUnauthenticated }: Props) {
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const [extraWithheld, setExtraWithheld] = useState<string[]>([]);
  // Bumped on refresh so every card is rebuilt from the new listing.
  const [generation, setGeneration] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const { value, withheld } = await client.listFailures(workflowId);
      setLoad({
        kind: "loaded",
        status: value.status,
        failures: value.failures ?? [],
        events: value.event_failures ?? [],
        withheld,
      });
      setExtraWithheld([]);
      setGeneration((n) => n + 1);
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated?.();
        return;
      }
      setLoad({ kind: "error", message: describeError(error, "No such workflow.") });
    }
  }, [client, workflowId, onUnauthenticated]);

  useEffect(() => {
    setLoad({ kind: "loading" });
    void refresh();
  }, [refresh]);

  const onWithheld = useCallback(
    (paths: string[]) => {
      if (paths.length) setExtraWithheld((prev) => [...prev, ...paths]);
    },
    [],
  );

  return (
    <section className="panel" aria-label="Failures">
      <div className="panel-header">
        <h3>
          Failures{" "}
          {load.kind === "loaded" && (
            <span className="muted small" data-testid="workflow-status">
              workflow {load.status}
            </span>
          )}
        </h3>
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>
      {load.kind === "loading" && <p className="muted">Loading…</p>}
      {load.kind === "error" && (
        <p className="callout error" role="alert">
          {load.message}
        </p>
      )}
      {load.kind === "loaded" && (
        <>
          <WithheldNotice paths={[...load.withheld, ...extraWithheld]} />
          <IntegrationsHealth client={client} onUnauthenticated={onUnauthenticated} compact />
          {load.failures.length === 0 && load.events.length === 0 ? (
            <p className="empty">No open failures.</p>
          ) : (
            load.failures.map((failure) => (
              <FailureCard
                key={`${failure.failure_id}-${generation}`}
                client={client}
                initial={failure}
                onUnauthenticated={onUnauthenticated}
                onRefresh={() => void refresh()}
                onWithheld={onWithheld}
              />
            ))
          )}
          {load.events.length > 0 && (
            <>
              <h4>Events the worker gave up delivering</h4>
              <ul className="event-failures">
                {load.events.map((event) => (
                  <EventFailureRow key={event.failure_id} failure={event} />
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </section>
  );
}
