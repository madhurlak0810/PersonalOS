import { useState } from "react";

import { ApiError, NetworkError, type ApiClient, type PendingApproval } from "../api/client";
import {
  actionTypeLabel,
  formatDateTime,
  formatRelative,
  isExpired,
} from "../format";
import { IntentDetails } from "./IntentDetails";
import { StatusBadge } from "./StatusBadge";

/**
 * Where one approval stands on this screen.
 *
 * Every phase but `idle` is terminal until the reviewer refreshes. `stale`
 * means the action changed, its window closed, or it was decided elsewhere, so
 * what is on screen is no longer what a decision would apply to. Nothing is
 * ever resubmitted automatically.
 */
type Phase =
  | { kind: "idle" }
  | { kind: "submitting"; approved: boolean }
  | { kind: "decided"; approved: boolean; resumed: boolean }
  | { kind: "stale"; serverMessage: string | null }
  | { kind: "failed"; message: string };

export const REVIEW_AGAIN_MESSAGE =
  "This approval changed, expired or was already decided since this page loaded, so " +
  "nothing was recorded. Refresh the inbox and review the action again before deciding.";

export const EXPIRED_MESSAGE =
  "This approval's window has closed. It can no longer be approved; you can still " +
  "reject it, or refresh the inbox to see where it stands.";

interface Props {
  approval: PendingApproval;
  client: ApiClient;
  now: Date;
  onRefresh: () => void;
  onUnauthenticated: () => void;
}

export function ApprovalCard({ approval, client, now, onRefresh, onUnauthenticated }: Props) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const expired = isExpired(approval, now);
  // Any outcome locks the card until the reviewer refreshes the inbox, which
  // rebuilds it from what the API says now.
  const locked = phase.kind !== "idle";

  async function decide(approved: boolean) {
    // One submission per click, and none while one is in flight or after the
    // card has gone stale: the only way to try again is to refresh and review.
    if (locked) return;
    setPhase({ kind: "submitting", approved });
    try {
      const decision = await client.decide(approval, approved);
      setPhase({ kind: "decided", approved, resumed: decision.resumed });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated();
        setPhase({ kind: "failed", message: "Your API token was rejected; nothing was recorded." });
      } else if (error instanceof ApiError && (error.isConflict || error.isNotFound)) {
        setPhase({ kind: "stale", serverMessage: error.envelope?.message ?? null });
      } else if (error instanceof NetworkError) {
        setPhase({
          kind: "failed",
          message:
            "Could not reach the API, so it is not known whether the decision was recorded. " +
            "Refresh the inbox before deciding again.",
        });
      } else {
        const reference =
          error instanceof ApiError && error.envelope ? ` (reference ${error.envelope.context_id})` : "";
        setPhase({
          kind: "failed",
          message: `The decision was not recorded: ${
            error instanceof Error ? error.message : String(error)
          }${reference}.`,
        });
      }
    }
  }

  const status = phase.kind === "decided" && !phase.approved ? "DENIED" : approval.action_status;
  const statusDetail =
    phase.kind === "decided"
      ? phase.approved
        ? "Approved — not carried out yet. Nothing has been sent."
        : undefined
      : undefined;

  return (
    <article
      className={`card${phase.kind === "stale" ? " card-stale" : ""}`}
      aria-labelledby={`approval-${approval.id}-title`}
      data-testid="approval"
    >
      <header className="card-header">
        <StatusBadge status={status} detail={statusDetail} />
        <h3 id={`approval-${approval.id}-title`}>{actionTypeLabel(approval.action_type)}</h3>
        <span className={`risk risk-${approval.risk_level}`}>risk: {approval.risk_level}</span>
      </header>

      <p className="summary" data-testid="approval-summary">
        {approval.summary}
      </p>

      <dl className="fields">
        <dt>Target</dt>
        <dd data-testid="approval-target">{approval.target}</dd>
        <dt>Requested scopes</dt>
        <dd>
          {(approval.requested_scopes ?? []).length > 0 ? (
            <ul className="scopes">
              {(approval.requested_scopes ?? []).map((scope) => (
                <li key={scope}>
                  <code>{scope}</code>
                </li>
              ))}
            </ul>
          ) : (
            "none"
          )}
        </dd>
        <dt>Expires</dt>
        <dd data-testid="approval-expiry">
          {formatDateTime(approval.expires_at)} ({expired ? "expired" : formatRelative(approval.expires_at, now)})
        </dd>
        <dt>Requested</dt>
        <dd>{formatDateTime(approval.requested_at)}</dd>
      </dl>

      {(approval.consequences ?? []).length > 0 && (
        <section aria-label="What approving does">
          <h4>If you approve</h4>
          <ul className="consequences">
            {(approval.consequences ?? []).map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </section>
      )}

      <IntentDetails approval={approval} />

      <footer className="card-footer">
        <code className="muted hash" title="action_hash: a decision is bound to exactly this action">
          hash {approval.action_hash.slice(0, 16)}…
        </code>
        <div className="actions">
          <button
            type="button"
            className="approve"
            onClick={() => decide(true)}
            disabled={locked || expired}
          >
            {phase.kind === "submitting" && phase.approved ? "Approving…" : "Approve"}
          </button>
          <button
            type="button"
            className="reject"
            onClick={() => decide(false)}
            disabled={locked}
          >
            {phase.kind === "submitting" && !phase.approved ? "Rejecting…" : "Reject"}
          </button>
        </div>
      </footer>

      {expired && phase.kind === "idle" && (
        <p className="callout warning" role="status">
          {EXPIRED_MESSAGE}
        </p>
      )}
      {phase.kind === "stale" && (
        <div className="callout warning" role="alert">
          <p>{REVIEW_AGAIN_MESSAGE}</p>
          {phase.serverMessage && <p className="muted">API: {phase.serverMessage}</p>}
          <button type="button" onClick={onRefresh}>
            Refresh and review
          </button>
        </div>
      )}
      {phase.kind === "failed" && (
        <div className="callout error" role="alert">
          <p>{phase.message}</p>
          <button type="button" onClick={onRefresh}>
            Refresh
          </button>
        </div>
      )}
      {phase.kind === "decided" && (
        <p className="callout" role="status">
          {phase.approved
            ? "Approval recorded. The workflow carries the action out; it is not done until it shows as EXECUTED."
            : "Rejection recorded. Nothing will be sent."}
          {!phase.resumed && " The workflow resumes once every request raised with this one is answered."}
        </p>
      )}
    </article>
  );
}
