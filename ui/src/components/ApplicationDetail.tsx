/**
 * One application: `GET /v1/applications/{id}`.
 *
 * Its fit score and the evidence-backed match behind it (each matched
 * requirement cross-linked to the documents that cite the same evidence), its
 * tailored documents, recruiter events, open waits and approvals, history, and
 * the moves the API says it can make from here.
 */
import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  NetworkError,
  effectFor,
  type ApiClient,
  type ApplicationDetail as Detail,
  type ApplicationStatus,
  type ArtifactSummary,
  type TransitionRequest,
} from "../api/client";
import { actionTypeLabel, applicationStatusLabel, formatDateTime, formatScore } from "../format";
import { href } from "../router";
import { EffectPrompt, MoveFeedbackView, Unauthenticated, sendMove, type MoveFeedback } from "./Move";
import { StatusBadge } from "./StatusBadge";
import { WorkflowProgress } from "./WorkflowProgress";
import type { FeedTimings } from "../progress/useWorkflowProgress";

interface Props {
  client: ApiClient;
  applicationId: string;
  onUnauthenticated: () => void;
  /** Injected in tests. */
  timings?: FeedTimings;
}

type Load =
  | { kind: "loading" }
  | { kind: "loaded"; detail: Detail }
  | { kind: "error"; message: string };

/** Only http(s) links from untrusted data become clickable. */
function safeUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch {
    return null;
  }
}

/** The evidence ids an artifact cites (`evidence[].ref`). */
function evidenceRefs(artifact: ArtifactSummary): string[] {
  return (artifact.evidence ?? [])
    .map((item) => item.ref)
    .filter((ref): ref is string => typeof ref === "string" && ref.length > 0);
}

function jumpTo(elementId: string) {
  const target = document.getElementById(elementId);
  if (!target) return;
  target.scrollIntoView?.({ behavior: "smooth", block: "center" });
  target.focus({ preventScroll: true });
}

const evidenceAnchor = (id: string) => `evidence-${id}`;
const artifactAnchor = (id: string) => `artifact-${id}`;

export function ApplicationDetail({ client, applicationId, onUnauthenticated, timings }: Props) {
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const [prompt, setPrompt] = useState<ApplicationStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<MoveFeedback | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const detail = await client.getApplication(applicationId);
      setLoad({ kind: "loaded", detail });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated();
        return;
      }
      setLoad({
        kind: "error",
        message:
          error instanceof ApiError && error.isNotFound
            ? "No such application."
            : error instanceof NetworkError
              ? "Could not reach the PersonalOS API."
              : error instanceof Error
                ? error.message
                : String(error),
      });
    }
  }, [client, applicationId, onUnauthenticated]);

  useEffect(() => {
    setLoad({ kind: "loading" });
    setFeedback(null);
    setPrompt(null);
    setWorkflowId(null);
    void refresh();
  }, [refresh]);

  if (load.kind === "loading") return <p className="muted">Loading…</p>;
  if (load.kind === "error")
    return (
      <section>
        <p className="callout error" role="alert">
          {load.message}
        </p>
        <a href={href.board()}>Back to the board</a>
      </section>
    );

  const { detail } = load;
  const app = detail.application;
  const match = detail.match ?? null;
  const artifacts = detail.artifacts ?? [];
  const approvals = detail.pending_approvals ?? [];
  const checkpoints = detail.pending_checkpoints ?? [];
  const recruiterEvents = detail.recruiter_events ?? [];
  const history = detail.history ?? [];
  const postingUrl = safeUrl(app.url);
  const followWorkflow =
    workflowId ?? approvals[0]?.workflow_id ?? checkpoints.find((c) => c.workflow_id)?.workflow_id ?? null;

  const citedBy = (evidenceId: string) =>
    artifacts.filter((artifact) => evidenceRefs(artifact).includes(evidenceId));
  // The first row citing each evidence id is the one an artifact's link jumps to.
  const firstRow = new Map<string, number>();
  (match?.matched_requirements ?? []).forEach((m, i) => {
    if (!firstRow.has(m.evidence_id)) firstRow.set(m.evidence_id, i);
  });

  const send = async (request: TransitionRequest) => {
    setBusy(true);
    setFeedback(null);
    try {
      const { response, feedback: result } = await sendMove(client, app.application_id, request);
      setFeedback(result);
      setPrompt(null);
      if (response) {
        if (response.workflow_id) setWorkflowId(response.workflow_id);
        if (response.outcome === "applied") {
          // Show the new status at once, then re-read what follows from it.
          setLoad({
            kind: "loaded",
            detail: { ...detail, application: response.application, allowed_next_states: [] },
          });
        }
        await refresh();
      }
    } catch (error) {
      if (error instanceof Unauthenticated) onUnauthenticated();
      else setFeedback({ kind: "error", message: String(error) });
    } finally {
      setBusy(false);
    }
  };

  const requestMove = (to: ApplicationStatus) => {
    if (busy) return;
    setFeedback(null);
    if (effectFor(to)) setPrompt(to);
    else {
      setPrompt(null);
      void send({ to });
    }
  };

  return (
    <article className="detail" aria-labelledby="detail-title">
      <a href={href.board()} className="small">
        ← Board
      </a>
      <header className="detail-header">
        <h2 id="detail-title">
          {app.company} — {app.title}
        </h2>
        <span className="app-status" data-testid="application-status">
          {applicationStatusLabel(app.status)}
        </span>
      </header>
      <dl className="fields">
        {app.location && (
          <div className="field-row">
            <dt>Location</dt>
            <dd>{app.location}</dd>
          </div>
        )}
        {postingUrl && (
          <div className="field-row">
            <dt>Posting</dt>
            <dd>
              <a href={postingUrl} target="_blank" rel="noopener noreferrer">
                {postingUrl}
              </a>
            </dd>
          </div>
        )}
        <div className="field-row">
          <dt>Added</dt>
          <dd>{formatDateTime(app.created_at)}</dd>
        </div>
        {app.applied_at && (
          <div className="field-row">
            <dt>Applied</dt>
            <dd>{formatDateTime(app.applied_at)}</dd>
          </div>
        )}
      </dl>

      <section className="card" aria-labelledby="moves-title">
        <h3 id="moves-title">Move</h3>
        {detail.allowed_next_states.length === 0 ? (
          <p className="muted">No moves are possible from here.</p>
        ) : (
          <div className="actions wrap" data-testid="allowed-moves">
            {detail.allowed_next_states.map((state) => (
              <button key={state} type="button" disabled={busy} onClick={() => requestMove(state)}>
                {applicationStatusLabel(state)}
                {effectFor(state) ? " …" : ""}
              </button>
            ))}
          </div>
        )}
        {prompt && (
          <EffectPrompt
            label={`${app.company} — ${app.title}`}
            from={app.status}
            to={prompt}
            busy={busy}
            onCancel={() => setPrompt(null)}
            onConfirm={(request) => void send(request)}
          />
        )}
        {feedback && (
          <MoveFeedbackView feedback={feedback} onPick={requestMove} onDismiss={() => setFeedback(null)} />
        )}
      </section>

      {approvals.length > 0 && (
        <section className="card" aria-labelledby="approvals-title">
          <h3 id="approvals-title">Waiting for your approval</h3>
          <ul>
            {approvals.map((approval) => (
              <li key={approval.id} data-testid="pending-approval">
                {actionTypeLabel(approval.action_type)} <StatusBadge status={approval.action_status} />{" "}
                <span className="muted">expires {formatDateTime(approval.expires_at)}</span>
              </li>
            ))}
          </ul>
          <a href={href.approvals()}>Review in the approvals inbox</a>
        </section>
      )}

      {followWorkflow && (
        <WorkflowProgress
          key={followWorkflow}
          client={client}
          workflowId={followWorkflow}
          onUnauthenticated={onUnauthenticated}
          timings={timings}
          compact
        />
      )}

      <section className="card" aria-labelledby="fit-title">
        <h3 id="fit-title">Fit</h3>
        {detail.fit_score == null && !match ? (
          <p className="muted">No match has been recorded yet.</p>
        ) : (
          <>
            <p>
              <span className="score" data-testid="fit-score">
                {detail.fit_score != null ? formatScore(detail.fit_score) : "—"}
              </span>{" "}
              {match && <span className="recommendation">{match.recommendation}</span>}{" "}
              {detail.match_recorded_at && (
                <span className="muted small">scored {formatDateTime(detail.match_recorded_at)}</span>
              )}
            </p>
            {match && match.reasons.length > 0 && (
              <ul className="reasons">
                {match.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            )}
            <h4>Matched requirements</h4>
            {(match?.matched_requirements ?? []).length === 0 ? (
              <p className="muted">None.</p>
            ) : (
              <table className="requirements" data-testid="matched-requirements">
                <thead>
                  <tr>
                    <th>Requirement</th>
                    <th>Strength</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {match!.matched_requirements.map((item, i) => {
                    const cited = citedBy(item.evidence_id);
                    return (
                      <tr
                        key={`${item.evidence_id}:${i}`}
                        id={firstRow.get(item.evidence_id) === i ? evidenceAnchor(item.evidence_id) : undefined}
                        tabIndex={-1}
                        data-testid="matched-requirement"
                      >
                        <td>
                          {item.requirement}
                          {item.rationale && <div className="muted small">{item.rationale}</div>}
                        </td>
                        <td>{item.strength}</td>
                        <td>
                          <span className="evidence" data-testid="evidence-ref">
                            {item.evidence_type} <code>{item.evidence_id}</code>
                          </span>
                          {cited.length > 0 && (
                            <div className="small">
                              cited in{" "}
                              {cited.map((artifact, j) => (
                                <span key={artifact.artifact_version_id}>
                                  {j > 0 && ", "}
                                  <button
                                    type="button"
                                    className="link"
                                    onClick={() => jumpTo(artifactAnchor(artifact.artifact_version_id))}
                                  >
                                    {artifact.artifact_type} v{artifact.version}
                                  </button>
                                </span>
                              ))}
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            <h4>Missing requirements</h4>
            {(match?.missing_requirements ?? []).length === 0 ? (
              <p className="muted">None.</p>
            ) : (
              <ul data-testid="missing-requirements">
                {match!.missing_requirements.map((item, i) => (
                  <li key={i}>
                    {item.requirement} <span className={`severity severity-${item.severity}`}>{item.severity}</span>
                    {item.note && <span className="muted"> — {item.note}</span>}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>

      <section className="card" aria-labelledby="artifacts-title">
        <h3 id="artifacts-title">Documents</h3>
        {artifacts.length === 0 ? (
          <p className="muted">No tailored documents yet.</p>
        ) : (
          <ul data-testid="artifacts">
            {artifacts.map((artifact) => (
              <li
                key={artifact.artifact_version_id}
                id={artifactAnchor(artifact.artifact_version_id)}
                tabIndex={-1}
                data-testid="artifact"
              >
                <strong>
                  {artifact.artifact_type} v{artifact.version}
                </strong>{" "}
                <span className="muted small">
                  {formatDateTime(artifact.created_at)}
                  {artifact.generated_by ? ` · ${artifact.generated_by}` : ""}
                  {artifact.doc_id ? ` · ${artifact.doc_id}` : ""}
                </span>
                {evidenceRefs(artifact).length > 0 && (
                  <div className="small">
                    evidence:{" "}
                    {evidenceRefs(artifact).map((ref, j) => (
                      <span key={ref}>
                        {j > 0 && ", "}
                        {firstRow.has(ref) ? (
                          <button type="button" className="link" onClick={() => jumpTo(evidenceAnchor(ref))}>
                            <code>{ref}</code>
                          </button>
                        ) : (
                          <code>{ref}</code>
                        )}
                      </span>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card" aria-labelledby="recruiter-title">
        <h3 id="recruiter-title">Recruiter events</h3>
        {recruiterEvents.length === 0 ? (
          <p className="muted">Nothing from recruiters yet.</p>
        ) : (
          <ul data-testid="recruiter-events">
            {recruiterEvents.map((event) => (
              <li key={event.communication_event_id}>
                <strong>{event.classification}</strong>{" "}
                <span className="muted small">{formatDateTime(event.occurred_at)}</span>
                {(event.commitments ?? []).length > 0 && (
                  <ul>
                    {event.commitments.map((commitment, i) => (
                      <li key={i}>
                        {commitment.actor}: {commitment.action}
                        {commitment.due_at ? ` (by ${formatDateTime(commitment.due_at)})` : ""}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card" aria-labelledby="checkpoints-title">
        <h3 id="checkpoints-title">Open waits</h3>
        {checkpoints.length === 0 ? (
          <p className="muted">None.</p>
        ) : (
          <ul data-testid="pending-checkpoints">
            {checkpoints.map((checkpoint) => (
              <li key={checkpoint.checkpoint_id ?? checkpoint.dedupe_key}>
                <strong>{checkpoint.kind.replace(/_/g, " ")}</strong> — {checkpoint.reason}{" "}
                <span className="muted small">
                  fires {formatDateTime(checkpoint.trigger_at)}, expires{" "}
                  {formatDateTime(checkpoint.expires_at)} ({checkpoint.status})
                </span>
                {checkpoint.workflow_id && (
                  <>
                    {" "}
                    <a href={href.workflow(checkpoint.workflow_id)} className="small">
                      progress
                    </a>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card" aria-labelledby="history-title">
        <h3 id="history-title">History</h3>
        {history.length === 0 ? (
          <p className="muted">No moves yet.</p>
        ) : (
          <ol data-testid="history">
            {history.map((change) => (
              <li key={change.event_id}>
                {change.from_status ? applicationStatusLabel(change.from_status) : "—"} →{" "}
                {applicationStatusLabel(change.to_status)}{" "}
                <span className="muted small">
                  {formatDateTime(change.occurred_at)}
                  {change.actor ? ` by ${change.actor}` : ""}
                  {change.reason ? ` — ${change.reason}` : ""}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </article>
  );
}
