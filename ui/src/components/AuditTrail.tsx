/**
 * The audit trail: what happened, and what authorized it. Read-only.
 *
 * `GET /v1/workflows/{id}/audit` answers oldest first, each entry one step of
 * an action's chain -- intent, policy decision, approval, execution -- with the
 * provider's receipt on the execution's `tool_execution` link. Here each
 * action's steps are grouped by their shared `action_hash` and shown in the
 * order the API gave them, with the receipt as its own step after the
 * execution that produced it.
 *
 * Every payload is the API's redacted version, and has also been through
 * `withholdSecrets` in the client. Nothing on this screen writes.
 */
import { useCallback, useEffect, useState } from "react";

import { ApiError, NetworkError, type ApiClient, type AuditTrailEntry } from "../api/client";
import { AUDIT_STEP_LABELS, actionTypeLabel, formatDateTime, policyDecisionLabel } from "../format";
import { href } from "../router";
import { WithheldNotice } from "./WithheldNotice";

export type AuditStepKind = AuditTrailEntry["stage"] | "receipt";

export interface AuditStep {
  key: string;
  kind: AuditStepKind;
  entry: AuditTrailEntry;
}

/** The entries as steps, each execution with a receipt followed by a receipt step. */
export function auditSteps(entries: AuditTrailEntry[]): AuditStep[] {
  const steps: AuditStep[] = [];
  entries.forEach((entry, i) => {
    const key = entry.audit_event_id ?? `${entry.stage}-${entry.at}-${i}`;
    steps.push({ key, kind: entry.stage, entry });
    if (entry.stage === "execution" && entry.links?.tool_execution?.receipt) {
      steps.push({ key: `${key}-receipt`, kind: "receipt", entry });
    }
  });
  return steps;
}

export interface AuditChain {
  /** The action's fingerprint; `null` for events not tied to an action. */
  actionHash: string | null;
  steps: AuditStep[];
}

/** Steps grouped by action, in order of each action's first step; loose events last. */
export function auditChains(entries: AuditTrailEntry[]): AuditChain[] {
  const chains = new Map<string, AuditChain>();
  const loose: AuditStep[] = [];
  for (const step of auditSteps(entries)) {
    const hash = step.entry.action_hash ?? null;
    if (!hash) {
      loose.push(step);
      continue;
    }
    const chain = chains.get(hash) ?? { actionHash: hash, steps: [] };
    chain.steps.push(step);
    chains.set(hash, chain);
  }
  const ordered = [...chains.values()];
  if (loose.length) ordered.push({ actionHash: null, steps: loose });
  return ordered;
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

/** One step, every value rendered as text. */
export function AuditStepView({ step, showWorkflow }: { step: AuditStep; showWorkflow?: boolean }) {
  const { entry, kind } = step;
  const execution = entry.links?.tool_execution ?? null;
  const approval = entry.links?.approval ?? null;
  const decision = entry.links?.policy_decision ?? null;
  const at = kind === "receipt" ? (execution?.completed_at ?? entry.at) : entry.at;

  return (
    <li className={`audit-step audit-${kind}`} data-testid="audit-step" data-kind={kind}>
      <div className="audit-step-header">
        <strong className="audit-kind">{AUDIT_STEP_LABELS[kind] ?? kind}</strong>
        <time dateTime={at} className="muted small">
          {formatDateTime(at)}
        </time>
        {entry.result && (
          <span className={`audit-result audit-result-${entry.result}`}>{entry.result}</span>
        )}
      </div>
      <dl className="fields">
        {kind !== "receipt" && (
          <div className="field-row">
            <dt>Action</dt>
            <dd>
              <code>{entry.action}</code>
              {actionTypeLabel(entry.action) !== entry.action && ` (${actionTypeLabel(entry.action)})`}
            </dd>
          </div>
        )}
        <div className="field-row">
          <dt>Actor</dt>
          <dd>
            <code>{entry.actor}</code>
          </dd>
        </div>
        {entry.target_ref && (
          <div className="field-row">
            <dt>Target</dt>
            <dd>{entry.target_ref}</dd>
          </div>
        )}
        {kind === "policy_decision" && entry.policy_decision && (
          <div className="field-row">
            <dt>Verdict</dt>
            <dd>
              <code data-testid="policy-verdict">{policyDecisionLabel(entry.policy_decision)}</code>
              {decision?.requested_scopes?.length ? ` — scopes ${decision.requested_scopes.join(", ")}` : ""}
            </dd>
          </div>
        )}
        {kind === "intent" && decision && (
          <div className="field-row">
            <dt>Proposed by</dt>
            <dd>
              <code>{decision.principal}</code> via <code>{decision.tool}</code>
            </dd>
          </div>
        )}
        {kind === "approval" && approval && (
          <div className="field-row">
            <dt>Decision</dt>
            <dd data-testid="approval-decision">
              {approval.status}
              {approval.decided_by ? ` by ${approval.decided_by}` : ""}
              {approval.decided_at ? `, ${formatDateTime(approval.decided_at)}` : ""}
            </dd>
          </div>
        )}
        {kind === "execution" && execution && (
          <>
            <div className="field-row">
              <dt>Status</dt>
              <dd>
                {execution.status}, {execution.attempts} attempt{execution.attempts === 1 ? "" : "s"}
                {execution.approved_by ? `, approved by ${execution.approved_by}` : ""}
              </dd>
            </div>
            <div className="field-row">
              <dt>Idempotency key</dt>
              <dd>
                <code>{execution.idempotency_key}</code>
              </dd>
            </div>
            {execution.error && (
              <div className="field-row">
                <dt>Error</dt>
                <dd>
                  <pre>{execution.error}</pre>
                </dd>
              </div>
            )}
          </>
        )}
        {showWorkflow && entry.workflow_id && (
          <div className="field-row">
            <dt>Workflow</dt>
            <dd>
              <a href={href.workflow(entry.workflow_id)}>
                <code>{entry.workflow_id}</code>
              </a>
            </dd>
          </div>
        )}
      </dl>
      {kind === "receipt" && execution?.receipt && (
        <figure className="payload receipt">
          <figcaption className="small muted">
            Provider receipt for <code>{execution.tool_name}</code> (redacted)
          </figcaption>
          <pre data-testid="receipt">{json(execution.receipt)}</pre>
        </figure>
      )}
    </li>
  );
}

interface Props {
  client: ApiClient;
  workflowId: string;
  onUnauthenticated?: () => void;
}

type Load =
  | { kind: "loading" }
  | { kind: "loaded"; entries: AuditTrailEntry[]; withheld: string[] }
  | { kind: "error"; message: string };

/** One workflow's audit trail, `GET /v1/workflows/{id}/audit`. */
export function WorkflowAudit({ client, workflowId, onUnauthenticated }: Props) {
  const [load, setLoad] = useState<Load>({ kind: "loading" });

  const refresh = useCallback(async () => {
    try {
      const { value, withheld } = await client.getWorkflowAudit(workflowId);
      setLoad({ kind: "loaded", entries: value.entries ?? [], withheld });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated?.();
        return;
      }
      setLoad({ kind: "error", message: describeError(error, "No audit trail for this workflow.") });
    }
  }, [client, workflowId, onUnauthenticated]);

  useEffect(() => {
    setLoad({ kind: "loading" });
    void refresh();
  }, [refresh]);

  return (
    <section className="panel" aria-label="Audit trail">
      <div className="panel-header">
        <h3>Audit trail</h3>
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
          <WithheldNotice paths={load.withheld} />
          {load.entries.length === 0 ? (
            <p className="empty">Nothing recorded for this workflow yet.</p>
          ) : (
            auditChains(load.entries).map((chain) => (
              <div
                key={chain.actionHash ?? "events"}
                className="audit-chain"
                data-testid="audit-chain"
              >
                <h4>
                  {chain.actionHash ? (
                    <>
                      Action <code className="hash">{chain.actionHash.slice(0, 12)}</code>
                    </>
                  ) : (
                    "Other events"
                  )}
                </h4>
                <ol className="audit-steps">
                  {chain.steps.map((step) => (
                    <AuditStepView key={step.key} step={step} />
                  ))}
                </ol>
              </div>
            ))
          )}
        </>
      )}
    </section>
  );
}

export function describeError(error: unknown, notFound: string): string {
  if (error instanceof ApiError && error.isNotFound) return notFound;
  if (error instanceof NetworkError) return "Could not reach the PersonalOS API.";
  if (error instanceof Error) return error.message;
  return String(error);
}
