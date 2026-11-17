/**
 * The audit trail, searched: `GET /v1/audit`, newest first, a page at a time.
 *
 * Filters by workflow, application, actor, action and time range, each one
 * passed straight to the API. Read-only, like every audit view.
 */
import { useCallback, useEffect, useState, type FormEvent } from "react";

import { ApiError, type ApiClient, type AuditQuery, type AuditSearchResponse } from "../api/client";
import { AuditStepView, auditSteps, describeError } from "./AuditTrail";
import { WithheldNotice } from "./WithheldNotice";

interface Props {
  client: ApiClient;
  onUnauthenticated: () => void;
  /** Filters to start with, e.g. one workflow. */
  initial?: Partial<Filters>;
  pageSize?: number;
}

interface Filters {
  workflowId: string;
  applicationId: string;
  actor: string;
  action: string;
  /** `<input type="datetime-local">` values, in the viewer's time zone. */
  since: string;
  until: string;
}

const EMPTY: Filters = { workflowId: "", applicationId: "", actor: "", action: "", since: "", until: "" };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** A local `datetime-local` value as the API's naive-UTC timestamp. */
export function localDateTimeToApi(value: string): string | undefined {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return undefined;
  return date.toISOString().replace("Z", "");
}

function validate(filters: Filters): string | null {
  if (filters.workflowId && !UUID.test(filters.workflowId.trim())) return "The workflow id is not a UUID.";
  if (filters.applicationId && !UUID.test(filters.applicationId.trim()))
    return "The application id is not a UUID.";
  const since = localDateTimeToApi(filters.since);
  const until = localDateTimeToApi(filters.until);
  if (since && until && since > until) return "The time range ends before it starts.";
  return null;
}

function toQuery(filters: Filters, limit: number, offset: number): AuditQuery {
  return {
    workflowId: filters.workflowId.trim() || undefined,
    applicationId: filters.applicationId.trim() || undefined,
    actor: filters.actor.trim() || undefined,
    action: filters.action.trim() || undefined,
    since: localDateTimeToApi(filters.since),
    until: localDateTimeToApi(filters.until),
    limit,
    offset,
  };
}

type Load =
  | { kind: "loading" }
  | { kind: "loaded"; page: AuditSearchResponse; withheld: string[] }
  | { kind: "error"; message: string };

export function AuditSearch({ client, onUnauthenticated, initial, pageSize = 50 }: Props) {
  const [draft, setDraft] = useState<Filters>({ ...EMPTY, ...initial });
  const [applied, setApplied] = useState<Filters>({ ...EMPTY, ...initial });
  const [offset, setOffset] = useState(0);
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const [invalid, setInvalid] = useState<string | null>(null);

  const fetchPage = useCallback(async () => {
    setLoad({ kind: "loading" });
    try {
      const { value, withheld } = await client.searchAudit(toQuery(applied, pageSize, offset));
      setLoad({ kind: "loaded", page: value, withheld });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated();
        return;
      }
      setLoad({ kind: "error", message: describeError(error, "Nothing found.") });
    }
  }, [client, applied, offset, pageSize, onUnauthenticated]);

  useEffect(() => {
    void fetchPage();
  }, [fetchPage]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const problem = validate(draft);
    setInvalid(problem);
    if (problem) return;
    setApplied(draft);
    setOffset(0);
  };

  const field = (name: keyof Filters, label: string, type = "text", placeholder?: string) => (
    <label className="field">
      {label}
      <input
        type={type}
        value={draft[name]}
        placeholder={placeholder}
        onChange={(event) => setDraft({ ...draft, [name]: event.target.value })}
      />
    </label>
  );

  const page = load.kind === "loaded" ? load.page : null;
  const entries = page?.entries ?? [];

  return (
    <section aria-label="Audit trail search">
      <h2>Audit trail</h2>
      <form className="board-filters" onSubmit={submit} aria-label="Audit filters">
        {field("workflowId", "Workflow", "text", "workflow id")}
        {field("applicationId", "Application", "text", "application id")}
        {field("actor", "Actor")}
        {field("action", "Action", "text", "e.g. approval.APPROVED")}
        {field("since", "From", "datetime-local")}
        {field("until", "To", "datetime-local")}
        <div className="actions">
          <button type="submit" className="approve">
            Search
          </button>
          <button
            type="button"
            onClick={() => {
              setDraft(EMPTY);
              setApplied(EMPTY);
              setOffset(0);
              setInvalid(null);
            }}
          >
            Clear
          </button>
        </div>
      </form>
      {invalid && (
        <p className="callout error" role="alert">
          {invalid}
        </p>
      )}
      {load.kind === "error" && (
        <p className="callout error" role="alert">
          {load.message}
        </p>
      )}
      {load.kind === "loading" && <p className="muted">Loading…</p>}
      {page && (
        <>
          <WithheldNotice paths={load.kind === "loaded" ? load.withheld : []} />
          <div className="pager">
            <span className="muted" data-testid="audit-range">
              {page.total === 0
                ? "No audit events match."
                : `${page.offset + 1}–${page.offset + entries.length} of ${page.total}, newest first`}
            </span>
            <div className="actions">
              <button
                type="button"
                disabled={page.offset === 0}
                onClick={() => setOffset(Math.max(0, page.offset - page.limit))}
              >
                Newer
              </button>
              <button
                type="button"
                disabled={page.next_offset == null}
                onClick={() => page.next_offset != null && setOffset(page.next_offset)}
              >
                Older
              </button>
            </div>
          </div>
          {entries.length > 0 && (
            <ol className="audit-steps" data-testid="audit-results">
              {auditSteps(entries).map((step) => (
                <AuditStepView key={step.key} step={step} showWorkflow />
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}
