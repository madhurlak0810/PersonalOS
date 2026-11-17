/**
 * Approvals shaped exactly as `GET /v1/approvals` returns them, typed against
 * the generated schema so a fixture that drifts from the API fails to compile.
 */
import { vi } from "vitest";

import type {
  ApplicationCard,
  ApplicationDetail,
  ApplicationStatus,
  BoardResponse,
  DecisionResponse,
  PendingApproval,
  TransitionResponse,
  WorkflowProgressEvent,
} from "../src/api/client";
import { APPLICATION_STATUSES } from "../src/api/client";

export const API = "http://api.test";
export const TOKEN = "pos_at_test-token";

const inAnHour = () => new Date(Date.now() + 60 * 60 * 1000).toISOString().replace("Z", "");
const anHourAgo = () => new Date(Date.now() - 60 * 60 * 1000).toISOString().replace("Z", "");

export function emailApproval(overrides: Partial<PendingApproval> = {}): PendingApproval {
  return {
    id: "0b6c0d4e-1111-4000-8000-000000000001",
    workflow_id: "0b6c0d4e-2222-4000-8000-000000000002",
    thread_id: "job_search:abc",
    status: "pending",
    action_id: "0b6c0d4e-3333-4000-8000-000000000003",
    action_status: "AWAITING_APPROVAL",
    action_type: "send_recruiter_message",
    target: "recruiter@acme.example",
    summary: "Reply to Acme's recruiter with your availability",
    preview: {
      recipient: "recruiter@acme.example",
      subject: "Re: Backend Engineer at Acme",
      body_diff: [
        "--- current",
        "+++ proposed",
        "@@ -0,0 +1,2 @@",
        "+Hi Dana, thanks for reaching out.",
        "+I am free Tuesday afternoon.",
      ].join("\n"),
      attachments: [
        { name: "resume.pdf", artifact_type: "resume", sha256: "a".repeat(64), evidence_ids: [] },
      ],
      in_reply_to: "<msg-123@acme.example>",
    },
    payload: {
      recipient: "recruiter@acme.example",
      subject: "Re: Backend Engineer at Acme",
      body: "Hi Dana, thanks for reaching out.\nI am free Tuesday afternoon.",
    },
    consequences: [
      "Sends an email to recruiter@acme.example as a reply in thread <msg-123@acme.example>.",
      "It cannot be unsent; a correction would be a second message.",
    ],
    risk_level: "medium",
    requested_scopes: ["gmail.send"],
    action_hash: "f00dfeed".repeat(8),
    requested_at: anHourAgo(),
    expires_at: inAnHour(),
    ...overrides,
  };
}

export function calendarApproval(overrides: Partial<PendingApproval> = {}): PendingApproval {
  return emailApproval({
    id: "0b6c0d4e-1111-4000-8000-0000000000c1",
    action_type: "create_calendar_event",
    target: "primary",
    summary: "Put the Acme phone screen on your calendar",
    preview: { recipient: "primary", body_diff: "", attachments: [] },
    payload: {
      title: "Acme phone screen",
      starts_at: "2026-12-01T17:00:00Z",
      ends_at: "2026-12-01T18:00:00Z",
      time_zone: "America/New_York",
      role: "interview",
      logical_key: "acme-screen",
    },
    consequences: ["Adds an event to your calendar: primary."],
    requested_scopes: ["calendar.events"],
    action_hash: "ca1e0da5".repeat(8),
    ...overrides,
  });
}

export function expired(approval: PendingApproval): PendingApproval {
  return { ...approval, status: "expired", expires_at: anHourAgo() };
}

export function decision(approval: PendingApproval, approved: boolean): DecisionResponse {
  return {
    id: approval.id,
    workflow_id: approval.workflow_id,
    status: approved ? "approved" : "denied",
    outcome: approved ? "APPROVED" : "DENIED_BY_USER",
    action_hash: approval.action_hash,
    decided_by: "local",
    decided_at: new Date().toISOString(),
    resumed: true,
  };
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "X-Correlation-Id": "corr-1" },
  });
}

export interface Recorded {
  method: string;
  url: string;
  /** The query string, e.g. `?state=saved&offset=0`. */
  search: string;
  authorization: string | null;
  lastEventId: string | null;
  body: unknown;
}

type Handler = (body: unknown, request: Request) => Response | Promise<Response>;

/**
 * A fake API: `routes` maps "METHOD /path" to a response factory. Every request
 * is recorded, body parsed, so a test can assert on exactly what was sent.
 */
export function fakeApi(routes: Record<string, Handler>) {
  const requests: Recorded[] = [];
  const fetch = vi.fn(async (request: Request) => {
    const url = new URL(request.url);
    const text = request.method === "GET" ? "" : await request.clone().text();
    const body = text ? JSON.parse(text) : undefined;
    requests.push({
      method: request.method,
      url: url.pathname,
      search: url.search,
      authorization: request.headers.get("Authorization"),
      lastEventId: request.headers.get("Last-Event-ID"),
      body,
    });
    const handler = routes[`${request.method} ${url.pathname}`];
    if (!handler) return json({ error_code: "not_found", message: "no route", context_id: "x" }, 404);
    return handler(body, request);
  });
  return { fetch, requests };
}

// --- Applications ---------------------------------------------------------------

export const USER_ID = "0b6c0d4e-5555-4000-8000-000000000005";

export function applicationCard(overrides: Partial<ApplicationCard> = {}): ApplicationCard {
  return {
    application_id: "0a000000-0000-4000-8000-000000000001",
    job_posting_id: "0a000000-0000-4000-8000-0000000000f1",
    user_id: USER_ID,
    company: "Acme",
    title: "Backend Engineer",
    location: "Remote",
    url: "https://jobs.acme.example/backend",
    status: "discovered",
    status_updated_at: "2026-11-10T12:00:00",
    created_at: "2026-11-10T12:00:00",
    last_activity_at: "2026-11-12T09:30:00",
    ...overrides,
  };
}

/** A board with one column per status, every card in its own status's column. */
export function board(
  cards: ApplicationCard[],
  overrides: Partial<BoardResponse> = {},
  statuses: readonly ApplicationStatus[] = APPLICATION_STATUSES,
): BoardResponse {
  return {
    columns: statuses.map((status) => {
      const applications = cards.filter((card) => card.status === status);
      return { status, total: applications.length, applications };
    }),
    total: cards.length,
    limit: 50,
    offset: 0,
    next_offset: null,
    ...overrides,
  };
}

export function applied(card: ApplicationCard, to: ApplicationStatus): TransitionResponse {
  return {
    outcome: "applied",
    from_status: card.status,
    to_status: to,
    application: { ...card, status: to, status_updated_at: "2026-11-17T10:00:00" },
    event_id: "0e000000-0000-4000-8000-000000000001",
  };
}

export const WORKFLOW_ID = "0f000000-0000-4000-8000-000000000001";

export function queued(card: ApplicationCard, to: ApplicationStatus): TransitionResponse {
  return {
    outcome: "queued",
    from_status: card.status,
    to_status: to,
    effect: "submit_application",
    application: card,
    event_id: "0e000000-0000-4000-8000-000000000002",
    workflow_id: WORKFLOW_ID,
    thread_id: `application_transition:${card.application_id}`,
    command_id: "0c000000-0000-4000-8000-000000000001",
    policy_decision: "require_approval",
  };
}

export function invalidTransition(message: string, allowed: ApplicationStatus[]): Response {
  return json(
    {
      error_code: "validation_failed",
      message,
      context_id: "ctx-1",
      allowed_next_states: allowed,
    },
    422,
  );
}

export function applicationDetail(overrides: Partial<ApplicationDetail> = {}): ApplicationDetail {
  const card = applicationCard({ status: "ready_to_apply" });
  return {
    application: card,
    fit_score: 0.82,
    match: {
      dedupe_key: "acme:backend",
      score: 0.82,
      weighted_score: 0.82,
      recommendation: "APPLY",
      reasons: ["Strong Python and PostgreSQL background"],
      risks: [],
      constraints: [],
      tailoring_suggestions: [],
      ungrounded_claims: [],
      matched_requirements: [
        {
          requirement: "5+ years of Python",
          evidence_id: "ev-python",
          evidence_type: "resume",
          strength: "strong",
          rationale: "Six years at two employers",
        },
        {
          requirement: "PostgreSQL",
          evidence_id: "ev-postgres",
          evidence_type: "project",
          strength: "partial",
        },
      ],
      missing_requirements: [{ requirement: "Kubernetes", severity: "minor", note: "Nice to have" }],
    },
    match_recorded_at: "2026-11-11T08:00:00",
    allowed_next_states: ["applied", "skipped"],
    artifacts: [
      {
        artifact_version_id: "0d000000-0000-4000-8000-000000000001",
        artifact_type: "cover_letter",
        version: 2,
        created_at: "2026-11-11T09:00:00",
        generated_by: "draft_writer",
        evidence: [{ ref: "ev-python" }],
      },
    ],
    recruiter_events: [
      {
        communication_event_id: "0e000000-0000-4000-8000-0000000000a1",
        classification: "interview_invite",
        occurred_at: "2026-11-13T15:00:00",
        commitments: [
          {
            action: "Send availability",
            actor: "user",
            confidence: 0.9,
            source_message_id: "msg-1",
            due_at: "2026-11-15T17:00:00",
          },
        ],
      },
    ],
    pending_checkpoints: [],
    pending_approvals: [],
    history: [
      {
        event_id: "0e000000-0000-4000-8000-0000000000b1",
        from_status: "preparing",
        to_status: "ready_to_apply",
        occurred_at: "2026-11-11T09:05:00",
        actor: "worker",
      },
    ],
    ...overrides,
  };
}

// --- Workflow progress stream ---------------------------------------------------

let eventCounter = 0;

/** A progress event with a fresh, ordered id. */
export function progressEvent(
  type: WorkflowProgressEvent["type"],
  data: Record<string, unknown> = {},
  overrides: Partial<WorkflowProgressEvent> = {},
): WorkflowProgressEvent {
  eventCounter += 1;
  const n = String(eventCounter).padStart(12, "0");
  return {
    event_id: `0e7e0000-0000-4000-8000-${n}`,
    type,
    workflow_id: WORKFLOW_ID,
    thread_id: "job_search:abc",
    at: "2026-11-17T10:00:00",
    data,
    ...overrides,
  };
}

/** One event as the API frames it (`apps.api.routes.events.sse_frame`). */
export function frame(event: WorkflowProgressEvent): string {
  return `id: ${event.event_id}\nevent: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}

/** A `text/event-stream` response the test writes to, closes, or cuts. */
export function sseStream() {
  const encoder = new TextEncoder();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  let open = true;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
    cancel() {
      open = false;
    },
  });
  return {
    response: () =>
      new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    send(text: string) {
      if (open) controller.enqueue(encoder.encode(text));
    },
    close() {
      if (open) controller.close();
      open = false;
    },
    /** A dropped connection: the read fails mid-stream. */
    drop() {
      if (open) controller.error(new TypeError("network connection lost"));
      open = false;
    },
  };
}
