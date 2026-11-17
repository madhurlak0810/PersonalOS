/**
 * Approvals shaped exactly as `GET /v1/approvals` returns them, typed against
 * the generated schema so a fixture that drifts from the API fails to compile.
 */
import { vi } from "vitest";

import type { DecisionResponse, PendingApproval } from "../src/api/client";

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
  authorization: string | null;
  body: unknown;
}

/**
 * A fake API: `routes` maps "METHOD /path" to a response factory. Every request
 * is recorded, body parsed, so a test can assert on exactly what was sent.
 */
export function fakeApi(routes: Record<string, (body: unknown) => Response>) {
  const requests: Recorded[] = [];
  const fetch = vi.fn(async (request: Request) => {
    const url = new URL(request.url);
    const text = request.method === "GET" ? "" : await request.clone().text();
    const body = text ? JSON.parse(text) : undefined;
    requests.push({
      method: request.method,
      url: url.pathname,
      authorization: request.headers.get("Authorization"),
      body,
    });
    const handler = routes[`${request.method} ${url.pathname}`];
    if (!handler) return json({ error_code: "not_found", message: "no route", context_id: "x" }, 404);
    return handler(body);
  });
  return { fetch, requests };
}
