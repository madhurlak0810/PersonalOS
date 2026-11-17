import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createApiClient } from "../src/api/client";
import { AuditSearch } from "../src/components/AuditSearch";
import { WorkflowAudit } from "../src/components/AuditTrail";
import { API, AUDIT_WORKFLOW_ID, TOKEN, approvedSendTrail, fakeApi, json } from "./fixtures";

const SECRET = "ya29.a0AfH6SMBlive-google-access-token";

function client(fetch: (request: Request) => Promise<Response>) {
  return createApiClient({ baseUrl: API, token: TOKEN, fetch });
}

afterEach(() => vi.restoreAllMocks());

describe("a workflow's audit trail", () => {
  it("shows intent, policy decision (REQUIRE_APPROVAL), approval and receipt, in order", async () => {
    const api = fakeApi({
      [`GET /v1/workflows/${AUDIT_WORKFLOW_ID}/audit`]: () =>
        json({ workflow_id: AUDIT_WORKFLOW_ID, entries: approvedSendTrail() }),
    });
    render(<WorkflowAudit client={client(api.fetch)} workflowId={AUDIT_WORKFLOW_ID} />);

    const chain = await screen.findByTestId("audit-chain");
    const steps = within(chain).getAllByTestId("audit-step");
    const kinds = steps.map((step) => step.dataset.kind);
    expect(kinds).toEqual(["intent", "policy_decision", "approval", "policy_decision", "execution", "receipt"]);

    // The four the acceptance test names, in that order.
    const index = (kind: string) => kinds.indexOf(kind);
    expect(index("intent")).toBeLessThan(index("policy_decision"));
    expect(index("policy_decision")).toBeLessThan(index("approval"));
    expect(index("approval")).toBeLessThan(index("receipt"));

    expect(within(steps[1]!).getByTestId("policy-verdict")).toHaveTextContent(/^REQUIRE_APPROVAL$/);
    expect(within(steps[2]!).getByTestId("approval-decision")).toHaveTextContent("approved by local");
    const receipt = within(steps[5]!).getByTestId("receipt");
    expect(receipt).toHaveTextContent('"external_reference": "gmail-1"');

    // Read-only: one GET, nothing else.
    expect(api.requests.map((r) => `${r.method} ${r.url}`)).toEqual([
      `GET /v1/workflows/${AUDIT_WORKFLOW_ID}/audit`,
    ]);
    expect(within(chain).queryByRole("button")).toBeNull();
  });

  it("shows the API's redaction marker but never an unredacted credential", async () => {
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});
    const api = fakeApi({
      [`GET /v1/workflows/${AUDIT_WORKFLOW_ID}/audit`]: () =>
        json({
          workflow_id: AUDIT_WORKFLOW_ID,
          entries: approvedSendTrail({
            refresh_token: "[REDACTED]",
            detail: "sent with Authorization: Bearer [REDACTED]",
            // An API bug: these got through the redactor.
            access_token: SECRET,
            echoed: `upstream said Authorization: Bearer ${SECRET}`,
          }),
        }),
    });
    const { container } = render(
      <WorkflowAudit client={client(api.fetch)} workflowId={AUDIT_WORKFLOW_ID} />,
    );

    const receipt = await screen.findByTestId("receipt");
    expect(container.textContent).not.toContain(SECRET);
    expect(container.textContent).not.toContain("ya29.");
    expect(receipt).toHaveTextContent('"refresh_token": "[REDACTED]"');
    expect(receipt).toHaveTextContent("Authorization: Bearer [REDACTED]");
    expect(receipt).toHaveTextContent('"access_token": "[withheld]"');
    expect(receipt).toHaveTextContent('"echoed": "[withheld]"');
    expect(screen.getByTestId("withheld-notice")).toHaveTextContent("API bug");
    expect(errorLog).toHaveBeenCalled();
    expect(JSON.stringify(errorLog.mock.calls)).not.toContain(SECRET);
  });

  it("says so when the workflow has no trail", async () => {
    const api = fakeApi({});
    render(<WorkflowAudit client={client(api.fetch)} workflowId={AUDIT_WORKFLOW_ID} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("No audit trail");
  });
});

describe("searching the audit trail", () => {
  it("sends every filter, and pages with the API's offsets", async () => {
    const entries = approvedSendTrail().slice(2).reverse();
    const api = fakeApi({
      "GET /v1/audit": (_body, request) => {
        const offset = Number(new URL(request.url).searchParams.get("offset") ?? 0);
        return json({
          entries: offset === 0 ? entries.slice(0, 2) : entries.slice(2),
          total: 3,
          limit: 2,
          offset,
          next_offset: offset === 0 ? 2 : null,
        });
      },
    });
    render(<AuditSearch client={client(api.fetch)} onUnauthenticated={() => {}} pageSize={2} />);

    expect(await screen.findByTestId("audit-range")).toHaveTextContent("1–2 of 3");
    expect(api.requests[0]!.search).toBe("?limit=2&offset=0");

    await userEvent.type(screen.getByLabelText("Workflow"), AUDIT_WORKFLOW_ID);
    await userEvent.type(screen.getByLabelText("Actor"), "local");
    await userEvent.type(screen.getByLabelText("Action"), "approval.APPROVED");
    await userEvent.type(screen.getByLabelText("Application"), "0a000000-0000-4000-8000-000000000001");
    await userEvent.type(screen.getByLabelText("From"), "2026-11-17T09:00");
    await userEvent.type(screen.getByLabelText("To"), "2026-11-17T11:00");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(api.requests).toHaveLength(2));
    const params = new URLSearchParams(api.requests[1]!.search);
    expect(params.get("workflow_id")).toBe(AUDIT_WORKFLOW_ID);
    expect(params.get("application_id")).toBe("0a000000-0000-4000-8000-000000000001");
    expect(params.get("actor")).toBe("local");
    expect(params.get("action")).toBe("approval.APPROVED");
    // Local times, sent as the API's naive UTC.
    expect(params.get("since")).toBe(new Date("2026-11-17T09:00").toISOString().replace("Z", ""));
    expect(params.get("until")).toBe(new Date("2026-11-17T11:00").toISOString().replace("Z", ""));
    expect(params.get("offset")).toBe("0");

    await screen.findByText("1–2 of 3, newest first");
    await userEvent.click(screen.getByRole("button", { name: "Older" }));
    expect(await screen.findByText("3–3 of 3, newest first")).toBeVisible();
    expect(new URLSearchParams(api.requests[2]!.search).get("offset")).toBe("2");
    expect(new URLSearchParams(api.requests[2]!.search).get("actor")).toBe("local");
    expect(screen.getByRole("button", { name: "Older" })).toBeDisabled();
    expect(api.requests.every((r) => r.method === "GET")).toBe(true);
  });

  it("refuses a malformed workflow id or an inverted range before asking the API", async () => {
    const api = fakeApi({
      "GET /v1/audit": () => json({ entries: [], total: 0, limit: 50, offset: 0, next_offset: null }),
    });
    render(<AuditSearch client={client(api.fetch)} onUnauthenticated={() => {}} />);
    await screen.findByText("No audit events match.");

    await userEvent.type(screen.getByLabelText("Workflow"), "not-a-uuid");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(screen.getByRole("alert")).toHaveTextContent("not a UUID");

    await userEvent.clear(screen.getByLabelText("Workflow"));
    await userEvent.type(screen.getByLabelText("From"), "2026-11-17T11:00");
    await userEvent.type(screen.getByLabelText("To"), "2026-11-17T09:00");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(screen.getByRole("alert")).toHaveTextContent("ends before it starts");
    expect(api.requests).toHaveLength(1);
  });
});
