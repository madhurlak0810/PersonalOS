/**
 * The approvals inbox against a fake API. The acceptance criteria:
 *
 * - a pending approval renders with its diff, and approving it calls the API
 *   with that approval's `action_hash`;
 * - a 409 shows the review-again message and does not resubmit;
 * - HTML or script in a recruiter email renders as inert text.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createApiClient } from "../src/api/client";
import { REVIEW_AGAIN_MESSAGE, EXPIRED_MESSAGE } from "../src/components/ApprovalCard";
import { ApprovalsInbox } from "../src/components/ApprovalsInbox";
import {
  API,
  TOKEN,
  calendarApproval,
  decision,
  emailApproval,
  expired,
  fakeApi,
  json,
  type Recorded,
} from "./fixtures";

function renderInbox(fetch: (request: Request) => Promise<Response>, onUnauthenticated = () => {}) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(<ApprovalsInbox client={client} onUnauthenticated={onUnauthenticated} />);
}

const decisionsSent = (requests: Recorded[]) =>
  requests.filter((r) => r.method === "POST" && r.url.endsWith("/decision"));

describe("a pending approval", () => {
  it("renders with its intent diff, status, scopes and expiry", async () => {
    const approval = emailApproval();
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [approval] }) });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    expect(within(card).getByTestId("action-status")).toHaveTextContent("AWAITING_APPROVAL");
    expect(within(card).getByRole("heading", { name: "Send recruiter message" })).toBeVisible();
    expect(within(card).getByTestId("approval-target")).toHaveTextContent("recruiter@acme.example");
    expect(within(card).getByTestId("approval-summary")).toHaveTextContent(approval.summary);
    expect(within(card).getByText("risk: medium")).toBeVisible();
    expect(within(card).getByText("gmail.send")).toBeVisible();
    expect(within(card).getByTestId("approval-expiry")).toHaveTextContent(/in \d+ minutes|in 1 hour/);

    // The exact email: recipient, subject, reply-or-new-thread, body diff, attachments.
    expect(within(card).getByTestId("email-recipient")).toHaveTextContent("recruiter@acme.example");
    expect(within(card).getByTestId("email-subject")).toHaveTextContent("Re: Backend Engineer at Acme");
    expect(within(card).getByTestId("email-thread")).toHaveTextContent(
      "Reply in existing thread <msg-123@acme.example>",
    );
    const diff = within(card).getByTestId("body-diff");
    expect(diff).toHaveTextContent("+Hi Dana, thanks for reaching out.");
    expect(diff.querySelectorAll(".diff-add")).toHaveLength(2);
    expect(within(card).getByTestId("attachments")).toHaveTextContent("resume.pdf");

    // The request carried the bearer token.
    expect(api.requests[0]?.authorization).toBe(`Bearer ${TOKEN}`);
  });

  it("is approved with the action_hash it was listed with", async () => {
    const approval = emailApproval();
    const other = emailApproval({
      id: "0b6c0d4e-1111-4000-8000-000000000099",
      action_hash: "0ther".repeat(12),
      summary: "Another approval",
    });
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [approval, other] }),
      [`POST /v1/approvals/${approval.id}/decision`]: () => json(decision(approval, true)),
    });

    renderInbox(api.fetch);
    const [first] = await screen.findAllByTestId("approval");
    await userEvent.click(within(first!).getByRole("button", { name: "Approve" }));

    await within(first!).findByText(/Approval recorded/);
    const sent = decisionsSent(api.requests);
    expect(sent).toHaveLength(1);
    expect(sent[0]).toMatchObject({
      url: `/v1/approvals/${approval.id}/decision`,
      authorization: `Bearer ${TOKEN}`,
      body: { approved: true, action_hash: approval.action_hash },
    });
    // Approving is not doing: still AWAITING_APPROVAL, never shown as executed.
    const status = within(first!).getByTestId("action-status");
    expect(status).toHaveTextContent("AWAITING_APPROVAL");
    expect(status).toHaveTextContent("not carried out yet");
    expect(within(first!).getByRole("button", { name: "Approve" })).toBeDisabled();
  });

  it("is rejected with its action_hash and then reads DENIED", async () => {
    const approval = emailApproval();
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [approval] }),
      [`POST /v1/approvals/${approval.id}/decision`]: () => json(decision(approval, false)),
    });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");
    await userEvent.click(within(card).getByRole("button", { name: "Reject" }));

    await within(card).findByText(/Rejection recorded/);
    expect(decisionsSent(api.requests)[0]?.body).toEqual({
      approved: false,
      action_hash: approval.action_hash,
    });
    expect(within(card).getByTestId("action-status")).toHaveTextContent("DENIED");
  });
});

describe("a 409 from the decision", () => {
  it("shows the review-again message and does not resubmit", async () => {
    const approval = emailApproval();
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [approval] }),
      [`POST /v1/approvals/${approval.id}/decision`]: () =>
        json(
          {
            error_code: "conflict",
            message: "action_hash does not match the action this approval was requested for",
            context_id: "ctx-409",
          },
          409,
        ),
    });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");
    const approve = within(card).getByRole("button", { name: "Approve" });
    await userEvent.click(approve);

    const alert = await within(card).findByRole("alert");
    expect(alert).toHaveTextContent(REVIEW_AGAIN_MESSAGE);
    expect(alert).toHaveTextContent("action_hash does not match");
    // Locked: neither button can send it again, and nothing was retried.
    expect(within(card).getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(within(card).getByRole("button", { name: "Reject" })).toBeDisabled();
    await userEvent.click(within(card).getByRole("button", { name: "Approve" }));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(decisionsSent(api.requests)).toHaveLength(1);
  });

  it("refreshes only when asked, and then shows the action as it now stands", async () => {
    const before = emailApproval();
    const after = emailApproval({
      action_hash: "c4a9ed".repeat(10),
      summary: "Reply to Acme's recruiter (edited)",
    });
    let listing = before;
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [listing] }),
      [`POST /v1/approvals/${before.id}/decision`]: () =>
        json({ error_code: "conflict", message: "changed", context_id: "c" }, 409),
    });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");
    await userEvent.click(within(card).getByRole("button", { name: "Approve" }));
    await within(card).findByRole("alert");
    const listsBefore = api.requests.filter((r) => r.method === "GET").length;

    listing = after;
    await userEvent.click(within(card).getByRole("button", { name: "Refresh and review" }));

    const fresh = await screen.findByTestId("approval");
    await within(fresh).findByText("Reply to Acme's recruiter (edited)");
    expect(api.requests.filter((r) => r.method === "GET").length).toBe(listsBefore + 1);
    // The refreshed card starts unlocked, and still nothing was resubmitted.
    expect(within(fresh).getByRole("button", { name: "Approve" })).toBeEnabled();
    expect(within(fresh).queryByRole("alert")).toBeNull();
    expect(decisionsSent(api.requests)).toHaveLength(1);
  });
});

describe("an expired approval", () => {
  it("cannot be approved, says so, and can still be rejected", async () => {
    const approval = expired(emailApproval());
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [approval] }),
      [`POST /v1/approvals/${approval.id}/decision`]: () => json(decision(approval, false)),
    });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    expect(within(card).getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(within(card).getByRole("status")).toHaveTextContent(EXPIRED_MESSAGE);
    expect(within(card).getByTestId("approval-expiry")).toHaveTextContent("expired");
    await userEvent.click(within(card).getByRole("button", { name: "Reject" }));
    await within(card).findByText(/Rejection recorded/);
  });

  it("whose window closed after it was listed gets the 409 review message too", async () => {
    // Listed as pending, but the server says it expired by the time of the click.
    const approval = emailApproval();
    const api = fakeApi({
      "GET /v1/approvals": () => json({ approvals: [approval] }),
      [`POST /v1/approvals/${approval.id}/decision`]: () =>
        json(
          {
            error_code: "conflict",
            message: "this approval has expired and can no longer be approved",
            context_id: "c",
          },
          409,
        ),
    });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");
    await userEvent.click(within(card).getByRole("button", { name: "Approve" }));

    const alert = await within(card).findByRole("alert");
    expect(alert).toHaveTextContent(REVIEW_AGAIN_MESSAGE);
    expect(alert).toHaveTextContent("expired");
    expect(decisionsSent(api.requests)).toHaveLength(1);
  });
});

describe("untrusted text", () => {
  const hostile = '<script>window.__pwned = true</script><img src=x onerror="window.__pwned=true">';

  it("in a recruiter email body renders as inert text", async () => {
    const approval = emailApproval({
      summary: `Reply to <b>Acme</b> ${hostile}`,
      target: `<a href="javascript:alert(1)">recruiter</a>`,
      preview: {
        recipient: `<iframe src="https://evil.example"></iframe>`,
        subject: `Re: <style>body{display:none}</style>Offer`,
        body_diff: ["--- current", "+++ proposed", "@@ -0,0 +1 @@", `+${hostile}`].join("\n"),
        attachments: [{ name: `<svg onload="window.__pwned=true">.pdf`, evidence_ids: [] }],
        in_reply_to: `<img src=x onerror=alert(1)>`,
      },
      payload: { body: hostile },
      consequences: [`Sends an email ${hostile}`],
    });
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [approval] }) });

    const { container } = renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    // Shown as the characters it is made of...
    expect(within(card).getByTestId("body-diff")).toHaveTextContent(`+${hostile}`);
    expect(within(card).getByTestId("approval-summary")).toHaveTextContent(`Reply to <b>Acme</b>`);
    expect(within(card).getByTestId("email-subject")).toHaveTextContent("<style>body{display:none}</style>Offer");
    expect(within(card).getByTestId("action-payload").textContent).toContain(hostile.replace(/"/g, '\\"'));
    // ...and nothing in it became an element.
    for (const tag of ["script", "img", "iframe", "style", "svg", "b", "a"]) {
      expect(container.querySelector(tag), tag).toBeNull();
    }
    expect(container.querySelector("[onerror], [onload], [href^='javascript']")).toBeNull();
    expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
  });
});

describe("other action types", () => {
  it("a calendar write shows the event details from its payload", async () => {
    const approval = calendarApproval();
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [approval] }) });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    const event = within(card).getByTestId("calendar-event");
    expect(event).toHaveTextContent("Acme phone screen");
    expect(event).toHaveTextContent("2026-12-01T17:00:00Z");
    expect(event).toHaveTextContent("2026-12-01T18:00:00Z");
    expect(event).toHaveTextContent("America/New_York");
    expect(within(card).getByRole("heading", { name: "Create calendar event" })).toBeVisible();
  });

  it("a calendar write without a verified payload warns instead of guessing", async () => {
    const approval = calendarApproval({ payload: null });
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [approval] }) });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    expect(within(card).getByRole("note")).toHaveTextContent("did not return the event's details");
  });

  it("any other action shows its typed payload", async () => {
    const approval = emailApproval({
      action_type: "submit_application",
      target: "https://boards.example/acme/123",
      preview: {
        recipient: "https://boards.example/acme/123",
        body_diff: "",
        attachments: [{ name: "cover_letter.md", evidence_ids: [] }],
      },
      payload: { dedupe_key: "acme-backend", url: "https://boards.example/acme/123" },
    });
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [approval] }) });

    renderInbox(api.fetch);
    const card = await screen.findByTestId("approval");

    const payload = within(card).getByTestId("action-payload");
    expect(payload).toHaveTextContent('"dedupe_key": "acme-backend"');
    expect(payload).toBeVisible();
    expect(within(card).getByTestId("attachments")).toHaveTextContent("cover_letter.md");
  });
});

describe("the listing", () => {
  it("says when nothing is waiting", async () => {
    const api = fakeApi({ "GET /v1/approvals": () => json({ approvals: [] }) });
    renderInbox(api.fetch);
    expect(await screen.findByText("Nothing is waiting for your approval.")).toBeVisible();
  });

  it("hands a 401 to the caller rather than showing data", async () => {
    let signedOut = false;
    const api = fakeApi({
      "GET /v1/approvals": () =>
        json({ error_code: "unauthenticated", message: "not authenticated", context_id: "c" }, 401),
    });
    renderInbox(api.fetch, () => {
      signedOut = true;
    });
    await waitFor(() => expect(signedOut).toBe(true));
    expect(screen.queryByTestId("approval")).toBeNull();
  });
});
