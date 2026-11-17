import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createApiClient } from "../src/api/client";
import { ApplicationDetail } from "../src/components/ApplicationDetail";
import {
  API,
  TOKEN,
  WORKFLOW_ID,
  applicationDetail,
  emailApproval,
  fakeApi,
  invalidTransition,
  json,
  queued,
  sseStream,
} from "./fixtures";

const detail = applicationDetail();
const ID = detail.application.application_id;

function renderDetail(fetch: (request: Request) => Promise<Response>) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(
    <ApplicationDetail
      client={client}
      applicationId={ID}
      onUnauthenticated={() => {}}
      timings={{ reconnectDelayMs: 5, pollIntervalMs: 5 }}
    />,
  );
}

describe("an application's detail", () => {
  it("shows the fit score and the evidence-backed match", async () => {
    const api = fakeApi({ [`GET /v1/applications/${ID}`]: () => json(detail) });
    renderDetail(api.fetch);

    expect(await screen.findByRole("heading", { name: "Acme — Backend Engineer" })).toBeVisible();
    expect(screen.getByTestId("application-status")).toHaveTextContent("Ready to apply");
    expect(screen.getByTestId("fit-score")).toHaveTextContent("82%");
    expect(screen.getByText("APPLY")).toBeVisible();

    const rows = screen.getAllByTestId("matched-requirement");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("5+ years of Python");
    expect(within(rows[0]!).getByTestId("evidence-ref")).toHaveTextContent("resume ev-python");
    // Linked to the document that cites the same evidence, and back.
    expect(within(rows[0]!).getByRole("button", { name: "cover_letter v2" })).toBeVisible();
    const artifact = screen.getByTestId("artifact");
    expect(artifact).toHaveTextContent("cover_letter v2");
    await userEvent.click(within(artifact).getByRole("button", { name: "ev-python" }));
    expect(document.activeElement).toBe(rows[0]);
    await userEvent.click(within(rows[0]!).getByRole("button", { name: "cover_letter v2" }));
    expect(document.activeElement).toBe(artifact);

    expect(screen.getByTestId("missing-requirements")).toHaveTextContent("Kubernetes minor — Nice to have");
    expect(screen.getByTestId("recruiter-events")).toHaveTextContent("interview_invite");
    expect(screen.getByTestId("recruiter-events")).toHaveTextContent("user: Send availability");
    expect(screen.getByTestId("history")).toHaveTextContent("Preparing → Ready to apply");
    // The posting link opens without handing over this page.
    expect(screen.getByRole("link", { name: "https://jobs.acme.example/backend" })).toHaveAttribute(
      "rel",
      "noopener noreferrer",
    );
  });

  it("links its pending approvals and checkpoints, and follows the workflow", async () => {
    const approval = emailApproval({ workflow_id: WORKFLOW_ID, action_type: "submit_application" });
    const stream = sseStream();
    const api = fakeApi({
      [`GET /v1/applications/${ID}`]: () =>
        json(
          applicationDetail({
            pending_approvals: [approval],
            pending_checkpoints: [
              {
                application_id: ID,
                checkpoint_id: "0c000000-0000-4000-8000-0000000000c1",
                dedupe_key: "follow-up",
                kind: "no_response",
                reason: "No reply in 7 days",
                condition: { kind: "no_inbound_message" } as never,
                trigger_at: "2026-11-24T12:00:00",
                expires_at: "2026-12-01T12:00:00",
                status: "pending",
                thread_id: "follow_up:1",
                workflow_id: WORKFLOW_ID,
              },
            ],
          }),
        ),
      [`GET /v1/workflows/${WORKFLOW_ID}/events`]: () => stream.response(),
    });
    renderDetail(api.fetch);

    expect(await screen.findByTestId("pending-approval")).toHaveTextContent("Submit application");
    expect(screen.getByRole("link", { name: "Review in the approvals inbox" })).toHaveAttribute(
      "href",
      "#/approvals",
    );
    expect(screen.getByTestId("pending-checkpoints")).toHaveTextContent("no response — No reply in 7 days");
    expect(screen.getByRole("link", { name: "progress" })).toHaveAttribute("href", `#/workflows/${WORKFLOW_ID}`);
    await waitFor(() =>
      expect(api.requests.some((r) => r.url === `/v1/workflows/${WORKFLOW_ID}/events`)).toBe(true),
    );
  });

  it("offers the API's allowed next states and shows a refusal", async () => {
    const api = fakeApi({
      [`GET /v1/applications/${ID}`]: () => json(detail),
      [`POST /v1/applications/${ID}/transition`]: () =>
        invalidTransition("cannot transition application from 'ready_to_apply' to 'skipped'", ["applied"]),
    });
    renderDetail(api.fetch);

    const moves = await screen.findByTestId("allowed-moves");
    expect(within(moves).getAllByRole("button").map((b) => b.textContent)).toEqual(["Applied …", "Skipped"]);
    await userEvent.click(within(moves).getByRole("button", { name: "Skipped" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Allowed next states: Applied");
    expect(screen.getByTestId("application-status")).toHaveTextContent("Ready to apply");
  });

  it("queues a submission for approval instead of applying it", async () => {
    const stream = sseStream();
    const api = fakeApi({
      [`GET /v1/applications/${ID}`]: () => json(detail),
      [`POST /v1/applications/${ID}/transition`]: () => json(queued(detail.application, "applied"), 202),
      [`GET /v1/workflows/${WORKFLOW_ID}/events`]: () => stream.response(),
    });
    renderDetail(api.fetch);

    await userEvent.click(await screen.findByRole("button", { name: "Applied …" }));
    const prompt = screen.getByTestId("effect-prompt");
    await userEvent.click(within(prompt).getByLabelText(/Submit it through PersonalOS/));
    await userEvent.type(within(prompt).getByLabelText(/Cover note/), "Looking forward to it.");
    await userEvent.click(within(prompt).getByRole("button", { name: "Queue for approval" }));

    expect(await screen.findByTestId("move-feedback")).toHaveTextContent("Queued for approval");
    expect(screen.getByTestId("application-status")).toHaveTextContent("Ready to apply");
    const post = api.requests.find((r) => r.method === "POST");
    expect(post?.body).toEqual({
      to: "applied",
      reason: null,
      effect: "submit_application",
      message: "Looking forward to it.",
    });
    // Its progress is followed in place.
    await waitFor(() =>
      expect(api.requests.some((r) => r.url === `/v1/workflows/${WORKFLOW_ID}/events`)).toBe(true),
    );
  });
});
