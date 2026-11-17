import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createApiClient } from "../src/api/client";
import { WorkflowFailures } from "../src/components/WorkflowFailures";
import {
  API,
  FAILED_WORKFLOW_ID,
  TOKEN,
  failure,
  failures,
  fakeApi,
  integrations,
  json,
  policyDenial,
  retried,
  revokedOAuth,
} from "./fixtures";

const LIST = `GET /v1/workflows/${FAILED_WORKFLOW_ID}/failures`;
const RETRY = `POST /v1/workflows/${FAILED_WORKFLOW_ID}/retry`;
const DISMISS = `POST /v1/workflows/${FAILED_WORKFLOW_ID}/dismiss`;

function renderFailures(fetch: (request: Request) => Promise<Response>) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(<WorkflowFailures client={client} workflowId={FAILED_WORKFLOW_ID} />);
}

const calls = (api: ReturnType<typeof fakeApi>, route: string) =>
  api.requests.filter((r) => `${r.method} ${r.url}` === route);

describe("a workflow's failures", () => {
  it("shows the class, failing step, retryable flag, sanitized error and attempts", async () => {
    const api = fakeApi({ [LIST]: () => json(failures([failure({ attempts: 2 })])) });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    expect(within(card).getByTestId("failure-status")).toHaveTextContent("FAILED");
    expect(within(card).getByTestId("failure-class")).toHaveTextContent("provider_transient");
    expect(card).toHaveTextContent("fetch_postings");
    expect(within(card).getByTestId("failure-retryable")).toHaveTextContent("Yes");
    expect(within(card).getByTestId("failure-error")).toHaveTextContent("Greenhouse timed out after 30s");
    expect(within(card).getByTestId("failure-attempts")).toHaveTextContent("2 of 5");
  });

  it("shows a policy denial as denied, with no retry control", async () => {
    // Even if the API mistakenly offered one, a denial is never retried.
    const denial = policyDenial({ retryable: true, actions: ["retry", "dismiss"] });
    const api = fakeApi({ [LIST]: () => json(failures([denial])) });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    expect(card).toHaveAttribute("data-status", "denied");
    expect(within(card).getByTestId("failure-status")).toHaveTextContent("DENIED");
    expect(within(card).getByTestId("failure-status")).not.toHaveTextContent("FAILED");
    expect(within(card).queryByRole("button", { name: /retry/i })).toBeNull();
    expect(within(card).getByRole("button", { name: "Dismiss" })).toBeVisible();
    expect(calls(api, RETRY)).toHaveLength(0);
  });

  it("retries a recoverable failure once, and updates from the API's answer", async () => {
    const item = failure({ attempts: 2 });
    let resolveRetry!: (response: Response) => void;
    const api = fakeApi({
      [LIST]: () => json(failures([item])),
      [RETRY]: () => new Promise<Response>((resolve) => (resolveRetry = resolve)),
    });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    const button = within(card).getByRole("button", { name: "Retry step" });
    await userEvent.click(button);
    // In flight: the button is disabled, so a second click sends nothing.
    expect(button).toBeDisabled();
    await userEvent.click(button);
    resolveRetry(json(retried(item), 202));

    expect(await within(card).findByTestId("retry-queued")).toHaveTextContent(
      "attempt 3 of step fetch_postings",
    );
    const sent = calls(api, RETRY);
    expect(sent).toHaveLength(1);
    expect(sent[0]!.body).toEqual({ failure_id: item.failure_id });

    expect(card).toHaveAttribute("data-status", "retry_queued");
    expect(within(card).getByTestId("failure-status")).toHaveTextContent("RETRY QUEUED");
    expect(within(card).getByTestId("failure-attempts")).toHaveTextContent("3 of 5");
    expect(within(card).getByTestId("failure-retryable")).toHaveTextContent("No — a retry is already queued");
    expect(within(card).queryByRole("button", { name: "Retry step" })).toBeNull();
    // Updated from the response, not by re-reading the listing.
    expect(calls(api, LIST)).toHaveLength(1);
  });

  it("shows a 409 from retry without resending it", async () => {
    const api = fakeApi({
      [LIST]: () => json(failures([failure()])),
      [RETRY]: () =>
        json({ error_code: "conflict", message: "a retry is already queued", context_id: "c" }, 409),
    });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    await userEvent.click(within(card).getByRole("button", { name: "Retry step" }));
    expect(await within(card).findByRole("alert")).toHaveTextContent("a retry is already queued");
    expect(within(card).queryByRole("button", { name: "Retry step" })).toBeNull();
    expect(calls(api, RETRY)).toHaveLength(1);

    await userEvent.click(within(card).getByRole("button", { name: "Refresh failures" }));
    await waitFor(() => expect(calls(api, LIST)).toHaveLength(2));
    expect(calls(api, RETRY)).toHaveLength(1);
  });

  it("asks for a reconnect instead of a retry when the provider refused permanently", async () => {
    const api = fakeApi({ [LIST]: () => json(failures([revokedOAuth()])) });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    const prompt = within(card).getByTestId("reconnect-prompt");
    expect(prompt).toHaveTextContent("Reconnect google");
    expect(prompt).toHaveTextContent("python -m personalos.cli auth google");
    expect(within(prompt).getByRole("link", { name: "Integration health" })).toHaveAttribute(
      "href",
      "#/integrations",
    );
    expect(within(card).queryByRole("button", { name: /retry/i })).toBeNull();
    expect(within(card).getByRole("button", { name: "Dismiss" })).toBeVisible();
  });

  it("dismisses a failure and shows who closed it", async () => {
    const item = policyDenial();
    const api = fakeApi({
      [LIST]: () => json(failures([item])),
      [DISMISS]: () =>
        json({
          workflow_id: FAILED_WORKFLOW_ID,
          failure_id: item.failure_id,
          status: "dismissed",
          dismissed_by: "local",
          dismissed_at: "2026-11-17T10:00:00",
        }),
    });
    renderFailures(api.fetch);

    const card = await screen.findByTestId("failure");
    await userEvent.click(within(card).getByRole("button", { name: "Dismiss" }));
    expect(await within(card).findByTestId("dismissed")).toHaveTextContent("Dismissed by local");
    expect(calls(api, DISMISS)[0]!.body).toEqual({ failure_id: item.failure_id });
    expect(within(card).queryByRole("button")).toBeNull();
  });

  it("names degraded integrations next to the failures when health is available", async () => {
    const api = fakeApi({
      [LIST]: () => json(failures([failure()])),
      "GET /v1/integrations": () => json({ integrations: integrations() }),
    });
    renderFailures(api.fetch);

    const banner = await screen.findByTestId("integrations-degraded");
    expect(banner).toHaveTextContent("gmail (degraded: the refresh token was revoked)");
    expect(banner).toHaveTextContent("greenhouse (down: timed out)");
    expect(banner).not.toHaveTextContent("ollama");
  });

  it("withholds a credential that reached the error text", async () => {
    const leaked = failure({ last_error: "401 from provider: refresh_token=1//0gLIVE-refresh-token-value-xyz" });
    const api = fakeApi({ [LIST]: () => json(failures([leaked])) });
    const { container } = renderFailures(api.fetch);

    expect(await screen.findByTestId("failure-error")).toHaveTextContent("[withheld]");
    expect(container.textContent).not.toContain("1//0gLIVE");
    expect(screen.getByTestId("withheld-notice")).toHaveTextContent("failures[0].last_error");
  });
});
