/**
 * Live workflow progress against a fake event stream. The acceptance criteria:
 *
 * - the view receives step events live;
 * - after a simulated disconnect it resumes from `Last-Event-ID`, with no
 *   event missed or shown twice;
 * - an `approval_requested` event produces a visible prompt linking to the
 *   approvals inbox.
 *
 * Plus the fallback: a stream that keeps dropping degrades to polling.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createApiClient, type WorkflowProgressEvent } from "../src/api/client";
import { WorkflowProgress } from "../src/components/WorkflowProgress";
import type { FeedTimings } from "../src/progress/useWorkflowProgress";
import {
  API,
  TOKEN,
  WORKFLOW_ID,
  fakeApi,
  frame,
  json,
  progressEvent,
  sseStream,
  type Recorded,
} from "./fixtures";

const EVENTS_PATH = `GET /v1/workflows/${WORKFLOW_ID}/events`;
const FAST: FeedTimings = {
  reconnectDelayMs: 5,
  pollIntervalMs: 5,
  maxStreamFailures: 3,
  liveRetryAfterPolls: 1000,
};

function renderProgress(fetch: (request: Request) => Promise<Response>, timings = FAST) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(<WorkflowProgress client={client} workflowId={WORKFLOW_ID} timings={timings} />);
}

const shownIds = () => screen.queryAllByTestId("progress-event").map((el) => el.dataset.eventId);
const streamRequests = (requests: Recorded[]) =>
  requests.filter((r) => r.url === `/v1/workflows/${WORKFLOW_ID}/events`);

/** Hands out one controllable stream per connection, in order. */
function streams(count: number) {
  const made = Array.from({ length: count }, () => sseStream());
  let next = 0;
  return {
    made,
    handler: () => {
      const stream = made[next++];
      if (!stream) return json({ error_code: "unavailable", message: "no more", context_id: "x" }, 503);
      return stream.response();
    },
  };
}

describe("workflow progress", () => {
  it("shows step events as they arrive", async () => {
    const { made, handler } = streams(1);
    const api = fakeApi({ [EVENTS_PATH]: handler });
    renderProgress(api.fetch);

    await waitFor(() => expect(screen.getByTestId("feed-status")).toHaveAttribute("data-status", "live"));
    expect(screen.getByText("No progress yet.")).toBeVisible();

    const started = progressEvent("step_started", { step: "search_jobs" });
    made[0]!.send("retry: 5\n\n");
    made[0]!.send(frame(started));
    expect(await screen.findByText("Started search_jobs")).toBeVisible();

    // A keepalive comment, then a frame split across two chunks.
    const completed = progressEvent("step_completed", { step: "search_jobs" });
    const text = frame(completed);
    made[0]!.send(": keepalive\n\n");
    made[0]!.send(text.slice(0, 20));
    made[0]!.send(text.slice(20));
    expect(await screen.findByText("Completed search_jobs")).toBeVisible();

    expect(shownIds()).toEqual([started.event_id, completed.event_id]);
    // Authenticated by header, as a fetch-based client; no token in the URL.
    const [first] = streamRequests(api.requests);
    expect(first?.authorization).toBe(`Bearer ${TOKEN}`);
    expect(first?.search).not.toContain(TOKEN);
    expect(first?.lastEventId).toBeNull();
  });

  it("resumes from Last-Event-ID after a disconnect, without gaps or duplicates", async () => {
    // A server that honours the cursor, like `apps/api/routes/events.py`.
    const log: WorkflowProgressEvent[] = [
      progressEvent("step_started", { step: "search_jobs" }),
      progressEvent("step_completed", { step: "search_jobs" }),
    ];
    const connections: ReturnType<typeof sseStream>[] = [];
    const api = fakeApi({
      [EVENTS_PATH]: (_body, request) => {
        const cursor = request.headers.get("Last-Event-ID");
        const start = cursor ? log.findIndex((e) => e.event_id === cursor) + 1 : 0;
        const stream = sseStream();
        connections.push(stream);
        stream.send("retry: 5\n\n");
        for (const event of log.slice(start)) stream.send(frame(event));
        return stream.response();
      },
    });
    renderProgress(api.fetch);

    await waitFor(() => expect(shownIds()).toEqual(log.map((e) => e.event_id)));

    // The connection drops; while it is down, the run moves on.
    connections[0]!.drop();
    log.push(progressEvent("step_started", { step: "score_jobs" }));

    await waitFor(() => expect(streamRequests(api.requests)).toHaveLength(2));
    expect(streamRequests(api.requests)[1]?.lastEventId).toBe(log[1]!.event_id);
    await waitFor(() => expect(shownIds()).toHaveLength(3));
    expect(screen.getByTestId("feed-status")).toHaveAttribute("data-status", "live");

    // And it carries on live on the new connection.
    const completed = progressEvent("step_completed", { step: "score_jobs" });
    log.push(completed);
    connections[1]!.send(frame(completed));
    await waitFor(() => expect(shownIds()).toEqual(log.map((e) => e.event_id)));
    expect(new Set(shownIds()).size).toBe(shownIds().length);
  });

  it("never shows an event twice, even if a server replays it", async () => {
    const first = progressEvent("step_started", { step: "draft" });
    const second = progressEvent("step_completed", { step: "draft" });
    const { made, handler } = streams(2);
    const api = fakeApi({ [EVENTS_PATH]: handler });
    renderProgress(api.fetch);

    await waitFor(() => expect(streamRequests(api.requests)).toHaveLength(1));
    made[0]!.send(frame(first));
    await waitFor(() => expect(shownIds()).toEqual([first.event_id]));
    made[0]!.close();

    await waitFor(() => expect(streamRequests(api.requests)).toHaveLength(2));
    expect(streamRequests(api.requests)[1]?.lastEventId).toBe(first.event_id);
    made[1]!.send(frame(first));
    made[1]!.send(frame(second));
    await waitFor(() => expect(shownIds()).toEqual([first.event_id, second.event_id]));
  });

  it("turns approval_requested into a prompt linking to the approvals inbox", async () => {
    const { made, handler } = streams(1);
    const api = fakeApi({ [EVENTS_PATH]: handler });
    renderProgress(api.fetch);
    await waitFor(() => expect(streamRequests(api.requests)).toHaveLength(1));

    const requested = progressEvent("approval_requested", {
      step: "approval_checkpoint",
      interrupt_id: "int-1",
      stage: "submission",
      requests: [{ request_id: "r1", kind: "submit_application", risk: "high" }],
    });
    made[0]!.send(frame(requested));

    const prompt = await screen.findByTestId("approval-prompt");
    expect(prompt).toBeVisible();
    expect(prompt).toHaveTextContent("Approval needed at approval_checkpoint: submit_application, risk high");
    expect(within(prompt).getByRole("link", { name: "Review it in the approvals inbox" })).toHaveAttribute(
      "href",
      "#/approvals",
    );

    // Once answered, the prompt goes; the history stays.
    made[0]!.send(
      frame(
        progressEvent("approval_resolved", {
          steps: ["approval_checkpoint"],
          interrupt_ids: ["int-1"],
          resolved_by: "local",
          decisions: [{ request_id: "r1", verdict: "approved" }],
        }),
      ),
    );
    await waitFor(() => expect(screen.queryByTestId("approval-prompt")).toBeNull());
    expect(screen.getByText("Approval answered by local: approved")).toBeVisible();
  });

  it("shows a failure event", async () => {
    const { made, handler } = streams(1);
    const api = fakeApi({ [EVENTS_PATH]: handler });
    renderProgress(api.fetch);
    await waitFor(() => expect(streamRequests(api.requests)).toHaveLength(1));

    made[0]!.send(
      frame(
        progressEvent("failure", {
          step: "draft_cover_letter",
          failure_class: "transient",
          message: "provider timed out",
        }),
      ),
    );
    expect(
      await screen.findByText("Failed at draft_cover_letter (transient): provider timed out"),
    ).toBeVisible();
  });

  it("falls back to polling with the same cursor when the stream keeps dropping", async () => {
    const event = progressEvent("step_started", { step: "search_jobs" });
    const later = progressEvent("step_completed", { step: "search_jobs" });
    let polls = 0;
    const api = fakeApi({
      [EVENTS_PATH]: (_body, request) => {
        const follow = new URL(request.url).searchParams.get("follow");
        if (follow === "true") {
          return json({ error_code: "unavailable", message: "proxy cut", context_id: "x" }, 502);
        }
        polls += 1;
        const stream = sseStream();
        if (polls === 1) stream.send(frame(event));
        if (polls === 2) stream.send(frame(later));
        stream.close();
        return stream.response();
      },
    });
    renderProgress(api.fetch);

    await waitFor(() => expect(screen.getByTestId("feed-status")).toHaveAttribute("data-status", "polling"));
    await waitFor(() => expect(shownIds()).toEqual([event.event_id, later.event_id]));
    const pollsSent = streamRequests(api.requests).filter((r) => r.search.includes("follow=false"));
    expect(pollsSent[0]?.lastEventId).toBeNull();
    expect(pollsSent[1]?.lastEventId).toBe(event.event_id);
    // Three live attempts first, then polls.
    expect(
      streamRequests(api.requests)
        .slice(0, 3)
        .every((r) => r.search.includes("follow=true")),
    ).toBe(true);
  });

  it("stops, and says so, when the API refuses the cursor", async () => {
    const api = fakeApi({
      [EVENTS_PATH]: () =>
        json({ error_code: "validation_failed", message: "Last-Event-ID 'x' is not an event id", context_id: "c" }, 422),
    });
    renderProgress(api.fetch);

    expect(await screen.findByRole("alert")).toHaveTextContent("is not an event id");
    expect(screen.getByTestId("feed-status")).toHaveAttribute("data-status", "failed");
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(streamRequests(api.requests)).toHaveLength(1);
  });
});
