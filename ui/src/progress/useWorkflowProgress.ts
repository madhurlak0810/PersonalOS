import { useEffect, useState } from "react";

import { ApiError, type ApiClient, type WorkflowProgressEvent } from "../api/client";
import { ProgressFeed, type FeedOptions, type FeedStatus } from "./feed";

/** Reconnect and polling timings; tests shorten them. */
export type FeedTimings = Pick<
  FeedOptions,
  "reconnectDelayMs" | "maxStreamFailures" | "pollIntervalMs" | "liveRetryAfterPolls"
>;

export interface WorkflowProgressState {
  events: WorkflowProgressEvent[];
  status: FeedStatus;
  error: string | null;
}

/** Follow `workflowId`'s events for as long as the component is mounted. */
export function useWorkflowProgress(
  client: ApiClient,
  workflowId: string,
  { timings, onUnauthenticated }: { timings?: FeedTimings; onUnauthenticated?: () => void } = {},
): WorkflowProgressState {
  const [state, setState] = useState<WorkflowProgressState>({
    events: [],
    status: "connecting",
    error: null,
  });

  useEffect(() => {
    setState({ events: [], status: "connecting", error: null });
    const feed = new ProgressFeed({
      ...timings,
      open: ({ lastEventId, follow }) =>
        client.openWorkflowEvents(workflowId, { lastEventId, follow }),
      onEvent: (event) => setState((prev) => ({ ...prev, events: [...prev.events, event] })),
      onStatus: (status) => setState((prev) => ({ ...prev, status })),
      onFatal: (error) => {
        if (error instanceof ApiError && error.isUnauthenticated) {
          onUnauthenticated?.();
          return;
        }
        const message =
          error instanceof ApiError && error.isNotFound
            ? "This workflow does not exist."
            : error instanceof Error
              ? error.message
              : String(error);
        setState((prev) => ({ ...prev, error: message }));
      },
    });
    feed.start();
    return () => feed.stop();
    // `timings` is configuration, read once per workflow, not a dependency.
  },[client, workflowId, onUnauthenticated]);

  return state;
}
