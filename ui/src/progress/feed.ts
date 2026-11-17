/**
 * A workflow's progress, followed live: `GET /v1/workflows/{id}/events`.
 *
 * - **Exact resumption.** Every connection after the first sends
 *   `Last-Event-ID` with the id of the last event delivered, so the API sends
 *   everything after it and nothing before. Events are also de-duplicated by
 *   id here, so even a server that replayed one would not show it twice.
 * - **Reconnects on its own.** A dropped or ended stream is reopened after the
 *   delay the server asked for (`retry:`).
 * - **Degrades to polling.** After `maxStreamFailures` drops in a row with no
 *   new event between them, it stops holding a stream open and instead asks
 *   for the backlog (`?follow=false`) every `pollIntervalMs`, with the same
 *   cursor. Every `liveRetryAfterPolls` polls it tries a live stream again.
 * - **Stops on answers that will not change.** 401 (the token), 404 (no such
 *   workflow) and 422 (a cursor the API does not recognise) are reported and
 *   end the feed rather than being retried forever.
 *
 * Framework-free, so the reconnect logic is tested without rendering.
 */
import { ApiError, type WorkflowProgressEvent } from "../api/client";
import { SseParser } from "../api/sse";

export type FeedStatus =
  /** Opening the first connection. */
  | "connecting"
  /** A stream is open and events arrive as they happen. */
  | "live"
  /** The stream dropped; reopening it from the last event. */
  | "reconnecting"
  /** The stream kept dropping; asking for new events every few seconds instead. */
  | "polling"
  /** Stopped after an error that retrying will not fix. */
  | "failed"
  | "stopped";

export type OpenStream = (options: {
  lastEventId: string | null;
  follow: boolean;
}) => Promise<ReadableStream<Uint8Array>>;

export interface FeedOptions {
  open: OpenStream;
  onEvent: (event: WorkflowProgressEvent) => void;
  onStatus?: (status: FeedStatus) => void;
  /** A non-retryable failure; the feed has stopped. */
  onFatal?: (error: unknown) => void;
  /** Start after this event, e.g. one already shown. */
  lastEventId?: string | null;
  /** Used until the server sends `retry:`. */
  reconnectDelayMs?: number;
  maxStreamFailures?: number;
  pollIntervalMs?: number;
  liveRetryAfterPolls?: number;
}

const PROGRESS_TYPES = new Set([
  "step_started",
  "step_completed",
  "approval_requested",
  "approval_resolved",
  "failure",
  "workflow_completed",
]);

/** Errors a retry will not fix: stop and say so. */
export function isFatal(error: unknown): boolean {
  return error instanceof ApiError && [401, 403, 404, 422].includes(error.status);
}

export class ProgressFeed {
  private readonly options: Required<Omit<FeedOptions, "onStatus" | "onFatal" | "lastEventId">> &
    Pick<FeedOptions, "onStatus" | "onFatal">;
  private readonly seen = new Set<string>();
  private cursor: string | null;
  private retryMs: number;
  private stopped = false;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private wake: (() => void) | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private status: FeedStatus | null = null;

  constructor(options: FeedOptions) {
    this.options = {
      reconnectDelayMs: 3000,
      maxStreamFailures: 3,
      pollIntervalMs: 5000,
      liveRetryAfterPolls: 6,
      ...options,
    };
    this.cursor = options.lastEventId ?? null;
    this.retryMs = this.options.reconnectDelayMs;
  }

  /** The id the next connection will resume after. */
  get lastEventId(): string | null {
    return this.cursor;
  }

  start(): void {
    void this.run();
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    void this.reader?.cancel().catch(() => {});
    if (this.timer !== null) clearTimeout(this.timer);
    this.wake?.();
    this.setStatus("stopped");
  }

  private setStatus(status: FeedStatus): void {
    if (this.status === status) return;
    this.status = status;
    this.options.onStatus?.(status);
  }

  private async run(): Promise<void> {
    let mode: "stream" | "poll" = "stream";
    let failures = 0;
    let polls = 0;
    let connected = false;
    while (!this.stopped) {
      const follow = mode === "stream";
      this.setStatus(follow ? (connected ? "reconnecting" : "connecting") : "polling");
      let delivered = 0;
      try {
        const stream = await this.options.open({ lastEventId: this.cursor, follow });
        if (this.stopped) {
          void stream.cancel().catch(() => {});
          return;
        }
        if (follow) {
          connected = true;
          this.setStatus("live");
        }
        delivered = await this.consume(stream);
      } catch (error) {
        if (this.stopped) return;
        if (isFatal(error)) {
          this.stopped = true;
          this.setStatus("failed");
          this.options.onFatal?.(error);
          return;
        }
        // Anything else (network, 5xx, a stream cut mid-read) is worth retrying.
      }
      if (this.stopped) return;

      if (mode === "stream") {
        failures = delivered > 0 ? 1 : failures + 1;
        if (failures >= this.options.maxStreamFailures) {
          mode = "poll";
          polls = 0;
          this.setStatus("polling");
          await this.sleep(this.options.pollIntervalMs);
        } else {
          await this.sleep(this.retryMs);
        }
      } else {
        polls += 1;
        if (polls >= this.options.liveRetryAfterPolls) {
          mode = "stream";
          failures = this.options.maxStreamFailures - 1;
        }
        await this.sleep(this.options.pollIntervalMs);
      }
    }
  }

  /** Read one connection to its end; returns how many new events it delivered. */
  private async consume(stream: ReadableStream<Uint8Array>): Promise<number> {
    const parser = new SseParser();
    const decoder = new TextDecoder();
    const reader = stream.getReader();
    this.reader = reader;
    let delivered = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (this.stopped) break;
        const text = done ? decoder.decode() : decoder.decode(value, { stream: true });
        for (const message of parser.push(text)) {
          if (this.deliver(message.id, message.event, message.data)) delivered += 1;
        }
        if (parser.retryMs !== null) this.retryMs = parser.retryMs;
        if (done) break;
      }
    } finally {
      this.reader = null;
      reader.releaseLock();
    }
    return delivered;
  }

  private deliver(id: string, kind: string, data: string): boolean {
    if (!PROGRESS_TYPES.has(kind)) return false;
    let event: WorkflowProgressEvent;
    try {
      event = JSON.parse(data) as WorkflowProgressEvent;
    } catch {
      return false;
    }
    const eventId = event.event_id || id;
    if (!eventId || this.seen.has(eventId)) {
      // Already shown: never twice. Still move the cursor past it.
      if (eventId) this.cursor = eventId;
      return false;
    }
    this.seen.add(eventId);
    this.cursor = eventId;
    this.options.onEvent({ ...event, event_id: eventId });
    return true;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
      this.wake = resolve;
      this.timer = setTimeout(() => {
        this.timer = null;
        this.wake = null;
        resolve();
      }, ms);
    });
  }
}
