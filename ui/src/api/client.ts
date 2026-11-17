/**
 * The typed PersonalOS API client.
 *
 * Every request and response type comes from `schema.d.ts`, which
 * `openapi-typescript` generates from `ui/openapi.json`, which
 * `scripts/export_openapi.py` writes from the FastAPI app itself. CI checks both
 * steps (`npm run api:check`), so a route or model change that the UI has not
 * caught up with fails the build instead of failing at runtime.
 *
 * Errors come back as `ApiError`, carrying the API's error envelope
 * (`error_code`, `message`, `context_id`) when it sent one. Nothing here retries:
 * a decision that fails is reported to the reviewer, who decides what to do.
 */
import createClient from "openapi-fetch";

import { withholdSecrets, type Guarded } from "../secrets";
import type { components, paths } from "./schema";

export type PendingApproval = components["schemas"]["PendingApprovalItem"];
export type ActionStatus = components["schemas"]["ActionStatus"];
export type ActionPreview = components["schemas"]["ActionPreview"];
export type AttachmentPreview = components["schemas"]["AttachmentPreview"];
export type ApprovalState = components["schemas"]["ApprovalState"];
export type DecisionResponse = components["schemas"]["DecisionResponse"];
export type WhoAmI = components["schemas"]["WhoAmIResponse"];
export type ApplicationStatus = components["schemas"]["ApplicationStatus"];
export type ApplicationCard = components["schemas"]["ApplicationCard"];
export type BoardResponse = components["schemas"]["BoardResponse"];
export type BoardColumn = components["schemas"]["BoardColumnView"];
export type ApplicationDetail = components["schemas"]["ApplicationDetailResponse"];
export type TransitionEffect = components["schemas"]["TransitionEffect"];
export type TransitionRequest = components["schemas"]["TransitionRequest"];
export type TransitionResponse = components["schemas"]["TransitionResponse"];
export type MatchedRequirement = components["schemas"]["MatchedRequirement"];
export type MissingRequirement = components["schemas"]["MissingRequirement"];
export type ArtifactSummary = components["schemas"]["ArtifactSummary"];
export type RecruiterEventSummary = components["schemas"]["RecruiterEventSummary"];
export type PendingCheckpoint = components["schemas"]["PendingCheckpoint"];
export type AuditTrailEntry = components["schemas"]["AuditTrailEntry"];
export type AuditStage = components["schemas"]["AuditStage"];
export type AuditSearchResponse = components["schemas"]["AuditSearchResponse"];
export type WorkflowAuditResponse = components["schemas"]["WorkflowAuditResponse"];
export type WorkflowFailure = components["schemas"]["WorkflowFailure"];
export type WorkflowFailuresResponse = components["schemas"]["WorkflowFailuresResponse"];
export type EventFailure = components["schemas"]["EventFailure"];
export type FailureClass = components["schemas"]["FailureClass"];
export type FailureStatus = components["schemas"]["FailureStatus"];
export type FailedOperation = components["schemas"]["FailedOperation"];
export type RetryResponse = components["schemas"]["RetryResponse"];
export type DismissResponse = components["schemas"]["DismissResponse"];
export type IntegrationStatus = components["schemas"]["IntegrationStatus"];
export type IntegrationHealth = components["schemas"]["IntegrationHealth"];
export type { Guarded };

/** `GET /v1/audit` filters. Times are API timestamps (UTC, no offset). */
export interface AuditQuery {
  workflowId?: string;
  applicationId?: string;
  actor?: string;
  action?: string;
  since?: string;
  until?: string;
  limit?: number;
  offset?: number;
}

/**
 * Every `ApplicationStatus`, in the order the API declares them: lifecycle
 * order, holding states last. This is also the board's column order.
 */
export const APPLICATION_STATUSES: readonly ApplicationStatus[] = [
  "discovered",
  "saved",
  "preparing",
  "ready_to_apply",
  "applied",
  "response",
  "interviewing",
  "offer",
  "accepted",
  "declined",
  "rejected",
  "withdrawn",
  "skipped",
  "follow_up_pending",
  "stalled",
];

/**
 * The statuses a move may carry an external effect to
 * (`personalos.domain.applications.EFFECT_TARGETS`). A move to one of these is
 * never sent without the user choosing, explicitly, whether the effect should
 * happen (through the approval flow) or the move only records what already did.
 */
export const EFFECT_TARGETS: Readonly<Record<TransitionEffect, readonly ApplicationStatus[]>> = {
  submit_application: ["applied"],
  notify_recruiter: ["withdrawn", "declined", "accepted"],
};

export function effectFor(status: ApplicationStatus): TransitionEffect | null {
  for (const [effect, targets] of Object.entries(EFFECT_TARGETS) as [
    TransitionEffect,
    readonly ApplicationStatus[],
  ][]) {
    if (targets.includes(status)) return effect;
  }
  return null;
}

/** `GET /v1/applications` filters. Dates are API timestamps (UTC, no offset). */
export interface BoardQuery {
  states?: ApplicationStatus[];
  company?: string;
  createdFrom?: string;
  createdTo?: string;
  limit?: number;
  offset?: number;
}

/** The kinds `GET /v1/workflows/{id}/events` streams (`ProgressEventType`). */
export type ProgressEventType =
  | "step_started"
  | "step_completed"
  | "approval_requested"
  | "approval_resolved"
  | "failure"
  | "workflow_completed";

/**
 * One progress event (`personalos.domain.progress.WorkflowProgressEvent`).
 * The stream is `text/event-stream`, so OpenAPI has no schema for its frames;
 * this mirrors the Python model by hand.
 */
export interface WorkflowProgressEvent {
  event_id: string;
  type: ProgressEventType;
  workflow_id: string;
  correlation_id?: string | null;
  thread_id?: string | null;
  at: string;
  data: Record<string, unknown>;
}

/** Every `ActionStatus`, in the order the API's enum declares them. */
export const ACTION_STATUSES: readonly ActionStatus[] = [
  "RECOMMENDED",
  "AWAITING_APPROVAL",
  "EXECUTED",
  "DENIED",
  "FAILED",
];

/** The body every API failure carries (`PersonalOSError.to_envelope`). */
export interface ErrorEnvelope {
  error_code: string;
  message: string;
  context_id: string;
  /** On a refused application move (422): the states it may move to instead. */
  allowed_next_states?: ApplicationStatus[];
}

export class ApiError extends Error {
  readonly status: number;
  readonly envelope: ErrorEnvelope | null;
  /** `X-Correlation-Id`, to quote when reporting the failure. */
  readonly correlationId: string | null;

  constructor(status: number, envelope: ErrorEnvelope | null, correlationId: string | null) {
    super(envelope?.message ?? `request failed with HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.envelope = envelope;
    this.correlationId = correlationId;
  }

  get isUnauthenticated(): boolean {
    return this.status === 401;
  }

  /** A 409: the approval changed, expired, or was already decided. */
  get isConflict(): boolean {
    return this.status === 409;
  }

  get isNotFound(): boolean {
    return this.status === 404;
  }

  /**
   * The states the API said an application may move to, when it refused a
   * move off the lifecycle. `null` when the error was something else.
   */
  get allowedNextStates(): ApplicationStatus[] | null {
    const states = this.envelope?.allowed_next_states;
    return this.status === 422 && Array.isArray(states) ? states : null;
  }
}

/** The request never got an HTTP answer: the API is down, or CORS refused it. */
export class NetworkError extends Error {
  constructor(cause: unknown) {
    super("could not reach the PersonalOS API");
    this.name = "NetworkError";
    this.cause = cause;
  }
}

export interface ApiClientOptions {
  baseUrl: string;
  token: string;
  /** Injected in tests; the browser's `fetch` otherwise. */
  fetch?: (request: Request) => Promise<Response>;
}

function isEnvelope(value: unknown): value is ErrorEnvelope {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Record<string, unknown>).error_code === "string" &&
    typeof (value as Record<string, unknown>).message === "string"
  );
}

function toApiError(response: Response, error: unknown): ApiError {
  return new ApiError(
    response.status,
    isEnvelope(error) ? error : null,
    response.headers.get("X-Correlation-Id"),
  );
}

export function createApiClient({ baseUrl, token, fetch: fetchImpl }: ApiClientOptions) {
  const client = createClient<paths>({
    baseUrl: baseUrl.replace(/\/+$/, ""),
    fetch: async (request: Request) => {
      try {
        return await (fetchImpl ?? globalThis.fetch)(request);
      } catch (cause) {
        throw new NetworkError(cause);
      }
    },
  });
  client.use({
    onRequest({ request }) {
      // The bearer token goes in this header and nowhere else: never a URL,
      // never a cookie (the API's CORS policy has credentials off).
      request.headers.set("Authorization", `Bearer ${token}`);
      return request;
    },
  });

  return {
    /** The verified caller; a 401 here means the token is wrong. */
    async whoami(): Promise<WhoAmI> {
      const { data, error, response } = await client.GET("/v1/auth/whoami");
      if (data === undefined) throw toApiError(response, error);
      return data;
    },

    /** Every approval still owed an answer, oldest first. */
    async listApprovals(): Promise<PendingApproval[]> {
      const { data, error, response } = await client.GET("/v1/approvals");
      if (data === undefined) throw toApiError(response, error);
      return data.approvals ?? [];
    },

    /**
     * Approve or reject `approval`, bound to the hash it was listed with.
     *
     * Takes the approval itself rather than an id and a hash, so the hash sent
     * is always the one belonging to the approval on screen. A 409 (the action
     * changed, the window closed, or it was decided the other way) is thrown as
     * an `ApiError` and is never retried.
     */
    async decide(approval: PendingApproval, approved: boolean): Promise<DecisionResponse> {
      const { data, error, response } = await client.POST(
        "/v1/approvals/{approval_id}/decision",
        {
          params: { path: { approval_id: approval.id } },
          body: { approved, action_hash: approval.action_hash },
        },
      );
      if (data === undefined) throw toApiError(response, error);
      return data;
    },

    /** The board: one column per lifecycle state, a page at a time. */
    async listApplications(query: BoardQuery = {}): Promise<BoardResponse> {
      const { data, error, response } = await client.GET("/v1/applications", {
        params: {
          query: {
            state: query.states?.length ? query.states : undefined,
            company: query.company || undefined,
            created_from: query.createdFrom || undefined,
            created_to: query.createdTo || undefined,
            limit: query.limit,
            offset: query.offset,
          },
        },
      });
      if (data === undefined) throw toApiError(response, error);
      return data;
    },

    /** One application in full: fit, evidence, documents, recruiter events, open waits. */
    async getApplication(applicationId: string): Promise<ApplicationDetail> {
      const { data, error, response } = await client.GET("/v1/applications/{application_id}", {
        params: { path: { application_id: applicationId } },
      });
      if (data === undefined) throw toApiError(response, error);
      return data;
    },

    /**
     * Ask for a lifecycle move. Resolves on 200 (`outcome: "applied"`, the card
     * as it now stands) and 202 (`outcome: "queued"`: the effect waits for
     * approval and the status has not changed). A move off the lifecycle is an
     * `ApiError` with `allowedNextStates`; nothing here retries.
     */
    async transition(applicationId: string, body: TransitionRequest): Promise<TransitionResponse> {
      const { data, error, response } = await client.POST(
        "/v1/applications/{application_id}/transition",
        { params: { path: { application_id: applicationId } }, body },
      );
      if (data === undefined) throw toApiError(response, error);
      return data;
    },

    // --- Audit, failures and integrations -------------------------------------
    //
    // Every one of these responses is passed through `withholdSecrets` before
    // it is returned: the API redacts them, so a credential that survives is an
    // API bug, and it is replaced rather than handed to a component. `withheld`
    // says where, so the screen can say so.

    /** One workflow's whole audit trail, oldest first. Read-only. */
    async getWorkflowAudit(workflowId: string): Promise<Guarded<WorkflowAuditResponse>> {
      const { data, error, response } = await client.GET("/v1/workflows/{workflow_id}/audit", {
        params: { path: { workflow_id: workflowId } },
      });
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data);
    },

    /** A page of the audit trail matching every filter given, newest first. Read-only. */
    async searchAudit(query: AuditQuery = {}): Promise<Guarded<AuditSearchResponse>> {
      const { data, error, response } = await client.GET("/v1/audit", {
        params: {
          query: {
            workflow_id: query.workflowId || undefined,
            application_id: query.applicationId || undefined,
            actor: query.actor || undefined,
            action: query.action || undefined,
            since: query.since || undefined,
            until: query.until || undefined,
            limit: query.limit,
            offset: query.offset,
          },
        },
      });
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data);
    },

    /** A workflow's failed steps, and the events the worker gave up delivering. */
    async listFailures(
      workflowId: string,
      { includeDismissed = false }: { includeDismissed?: boolean } = {},
    ): Promise<Guarded<WorkflowFailuresResponse>> {
      const { data, error, response } = await client.GET("/v1/workflows/{workflow_id}/failures", {
        params: {
          path: { workflow_id: workflowId },
          query: { include_dismissed: includeDismissed || undefined },
        },
      });
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data);
    },

    /**
     * Queue one re-run of `failure`'s step. Sent once: a 409 (not retryable, or
     * already being retried) is thrown as an `ApiError` and never resent.
     */
    async retryFailure(failure: WorkflowFailure): Promise<Guarded<RetryResponse>> {
      const { data, error, response } = await client.POST("/v1/workflows/{workflow_id}/retry", {
        params: { path: { workflow_id: failure.workflow_id } },
        body: { failure_id: failure.failure_id },
      });
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data);
    },

    /** Close `failure` without retrying it. */
    async dismissFailure(failure: WorkflowFailure): Promise<DismissResponse> {
      const { data, error, response } = await client.POST("/v1/workflows/{workflow_id}/dismiss", {
        params: { path: { workflow_id: failure.workflow_id } },
        body: { failure_id: failure.failure_id },
      });
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data).value;
    },

    /** Every integration's connection and health state (Phase J health checks). */
    async listIntegrations(): Promise<Guarded<IntegrationStatus[]>> {
      const { data, error, response } = await client.GET("/v1/integrations");
      if (data === undefined) throw toApiError(response, error);
      return withholdSecrets(data.integrations ?? []);
    },

    /**
     * Open a workflow's progress stream, resuming after `lastEventId`.
     *
     * Sent with `fetch`, not `EventSource`, so the bearer token travels in its
     * header (never in a URL) and `Last-Event-ID` is set by us, exactly. With
     * `follow: false` the API sends the backlog and closes: one poll.
     * The caller reads, and cancels, the returned body.
     */
    async openWorkflowEvents(
      workflowId: string,
      { lastEventId, follow }: { lastEventId: string | null; follow: boolean },
    ): Promise<ReadableStream<Uint8Array>> {
      const { data, error, response } = await client.GET("/v1/workflows/{workflow_id}/events", {
        params: {
          path: { workflow_id: workflowId },
          query: { follow },
          header: lastEventId ? { "Last-Event-ID": lastEventId } : undefined,
        },
        headers: { Accept: "text/event-stream" },
        parseAs: "stream",
      });
      if (!response.ok) throw toApiError(response, error);
      if (!data) throw new NetworkError(new Error("the event stream had no body"));
      return data as ReadableStream<Uint8Array>;
    },
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
