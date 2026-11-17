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

import type { components, paths } from "./schema";

export type PendingApproval = components["schemas"]["PendingApprovalItem"];
export type ActionStatus = components["schemas"]["ActionStatus"];
export type ActionPreview = components["schemas"]["ActionPreview"];
export type AttachmentPreview = components["schemas"]["AttachmentPreview"];
export type ApprovalState = components["schemas"]["ApprovalState"];
export type DecisionResponse = components["schemas"]["DecisionResponse"];
export type WhoAmI = components["schemas"]["WhoAmIResponse"];

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
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
