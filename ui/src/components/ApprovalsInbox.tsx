import { useCallback, useEffect, useState } from "react";

import { ApiError, NetworkError, type ApiClient, type PendingApproval } from "../api/client";
import { ApprovalCard } from "./ApprovalCard";

type Load =
  | { kind: "loading" }
  | { kind: "loaded"; approvals: PendingApproval[] }
  | { kind: "error"; message: string };

interface Props {
  client: ApiClient;
  onUnauthenticated: () => void;
  /** Injected in tests; how often "expires in" and expiry locks are re-evaluated. */
  clockIntervalMs?: number;
}

function useNow(intervalMs: number): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}

/** `GET /v1/approvals`, one card per pending approval. Refreshes only when asked. */
export function ApprovalsInbox({ client, onUnauthenticated, clockIntervalMs = 15_000 }: Props) {
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  // Bumped on every refresh, so each card is rebuilt from the fresh listing and
  // a card locked by a 409 does not carry its state over to the new data.
  const [generation, setGeneration] = useState(0);
  const now = useNow(clockIntervalMs);

  const refresh = useCallback(async () => {
    setLoad({ kind: "loading" });
    try {
      const approvals = await client.listApprovals();
      setLoad({ kind: "loaded", approvals });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated();
        return;
      }
      const message =
        error instanceof NetworkError
          ? "Could not reach the PersonalOS API. Is `python -m personalos.cli api` running, and does API_CORS_ORIGINS include this page's origin?"
          : error instanceof Error
            ? error.message
            : String(error);
      setLoad({ kind: "error", message });
    } finally {
      setGeneration((value) => value + 1);
    }
  }, [client, onUnauthenticated]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <section className="inbox" aria-labelledby="inbox-title">
      <div className="inbox-header">
        <h2 id="inbox-title">Approvals inbox</h2>
        <button type="button" onClick={() => void refresh()} disabled={load.kind === "loading"}>
          {load.kind === "loading" ? "Loading…" : "Refresh"}
        </button>
      </div>
      {load.kind === "error" && (
        <p className="callout error" role="alert">
          {load.message}
        </p>
      )}
      {load.kind === "loaded" && load.approvals.length === 0 && (
        <p className="empty">Nothing is waiting for your approval.</p>
      )}
      {load.kind === "loaded" &&
        load.approvals.map((approval) => (
          <ApprovalCard
            key={`${generation}:${approval.id}:${approval.action_hash}`}
            approval={approval}
            client={client}
            now={now}
            onRefresh={() => void refresh()}
            onUnauthenticated={onUnauthenticated}
          />
        ))}
    </section>
  );
}
