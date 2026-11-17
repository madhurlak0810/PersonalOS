/**
 * Integration health: `GET /v1/integrations` (Phase J health checks).
 *
 * Each integration's connection and its last health check -- `healthy`,
 * `degraded` or `down` -- with the reason, the last error (redacted), when it
 * last worked, and what fixes it: a reconnect, or nothing but waiting for the
 * worker's next check. No token or credential is shown; a Google account's
 * `credential_ref` is left off on purpose, and anything credential-shaped the
 * API sends by mistake is withheld by the client.
 *
 * `compact` shows only the integrations that are not healthy, and nothing at
 * all when every one is (or when the API has no health to give), for embedding
 * next to a workflow's failures.
 */
import { useCallback, useEffect, useState } from "react";

import { ApiError, type ApiClient, type IntegrationStatus } from "../api/client";
import { formatDateTime } from "../format";
import { href } from "../router";
import { describeError } from "./AuditTrail";
import { WithheldNotice } from "./WithheldNotice";

interface Props {
  client: ApiClient;
  onUnauthenticated?: () => void;
  compact?: boolean;
}

type Load =
  | { kind: "loading" }
  | { kind: "loaded"; integrations: IntegrationStatus[]; withheld: string[] }
  | { kind: "unavailable"; message: string };

/** How to reconnect a provider. Google is the only one with a consent flow. */
export function reconnectHint(provider: string): string {
  return provider === "google" || provider.startsWith("gmail") || provider.startsWith("google")
    ? "python -m personalos.cli auth google"
    : `reconnect ${provider} and wait for the next health check`;
}

export function isUnhealthy(integration: IntegrationStatus): boolean {
  return (
    integration.reconnect_required === true ||
    integration.health === "degraded" ||
    integration.health === "down"
  );
}

function healthText(integration: IntegrationStatus): string {
  return integration.health ?? "not checked";
}

function connectionText(integration: IntegrationStatus): string {
  if (integration.reconnect_required) return "Reconnect required";
  return integration.connected ? "Connected" : "Not connected";
}

export function IntegrationsHealth({ client, onUnauthenticated, compact }: Props) {
  const [load, setLoad] = useState<Load>({ kind: "loading" });

  const refresh = useCallback(async () => {
    try {
      const { value, withheld } = await client.listIntegrations();
      setLoad({ kind: "loaded", integrations: value, withheld });
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated?.();
        return;
      }
      // "When available": an API without health checks is not an error here.
      setLoad({
        kind: "unavailable",
        message: describeError(error, "This API does not report integration health."),
      });
    }
  }, [client, onUnauthenticated]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (compact) {
    if (load.kind !== "loaded") return null;
    const unhealthy = load.integrations.filter(isUnhealthy);
    if (unhealthy.length === 0 && load.withheld.length === 0) return null;
    return (
      <div className="callout warning" role="status" data-testid="integrations-degraded">
        <WithheldNotice paths={load.withheld} />
        {unhealthy.length > 0 && (
          <p>
            <strong>Integrations not healthy:</strong>{" "}
            {unhealthy
              .map((i) => `${i.provider} (${healthText(i)}${i.health_reason ? `: ${i.health_reason}` : ""})`)
              .join("; ")}
            .{" "}
            <a href={href.integrations()}>Integration health</a>
          </p>
        )}
      </div>
    );
  }

  return (
    <section aria-label="Integration health">
      <div className="panel-header">
        <h2>Integrations</h2>
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>
      {load.kind === "loading" && <p className="muted">Loading…</p>}
      {load.kind === "unavailable" && (
        <p className="callout" role="status" data-testid="integrations-unavailable">
          Integration health is not available: {load.message}
        </p>
      )}
      {load.kind === "loaded" && (
        <>
          <WithheldNotice paths={load.withheld} />
          {load.integrations.length === 0 ? (
            <p className="empty">No integrations reported.</p>
          ) : (
            <table className="requirements integrations">
              <thead>
                <tr>
                  <th scope="col">Integration</th>
                  <th scope="col">Connection</th>
                  <th scope="col">Health</th>
                  <th scope="col">Details</th>
                </tr>
              </thead>
              <tbody>
                {load.integrations.map((integration) => (
                  <tr
                    key={integration.provider}
                    data-testid="integration"
                    data-provider={integration.provider}
                    data-health={integration.health ?? "unknown"}
                  >
                    <th scope="row">
                      <code>{integration.provider}</code>
                    </th>
                    <td>{connectionText(integration)}</td>
                    <td>
                      <span className={`health health-${integration.health ?? "unknown"}`}>
                        {healthText(integration)}
                      </span>
                    </td>
                    <td>
                      {integration.health_reason && <div>{integration.health_reason}</div>}
                      {integration.action === "reconnect" || integration.reconnect_required ? (
                        <div className="callout warning small" data-testid="reconnect-prompt">
                          Needs reconnecting; nothing retries it meanwhile. Run{" "}
                          <code>{reconnectHint(integration.provider)}</code>.
                        </div>
                      ) : integration.action === "retry" ? (
                        <div className="small muted">The worker's next health check will try again.</div>
                      ) : null}
                      {integration.last_error && (
                        <div className="small">
                          Last error: <pre>{integration.last_error}</pre>
                        </div>
                      )}
                      <div className="small muted">
                        {integration.last_success_at
                          ? `Last worked ${formatDateTime(integration.last_success_at)}`
                          : "Never worked yet"}
                        {integration.last_checked_at
                          ? ` · checked ${formatDateTime(integration.last_checked_at)}`
                          : ""}
                      </div>
                      {(integration.accounts ?? []).length > 0 && (
                        <ul className="small accounts">
                          {(integration.accounts ?? []).map((account) => (
                            <li key={account.account}>
                              {account.account} — {account.status}
                              {account.consent_expires_at
                                ? `, consent expires ${formatDateTime(account.consent_expires_at)}`
                                : ""}
                              {account.capabilities?.length ? ` (${account.capabilities.join(", ")})` : ""}
                            </li>
                          ))}
                        </ul>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </section>
  );
}
