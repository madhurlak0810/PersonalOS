import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createApiClient } from "../src/api/client";
import { IntegrationsHealth } from "../src/components/IntegrationsHealth";
import { API, TOKEN, fakeApi, integrations, json } from "./fixtures";

function renderHealth(fetch: (request: Request) => Promise<Response>) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(<IntegrationsHealth client={client} />);
}

describe("integration health", () => {
  it("shows each integration's health, including degraded and down", async () => {
    const api = fakeApi({ "GET /v1/integrations": () => json({ integrations: integrations() }) });
    const { container } = renderHealth(api.fetch);

    const rows = await screen.findAllByTestId("integration");
    expect(rows.map((row) => [row.dataset.provider, row.dataset.health])).toEqual([
      ["google", "unknown"],
      ["gmail", "degraded"],
      ["greenhouse", "down"],
      ["ollama", "healthy"],
    ]);

    const gmail = rows[1]!;
    expect(gmail).toHaveTextContent("Reconnect required");
    expect(gmail).toHaveTextContent("the refresh token was revoked");
    expect(within(gmail).getByTestId("reconnect-prompt")).toHaveTextContent(
      "python -m personalos.cli auth google",
    );
    expect(rows[2]).toHaveTextContent("next health check will try again");
    expect(rows[0]).toHaveTextContent("me@example.com — active");
    // A credential reference is not a secret, but it is not shown either.
    expect(container.textContent).not.toContain("cred://");
  });

  it("says health is unavailable when the API has none", async () => {
    const api = fakeApi({
      "GET /v1/integrations": () =>
        json({ error_code: "port_not_configured", message: "not configured", context_id: "c" }, 503),
    });
    renderHealth(api.fetch);
    expect(await screen.findByTestId("integrations-unavailable")).toHaveTextContent("not available");
  });

  it("withholds a token that came back in a last error", async () => {
    const list = integrations();
    list[1] = { ...list[1]!, last_error: "rejected Authorization: Bearer pos_at_abcdefghijklmnopqrstuvwxyz0123" };
    const api = fakeApi({ "GET /v1/integrations": () => json({ integrations: list }) });
    const { container } = renderHealth(api.fetch);

    await screen.findAllByTestId("integration");
    expect(container.textContent).not.toContain("pos_at_");
    expect(screen.getByTestId("withheld-notice")).toBeVisible();
  });
});
