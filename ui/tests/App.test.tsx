import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { API, TOKEN, applicationCard, board, emailApproval, fakeApi, json } from "./fixtures";

const unauthenticated = () =>
  json({ error_code: "unauthenticated", message: "not authenticated", context_id: "c" }, 401);

describe("token handling", () => {
  it("asks for the token, checks it with whoami, then shows the inbox", async () => {
    const api = fakeApi({
      "GET /v1/auth/whoami": () => json({ actor_id: "local", method: "local_token" }),
      "GET /v1/approvals": () => json({ approvals: [emailApproval()] }),
    });
    render(<App apiBaseUrl={API} fetch={api.fetch} />);

    await userEvent.type(screen.getByLabelText("API token"), `  Bearer ${TOKEN}  `);
    await userEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(await screen.findByTestId("approval")).toBeVisible();
    expect(screen.getByText("local")).toBeVisible();
    expect(api.requests.map((r) => r.authorization)).toEqual(
      api.requests.map(() => `Bearer ${TOKEN}`),
    );
    expect(window.sessionStorage.getItem("personalos.api-token")).toBe(TOKEN);
    // Never in the URL.
    expect(api.requests.every((r) => !r.url.includes(TOKEN))).toBe(true);
  });

  it("rejects a token the API does not accept, and stores nothing", async () => {
    const api = fakeApi({ "GET /v1/auth/whoami": unauthenticated });
    render(<App apiBaseUrl={API} fetch={api.fetch} />);

    await userEvent.type(screen.getByLabelText("API token"), "pos_at_wrong");
    await userEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("not accepted");
    expect(window.sessionStorage.getItem("personalos.api-token")).toBeNull();
  });

  it("goes back to the token form when a stored token stops working", async () => {
    window.sessionStorage.setItem("personalos.api-token", TOKEN);
    const api = fakeApi({
      "GET /v1/auth/whoami": () => json({ actor_id: "local", method: "local_token" }),
      "GET /v1/approvals": unauthenticated,
    });
    render(<App apiBaseUrl={API} fetch={api.fetch} />);

    // The inbox loaded once (whoami passed), then its listing got the 401.
    expect(await screen.findByText(/rotated/)).toBeVisible();
    expect(screen.getByLabelText("API token")).toBeVisible();
    expect(api.requests.map((r) => r.url)).toEqual(["/v1/auth/whoami", "/v1/approvals"]);
    await waitFor(() => expect(window.sessionStorage.getItem("personalos.api-token")).toBeNull());
  });
});

describe("navigation", () => {
  it("switches between the approvals inbox and the board", async () => {
    window.sessionStorage.setItem("personalos.api-token", TOKEN);
    window.location.hash = "";
    const api = fakeApi({
      "GET /v1/auth/whoami": () => json({ actor_id: "local", method: "local_token" }),
      "GET /v1/approvals": () => json({ approvals: [] }),
      "GET /v1/applications": () => json(board([applicationCard()])),
    });
    render(<App apiBaseUrl={API} fetch={api.fetch} />);

    expect(await screen.findByRole("heading", { name: "Approvals inbox" })).toBeVisible();
    await userEvent.click(screen.getByRole("link", { name: "Applications" }));
    expect(await screen.findByRole("heading", { name: "Applications" })).toBeVisible();
    expect(await screen.findByTestId("board-card")).toHaveTextContent("Acme");
    expect(screen.getByRole("link", { name: "Applications" })).toHaveAttribute("aria-current", "page");
    window.location.hash = "";
  });
});
