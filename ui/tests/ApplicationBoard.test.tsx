/**
 * The application board against a fake API. The acceptance criterion:
 *
 * - moving a card to an invalid state surfaces the API's allowed-states
 *   message and leaves the card in place.
 *
 * And the rest of the board's contract: columns by state, filters and pages
 * sent as query parameters, an applied move reflected at once, and an
 * effectful move never sent without an explicit choice.
 */
import { createEvent, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createApiClient } from "../src/api/client";
import { ApplicationBoard } from "../src/components/ApplicationBoard";
import {
  API,
  TOKEN,
  WORKFLOW_ID,
  applicationCard,
  applied,
  board,
  fakeApi,
  invalidTransition,
  json,
  queued,
  type Recorded,
} from "./fixtures";

function renderBoard(fetch: (request: Request) => Promise<Response>) {
  const client = createApiClient({ baseUrl: API, token: TOKEN, fetch });
  return render(<ApplicationBoard client={client} onUnauthenticated={() => {}} />);
}

const column = (status: string) =>
  screen.getAllByTestId("board-column").find((el) => el.dataset.status === status)!;

const cardIn = (status: string, id: string) =>
  within(column(status))
    .queryAllByTestId("board-card")
    .find((el) => el.dataset.applicationId === id) ?? null;

const transitions = (requests: Recorded[]) =>
  requests.filter((r) => r.method === "POST" && r.url.endsWith("/transition"));
const listings = (requests: Recorded[]) =>
  requests.filter((r) => r.method === "GET" && r.url === "/v1/applications");

const acme = applicationCard();
const globex = applicationCard({
  application_id: "0a000000-0000-4000-8000-000000000002",
  company: "Globex",
  title: "Data Engineer",
  status: "interviewing",
});

describe("the board", () => {
  it("groups applications into lifecycle columns, in order", async () => {
    const api = fakeApi({ "GET /v1/applications": () => json(board([acme, globex])) });
    renderBoard(api.fetch);

    await screen.findAllByTestId("board-card");
    const statuses = screen.getAllByTestId("board-column").map((el) => el.dataset.status);
    expect(statuses.slice(0, 8)).toEqual([
      "discovered",
      "saved",
      "preparing",
      "ready_to_apply",
      "applied",
      "response",
      "interviewing",
      "offer",
    ]);
    expect(statuses).toEqual(expect.arrayContaining(["rejected", "withdrawn", "skipped", "stalled"]));
    expect(cardIn("discovered", acme.application_id)).toHaveTextContent("Acme");
    expect(cardIn("interviewing", globex.application_id)).toHaveTextContent("Globex");
    expect(within(column("discovered")).getByTestId("column-total")).toHaveTextContent("1");
    // Each card links to its detail view.
    expect(
      within(cardIn("discovered", acme.application_id)!).getByRole("link", { name: /Acme/ }),
    ).toHaveAttribute("href", `#/applications/${acme.application_id}`);
    expect(api.requests[0]?.authorization).toBe(`Bearer ${TOKEN}`);
  });

  it("sends the filters and page as query parameters", async () => {
    const api = fakeApi({
      "GET /v1/applications": (_body, request) => {
        const offset = Number(new URL(request.url).searchParams.get("offset") ?? 0);
        return json(board([acme], { total: 120, offset, next_offset: offset + 50 }));
      },
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.click(screen.getByLabelText("Interviewing"));
    await userEvent.click(screen.getByLabelText("Saved"));
    await userEvent.type(screen.getByLabelText("Company"), "Acme");
    fireEvent.change(screen.getByLabelText("Created from"), { target: { value: "2026-11-01" } });
    fireEvent.change(screen.getByLabelText("Created to"), { target: { value: "2026-11-30" } });
    await userEvent.click(screen.getByRole("button", { name: "Apply filters" }));

    await waitFor(() => expect(listings(api.requests)).toHaveLength(2));
    const params = new URLSearchParams(listings(api.requests)[1]!.search);
    // In lifecycle order, whatever order they were ticked in.
    expect(params.getAll("state")).toEqual(["saved", "interviewing"]);
    expect(params.get("company")).toBe("Acme");
    expect(params.get("created_from")).toBe("2026-11-01T00:00:00");
    expect(params.get("created_to")).toBe("2026-11-30T23:59:59.999999");
    expect(params.get("offset")).toBe("0");

    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => expect(listings(api.requests)).toHaveLength(3));
    const next = new URLSearchParams(listings(api.requests)[2]!.search);
    expect(next.get("offset")).toBe("50");
    expect(next.getAll("state")).toEqual(["saved", "interviewing"]);
    expect(await screen.findByTestId("board-range")).toHaveTextContent("Showing 51–51 of 120");
  });
});

describe("moving a card", () => {
  it("to an invalid state shows the API's allowed states and leaves the card in place", async () => {
    const message = "cannot transition application from 'discovered' to 'offer'";
    const api = fakeApi({
      "GET /v1/applications": () => json(board([acme])),
      [`POST /v1/applications/${acme.application_id}/transition`]: () =>
        invalidTransition(message, ["saved", "skipped"]),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.selectOptions(
      screen.getByLabelText("Move Acme — Backend Engineer to"),
      "offer",
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(message);
    expect(within(alert).getByTestId("allowed-states")).toHaveTextContent(
      "Allowed next states: Saved, Skipped",
    );
    // The card did not move: still in Discovered, not in Offer.
    expect(cardIn("discovered", acme.application_id)).not.toBeNull();
    expect(cardIn("offer", acme.application_id)).toBeNull();
    expect(within(column("discovered")).getByTestId("column-total")).toHaveTextContent("1");
    expect(within(column("offer")).getByTestId("column-total")).toHaveTextContent("0");
    // The move was asked for once, as is, and not retried.
    expect(transitions(api.requests).map((r) => r.body)).toEqual([{ to: "offer" }]);
  });

  it("offers the allowed states as the next move", async () => {
    const api = fakeApi({
      "GET /v1/applications": () => json(board([acme])),
      [`POST /v1/applications/${acme.application_id}/transition`]: (body) =>
        (body as { to: string }).to === "saved"
          ? json(applied(acme, "saved"))
          : invalidTransition("not allowed", ["saved", "skipped"]),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.selectOptions(screen.getByLabelText("Move Acme — Backend Engineer to"), "offer");
    await userEvent.click(await screen.findByRole("button", { name: "Saved" }));

    await waitFor(() => expect(cardIn("saved", acme.application_id)).not.toBeNull());
    expect(transitions(api.requests).map((r) => r.body)).toEqual([{ to: "offer" }, { to: "saved" }]);
  });

  it("shows an applied move at once, from the card the API returned", async () => {
    const api = fakeApi({
      "GET /v1/applications": () => json(board([acme])),
      [`POST /v1/applications/${acme.application_id}/transition`]: () => json(applied(acme, "saved")),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.selectOptions(screen.getByLabelText("Move Acme — Backend Engineer to"), "saved");

    await waitFor(() => expect(cardIn("saved", acme.application_id)).not.toBeNull());
    expect(cardIn("discovered", acme.application_id)).toBeNull();
    expect(within(column("saved")).getByTestId("column-total")).toHaveTextContent("1");
    expect(within(column("discovered")).getByTestId("column-total")).toHaveTextContent("0");
    expect(screen.getByTestId("move-feedback")).toHaveTextContent("Moved to Saved");
    // No re-read was needed to show it.
    expect(listings(api.requests)).toHaveLength(1);
  });

  it("by drag and drop sends the same move", async () => {
    const api = fakeApi({
      "GET /v1/applications": () => json(board([acme])),
      [`POST /v1/applications/${acme.application_id}/transition`]: () => json(applied(acme, "saved")),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    const data = new Map<string, string>();
    const dataTransfer = {
      setData: (type: string, value: string) => data.set(type, value),
      getData: (type: string) => data.get(type) ?? "",
      effectAllowed: "",
      dropEffect: "",
    };
    const card = cardIn("discovered", acme.application_id)!;
    fireEvent(card, Object.assign(createEvent.dragStart(card), { dataTransfer }));
    const target = column("saved");
    fireEvent(target, Object.assign(createEvent.dragOver(target), { dataTransfer }));
    fireEvent(target, Object.assign(createEvent.drop(target), { dataTransfer }));

    await waitFor(() => expect(cardIn("saved", acme.application_id)).not.toBeNull());
    expect(transitions(api.requests).map((r) => r.body)).toEqual([{ to: "saved" }]);
  });

  it("with a possible external effect asks first, and an approved-only effect leaves the card", async () => {
    const ready = applicationCard({ status: "ready_to_apply" });
    const api = fakeApi({
      "GET /v1/applications": () => json(board([ready])),
      [`POST /v1/applications/${ready.application_id}/transition`]: () =>
        json(queued(ready, "applied"), 202),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.selectOptions(screen.getByLabelText("Move Acme — Backend Engineer to"), "applied");

    // Nothing is sent until a choice is made, and nothing is pre-selected.
    const prompt = await screen.findByTestId("effect-prompt");
    expect(transitions(api.requests)).toHaveLength(0);
    const confirm = within(prompt).getByRole("button", { name: /Move|Queue/ });
    expect(confirm).toBeDisabled();

    await userEvent.click(within(prompt).getByLabelText(/Submit it through PersonalOS/));
    await userEvent.click(within(prompt).getByRole("button", { name: "Queue for approval" }));

    const feedback = await screen.findByTestId("move-feedback");
    expect(feedback).toHaveTextContent("Queued for approval: submit the application");
    expect(within(feedback).getByRole("link", { name: "Open the approvals inbox" })).toHaveAttribute(
      "href",
      "#/approvals",
    );
    expect(within(feedback).getByRole("link", { name: "Follow its progress" })).toHaveAttribute(
      "href",
      `#/workflows/${WORKFLOW_ID}`,
    );
    // Not applied: the card stays in Ready to apply until the approval has run.
    expect(cardIn("ready_to_apply", ready.application_id)).not.toBeNull();
    expect(cardIn("applied", ready.application_id)).toBeNull();
    expect(transitions(api.requests).map((r) => r.body)).toEqual([
      { to: "applied", reason: null, effect: "submit_application" },
    ]);
  });

  it("that only records what already happened is sent without an effect", async () => {
    const offer = applicationCard({ status: "offer" });
    const api = fakeApi({
      "GET /v1/applications": () => json(board([offer])),
      [`POST /v1/applications/${offer.application_id}/transition`]: () =>
        json(applied(offer, "declined")),
    });
    renderBoard(api.fetch);
    await screen.findAllByTestId("board-card");

    await userEvent.selectOptions(screen.getByLabelText("Move Acme — Backend Engineer to"), "declined");
    const prompt = await screen.findByTestId("effect-prompt");
    await userEvent.click(within(prompt).getByLabelText(/Only record it/));
    await userEvent.type(within(prompt).getByLabelText("Reason (optional)"), "Took another offer");
    await userEvent.click(within(prompt).getByRole("button", { name: "Move" }));

    await waitFor(() => expect(cardIn("declined", offer.application_id)).not.toBeNull());
    expect(transitions(api.requests).map((r) => r.body)).toEqual([
      { to: "declined", reason: "Took another offer" },
    ]);
  });
});
