# PersonalOS UI

The operator UI: Vite, React and TypeScript. Screens, routed by the URL hash:

- `#/approvals` (the default): the approvals inbox (Phase M1), which lists
  `GET /v1/approvals` and records decisions through `POST /v1/approvals/{id}/decision`.
- `#/board`: the application board (Phase M2), a kanban of `GET /v1/applications`
  with state, company and date filters and pages. `#/applications/{id}` is one
  application (`GET /v1/applications/{id}`): fit score, matched and missing
  requirements with their evidence, documents, recruiter events, open waits and
  pending approvals.
- `#/workflows/{id}`: one workflow (Phase M2, M3): its live progress from
  `GET /v1/workflows/{id}/events`, its failures from `GET /v1/workflows/{id}/failures`
  with Retry and Dismiss, and its audit trail from `GET /v1/workflows/{id}/audit`.
- `#/audit`: the audit trail searched (`GET /v1/audit`) by workflow, application,
  actor, action and time range, a page at a time, newest first.
- `#/integrations`: integration health from `GET /v1/integrations` (Phase J checks),
  `healthy`, `degraded` or `down`, with the reason and what fixes it.

```bash
npm ci
npm run dev          # http://localhost:5173, against the API at VITE_API_BASE_URL
npm test             # component tests (Vitest + Testing Library, jsdom)
npm run typecheck
npm run build        # dist/; `npm run preview` serves it with the production CSP
```

Start the API first (`python -m personalos.cli api`), then paste the token from
`python -m personalos.cli auth token` into the page. The token stays in this tab's
`sessionStorage` and is sent only as `Authorization: Bearer`. It never goes in a URL
or a cookie. The API's `API_CORS_ORIGINS` must include the UI's origin
(`http://localhost:5173` by default). To call a different API, set `VITE_API_BASE_URL`
(see `.env.example`).

## The typed API client

The client's types are generated from the API itself, so the UI cannot quietly drift
from it:

1. `python ../scripts/export_openapi.py` (or `npm run api:export`) writes
   `openapi.json` from `apps.api.main.create_app()`.
2. `npm run api:generate` turns it into `src/api/schema.d.ts` with `openapi-typescript`.

`npm run api:check` runs both checks. CI fails if either file is stale. Requests go
through `openapi-fetch` in `src/api/client.ts`.

## Rules the inbox keeps

- **A decision is bound to what is on screen.** `client.decide(approval, approved)`
  takes the displayed approval and sends its `action_hash`.
- **A 409 is never retried.** When the action changed, expired or was decided
  elsewhere, the card locks and asks the reviewer to refresh and review it again.
  Refreshing rebuilds every card from the new listing.
- **Statuses are the API's.** Every item shows its `ActionStatus` verbatim. Approving
  leaves an action `AWAITING_APPROVAL` ("not carried out yet"), and nothing on this
  screen is ever shown as `EXECUTED`.
- **Untrusted text is inert.** Email bodies, summaries and payloads render only as React
  text nodes. `tests/no-html-sinks.test.ts` fails the build if any source file uses an
  HTML-parsing API.

## Rules the board keeps

- **A move is the API's to allow.** Cards move by drag and drop or the "Move to" menu,
  both through `POST /v1/applications/{id}/transition`. A refused move (422) shows the
  API's message and the `allowed_next_states` it returned, and the card stays put.
- **An applied move shows at once.** The card is placed from the `application` the
  API returned, without re-reading the board.
- **Nothing outside PersonalOS happens silently.** A move to `applied`, `withdrawn`,
  `declined` or `accepted` could carry an effect (submitting, telling the recruiter).
  It is not sent until the user picks, explicitly, between having PersonalOS do it,
  which is queued for approval (202) and leaves the card where it is, or recording
  only what already happened. The queued answer links to the approvals inbox and
  the run's progress.

## Following a workflow

`src/progress/feed.ts` reads the event stream with `fetch`, not `EventSource`, so the
bearer token travels in its header and `Last-Event-ID` is set exactly:

- every reconnect resumes after the last event delivered, and events are also
  de-duplicated by id, so nothing is missed or shown twice;
- a dropped stream is reopened after the server's `retry:` delay; after three drops
  in a row with nothing new, it polls `?follow=false` with the same cursor instead,
  and tries a live stream again every so often;
- 401, 404 and 422 (a cursor the API does not know) stop the feed and say so.

An `approval_requested` event shows a prompt linking to the approvals inbox until
the matching `approval_resolved` arrives.

## Rules the audit and failure views keep

- **The audit trail is read-only.** Each action's steps are grouped by `action_hash`
  and shown in the API's order: intent, policy decision (`REQUIRE_APPROVAL`, …),
  approval, execution, then the provider's receipt as its own step. No audit view has
  a control that writes.
- **A denial is denied, not failed.** A `policy_denial` shows as `DENIED` and never
  gets a Retry, even if the response offered one. It can only be dismissed.
- **A provider-permanent failure asks for a reconnect.** Revoked OAuth or a missing
  scope shows how to reconnect and links to integration health, with no Retry, so it
  cannot loop.
- **Retry is the API's to allow, and is sent once.** It is shown only for a `failed`
  step that is `retryable` with a `retry` action. The button is disabled while the
  request is in flight, and the card is updated from the `RetryResponse` (attempt,
  `retry_queued`). A 409 is shown and locks the card until a refresh. It is never
  resent.
- **Credentials are never rendered.** The API redacts these payloads. As a second
  line, `src/secrets.ts` checks every audit, failure and integration response before
  a component sees it. A value under a credential key (`access_token`,
  `Authorization`, …) or a token-shaped string (bearer, `ya29.`, `pos_at_`, PEM, …)
  that is not already `[REDACTED]` is replaced with `[withheld]`. The screen then
  reports an API bug, naming the field but not the value. A Google account's
  `credential_ref` is never shown.
