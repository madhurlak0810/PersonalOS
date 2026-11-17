# PersonalOS UI

The operator UI: Vite, React and TypeScript. Its first screen is the approvals inbox
(Phase M1), which lists `GET /v1/approvals` and records decisions through
`POST /v1/approvals/{id}/decision`.

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
