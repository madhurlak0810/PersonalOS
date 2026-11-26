# Alerts and runbooks

The alert rules are in `deploy/prometheus/rules/personalos.yml`. The compose
stack's `monitoring` profile evaluates them ([METRICS.md](METRICS.md#start-it)).
Each alert below says what it means and the first thing to check. Each rule's
`runbook` annotation links to its section here.

- [How alerts reach you](#how-alerts-reach-you)
- [Sending alerts elsewhere](#sending-alerts-elsewhere)
- Alerts: [WorkflowStalled](#workflowstalled) ·
  [WorkflowFailureRateHigh](#workflowfailureratehigh) ·
  [DeadLettersGrowing](#deadlettersgrowing) ·
  [OutboxBacklogGrowing](#outboxbackloggrowing) ·
  [ApiErrorRateHigh](#apierrorratehigh) ·
  [ModelErrorRateHigh](#modelerrorratehigh) ·
  [IntegrationUnhealthy](#integrationunhealthy) ·
  [OperationalStateMissing](#operationalstatemissing)

## How alerts reach you

Nothing outside the stack is needed:

1. Prometheus evaluates the rules every 15 seconds and sends alerts that fire
   to Alertmanager.
2. Alertmanager sends each one, and later its resolution, to `alert-receiver`
   (`apps/alerts/receiver.py`). The receiver writes one structured log line
   per alert:

   ```bash
   docker compose logs -f alert-receiver
   # {"level": "WARNING", "event": "alert.firing", "alertname": "WorkflowStalled",
   #  "severity": "critical", "workflow_id": "…", "reason": "no_progress",
   #  "summary": "Workflow … is stalled (no_progress) for 16m 0s",
   #  "runbook": "docs/ALERTS.md#workflowstalled", …}
   ```

   Resolutions are logged as `alert.resolved` at INFO. The alerts firing right
   now are also served as JSON at `GET http://alert-receiver:9095/alerts`
   inside the stack's network.
3. Every Grafana dashboard starts with a **Firing alerts** table, read from
   Prometheus. It shows the alert even if the receiver is down.

Alert labels carry only ids and code-defined names (`workflow_id`,
`integration`, `route`, `reason`), never user content. See
[METRICS.md](METRICS.md#labels-and-what-may-never-be-in-them).

## Sending alerts elsewhere

An external channel is optional. To add one, copy
`deploy/alertmanager/alertmanager.yml` to `alertmanager.local.yml` at the
repository root (gitignored) or anywhere outside the repository. Add a
receiver to the copy, and route to it as well as to `local`:

```yaml
route:
  receiver: local
  routes:
    - receiver: local
      continue: true
    - receiver: mail
      matchers: ['severity="critical"']
receivers:
  - name: local
    webhook_configs:
      - url: http://alert-receiver:9095/alerts
        send_resolved: true
  - name: mail
    email_configs:
      - to: you@example.com
        from: personalos@example.com
        smarthost: smtp.example.com:587
        auth_username: personalos@example.com
        auth_password_file: /run/secrets/smtp-password
```

Then start the stack with your file:

```bash
PERSONALOS_ALERTMANAGER_CONFIG=/path/to/alertmanager.yml docker compose --profile monitoring up -d
```

Alertmanager is on the internal network only. A receiver outside the machine
also needs the `egress` network added to the `alertmanager` service in a
`compose.override.yml`. Pass any password as a Docker secret file
(`*_file` settings), never inline. The same rule applies to every secret in
the stack ([CONTAINERS.md](CONTAINERS.md#secrets-in-containers)).

## WorkflowStalled

**What it means.** A workflow has stopped moving. The `reason` label says how:

- `no_progress`: the workflow is unfinished and not waiting on a person.
  Nothing has been recorded for it (no step, no status change, no new
  command) for `WORKFLOW_STALL_SECONDS` (default 15 minutes).
- `approval_overdue`: the workflow is parked at an approval whose expiry passed
  more than `APPROVAL_OVERDUE_GRACE_SECONDS` ago. The worker's expiry sweep
  should have closed it.

The alert names the workflow (`workflow_id`). It clears by itself once the
workflow moves again.

**First thing to check.** Whether a worker is running and claiming work:

```bash
docker compose ps worker                     # running and (healthy)?
docker compose logs --since 30m worker | grep '<workflow_id>'
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/v1/workflows/<workflow_id> | jq
```

On the **Worker and outbox** dashboard, a command queue depth above zero with
no commands settled means the worker is down or stuck. Restart it with
`docker compose restart worker`. A restarted worker reclaims the command and
continues from the last checkpoint. If the worker is healthy and the workflow
still does not move, read its failures
(`GET /v1/workflows/<workflow_id>/failures`) and retry or dismiss them. For
`approval_overdue`, look for `approval_expiry` sweep errors in the worker log.

## WorkflowFailureRateHigh

**What it means.** More than a quarter of one graph's runs (at least 4 in 30
minutes) have ended `failed` for 15 minutes.

**First thing to check.** The failures themselves:
`GET /v1/workflows/{id}/failures` for a recent workflow, or the worker log at
`level=ERROR` with `event=workflow.failed`. If they all name one integration,
see [IntegrationUnhealthy](#integrationunhealthy). If they name a port that is
not configured (`PortNotConfigured`), the deployment is missing settings, not
experiencing an outage.

## DeadLettersGrowing

**What it means.** Outbox events have been parked as `dead_letter` (their retry
budget ran out) or `quarantined` (malformed) within the last 15 minutes. A
parked event is one that some consumer never acted on.

**First thing to check.** Which events, and why:
`GET /v1/failures/events` lists parked events with their redacted last error.
Fix the cause, then requeue each with `POST /v1/failures/events/{failure_id}/retry`.
A quarantined event will fail the same way again, so read its error before
you requeue it.

## OutboxBacklogGrowing

**What it means.** Either more than 20 undelivered outbox events have been
growing for 10 minutes, or the oldest undelivered event is over 15 minutes
old. Events are being written faster than they are delivered, or not
delivered at all.

**First thing to check.** Whether the worker's dispatcher is running. Look in
`docker compose logs worker` for `delivering event … failed` lines, and check
**Outbox deliveries per second by outcome** on the Worker dashboard. If
deliveries are zero, restart the worker. If they are mostly
`retry_scheduled`, the consumer is failing: read the errors and see
[DeadLettersGrowing](#deadlettersgrowing).

## ApiErrorRateHigh

**What it means.** One route has answered more than 5% of its requests (at
least 10 in 10 minutes) with a 5xx for 10 minutes.

**First thing to check.** The API log for that route at `level=ERROR`:
`docker compose logs --since 30m api`. Every line carries the request's
`correlation_id`, which the client also received in `X-Correlation-Id`. A
database error points at Postgres (`docker compose ps postgres`). Anything
else is a bug.

## ModelErrorRateHigh

**What it means.** More than 20% of calls to one model provider (at least 5
in 15 minutes) have errored or returned answers that do not match the schema,
for 15 minutes.

**First thing to check.** Whether the model server is up. `rules` never errors.
For `ollama`, run `docker compose ps ollama` and check that the model in
`LLM_GATEWAY_CONFIG` is pulled. On the Models dashboard, `outcome="error"`
means the provider failed and `schema_invalid` means the model answered in the
wrong shape. The second usually means the model is too small for the
prompt. Evaluate another with `python -m evals.harness --config …`.

## IntegrationUnhealthy

**What it means.** An integration (a job board, Gmail, Google Calendar, the
embedder, a local model server) has been `degraded` or `down` for at least 30
minutes. Work that needs it is held, not failed, until it recovers (Phase J8).

**First thing to check.**
`curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/v1/integrations | jq`.
It gives the reason and what to do. `reconnect` means the credential was
revoked or expired: reconnect the account (`personalos.cli auth google`).
Anything else is the provider being unreachable. Check the network and the
provider's status, and the next scheduled check will clear it.

## OperationalStateMissing

**What it means.** No operational gauges have arrived for 5 minutes. Without
them, [WorkflowStalled](#workflowstalled) cannot fire, so this alert covers
for it. Either the API is down, its metrics are off, or its database sampling
keeps failing.

**First thing to check.** `docker compose ps api`, then
`docker compose logs api | grep 'could not sample operational state'`. Also
check that `METRICS_ENABLED=true` and `METRICS_STATE_INTERVAL_SECONDS` is above
0, and that the collector is up (`docker compose ps otel-collector`).
