# Metrics and dashboards

PersonalOS records OpenTelemetry metrics in the API and the worker and pushes
them to the collector, which exposes them to Prometheus. The compose stack's
`monitoring` profile runs Prometheus, Alertmanager and Grafana with dashboards
and alert rules kept as code in `deploy/`. Alerts and their runbooks are in
[ALERTS.md](ALERTS.md).

- [Start it](#start-it)
- [How metrics flow](#how-metrics-flow)
- [The catalog](#the-catalog)
- [Labels and what may never be in them](#labels-and-what-may-never-be-in-them)
- [Stalled workflows](#stalled-workflows)
- [Quick and long-running steps](#quick-and-long-running-steps)
- [Dashboards](#dashboards)
- [Adding, renaming or removing a metric](#adding-renaming-or-removing-a-metric)
- [What CI checks](#what-ci-checks)

## Start it

```bash
docker compose --profile monitoring up -d --build
open http://127.0.0.1:3000        # Grafana: the PersonalOS folder
docker compose logs -f alert-receiver   # one line per firing or resolved alert
```

Grafana is published on `127.0.0.1` only (`PERSONALOS_GRAFANA_PORT`, default
3000). Anyone on this machine can view it. There is no login form and no admin
over HTTP, and the dashboards are read-only. Prometheus, Alertmanager and the
alert receiver have no host port. Outside the compose stack, set
`METRICS_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT` to an OTLP/HTTP
collector.

## How metrics flow

```
API, worker ──OTLP/HTTP every 15 s──▶ otel-collector ──:8889──▶ Prometheus ──▶ Grafana
                                                                    │
                                                        rules ──▶ Alertmanager ──▶ alert-receiver (log)
```

- **Counters and histograms** are recorded where things happen: the request
  middleware (`apps/api/metrics.py`), the workflow runner, every graph node
  (`tracing.node_span`), the command loop, the outbox dispatcher, the approval
  queue, every model call (`tracing.model_span`), and integration health.
- **Gauges of operational state** are sampled from the database by the API
  every `METRICS_STATE_INTERVAL_SECONDS` (`personalos/observability/
  state_metrics.py`). These are queue depth, outbox backlog, dead letters,
  active and stalled workflows, and stored integration health. The API takes
  these samples, not the worker, so a stalled-workflow alert still fires when
  the worker is the thing that stopped.
- The collector names each series for Prometheus. Dots become underscores, a
  unit of seconds appends `_seconds`, and counters get `_total`. Each series
  carries `job` (`personalos-api` or `personalos-worker`) and `instance`.

## The catalog

This table is the metric contract. It is the prose form of
`personalos.observability.metrics.CATALOG`, and
`tests/contracts/test_metrics_contract.py` fails when the two differ in any
row. Dashboards and alert rules query the **Prometheus** name. For histograms
they query its `_bucket`, `_sum` and `_count` series.

<!-- metrics-catalog:start -->
| Prometheus | OpenTelemetry | Kind | Unit | Labels | Emitted by | Meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `personalos_api_request_duration_seconds` | `personalos.api.request.duration` | histogram | s | `route`, `method`, `status_class` | api | Time to answer one API request, by route template, method and status class |
| `personalos_api_limit_hits_total` | `personalos.api.limit_hits` | counter | — | `route`, `limit` | api | Requests refused by an API limit (Phase L2), by route and limit |
| `personalos_workflow_runs_total` | `personalos.workflow.runs` | counter | — | `graph`, `outcome` | worker | Graph invocations finished, by graph and outcome (completed, paused, failed) |
| `personalos_workflow_run_duration_seconds` | `personalos.workflow.run.duration` | histogram | s | `graph`, `outcome` | worker | Time one graph invocation ran, from claim to completion, pause or failure |
| `personalos_workflow_step_duration_seconds` | `personalos.workflow.step.duration` | histogram | s | `graph`, `step`, `step_class`, `outcome` | worker | Time one step ran, by graph, step, step class (quick, long_running) and outcome |
| `personalos_approval_wait_duration_seconds` | `personalos.approval.wait.duration` | histogram | s | `outcome`, `action_kind` | any | Time from an approval being requested to it being decided or expired |
| `personalos_worker_commands_total` | `personalos.worker.commands` | counter | — | `kind`, `outcome` | worker | Workflow commands the worker settled, by kind and outcome (done, failed, held, lease_busy) |
| `personalos_worker_retries_total` | `personalos.worker.retries` | counter | — | `source` | worker | Work tried again, by source (command_redelivery, operator_retry, outbox_delivery, tool_dispatch) |
| `personalos_outbox_deliveries_total` | `personalos.outbox.deliveries` | counter | — | `event_type`, `outcome` | worker | Outbox event deliveries, by event type and outcome (acked, retry_scheduled, dead_lettered, quarantined, claim_lost) |
| `personalos_model_request_duration_seconds` | `personalos.model.request.duration` | histogram | s | `provider`, `operation`, `outcome` | any | Time one model call attempt took, by provider, operation and outcome |
| `personalos_integration_health_checks_total` | `personalos.integration.health.checks` | counter | — | `integration`, `status`, `source` | any | Integration health outcomes recorded, by integration, status and source |
| `personalos_integration_health_transitions_total` | `personalos.integration.health.transitions` | counter | — | `integration`, `from_status`, `status` | any | Integration health state changes |
| `personalos_integration_health_status` | `personalos.integration.health.status` | gauge | — | `integration` | any | Integration health: 0 healthy, 1 degraded, 2 down |
| `personalos_worker_queue_depth` | `personalos.worker.queue.depth` | gauge | — | `state` | state | Workflow commands not yet settled, by state (pending, in_progress) |
| `personalos_outbox_backlog` | `personalos.outbox.backlog` | gauge | — | `state` | state | Outbox events (other than workflow commands) not yet delivered, by state |
| `personalos_outbox_oldest_pending_age_seconds` | `personalos.outbox.oldest_pending.age` | gauge | s | — | state | Age of the oldest undelivered outbox event; 0 when there is none |
| `personalos_outbox_parked` | `personalos.outbox.parked` | gauge | — | `status` | state | Outbox events parked for an operator, by status (dead_letter, quarantined) |
| `personalos_workflows_active` | `personalos.workflows.active` | gauge | — | — | state | Workflows with a thread running or a command queued |
| `personalos_workflows_stalled` | `personalos.workflows.stalled` | gauge | — | `reason` | state | Stalled workflows, by reason (no_progress, approval_overdue); 0 when none |
| `personalos_workflow_stalled_for_seconds` | `personalos.workflow.stalled_for` | gauge | s | `workflow_id`, `reason` | state | How long each stalled workflow has gone without progress (only stalled ones) |
<!-- metrics-catalog:end -->

"Emitted by": `api` and `worker` are the process. `state` means the API
samples it from the database. `any` means either process can record it, and
the dashboards aggregate across `job`.

Label values:

| Label | Values |
| --- | --- |
| `route` | the route template, e.g. `/v1/workflows/{workflow_id}`; `{unmatched}` for a request no route matched |
| `status_class` | `2xx`, `3xx`, `4xx`, `5xx` |
| `graph` | `supervisor`, `job_search`, `communications`, `calendar`, `file_task` |
| `outcome` (runs) | `completed`, `paused` (stopped at an approval), `failed` |
| `outcome` (steps) | `completed`, `interrupted`, `failed` |
| `outcome` (approvals) | `APPROVED`, `DENIED_BY_USER`, `EXPIRED` |
| `outcome` (models) | `ok`, `schema_invalid`, `error` |
| `step_class` | `quick`, `long_running` ([below](#quick-and-long-running-steps)) |
| `provider` | `rules`, `anthropic`, `ollama`, `openai_compatible`, `unknown` |
| `reason` | `no_progress`, `approval_overdue` |
| `state` | `pending`, `in_progress` |
| `status` (outbox) | `dead_letter`, `quarantined` |
| `status` (health) | `healthy`, `degraded`, `down` |

Counters of rare events start at zero when the worker starts, so a rate over
them is 0 rather than no data. These are each graph's run outcomes, each
command kind's outcomes, each retry source, and the delivery outcomes (under
`event_type="any"`).

## Labels and what may never be in them

A label is an id or a name defined in code. It is never something a user, a
recruiter or a model wrote. Every value passes through
`metrics.clean_labels` before it reaches an instrument:

- only the labels the catalog declares for that metric are kept;
- `workflow_id` must be a UUID, or it is recorded as `invalid`;
- every other value must be a short identifier: letters, digits, spaces and
  `_ . : / { } -`, at most 128 characters. The Phase D redactor must also find
  no secret in it. Otherwise it is recorded as `other`. An email address
  (`@`), a message body (commas, line breaks) or a token is therefore never
  recorded as itself.

The SDK views (`metrics_setup.catalog_views`) apply the label allow-list a
second time inside the SDK. Spans and log lines are redacted on their way out
as before (`trace_redaction`, `log_redaction`).
`tests/adversarial/test_telemetry_redaction.py` seeds a fake secret and a fake
email body through every path and checks that neither appears in a metric
label, a log line or a span attribute.

## Stalled workflows

`personalos/persistence/operational_state.py` decides this on every sample.

A workflow is **active** while one of its threads is `pending` or `running`,
or a command for it is queued or claimed. An active workflow is **stalled**
in either of these cases:

- **`no_progress`**: it is not parked at an approval and not waiting on a
  follow-up that has not come due, and the newest of its progress events,
  thread status changes and queued commands is older than
  `WORKFLOW_STALL_SECONDS` (default 900). A command queued while no worker is
  running ends up here.
- **`approval_overdue`**: it is parked at an approval, nobody has queued an
  answer, and the earliest expiry among the requests passed more than
  `APPROVAL_OVERDUE_GRACE_SECONDS` (default 300) ago. The worker's expiry
  sweep should have closed it by then.

Waiting for a person to answer an approval before it expires is not a stall.
A stalled workflow that moves again drops out of the next sample, and the
alert clears.

## Quick and long-running steps

Step latency is charted in two panels, because a quick step that slows down
is a different problem from a slow one. `LONG_RUNNING_STEPS` in
`personalos/observability/metrics.py` lists, by `(graph, node)`, the steps
that wait on a job board, a mailbox, a calendar, a model or a whole subgraph.
Every other step is `quick`. The contract test checks that every listed step
is a registered node, so renaming a node without updating the list fails CI.

## Dashboards

`deploy/grafana/dashboards/*.json` is generated by
`scripts/build_dashboards.py`, where each panel is a title and its PromQL.
Grafana loads the files read-only into a **PersonalOS** folder:

| Dashboard | Panels |
| --- | --- |
| API | requests per route, p95 latency per route, 5xx share per route, responses by status class, API limit hits (Phase L2) |
| Workflows | active and stalled workflows, throughput by graph and outcome, completion rate, run duration, approval wait, step latency for quick and long-running steps |
| Worker and outbox | command queue depth, commands settled, retries by source, dead letters and quarantined events, outbox backlog, oldest pending event, deliveries by outcome |
| Models and integrations | model latency, error rate and calls per provider; integration health (Phase J8), check outcomes and transitions |

Every dashboard starts with a **Firing alerts** table.

To change a dashboard, edit `scripts/build_dashboards.py`, then run
`python scripts/build_dashboards.py` and commit the JSON it writes.

To check that every panel has data on a running stack:

```bash
python scripts/check_dashboards.py --grafana http://127.0.0.1:3000
```

## Adding, renaming or removing a metric

1. Change the `MetricSpec` in `personalos/observability/metrics.py`, and the
   call sites that record it.
2. Change its row in [the catalog](#the-catalog) above.
3. Change every panel in `scripts/build_dashboards.py` and every rule in
   `deploy/prometheus/rules/` that queries it, then regenerate the
   dashboards.

If you miss a step, `tests/contracts/test_metrics_contract.py` fails. A
renamed metric then breaks CI instead of leaving a dashboard empty.

## What CI checks

- `tests/contracts/test_metrics_contract.py`: the catalog and this document
  agree, every series a dashboard or rule queries is in the catalog with the
  labels it uses, every catalog metric is on a dashboard, every alert has a
  runbook in ALERTS.md, and every long-running step exists.
- `tests/contracts/test_metrics_emission.py`: a fixture workflow, run through
  the API and a worker in-process, records the metrics each panel needs.
- `tests/adversarial/test_telemetry_redaction.py`: no seeded secret or email
  body reaches a metric label, a log line or a span attribute.
- `scripts/build_dashboards.py --check`: the committed JSON is current.
- `promtool test rules deploy/prometheus/tests/*.test.yml`: each alert fires
  on the series it is meant for. The stalled-workflow alert also clears.
- `scripts/smoke_observability.py`, in the `containers` job, brings up the
  stack with `--profile monitoring` and runs the quick start's workflow. It
  then queries every panel through Grafana. It also stops the worker, queues
  a workflow and waits for `WorkflowStalled` to fire and reach the alert
  receiver. Last, it restarts the worker and waits for the alert to clear.
