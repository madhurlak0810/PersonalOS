# Adding a domain

A domain is a system the Supervisor can hand a request to: today `job`,
`file`, `communications` and `calendar`. Adding one means a misroute can now
take a real action on a new system, so the steps below are mostly about making
sure that can't happen silently. Each step names the check that fails if you
skip it.

The running example is a hypothetical `notes` domain: a subgraph over the
person's note-taking app, which can read notes and propose `append_note`.

## 1. The routing label

| Where | What |
| --- | --- |
| `personalos/domain/routing.py` | Add `RouteDomain.NOTES`. Add it to `SUPPORTED_ROUTE_DOMAINS` only once steps 2-5 are done: until then it's schema-valid but `resolve_route` reports it `unsupported`. |
| `personalos/models/routing.py` | `_DOMAIN_DESCRIPTIONS[NOTES]`: one line for the LLM classifier's prompt. `_DOMAIN_KEYWORDS[NOTES]`: the keyword fallback's phrases, with `_WEAK` weight for words that name a topic rather than the system ("idea" is weak, "notes app" is not). |
| `personalos/models/task_planning.py` | `clause_domain` patterns, `_STEP_NAMES` and `DOMAIN_OUTPUTS`, so the rule-based planner can make a cross-domain plan with a notes step. |
| `personalos/graphs/supervisor.py` | `_DOMAIN_NAMES[NOTES]`: what a clarification calls it ("your notes"). |

The classifier's structured output already carries `secondary_domains`,
`ambiguous` and `access_escalation`, so the new domain gets multi-domain and
ambiguity handling without a schema change. Requests that ask to skip approval
or widen access are caught by `screen_request` before any domain sees them.
That screen is deterministic and doesn't depend on the classifier.

*Fails if skipped:* `tests/unit/test_routing_classifier.py::test_every_route_domain_has_keywords`,
and `test_every_route_domain_is_supported_in_this_build` once you add it to
the supported set.

## 2. The subgraph and its registration

1. Write the graph in `personalos/graphs/notes.py` with a `GRAPH_NAME` and
   every node wrapped in `traced_node(name, method, graph=GRAPH_NAME)`. Every
   write is proposed as an `ActionIntent` and waits at an approval interrupt,
   the same way the Calendar and Communications subgraphs do it.
2. Give every node a `NodeContract` in `personalos/graphs/contracts.py` and a
   test class marked `@covers_node(NOTES_GRAPH, node)`.
3. In the Supervisor, add a `NotesSubgraphRunner` port, a
   `RUN_NOTES_SUBGRAPH` node, a fixed `_build_notes_dag`, and the branches in
   `_plan_work` and `_route_after_planning`. Add the optional subgraph to the
   `optional` map in `__init__` so that a Supervisor built without it asks back
   instead of routing.
4. Wire it in `apps/worker/graphs.py`: the `notes_subgraph=` argument, its
   entry in `step_graphs` so plans can include it, and a thread-id recipe in
   `personalos/graphs/plan_steps.py::STEP_THREAD_IDS`.
5. In `evals/world.py`, add an echo subgraph to `SupervisorWorld` (and to
   `SupervisorWorld.route`) and a `notes(gateway, case)` builder in
   `DOMAIN_WORLDS` that takes its model-backed ports from the gateway.

*Fails if skipped:* `tests/contracts/test_node_contract_coverage.py` (the
"Node contract gate" step), and the routing suite reports the domain's cases
as `clarify`.

## 3. Policy

Nothing a new domain can do is reachable until policy says so. In order:

1. **Tools.** Add each tool's argument surface to a `*_TOOL_ARGUMENTS` table in
   `personalos/policy/rules.py` (that is the allowlist), and give each one a
   permission class and scopes in `personalos/policy/permissions.py`. Reads
   are `READ_*`. A write another person can see is `WRITE_EXTERNAL`, and one
   that destroys data is `DESTRUCTIVE`.
2. **Who may call them.** Add a `NOTES_REQUESTER_PREFIX`, a
   `NOTES_NODE_TOOL_ALLOWLISTS` mapping (each node gets only the tools it
   needs; the node that reads note content gets no write), and a
   `RequesterToolScopeRule` for it in `default_rules()`.
3. **The action kind.** Add `ActionKind.APPEND_NOTE` in
   `personalos/domain/job_search.py` with its `ACTION_RISK_PROFILES` entry,
   then add its row to `personalos/policy/actions.py::ACTION_POLICIES`. That
   row says whether `ToolExecutor` carries it out (`EXECUTOR`, evaluated as its
   first probe's tool) or its own subgraph does (`SUBGRAPH`), and lists probes:
   the riskiest calls the action makes, with the requester that makes them.
   `ACTION_TOOLS` in the executor is derived from this table.
4. **The label.** Add `append_note` to `action_kinds` in
   `evals/golden/approval_boundaries.json` with its `consequential`,
   `external_write` and `domain` labels. Then add policy cases for the new
   tools there: a read that is allowed, a write that is held, and a write
   attempted by a node that must not make it.

*Fails if skipped:* `tests/contracts/test_action_policy_coverage.py` (the
"Action policy gate" step). It fails when a kind has no row, when a probe's
tool is unclassified or not allowlisted, or when policy would let any probe
run without approval. It also fails when the eval labels are missing.

## 4. Golden cases

All synthetic: invented names, `*.example` addresses, no real notes, mail,
documents or calendars.

1. **Routing.** Create `evals/golden/routing/notes.json` with
   `"domain": "notes"` and add it to `evals/dataset.py::ROUTING_FILES`. It
   needs at least one case of each `REQUIRED_ROUTING_KINDS`, which are clear,
   ambiguous, cross-domain and adversarial. Add out-of-scope and negative
   cases too. Each case's `expected` has the route a careful reader would
   take; `never_routes` lists the routes it must not reach. Adversarial cases
   (asking to skip approval, act unattended or widen access) expect `clarify`
   with reason `access_escalation`, and must never reach any write route.
2. **Workflows.** Add end-to-end cases to
   `evals/golden/domain_workflows.json`: a read that proposes nothing, and a
   write in three variants: approved, rejected and pending. Add one more with
   `fault` set, which must recover by resuming and must not duplicate its
   effect.
3. **Metrics.** Add `"notes"` to `evals/metrics.py::DOMAINS`, and its nodes to
   `STEP_NODES` under an interactive step or a long-running one.

Write the outcome a careful person expects, not the one the system gives
today. A new case that fails is a finding, not a broken test.

*Fails if skipped:* `tests/unit/test_routing_golden.py` (each domain has every
required kind; adversarial cases forbid every write route), and the floors in
step 5. A critical floor with no data counts as a breach.

## 5. Floors and the baseline

Add the domain's floors to `evals/thresholds.json`:

- `routing_accuracy_notes`: `min`, critical. A misroute is a real action on
  the wrong system.
- `approval_recall_notes`: `min: 1.0`, critical.
- `external_write_approval_recall_notes`: `min: 1.0`, critical, if any of its
  actions reach someone else.
- `approval_precision_notes`, `tool_selection_accuracy_notes`,
  `workflow_completion_rate_notes`: warnings.

Thresholds live only in that file. Tests and CI read them from there, and
`--thresholds` points a run at another file (for example, to benchmark a weak
local model without failing on targets it is expected to miss).

Then rerun the harness and accept the result:

```
python -m evals.harness                    # read the per-domain table and the confusion matrix
python -m evals.harness --update-baseline  # once the numbers are what you expect
```

Do this for every config whose baseline you keep. CI runs `rules`; the
`Quality metrics gate` step fails on a baseline regression, and the
`Per-domain floors` step fails on any critical floor.

## The checks, in one place

| If you skip... | ...this fails |
| --- | --- |
| keywords for the new `RouteDomain` | `test_routing_classifier.py::test_every_route_domain_has_keywords` |
| a node contract or contract test | Node contract gate |
| a policy row, a permission class, an allowlist entry, or a rule that holds the write | Action policy gate |
| an eval label for the action kind | Action policy gate (`test_every_action_kind_is_labelled_for_the_approval_evals`) |
| a required kind of routing case | `test_routing_golden.py` |
| routing or approval quality in the domain | Per-domain floors (`evals/thresholds.json`) |
| re-baselining after changing golden data | `test_eval_harness.py::test_the_rules_run_matches_the_checked_in_baseline` |
