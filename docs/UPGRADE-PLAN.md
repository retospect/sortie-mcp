# sortie-mcp — Upgrade Plan

> Consolidated roadmap for the next three releases of `sortie-mcp`
> (currently `0.1.13`). Covers multi-runner safety, fair-share scheduling,
> Feynman-inspired role/workflow primitives, and MCP token-economy fixes.

*Status: drafted 2026-04-18. Owner: @bots.*
*Related: `docs/sortie-spec.md` (git-native mode, separate package `sortie`).*

---

## 1. Goals

1. **Multi-runner safety**: two `sortie-runner` processes on different hosts
   must be able to share a DB without double-dispatching a step or
   silently resurrecting a zombie.
2. **Concurrent editing of shared resources**: N agents can edit different
   paragraphs of `ch01.tex` simultaneously; two agents cannot edit the
   same paragraph; a full-chapter reformat waits for all paragraph locks
   to drain.
3. **Fair compute distribution**: with 4 active campaigns (A, B, C, D) and
   4 runner slots, each campaign gets ~1 slot on average. Today's tick
   re-rolls the allocation each cron run and can starve long-lived
   campaigns against new ones.
4. **Bounded token cost per step**: `get_my_context` should cost single-digit
   thousands of tokens, not 10–30k.
5. **Typed role outputs**: researcher/writer/reviewer/verifier each have a
   fixed output shape, making composition (workflow templates) tractable.
6. **Workflow templates over free-form planning**: five canonical DAG
   shapes the planner *instantiates* rather than re-inventing.

## 2. Non-goals (explicit)

- Rewriting `sortie` (the git-native package) — that stays as-is.
- A web dashboard. `sortie ls` + `sortie status` + MCP tools remain the
  surface.
- Replacing the planner LLM with a rule engine. Planner is still an LLM;
  we give it better primitives and narrower scope.
- Multi-tenant ACLs. Campaign `user_id` stays advisory; real auth lives
  at the MCP transport.

## 3. Current state (one page)

- **DB-backed** (`asyncpg` + `pgvector`), single schema (default `sortie`),
  tables: `campaigns`, `campaign_steps`, `campaign_notes`,
  `notifications`. Migration idempotent on boot.
- **Step claim**: atomic `UPDATE WHERE status='pending' RETURNING *` — safe
  vs. races but no owner/heartbeat. Recovery via
  `reset_zombies(timeout_minutes)`.
- **Campaign scan** uses `FOR UPDATE SKIP LOCKED` + priority ordering.
- **Scheduler** in `Runner.tick()` allocates slots by priority tier
  (urgent=1.0, high=0.75, normal=0.5, low=0.25, bg=0.25) then round-robin
  inside each tier. No cross-tick memory; no per-campaign compute
  accounting.
- **Agent MCP surface**: 18 tools in one namespace (no role scoping),
  `step_id` passed on every call, `get_my_context` returns full upstream
  outputs + 10 full notes. Embeddings TODO → `search_notes` falls back to
  recency. Several token leaks detailed in §7.
- **Step types**: `atomic`, `parallel_group`, `sequence`, `for_each`.
  `spawn_and_continue` for DAG splice, `abort_branch` for scoped early
  return, `request_input`/`provide_input` for coordinator gating.

## 4. Focus areas

### 4.1 Multi-runner safety and fine-grained locks

Two layers of locking.

#### Layer A — claim ownership on `campaign_steps`

Schema additions:

```sql
ALTER TABLE sortie.campaign_steps
  ADD COLUMN claim_owner text,       -- 'balthazar/runner-pid-4711'
  ADD COLUMN claim_token uuid,       -- regenerated on each claim
  ADD COLUMN heartbeat_at timestamptz;

CREATE INDEX idx_steps_heartbeat
  ON sortie.campaign_steps (heartbeat_at)
  WHERE status = 'running';
```

Behaviour:
- `claim_step(step_id, owner)` stamps `claim_owner`, fresh `claim_token`,
  `heartbeat_at = now()`, `started_at = now()`.
- `complete_step` / `fail_step` / `request_input` require matching
  `claim_token` (so a resurrected zombie cannot overwrite a successful
  re-run). Mismatch → return `{status: "stale_claim"}`, no-op.
- `reset_zombies(timeout_minutes)` switches from `started_at` to
  `heartbeat_at` as the staleness signal. A long-running but healthy
  step keeps heart-beating; a stuck one is recovered promptly.
- New MCP tool `heartbeat()` (session-scoped: no args). Worker agents
  call every 30s. Also extends any resource leases (§ Layer B).

#### Layer B — declarative resource leases

New table:

```sql
CREATE TABLE sortie.resource_leases (
    resource_key text PRIMARY KEY,
    step_id      integer NOT NULL REFERENCES sortie.campaign_steps(id),
    owner        text NOT NULL,
    mode         text NOT NULL DEFAULT 'exclusive', -- or 'shared'
    acquired_at  timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL
);
CREATE INDEX ON sortie.resource_leases (step_id);
CREATE INDEX ON sortie.resource_leases USING btree
    (resource_key text_pattern_ops);
```

`resource_key` is opaque but hierarchical, `§` as path separator.
Examples:
- `file:content/books/nanobuds/chapters/ch01.tex`
- `file:content/books/nanobuds/chapters/ch01.tex§PLXDX`
- `campaign:<uuid>§strategy`

Conflict matrix (checked in `try_claim_with_locks`):

| Held  \ Requested | `K` excl | `K` shared | `K§child` excl | `K§child` shared |
|---|---|---|---|---|
| `K` exclusive    | ✗ | ✗ | ✗ | ✗ |
| `K` shared       | ✗ | ✓ | ✗ | ✓ |
| `K§child` excl   | ✗ | ✗ | ✗ (same key) / ✓ (sibling) | ✗/✓ |
| `K§child` shared | ✗ | ✓ | ✗/✓ | ✓ |

Atomic acquire (all-or-nothing):

```python
async def try_claim_with_locks(
    step_id: int,
    owner: str,
    keys: list[str],
    ttl_sec: int = 900,
) -> Step | None:
    """Claim the step row and all resource leases atomically.

    Returns the claimed Step, or None if either the step was already
    taken OR any requested lock conflicts. Under contention the caller
    retries with a different step.
    """
```

Wrap in `isolation='serializable'`; retry on `serialization_failure`.
Release leases in `complete_step` / `fail_step` / `reset_zombies` via
`DELETE FROM resource_leases WHERE step_id = $1`.

**StepPlan additions**: `requires_locks: list[str] = []`. Planner emits
these; `add_step` persists them in a new column
`campaign_steps.requires_locks text[]`.

**Scheduler integration**: `dispatch_campaign` swaps `claim_step` for
`try_claim_with_locks`. Lock-busy and lost-the-race are indistinguishable
at the caller: "couldn't get it, try next ready step". Steps never
starve because pending rows are revisited each tick.

#### Open decisions

- **Advisory locks vs. table**: we pick the table for visibility and
  hierarchical semantics. `pg_try_advisory_xact_lock(hash)` is an easy
  fallback for very-hot resources if the table becomes a hotspot. Not
  planned for v0.2.
- **Key format discipline**: add a helper `make_lock_key(kind, path,
  slug?)` in `sortie_mcp.locks` so agents don't invent their own scheme.

### 4.2 Fair-share scheduler

Replace the per-tick priority-fraction allocator with a per-campaign
compute ledger persisted across ticks.

Schema:

```sql
ALTER TABLE sortie.campaigns
  ADD COLUMN slot_seconds_used real NOT NULL DEFAULT 0,
  ADD COLUMN weight real NOT NULL DEFAULT 1.0;   -- priority-derived multiplier
```

On `complete_step` / `fail_step`:
```sql
UPDATE sortie.campaigns
SET slot_seconds_used = slot_seconds_used + $2,
    tokens_used       = tokens_used + $3
WHERE id = $1;
```

`weight` mapping (priority → weight):

| Priority | Weight |
|---|---|
| urgent | 8 |
| high | 4 |
| normal | 2 |
| low | 1 |
| background | 0.5 |

Picker (weighted deficit round-robin):

```python
def pick_next_campaign(campaigns, capacity):
    # Virtual time per campaign = slot_seconds_used / weight.
    # Always pick the campaign with minimum virtual time that has
    # a ready step under its resource lease budget. Ties broken
    # by next_action_at asc.
    candidates = [c for c in campaigns if c.has_ready_work]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (c.slot_seconds_used / c.weight, c.next_action_at),
    )
```

Properties:
- A new urgent campaign still gets prompt service (`weight=8` keeps its
  virtual time low for many ticks).
- A greedy background campaign can't monopolise — every completed step
  charges it, raising virtual time relative to siblings.
- Explicit and debuggable: `slot_seconds_used / weight` is a single
  scalar per campaign that `sortie ls` can display.

Tick algorithm becomes:

```python
while slots_remaining > 0:
    c = pick_next_campaign(due_campaigns, slots_remaining)
    if c is None: break
    if not dispatch_campaign(c, max_slots=1):  # returns 0 if lock-busy
        mark_unavailable(c)                    # for this tick only
        continue
    slots_remaining -= 1
```

One slot at a time per iteration keeps the accounting tight.

### 4.3 Feynman-derived primitives

Six additions, ordered by value.

#### V1 — Verifier role

- New hermes profile: `verifier` (low-temperature, narrow toolset:
  `precis`, `perplexity.verify_url`, file-read). Prompt at
  `grimoire/agents/verifier.md`, contract:

  ```text
  INPUT:  artifact_path + source_refs[]
  TOOLS:  precis.get, precis.search, perplexity.verify_url,
          fs.read; NO generation tools.
  OUTPUT: JSON list of
          {claim, source_ref, status, evidence_quote?, severity}
          status ∈ {verified, unsupported, dead_link, number_mismatch,
                    conflicting_source}
          severity ∈ {info, warning, blocker}
  ```
- `Step.verify_after: bool` — when true, `complete_step` auto-inserts
  a verifier step depending on this one. Saves the planner a DAG
  decision.

#### V2 — Workflow template registry

New module `sortie_mcp/workflows.py`:

```python
def deep_research(topic: str, max_researchers: int = 3, ...) -> list[StepPlan]
def lit_review(topic: str, sources: list[str] | None = None) -> list[StepPlan]
def review_artifact(artifact_path: str, criteria: list[str]) -> list[StepPlan]
def autoresearch(
    program: str,             # path to program.md
    metric_name: str,
    benchmark_command: str,
    scope: str,
    max_iterations: int,
) -> list[StepPlan]
def watch(topic: str, interval_hours: int) -> list[StepPlan]

WORKFLOWS: dict[str, Callable[..., list[StepPlan]]] = {
    "deep_research": deep_research,
    "lit_review": lit_review,
    "review": review_artifact,
    "autoresearch": autoresearch,
    "watch": watch,
}
```

New MCP tool `create_campaign_from_template(template, params)`:
validates params against template signature, emits a pre-built StepPlan
tree, calls `db.create_campaign` + `db.add_step` for each root.

Mirror each template with a human-readable doc under
`grimoire/workflows/*.md` (deep-research, lit-review, review-artifact,
autoresearch, watch) so humans know which template to request.

#### V3 — Typed success contract

```sql
ALTER TABLE sortie.campaigns
  ADD COLUMN success_metric text,
  ADD COLUMN benchmark_command text,
  ADD COLUMN scope text,
  ADD COLUMN max_iterations integer;
```

Templates may require all four non-null (autoresearch). Free-form
`create_campaign` leaves them NULL. The runner provides a new
MCP tool `check_success(campaign_id)` returning
`{met: bool, metric_value: number?, iterations_used: int}` — the
planner calls it between iterations and decides `done`.

#### V4 — Knowledge cards

```sql
CREATE TABLE sortie.knowledge_cards (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id uuid NOT NULL REFERENCES sortie.campaigns(id),
    step_id integer REFERENCES sortie.campaign_steps(id),
    claim text NOT NULL,
    source_ref text NOT NULL,           -- DOI, acatome slug, URL, path
    quote text,
    confidence real DEFAULT 0.5,
    verified_status text,               -- NULL until verifier runs
    verified_at timestamptz,
    embedding vector(384),
    created_at timestamptz DEFAULT now()
);
CREATE INDEX ON sortie.knowledge_cards (campaign_id);
CREATE INDEX ON sortie.knowledge_cards (step_id);
```

- `researcher` emits cards (MCP: `add_knowledge_card`).
- `verifier` updates `verified_status`.
- `writer` reads cards via `get_knowledge_cards(campaign_id,
  embedding_query?, min_confidence?)` and cites by `source_ref`.
- `campaign_notes` stays for free-form observations (not intended as
  writing source material).

#### V5 — Artifacts and provenance

```sql
CREATE TABLE sortie.artifacts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id uuid NOT NULL REFERENCES sortie.campaigns(id),
    step_id integer REFERENCES sortie.campaign_steps(id),
    slug text NOT NULL,
    kind text NOT NULL,                 -- brief|draft|review|provenance|figure|dataset
    path text,                          -- NFS path or URI
    mime text,
    provenance_refs uuid[],             -- → knowledge_cards.id
    created_at timestamptz DEFAULT now()
);
CREATE UNIQUE INDEX ON sortie.artifacts (campaign_id, slug);
CREATE INDEX ON sortie.artifacts (step_id);
```

Every writer step emits at least one primary artifact plus one
`kind='provenance'` sidecar listing `knowledge_cards` cited.

MCP: `add_artifact`, `list_artifacts(campaign_id)`, `get_artifact(slug)`.

#### V6 — Iteration log (autoresearch campaigns)

Either a dedicated table or an enforced schema on `campaign_notes`:

```sql
CREATE TABLE sortie.iterations (
    id serial PRIMARY KEY,
    campaign_id uuid NOT NULL REFERENCES sortie.campaigns(id),
    iteration integer NOT NULL,
    hypothesis text,
    change_summary text,
    metric_before real,
    metric_after real,
    decision text,                      -- kept | reverted | inconclusive
    created_at timestamptz DEFAULT now(),
    UNIQUE (campaign_id, iteration)
);
```

MCP: `log_iteration(campaign_id, iteration, hypothesis, change_summary,
metric_before, metric_after, decision)`. The autoresearch template's
loop calls it on every cycle.

### 4.4 Token-economy fixes (no schema change)

These ride on top of the table changes and land in `server.py`
primarily.

1. **Preview-plus-seek upstream outputs.**
   `get_my_context.upstream_context[].output` becomes
   `output_preview` (first 400 chars) + `full_length: int`.
   New tool `read_step_output(step_id)`.
2. **Role-scoped tool registration.** Env `SORTIE_ROLE=worker|coordinator`
   selects which `@mcp.tool()` blocks are registered. Default
   `coordinator` for backward compat with Asa; hermes worker sessions
   set `worker`.
3. **Session-bound step_id.** Read `SORTIE_STEP_ID` from env at server
   init (one server per session). Drop the `step_id` arg from
   `add_note`, `complete_step`, `fail_step`, `request_input`,
   `spawn_and_continue`, `abort_branch`, `get_my_context`,
   `heartbeat`. Coordinator-side tools unaffected.
4. **Wire embeddings.** `add_note` and `add_knowledge_card` embed via
   `LITELLM_URL/v1/embeddings` (model configurable, default
   `text-embedding-3-small`). `search_notes` and
   `get_knowledge_cards` actually do cosine search.
5. **Cap `strategy` / `progress`.** `steer_campaign` and planner
   updates truncate to 2k chars via an LLM-compaction fallback when
   exceeded.
6. **Compact `get_campaign`.** Default returns
   `{status_counts, top_n_recent, strategy, progress}`. Full step list
   gated by `include="steps,notes"`.
7. **Move per-tool "Next:" hints** out of docstrings into the FastMCP
   `instructions` string.

## 5. Migration strategy

All schema work is additive (new tables, new columns with defaults).
`db.migrate()` in `sortie_mcp/db.py` owns ordering:

```text
m1. ALTER campaign_steps ADD COLUMN claim_owner, claim_token, heartbeat_at, requires_locks
m2. CREATE TABLE resource_leases
m3. ALTER campaigns ADD COLUMN slot_seconds_used, weight, success_metric,
    benchmark_command, scope, max_iterations
m4. CREATE TABLE knowledge_cards
m5. CREATE TABLE artifacts
m6. CREATE TABLE iterations
m7. ALTER campaign_steps ADD COLUMN verify_after
```

No data backfill required. `SCHEMA_SQL` stays a single `CREATE TABLE IF
NOT EXISTS` bundle; ALTER statements become idempotent by wrapping in
`DO $$ BEGIN ... EXCEPTION WHEN duplicate_column THEN NULL; END $$;` or
by using `ADD COLUMN IF NOT EXISTS` (PG 9.6+).

Ansible template `sortie_tables.sql.j2` mirrors the Python migration —
update both in the same commit.

## 6. Rollout order

### v0.2.0 — Locks + token fixes (safe, high-value, no API breakage)

- A1. Add `claim_owner` / `claim_token` / `heartbeat_at` columns + methods
- A2. Add `heartbeat()` MCP tool
- A3. Add `resource_leases` table + `try_claim_with_locks` + `requires_locks`
- A4. Preview-plus-seek + `read_step_output` tool
- A5. Role-scoped tool registration + session-bound `step_id`
- A6. Wire embeddings (behind `SORTIE_EMBEDDINGS_ENABLED=1` for rollout)

### v0.3.0 — Fair-share + Feynman primitives

- B1. Add `slot_seconds_used` / `weight` columns; runner uses WDRR picker
- B2. Verifier role + `verify_after` flag
- B3. Workflow template registry + `create_campaign_from_template`
- B4. Typed success contract + `check_success` tool
- B5. Knowledge cards table + `add_knowledge_card` / `get_knowledge_cards`

### v0.4.0 — Artifacts + iteration log + prompt rewrites

- C1. Artifacts table + provenance sidecars
- C2. Iterations table + `log_iteration` tool
- C3. `grimoire/agents/*.md` imperative-style rewrite pass
- C4. `grimoire/workflows/*.md` companion docs for each template
- C5. Split `cancel_campaign(keep_artifacts)` vs `purge_campaign`
- C6. `confirm_plan(campaign_id)` formalising dry-run-then-approve

## 7. Test plan

New test files under `pips/packages/sortie-mcp/tests/`:

- `test_locks.py` — claim races (`claim_owner` uniqueness under
  parallel runners), heartbeat-based zombie reset, hierarchical lease
  conflicts, shared-vs-exclusive matrix, TTL expiry, atomic
  all-or-nothing acquire.
- `test_scheduler.py` — WDRR fairness over 100 synthetic ticks with
  mixed priorities, starvation resistance, lock-busy falls through.
- `test_workflows.py` — each template produces a valid StepPlan tree,
  parameter validation rejects missing required fields.
- `test_knowledge_cards.py` — embedding insertion, cosine search,
  verifier updates.
- `test_artifacts.py` — slug uniqueness per campaign, provenance
  link integrity.
- Extensions to `test_server.py` — role-scoped tool lists, session
  `step_id`, preview-plus-seek semantics.
- Extensions to `test_runner.py` — dispatch with lock conflicts,
  heartbeat extends leases.

Integration test (new, `tests/scenarios/test_lit_review.py`): end-to-end
`create_campaign_from_template('lit_review', {...})` on an ephemeral
database with mocked LLM, asserts researcher→verifier→writer chain
produces an artifact + provenance + knowledge cards.

## 8. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| `resource_leases` becomes a hot row on a shared file | Hash-partition by prefix if p99 > 10 ms; advisory-lock fallback documented |
| WDRR starves a low-weight campaign behind a huge urgent one | `slot_seconds_used` is always-increasing; add an aging term `c.slot_seconds_used *= 0.99` per day if observed |
| Workflow templates diverge from planner output shape | Templates produce the same `StepPlan` dataclass the planner emits; single code path downstream |
| Embedding endpoint flaky → writes stall | Embed is best-effort: store row with `embedding = NULL`, backfill job reruns embeddings. `search_notes` degrades to recency as today |
| Tool-surface split breaks existing Asa code | Env-gated. Default matches current behaviour when `SORTIE_ROLE` unset |

## 9. Companion work outside sortie-mcp

Tracked for cross-referencing only; not in this plan's scope.

- `hermes` profile definitions for the `verifier` role.
- `grimoire/agents/verifier.md` prompt.
- `grimoire/workflows/{deep-research,lit-review,review,autoresearch,watch}.md`.
- `ansible/roles/sortie` SQL template + env.j2 updates for new env vars
  (`SORTIE_ROLE`, `SORTIE_EMBEDDINGS_ENABLED`).

## 10. Open questions

1. **Lock key vocabulary**: do we want a single scheme (`kind:path§slug`)
   or per-domain helpers (`file_lock_key(path, slug)`)? Proposal: both,
   with a linter in tests ensuring only helper-produced keys appear in
   `StepPlan.requires_locks` in-tree.
2. **Priority vs weight**: do we keep `priority` as a string enum *and*
   a derived `weight` column, or make `weight` the single source of
   truth? Proposal: keep both for human readability; weight is the
   computed field, priority is user-facing.
3. **Verifier temperature / model**: fixed low temperature on a small
   model, or let `hermes` route? Proposal: hermes routes; verifier
   profile in `hermes` config sets `temperature=0.1`, `max_tokens=2048`.
4. **Autoresearch loop granularity**: one step per iteration, or a
   single long-running step that calls `log_iteration` internally?
   Proposal: one step per iteration so lock/fair-share/retry all work
   uniformly and the DB is the audit log.

---

*When this plan ships, delete this file or prefix with `done-`.*
