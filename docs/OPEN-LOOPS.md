# Open loops — campaign / feynman work in progress

*Written 2026-04-23 after the Path B (Asa-as-coordinator) + Feynman Fix 1 (detach pattern) sessions. This file is the honest "what is not yet done" ledger. It supersedes any reassuring language in chat transcripts.*

Structure:

1. **Release blockers** — shipped code that does nothing until a package is published.
2. **Integration gaps** — code that should work but has never been exercised end-to-end.
3. **Half-wired pieces** — WIP state that needs to land before anything downstream makes sense.
4. **Feature work pending** — items from `docs/sortie-mcp-plan.md` and `docs/feynman-mcps-plan.md` that were mapped to this workstream but not done.
5. **Feynman follow-ups** — smaller items on top of Fix 1.
6. **Testing / docs gaps**.

---

## 1. Release blockers

The cluster runs `sortie-mcp` from PyPI via `uv pip install sortie-mcp` (unpinned, no version). The current PyPI release is **`v0.1.13`**. Four commits landed on `main` of the in-tree repo since then and none have been published:

```
7fb5201  success-contract: typed success fields + check_success tool
61f9a69  fair-share: WDRR scheduler with slot_seconds_used/weight ledger
a599e01  embeddings: LiteLLM-backed semantic note search behind feature flag
325e436  v0.2.0: claim ownership, resource leases, token economy, role gating
07372f8  (tag: v0.1.13, origin/main)   ← what PyPI + the cluster currently run
```

**Consequence for Path B (Asa wiring)**: the `SORTIE_ROLE` gating, the `SORTIE_STEP_ID`/`SORTIE_CLAIM_TOKEN` session binding, `heartbeat`, resource leases, fair-share, and `check_success` are all **absent** from the deployed package. `ansible-playbook playbooks/22-sortie.yml` today would install v0.1.13 against Asa's config that expects v0.2.0+ semantics. She would see the old 18-tool flat surface; workers would be unable to heartbeat; my config.yaml.j2 `SORTIE_ROLE=coordinator` env var would be silently ignored.

### L1. Release `sortie-mcp`

**L1a.** Bump `pips/packages/sortie-mcp/pyproject.toml` from `0.2.0` to `0.3.0` (the four head commits are API-additive on top of v0.2.0).

**L1b.** Run the `/release` workflow for `sortie-mcp`. Tag `v0.3.0`, publish to PyPI via trusted-publisher.

**L1c.** Either pin ansible to `sortie-mcp==0.3.0` or accept "latest" and let the next deploy pick it up.

**L1d.** `ansible-playbook playbooks/22-sortie.yml playbooks/21-hermes.yml` on the cluster.

### L2. Release `feynman-mcp`

**L2a.** `pips/packages/feynman-mcp/` is a brand-new **local-only** git repo (I ran `git init` there as part of Fix 1). There is no GitHub remote and no PyPI release. The ansible role that installs it needs to find it somewhere.

**L2b.** Create `retospect/feynman-mcp` on GitHub, set the remote, push `main`. Add CI + trusted-publisher config mirroring `sortie-mcp`.

**L2c.** Add `feynman-mcp` to `pips/packages/clone-all.sh` so other contributors clone it.

**L2d.** Run `/release` → PyPI `feynman-mcp==0.2.0`.

**L2e.** Check: does an ansible role install `feynman-mcp` today? I referenced `playbooks/24-feynman.yml` in my summary but never verified it exists / contains an install step. Need to grep and, if missing, add an install task to the shared `mcps` role.

---

## 2. Integration gaps

### I1. End-to-end campaign flow has never been exercised

The individual pieces (create_campaign in coordinator, dispatch_step in runner, complete_step in worker) are each unit-tested in `sortie-mcp`. The full loop — Asa creates → runner picks up on balthazar → dispatches to hermes on melchior → worker profile runs → worker calls `complete_step` → Asa sees `done` — has never been run. We do not know that it works.

**Smallest useful integration test:** a "hello world" campaign with a single atomic step that writes a one-line note and completes. Goal: watcher on balthazar picks it up within 15 min, step transitions `pending → running → done`, `get_campaign` on the coordinator side reflects the final state.

### I2. Runner env injection verified by code-read only

`@/Users/bots/Documents/openclaw-cluster/pips/packages/sortie-mcp/src/sortie_mcp/runner.py:204-220` constructs the `env` dict passed to the Hermes execute endpoint. I verified the strings but not that Hermes actually honours them. If Hermes ignores the `env` field in the request body, workers will not receive `SORTIE_STEP_ID` and will all see "no step_id in session" errors.

**Check**: read the Hermes execute handler; confirm it propagates `env` into the spawned agent process.

### I3. Discord channel routing

Campaigns have a `channel` field (`research`, `writing`, …) and `notifications` rows are written by the runner. Nothing today routes those into Discord. The old OpenClaw discord bridge is decommissioned; Hermes has a discord integration but I don't know if it subscribes to `sortie.notifications` or polls.

**Consequence**: a campaign can complete overnight and the user will never hear about it unless they run `sortie__get_updates` themselves. The whole "spend the night on this" pitch needs this to land.

### I4. Asa's `waiting_input` handling

`asa.md` tells Asa to call `sortie__provide_input(id, step_id, answer)` when she sees a `waiting_input` step in `get_updates`. But Asa has no daemon / cron — she only runs when a user DMs her. If a step asks for input and the user never opens the chat, the campaign stalls silently. Needs either:

- A "Asa nags you on Discord when a step is waiting" daemon, or
- An understood convention ("you must check back in or campaigns halt"), documented.

Likely related to I3: if channel notifications work, this goes away.

---

## 3. Half-wired pieces

### W1. Unstaged hermes profile changes

Two profile files were on disk untracked before this session and I edited them (added the sortie worker block) without staging. They remain unstaged:

- `@/Users/bots/Documents/openclaw-cluster/ansible/roles/hermes/templates/profiles/deep-researcher.yaml.j2` (new file + my sortie block)
- `@/Users/bots/Documents/openclaw-cluster/ansible/roles/hermes/templates/profiles/science.yaml.j2` (new file + my sortie block)
- `@/Users/bots/Documents/openclaw-cluster/ansible/roles/hermes/tasks/main.yml` (adds both profiles to three deployment loops)

Also unstaged: `@/Users/bots/Documents/openclaw-cluster/grimoire/agents/science.md` (science profile's SOUL).

These belong together as one "add deep-researcher and science profiles" commit. Until it lands, running `21-hermes.yml` errors: the deploy loop lists profiles that have no template file (or skips them and leaves Path B incomplete for those two profiles).

### W2. Worker SOUL prompts for sortie are missing everywhere except deep-researcher

I updated `@/Users/bots/Documents/openclaw-cluster/grimoire/agents/deep-researcher.md` to teach the heartbeat + detach loop. The equivalent "how to be a sortie worker" block is missing from:

- `grimoire/agents/writer.md`
- `grimoire/agents/researcher.md`
- `grimoire/agents/librarian.md`
- `grimoire/agents/coder.md`
- `grimoire/agents/code-reviewer.md`
- `grimoire/agents/science.md` (also pending W1)

Without these, when the runner dispatches a step to writer/researcher/etc., the worker's model does not know it should call `sortie__heartbeat` / `sortie__complete_step` / `sortie__add_note`. It will happily write a draft and never complete the step. Zombie reaper recovers, but the campaign stalls and accumulates failures.

**Shape of the shared block**: a 10-line "You are executing a campaign step" snippet that lists the handful of sortie worker tools, tells the model to call `heartbeat` every ~60s, and insists on `complete_step(summary=…)` at the end. Could live in a shared `grimoire/fragments/sortie-worker.md` included by each worker SOUL (or just duplicated, matching the existing grimoire convention of self-contained SOULs).

### W3. Profiles never considered for sortie-worker wiring

Path B wired writer, researcher, librarian, coder, code-reviewer. Not wired (deliberate gap — need a decision per profile):

- `quest` — queue-based paper acquisition. Probably fits `sortie__watch` campaigns.
- `flashcard` — interactive, not campaign-shaped. Probably leave un-wired.
- `coach` — one-shot, not campaign-shaped. Leave un-wired.
- `improver` — one-shot. Leave un-wired.

A 5-line decision matrix in the ansible README would prevent future re-debates.

### W4. Sortie B5 — knowledge cards partially begun

Two untracked migration files exist in the sortie repo:

```
src/sortie_mcp/migrations/0006.knowledge-cards.sql
src/sortie_mcp/migrations/0006.knowledge-cards.rollback.sql
```

Schema drafted; Python models / DB methods / MCP tools (`add_card`, `search_cards`, `verify_cards`) not started. See `docs/sortie-mcp-plan.md` §4.3 V4 for the intended shape.

---

## 4. Feature work pending

From `docs/sortie-mcp-plan.md`, mapped against what's done:

| ID | Item | Status |
|---|---|---|
| §4.1 Layer A | claim ownership, heartbeat, claim_token | **done**, commit `325e436`, unreleased |
| §4.1 Layer B | resource leases table + `try_claim_with_locks` | **done**, commit `325e436`, unreleased |
| §4.2 | fair-share scheduler (WDRR) | **done**, commit `61f9a69`, unreleased |
| §4.3 V1 | verifier role + profile + `verify_after` | **not started** |
| §4.3 V2 | workflow template registry + `create_campaign_from_template` | **not started** (this is "Path C" from the chat) |
| §4.3 V3 | typed success contract + `check_success` | **done**, commit `7fb5201`, unreleased |
| §4.3 V4 | knowledge cards | **schema drafted, code not started** (W4 above) |
| §7 (token econ) | trim `get_my_context` to single-digit k tokens | **not checked** — assume still unfixed |

### P1. V1 Verifier role

Would let the autoresearch template auto-insert a verifier step after a writer step — important for the "write + verify citations" loop the user actually wants. Requires a new `verifier` hermes profile with a restricted toolset.

### P2. V2 Workflow template registry (chat's "Path C")

The module `sortie_mcp/workflows.py` with `deep_research` / `lit_review` / `review_artifact` / `autoresearch` / `watch` constructors, plus the `create_campaign_from_template` MCP tool. Right now every campaign goes through the free-form LLM planner — which is slower, more expensive, and produces inconsistent DAG shapes for canonical work.

### P3. V4 Knowledge cards code

Schema is half-written (W4). Need `KnowledgeCard` model, DB methods (`add_card`, `list_cards`, `search_cards`, `mark_verified`), MCP tools (`knowledge__add_card`, `knowledge__search_cards`), and a worker prompt snippet that teaches agents to emit cards during research steps instead of freeform notes.

### P4. Token economy fixes

`docs/sortie-mcp-plan.md` §7 lists several paths to cut `get_my_context` from 10-30k tokens to single-digit k. Not inspected this session — likely still open.

---

## 5. Feynman follow-ups

On top of Fix 1 (detach pattern, shipped at `feynman-mcp v0.2.0`):

### F1. Partial output streaming

`poll_research` exposes `partial_output_len` (byte count) during a run, not the content. Agents can tell *whether* the child is producing output but not *what*. Fine for the common case; painful for long deepresearch runs where an agent might want to early-exit.

**Shape of fix**: extend `poll_research` with an optional `since_bytes: int = 0` parameter that returns the delta since the last poll. Registry already keeps `stdout_chunks` so this is cheap.

### F2. SIGTERM before SIGKILL

`_kill_quietly` in `jobs.py` goes straight to SIGKILL. Feynman spawns a Claude Code session + a research-orchestration sub-process tree; SIGKILL may orphan grandchildren or leave `~/.feynman/sessions/*` half-written. 3-line fix: send SIGTERM, wait 2s, then SIGKILL if still alive.

### F3. No survival across MCP restart

Acknowledged in the `jobs.py` module docstring. Fine for now (Feynman queries are idempotent) but if a worker dies mid-deepresearch the 5-min investment is lost. Would need job metadata persisted to the sortie DB (cross-package coupling) or a local SQLite — both overkill for v0.2.

### F4. Rate limit / concurrent-job cap

Nothing stops an agent from calling `start_research` 20 times in a row. Each spawns a Feynman CLI with its own Claude Code session. Need a per-registry cap (default 4?) and a queuing behaviour when the cap is hit.

### F5. Feynman CLI install path in ansible

I have not verified that `feynman-mcp v0.2.0` installs cleanly in the MCP venv on melchior. The binary path `/Users/hermes/.npm-global/bin/feynman` must exist (owned by the hermes user) and the `HOME` env passed by the MCP profile must point at `/Users/hermes` so the session store resolves. These are inherited from the existing v0.1 deploy but should be smoke-tested after release.

---

## 6. Testing / docs gaps

### T1. No integration smoke test

See I1. A bash or pytest smoke test that creates a minimal campaign and asserts progression to `done` within 5 minutes would catch 80% of post-deploy breakage. Could live in `ansible/roles/sortie/files/smoketest.sh` or as a `pytest -m integration` in sortie-mcp tests behind `TEST_HERMES_URL`.

### T2. Campaign debugging runbook

When a campaign stalls at 80% the user will need to know: which step is stuck, which worker claimed it, how to unstick. No runbook exists. At minimum a one-pager with:

- `SELECT * FROM sortie.campaign_steps WHERE status='running' AND heartbeat_at < now() - interval '5 min'` — find zombies
- `sortie__get_campaign(id)` return shape annotated with what each field means for debugging
- How to inspect `/Users/hermes/.hermes/logs/` for a specific step's agent output
- How to manually `reset_zombies` / `cancel_step` via psql

### T3. Architecture doc for the campaign path

`docs/agent-architecture.md` predates sortie. The campaign dispatch path (balthazar cron → DB → runner → Hermes execute → worker env → MCP tools) deserves a sequence diagram. Currently lives only in commit messages.

### T4. Asa prompt coverage

`asa.md` now has the campaign section but it has not been tested with real Asa (qwen3.5:9b). The model might struggle with the `get_updates` JSON shape or pick `delegate_task` when `create_campaign` would be better. Needs a handful of prompted-conversation transcripts against the running agent to verify.

---

## Summary of true blockers for the "ask Asa to run a gig" UX

Ranked by what breaks if you skip it:

1. **L1** — release sortie-mcp v0.3.0 to PyPI. Without this, nothing works.
2. **W1** — commit the deep-researcher + science profile WIP. Without this, `21-hermes.yml` is inconsistent.
3. **W2** — add the sortie worker snippet to writer / researcher / librarian / coder / code-reviewer SOULs. Without this, dispatched steps stall silently.
4. **I3** — Discord channel notifications. Without this, campaigns are fire-and-forget with no feedback.
5. **L2** — release feynman-mcp to PyPI + add ansible install. Without this, the deep-researcher worker has no way to actually call Feynman.

Everything else in this document is quality-of-life or future scope.
