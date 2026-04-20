"""sortie-mcp MCP server — campaign orchestration tools for AI agents.

Three perspectives on one server:
- Coordinator (Asa): create, list, get, steer, pause/resume/cancel campaigns
- Worker (agents): get_my_context, add_note, search_notes, complete_step, etc.
- Executor tools are Python internal API (see db.py), not MCP-exposed.

Role gating
-----------
Each process serves exactly one role, selected via ``$SORTIE_ROLE``:

- ``coordinator`` — coordinator-only tools registered. Agent-side tools
  (``get_my_context``, ``complete_step``, ``heartbeat``, …) are hidden
  so Asa's prompt doesn't waste tokens on tool schemas it can never use.
- ``worker`` — worker-only tools registered. Campaign-management tools
  are hidden from agents that should only be operating on their own step.
- ``both`` (default) — everything registered, as in v0.1.

Session binding
---------------
Worker tools default their ``step_id`` / ``claim_token`` arguments from
``$SORTIE_STEP_ID`` / ``$SORTIE_CLAIM_TOKEN`` so agents don't have to
re-state their own identity on every call. See :mod:`sortie_mcp.session`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from .db import DB
from .embeddings import embed_text, embeddings_enabled
from .models import (
    Campaign,
    CampaignStatus,
    Priority,
    StepStatus,
)
from .session import (
    resolve_step_id,
    session_claim_token,
    session_role,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "sortie-mcp",
    instructions="Campaign orchestration for AI agents — dependency DAGs, "
    "parallel fan-out, failure policies, embedded notes.",
)

_db: DB | None = None


async def get_db() -> DB:
    """Lazy-init the DB connection pool."""
    global _db
    if _db is None:
        dsn = os.environ.get("DATABASE_URL", "postgresql://localhost/sortie")
        schema = os.environ.get("SORTIE_SCHEMA", "sortie")
        _db = DB(dsn, schema=schema)
        await _db.connect()
        await _db.migrate()
    return _db


# ---------------------------------------------------------------------------
# Role-gated tool registration
# ---------------------------------------------------------------------------
#
# Reading ``SORTIE_ROLE`` ONCE at module import is deliberate: FastMCP
# registers tools eagerly via decorators, and the set of registered tools
# must be stable across an MCP session. Tests that want to exercise a
# specific role import server.py inside a ``monkeypatch.setenv`` block.

_ACTIVE_ROLE = session_role()

F = TypeVar("F", bound=Callable[..., Any])


def coordinator_tool() -> Callable[[F], F]:
    """Register a coordinator-facing tool (create/list/steer/…).

    No-op decorator when ``$SORTIE_ROLE=worker``.
    """

    def _decorate(fn: F) -> F:
        if _ACTIVE_ROLE in ("coordinator", "both"):
            return mcp.tool()(fn)  # type: ignore[return-value]
        return fn

    return _decorate


def worker_tool() -> Callable[[F], F]:
    """Register a worker-facing tool (get_my_context/complete_step/…).

    No-op decorator when ``$SORTIE_ROLE=coordinator``.
    """

    def _decorate(fn: F) -> F:
        if _ACTIVE_ROLE in ("worker", "both"):
            return mcp.tool()(fn)  # type: ignore[return-value]
        return fn

    return _decorate


# ---------------------------------------------------------------------------
# Preview helpers — token-economy for large step outputs
# ---------------------------------------------------------------------------

# Token budget for upstream outputs bundled into ``get_my_context``.
# Each upstream step gets up to HEAD chars from the top and TAIL chars
# from the end — agents can ``read_step_output(step_id)`` for the rest.
PREVIEW_HEAD_CHARS = int(os.environ.get("SORTIE_PREVIEW_HEAD", "600"))
PREVIEW_TAIL_CHARS = int(os.environ.get("SORTIE_PREVIEW_TAIL", "200"))


def _preview(text: str | None) -> dict[str, Any]:
    """Return head+tail preview with metadata.

    Shape::

        {
            "preview": "<head>\n…[ELIDED N chars]…\n<tail>",
            "total_chars": N,
            "truncated": True|False,
            "hint": "call read_step_output(<id>) for full text",  # only when truncated
        }

    For text shorter than ``HEAD + TAIL`` the full string is returned
    with ``truncated=False`` and no elision marker.
    """
    if not text:
        return {"preview": "", "total_chars": 0, "truncated": False}
    n = len(text)
    if n <= PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS:
        return {"preview": text, "total_chars": n, "truncated": False}
    elided = n - PREVIEW_HEAD_CHARS - PREVIEW_TAIL_CHARS
    head = text[:PREVIEW_HEAD_CHARS]
    tail = text[-PREVIEW_TAIL_CHARS:]
    return {
        "preview": f"{head}\n…[ELIDED {elided} chars]…\n{tail}",
        "total_chars": n,
        "truncated": True,
    }


# ---------------------------------------------------------------------------
# Coordinator tools (Asa)
# ---------------------------------------------------------------------------


@coordinator_tool()
async def create_campaign(
    goal: str,
    name: str | None = None,
    channel: str | None = None,
    priority: str = "normal",
    max_depth: int = 4,
    token_budget: int | None = None,
    dry_run: bool = False,
    # Typed success contract (v0.3 B4). Set together when the campaign
    # is measurable (autoresearch template, benchmark sweeps). Leave
    # unset for exploratory / free-form campaigns.
    success_metric: str | None = None,
    benchmark_command: str | None = None,
    scope: str | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Create a new campaign for long-running, multi-step work.

    Args:
        goal: What this campaign should accomplish.
        name: Short name for display. Auto-generated if omitted.
        channel: Discord channel for notifications.
        priority: urgent / high / normal / low / background.
        max_depth: Max nesting depth for subtasks (default 4).
        token_budget: Optional token limit. NULL = unlimited.
        dry_run: If true, create in paused status for review.
        success_metric: Short metric name emitted by the verifier /
            benchmark (e.g. "accuracy_at_1k"). Paired with
            ``benchmark_command``. Leave NULL for free-form campaigns.
        benchmark_command: Shell / Python invocation that produces a
            JSON line with the metric value. Metadata only — the
            runner does not execute it; worker steps do.
        scope: Freeform identifier narrowing the benchmark
            (e.g. "chapter-01", "test_subset_A").
        max_iterations: Hard cap on autoresearch-style loops. NULL
            means open-ended (planner decides).

    Returns: Campaign ID, name, status, next_action_at.

    Next: Use `get_campaign(id)` to check progress, or `steer_campaign(id, guidance)` to adjust.
    """
    db = await get_db()
    status = CampaignStatus.PAUSED if dry_run else CampaignStatus.ACTIVE
    campaign = await db.create_campaign(
        goal,
        name=name,
        channel=channel,
        max_depth=max_depth,
        token_budget=token_budget,
        priority=Priority(priority),
        status=status,
        success_metric=success_metric,
        benchmark_command=benchmark_command,
        scope=scope,
        max_iterations=max_iterations,
    )
    return {
        "id": str(campaign.id),
        "name": campaign.name,
        "status": campaign.status.value,
        "priority": campaign.priority.value,
        "next_action_at": str(campaign.next_action_at),
    }


@coordinator_tool()
async def list_campaigns(status: str | None = None) -> list[dict[str, Any]]:
    """List campaigns, optionally filtered by status.

    Args:
        status: Filter by status (active/paused/done/failed/cancelled). Omit for all.

    Returns: Array of {id, name, status, priority, progress}.
    """
    db = await get_db()
    cs = CampaignStatus(status) if status else None
    campaigns = await db.list_campaigns(status=cs)
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "status": c.status.value,
            "priority": c.priority.value,
            "progress": c.progress,
            "goal": c.goal[:200],
        }
        for c in campaigns
    ]


@coordinator_tool()
async def get_campaign(id: str) -> dict[str, Any]:
    """Get full campaign state: goal, strategy, progress, step tree, recent notes.

    Sets last_reported_at so next call only returns new activity.

    Args:
        id: Campaign UUID.

    Returns: Full campaign state with steps and recent notes.
    """
    db = await get_db()
    cid = UUID(id)
    campaign = await db.get_campaign(cid)
    if not campaign:
        return {"error": f"Campaign {id} not found"}

    steps = await db.get_steps(cid)
    notes = await db.get_notes(cid)

    # Only return notes since last report
    recent_notes = [
        n
        for n in notes
        if campaign.last_reported_at is None
        or (n.created_at and n.created_at > campaign.last_reported_at)
    ]

    await db.set_last_reported(cid)

    return {
        "id": str(campaign.id),
        "name": campaign.name,
        "goal": campaign.goal,
        "status": campaign.status.value,
        "priority": campaign.priority.value,
        "strategy": campaign.strategy,
        "progress": campaign.progress,
        "max_depth": campaign.max_depth,
        "tokens_used": campaign.tokens_used,
        "token_budget": campaign.token_budget,
        # Fair-share scheduler state (migration 0004) — surfaced so
        # humans/operators can see where the compute has gone and why
        # a campaign might be waiting behind a more-entitled sibling.
        "slot_seconds_used": campaign.slot_seconds_used,
        "weight": campaign.weight,
        "virtual_time": campaign.virtual_time,
        # Typed success contract (migration 0005) — NULL for free-form
        # campaigns, populated for autoresearch / templated campaigns.
        "success_metric": campaign.success_metric,
        "benchmark_command": campaign.benchmark_command,
        "scope": campaign.scope,
        "max_iterations": campaign.max_iterations,
        "steps": [
            {
                "id": s.id,
                "action": s.action,
                "agent": s.agent,
                "status": s.status.value,
                "step_type": s.step_type.value,
                "depth": s.depth,
                "output": s.output[:500] if s.output else None,
                "error": s.error,
                "parent_step_id": s.parent_step_id,
                "depends_on": s.depends_on,
            }
            for s in steps
        ],
        "recent_notes": [
            {"id": n.id, "content": n.content, "tags": n.tags, "agent": n.agent}
            for n in recent_notes[:20]
        ],
    }


@coordinator_tool()
async def get_updates(id: str | None = None) -> dict[str, Any]:
    """Get delta since last report: completed steps, failures, new notes.

    Args:
        id: Campaign UUID. Omit for updates across all active campaigns.

    Returns: Recent completions, failures, and notes.
    """
    db = await get_db()
    campaigns: list[Campaign]
    if id:
        found = await db.get_campaign(UUID(id))
        campaigns = [found] if found else []
    else:
        campaigns = await db.list_campaigns(status=CampaignStatus.ACTIVE)

    updates = []
    for c in campaigns:
        steps = await db.get_steps(c.id)
        notes = await db.get_notes(c.id)
        recent = [
            s
            for s in steps
            if c.last_reported_at is None
            or (s.completed_at and s.completed_at > c.last_reported_at)
        ]
        recent_notes = [
            n
            for n in notes
            if c.last_reported_at is None
            or (n.created_at and n.created_at > c.last_reported_at)
        ]
        await db.set_last_reported(c.id)
        updates.append(
            {
                "campaign_id": str(c.id),
                "name": c.name,
                "completed": [
                    {"id": s.id, "action": s.action, "status": s.status.value}
                    for s in recent
                    if s.status
                    in (StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED)
                ],
                "notes": [
                    {"content": n.content, "tags": n.tags} for n in recent_notes[:10]
                ],
            }
        )
    return {"updates": updates}


@coordinator_tool()
async def steer_campaign(id: str, guidance: str) -> dict[str, Any]:
    """Change campaign direction. Updates strategy for the planner.

    Args:
        id: Campaign UUID.
        guidance: New direction, constraints, or focus areas.

    Returns: Updated strategy.
    """
    db = await get_db()
    cid = UUID(id)
    campaign = await db.get_campaign(cid)
    if not campaign:
        return {"error": f"Campaign {id} not found"}

    new_strategy = f"{campaign.strategy or ''}\n\n[User guidance]: {guidance}".strip()
    updated = await db.update_campaign(cid, strategy=new_strategy)
    return {
        "id": str(cid),
        "strategy": updated.strategy if updated else None,
        "status": "updated",
    }


@coordinator_tool()
async def pause_campaign(id: str) -> dict[str, str]:
    """Pause a campaign. Running steps finish but no new ones start.

    Args:
        id: Campaign UUID.
    """
    db = await get_db()
    await db.update_campaign(UUID(id), status=CampaignStatus.PAUSED)
    return {"id": id, "status": "paused"}


@coordinator_tool()
async def resume_campaign(id: str) -> dict[str, str]:
    """Resume a paused campaign.

    Args:
        id: Campaign UUID.
    """
    db = await get_db()
    await db.update_campaign(UUID(id), status=CampaignStatus.ACTIVE)
    return {"id": id, "status": "active"}


@coordinator_tool()
async def provide_input(id: str, step_id: int, answer: str) -> dict[str, Any]:
    """Provide input to a step that is waiting for a human decision.

    Workers call ``request_input`` when they need guidance. This tool
    unblocks them by supplying the answer and returning the step to
    the ready queue.

    Args:
        id: Campaign UUID.
        step_id: The step waiting for input.
        answer: Your decision / the information the agent asked for.

    Returns: Updated step status.
    """
    db = await get_db()
    step = await db.provide_input(step_id, answer)
    if not step:
        return {"error": f"Step {step_id} not found or not in waiting_input status"}
    return {
        "step_id": step.id,
        "status": step.status.value,
        "input_provided": True,
    }


@coordinator_tool()
async def check_success(id: str) -> dict[str, Any]:
    """Evaluate whether a campaign has met its typed success contract.

    A campaign's success contract (``success_metric``,
    ``benchmark_command``, ``scope``, ``max_iterations``) is set at
    creation time by templates like ``autoresearch``. This tool gives
    the planner / coordinator a single scalar decision:

    * ``met=True``  — contract satisfied; coordinator should mark the
                      campaign ``done``.
    * ``met=False`` — keep iterating, or escalate if budget exhausted.
    * ``iterations_used`` — count of DONE atomic steps on this campaign
                      so far. Useful for comparing against
                      ``max_iterations``.
    * ``metric_value`` — the most recently recorded metric from
                      ``campaign_notes`` tagged ``metric:<success_metric>``
                      (best-effort parse; see below).

    **How the metric is discovered.** The runner / worker agents emit
    a note of the form::

        add_note(campaign_id, f"metric={value}", tags=["metric:<name>"])

    on every benchmark run. ``check_success`` scans the most recent
    such note, extracts the float after ``=``, and compares it against
    ``success_metric_threshold`` if set via ``steer_campaign(strategy=...)``.
    In v0.3 we only surface the most-recent value — the planner
    decides whether it's "good enough". A future revision may wire a
    numeric threshold column.

    Args:
        id: Campaign UUID.

    Returns:
        ``{met, metric_value, iterations_used, max_iterations,
           success_metric, notes_checked, reason}``

        ``reason`` is a short string explaining the decision — useful
        both for humans looking at logs and for the planner's own
        chain-of-thought.

    A campaign without a success contract returns
    ``{met: False, reason: "no_success_metric_configured"}``.

    Next: if ``met`` is True, call ``cancel_campaign(id)`` or
    ``steer_campaign(id, "wrap up")``. If False and
    ``iterations_used >= max_iterations``, escalate via the
    notification channel.
    """
    db = await get_db()
    cid = UUID(id)
    campaign = await db.get_campaign(cid)
    if not campaign:
        return {"error": f"Campaign {id} not found"}

    if not campaign.success_metric:
        return {
            "met": False,
            "reason": "no_success_metric_configured",
            "success_metric": None,
            "metric_value": None,
            "iterations_used": 0,
            "max_iterations": campaign.max_iterations,
        }

    # Count iterations = done atomic steps. ``skipped`` / ``failed``
    # don't count as successful iterations for budget accounting.
    done_steps = await db.get_steps(cid, status=StepStatus.DONE)
    iterations_used = len(done_steps)

    # Find the most recent "metric:<name>" tagged note; parse value.
    tag = f"metric:{campaign.success_metric}"
    notes = await db.get_notes(cid, tags=[tag])
    metric_value: float | None = None
    last_note_id: int | None = None
    if notes:
        # ``get_notes`` returns newest-first; grab the first parseable.
        for n in notes:
            val = _extract_metric_value(n.content)
            if val is not None:
                metric_value = val
                last_note_id = n.id
                break

    # Termination logic — kept deliberately conservative:
    #   * any recorded metric => ``met=True`` (planner decides threshold).
    #   * budget exhausted without a metric => ``met=False, reason=budget``.
    #   * otherwise => ``met=False, reason=still_running``.
    # Thresholding is a v0.4 follow-up; today the contract is existence-
    # of-signal, not magnitude-of-signal.
    if metric_value is not None:
        return {
            "met": True,
            "reason": "metric_recorded",
            "success_metric": campaign.success_metric,
            "metric_value": metric_value,
            "iterations_used": iterations_used,
            "max_iterations": campaign.max_iterations,
            "notes_checked": len(notes),
            "last_metric_note_id": last_note_id,
        }
    if (
        campaign.max_iterations is not None
        and iterations_used >= campaign.max_iterations
    ):
        return {
            "met": False,
            "reason": "max_iterations_reached_without_metric",
            "success_metric": campaign.success_metric,
            "metric_value": None,
            "iterations_used": iterations_used,
            "max_iterations": campaign.max_iterations,
            "notes_checked": len(notes),
        }
    return {
        "met": False,
        "reason": "still_running",
        "success_metric": campaign.success_metric,
        "metric_value": None,
        "iterations_used": iterations_used,
        "max_iterations": campaign.max_iterations,
        "notes_checked": len(notes),
    }


def _extract_metric_value(text: str) -> float | None:
    """Parse the most permissive ``key=number`` form from a note body.

    Accepts any of::

        metric=0.83
        accuracy_at_1k = 0.742
        metric: 12.5
        Final: 99.9%     (% is stripped)
        result=-3.2e-2

    Returns ``None`` if no numeric value could be extracted. Designed
    to be forgiving so workers don't need a rigid report format —
    producers write what's natural, ``check_success`` squints at it.
    """
    import re

    # Strip trailing percent sign; it's a unit, not a separator.
    candidate_patterns = [
        r"[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?",  # key=value or key:value
        r"\b(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?\s*$",  # trailing number
    ]
    for pat in candidate_patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


@coordinator_tool()
async def cancel_campaign(id: str) -> dict[str, Any]:
    """Cancel a campaign. All pending steps are skipped.

    Args:
        id: Campaign UUID.
    """
    db = await get_db()
    cid = UUID(id)
    await db.update_campaign(cid, status=CampaignStatus.CANCELLED)
    # Skip all pending steps
    pending = await db.get_steps(cid, status=StepStatus.PENDING)
    for step in pending:
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {db._t('campaign_steps')} SET status = 'skipped', error = 'Campaign cancelled' WHERE id = $1 AND status = 'pending'",
                step.id,
            )
    return {"id": id, "status": "cancelled", "steps_skipped": len(pending)}


# ---------------------------------------------------------------------------
# Worker tools (specialist agents executing steps)
# ---------------------------------------------------------------------------


@worker_tool()
async def get_my_context(step_id: int | None = None) -> dict[str, Any]:
    """Get campaign context for the step you're executing.

    Upstream outputs are returned as **previews** (head + tail + total
    char count) so the context bundle stays small. Call
    ``read_step_output(<id>)`` when you need the full body of a specific
    upstream output.

    Args:
        step_id: Your step ID. Inferred from ``$SORTIE_STEP_ID`` if unset.

    Returns: Campaign goal, your task, upstream output previews, notes.

    Next: Do your work, then call ``complete_step(summary)`` or
    ``fail_step(error)``.
    """
    resolved = resolve_step_id(step_id)
    if resolved is None:
        return {
            "error": "step_id not provided and $SORTIE_STEP_ID is unset",
        }
    db = await get_db()
    step = await db.get_step(resolved)
    if not step:
        return {"error": f"Step {resolved} not found"}

    campaign = await db.get_campaign(step.campaign_id)
    if not campaign:
        return {"error": "Campaign not found"}

    # Upstream outputs — preview-plus-seek to keep the context bundle small.
    upstream: list[dict[str, Any]] = []
    if step.depends_on:
        for dep_id in step.depends_on:
            dep = await db.get_step(dep_id)
            if dep and dep.output:
                entry: dict[str, Any] = {
                    "step_id": dep.id,
                    "action": dep.action,
                    **_preview(dep.output),
                }
                if entry.get("truncated"):
                    entry["hint"] = f"call read_step_output({dep.id}) for full text"
                upstream.append(entry)

    # Input (answer from a prior ``request_input``) — also preview'd.
    input_preview: dict[str, Any] | None = None
    if step.input:
        input_preview = _preview(step.input)
        if input_preview.get("truncated"):
            input_preview["hint"] = (
                f"call read_step_output({step.id}, field='input') for full text"
            )

    # Relevant notes (most recent).
    notes = await db.get_notes(step.campaign_id)
    recent_notes = notes[:10]

    return {
        "campaign_name": campaign.name,
        "campaign_goal": campaign.goal,
        "your_task": step.action,
        "your_step_id": step.id,
        "your_input": input_preview,
        "depth": step.depth,
        "max_depth": campaign.max_depth,
        "upstream_context": upstream,
        "relevant_notes": [
            {"content": n.content, "tags": n.tags, "agent": n.agent}
            for n in recent_notes
        ],
    }


@worker_tool()
async def read_step_output(
    step_id: int,
    offset: int = 0,
    limit: int = 8000,
    field: str = "output",
) -> dict[str, Any]:
    """Read a range from a step's ``output`` (or ``input``) field.

    Counterpart to the preview-plus-seek contract in ``get_my_context``:
    when an upstream preview is truncated, call this with the
    ``step_id`` from the preview to pull the full (or a slice of) text.

    Args:
        step_id: The step whose output you want to read. Does not have
            to be your own step — any step in the same campaign's DAG
            is readable.
        offset: 0-indexed character offset to start from.
        limit: Max characters to return (default 8000, capped at 32000
            to avoid blowing the agent's context).
        field: ``"output"`` (default) or ``"input"``.

    Returns::

        {
            "step_id": N,
            "field": "output",
            "content": "...",
            "offset": 0,
            "limit": 8000,
            "total_chars": N,
            "has_more": bool,  # True if offset+limit < total_chars
        }
    """
    if field not in ("output", "input"):
        return {"error": f"field must be 'output' or 'input', got {field!r}"}
    limit = max(1, min(limit, 32_000))
    offset = max(0, offset)

    db = await get_db()
    step = await db.get_step(step_id)
    if not step:
        return {"error": f"Step {step_id} not found"}

    text = step.output if field == "output" else step.input
    if text is None:
        return {
            "step_id": step_id,
            "field": field,
            "content": "",
            "offset": 0,
            "limit": limit,
            "total_chars": 0,
            "has_more": False,
        }
    total = len(text)
    slice_end = min(offset + limit, total)
    return {
        "step_id": step_id,
        "field": field,
        "content": text[offset:slice_end],
        "offset": offset,
        "limit": limit,
        "total_chars": total,
        "has_more": slice_end < total,
    }


@worker_tool()
async def heartbeat(
    step_id: int | None = None,
    extend_leases_sec: int = 900,
) -> dict[str, Any]:
    """Report that you're still alive and keep your claim + leases fresh.

    Long-running agents should call this every few minutes. The runner's
    ``reset_zombies`` sweep uses ``heartbeat_at`` to distinguish healthy
    workers from crashed ones. If you hold resource leases (see
    ``requires_locks`` on your step), the lease ``expires_at`` is bumped
    by ``extend_leases_sec`` from now so the lease reaper won't steal
    them out from under you.

    Returns ``{"status": "stale_claim"}`` if your claim token no longer
    matches — this means the zombie reset already repossessed your step.
    **Stop working immediately** if you see that: another runner is
    about to pick your step up.

    Args:
        step_id: Your step ID. Inferred from ``$SORTIE_STEP_ID`` if unset.
        extend_leases_sec: Seconds of TTL to give every lease you hold.
            Default 900 (15 min). Set to 0 for heartbeat-only.

    Returns: ``{"status": "ok" | "stale_claim", "step_id": N}``.
    """
    resolved = resolve_step_id(step_id)
    if resolved is None:
        return {
            "error": "step_id not provided and $SORTIE_STEP_ID is unset",
        }
    token = session_claim_token()
    if token is None:
        return {
            "error": "heartbeat requires $SORTIE_CLAIM_TOKEN to be set by the runner",
        }
    db = await get_db()
    extend = extend_leases_sec if extend_leases_sec > 0 else None
    step = await db.heartbeat(resolved, token, extend_leases_sec=extend)
    if step is None:
        return {"status": "stale_claim", "step_id": resolved}
    return {"status": "ok", "step_id": step.id}


@worker_tool()
async def add_note(
    campaign_id: str,
    content: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Record a noteworthy finding during step execution.

    Notes are embedded for semantic search across the campaign.

    Args:
        campaign_id: Campaign UUID.
        content: What you found. Be specific.
        tags: Optional tags for filtering (e.g. ["finding", "citation"]).

    Returns: Note ID and any similar existing notes.

    Next: Continue your work. Call `complete_step` when done.
    """
    db = await get_db()
    cid = UUID(campaign_id)
    # Feature-flagged: if $SORTIE_EMBEDDINGS_ENABLED is off, embed_text
    # returns None and the note gets a NULL embedding (back-compat).
    vec = await embed_text(content)
    note = await db.add_note(cid, content, tags=tags, embedding=vec)
    return {
        "note_id": note.id,
        "recorded": True,
        "embedded": vec is not None,
    }


@worker_tool()
async def search_notes(
    query: str,
    campaign_id: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search across campaign notes.

    Args:
        query: What to search for.
        campaign_id: Scope to a specific campaign. Omit for all.
        top_k: Number of results (default 5).

    Returns: Ranked results with content and tags. Each entry is
        annotated with ``"mode": "semantic"`` when cosine ranking was
        used, or ``"mode": "recency"`` when embeddings were disabled /
        unavailable and the server fell back to most-recent-first
        listing.
    """
    db = await get_db()
    cid = UUID(campaign_id) if campaign_id else None

    # Try the semantic path only when embeddings are on AND LiteLLM
    # gave us a non-None query vector. Otherwise fall back to recency.
    query_vec = await embed_text(query) if embeddings_enabled() else None
    if query_vec is not None:
        notes = await db.search_notes(query_vec, campaign_id=cid, top_k=top_k)
        return [
            {
                "id": n.id,
                "content": n.content,
                "tags": n.tags,
                "agent": n.agent,
                "mode": "semantic",
            }
            for n in notes
        ]

    # Fallback: recency-ordered listing. Scoped to campaign when given;
    # global searches return an empty list rather than silently scanning
    # every campaign (would be a DoS vector in a large cluster).
    notes = await db.get_notes(cid) if cid else []
    return [
        {
            "id": n.id,
            "content": n.content,
            "tags": n.tags,
            "agent": n.agent,
            "mode": "recency",
        }
        for n in notes[:top_k]
    ]


@worker_tool()
async def get_notes(
    campaign_id: str,
    tags: list[str] | None = None,
    step_id: int | None = None,
) -> list[dict[str, Any]]:
    """List notes filtered by tag or step.

    Args:
        campaign_id: Campaign UUID.
        tags: Filter by tags (OR match).
        step_id: Filter by step ID.

    Returns: Matching notes.
    """
    db = await get_db()
    notes = await db.get_notes(UUID(campaign_id), tags=tags, step_id=step_id)
    return [
        {"id": n.id, "content": n.content, "tags": n.tags, "agent": n.agent}
        for n in notes
    ]


@worker_tool()
async def complete_step(
    step_id: int | None = None,
    summary: str = "",
) -> dict[str, Any]:
    """Mark your step as done with a summary of what you accomplished.

    Args:
        step_id: Your step ID. Inferred from ``$SORTIE_STEP_ID`` if unset.
        summary: What you did and what you found. This becomes the step
                 output visible to downstream steps.

    Returns: Confirmation. If the step was already skipped (branch abort),
             returns ``{status: "skipped"}`` — your output is recorded for audit.
             Returns ``{status: "stale_claim"}`` if a zombie-reset already
             stole your claim — stop working.
    """
    resolved = resolve_step_id(step_id)
    if resolved is None:
        return {
            "error": "step_id not provided and $SORTIE_STEP_ID is unset",
        }
    db = await get_db()
    # Session claim_token enforces: "only the owner of this claim may
    # complete it". A zombie worker whose claim was reset sees ``None``.
    step = await db.complete_step(resolved, summary, claim_token=session_claim_token())
    if not step:
        # Distinguish "token mismatch" from "step missing". When a session
        # token was set but the update failed, it's almost certainly a
        # stale claim.
        if session_claim_token() is not None:
            return {"status": "stale_claim", "step_id": resolved}
        return {"error": f"Step {resolved} not found"}

    if step.status == StepStatus.SKIPPED:
        return {
            "status": "skipped",
            "reason": step.error or "Branch aborted — output recorded for audit",
            "output_recorded": True,
        }

    return {"status": "done", "step_id": resolved}


@worker_tool()
async def fail_step(
    step_id: int | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Report that you cannot complete your step.

    Args:
        step_id: Your step ID. Inferred from ``$SORTIE_STEP_ID`` if unset.
        error: What went wrong and why you can't continue.

    Returns: Whether the step can be retried or has failed permanently.
    """
    resolved = resolve_step_id(step_id)
    if resolved is None:
        return {
            "error": "step_id not provided and $SORTIE_STEP_ID is unset",
        }
    db = await get_db()
    step = await db.fail_step(resolved, error, claim_token=session_claim_token())
    if not step:
        if session_claim_token() is not None:
            return {"status": "stale_claim", "step_id": resolved}
        return {"error": f"Step {resolved} not found"}

    return {
        "status": step.status.value,
        "retry_count": step.retry_count,
        "max_retries": step.max_retries,
        "can_retry": step.retry_count < step.max_retries,
    }


@worker_tool()
async def request_input(
    step_id: int | None = None,
    question: str = "",
    partial_output: str | None = None,
) -> dict[str, Any]:
    """Pause your step and ask the coordinator for a decision.

    Use when you hit a fork that requires human judgement — e.g. which
    approach to take, whether to proceed with a risky action, or
    clarification on ambiguous requirements.

    Your step pauses until the coordinator calls ``provide_input``.
    When it resumes, the answer will be in your step's input field
    (visible via ``get_my_context``).

    Args:
        step_id: Your step ID. Inferred from ``$SORTIE_STEP_ID`` if unset.
        question: What you need decided. Be specific.
        partial_output: Optional summary of work done so far.

    Returns: Confirmation that the step is paused.
    """
    resolved = resolve_step_id(step_id)
    if resolved is None:
        return {
            "error": "step_id not provided and $SORTIE_STEP_ID is unset",
        }
    db = await get_db()
    step = await db.request_input(
        resolved,
        question,
        partial_output=partial_output,
        claim_token=session_claim_token(),
    )
    if not step:
        if session_claim_token() is not None:
            return {"status": "stale_claim", "step_id": resolved}
        return {"error": f"Step {resolved} not found or not in running status"}
    return {
        "status": "waiting_input",
        "step_id": step.id,
        "question": question,
        "message": "Step paused. The coordinator will provide input and your step will be re-dispatched.",
    }


@worker_tool()
async def spawn_and_continue(
    step_id: int,
    partial_output: str,
    subtasks: list[dict[str, str]],
    continuation: str,
) -> dict[str, Any]:
    """Split your work: spawn subtasks and a continuation that resumes after they complete.

    Use when you discover you need additional work done before you can finish.
    The DAG rewires automatically — your downstream dependents will wait for
    the continuation, not your partial result.

    Args:
        step_id: Your step ID.
        partial_output: What you've done so far.
        subtasks: List of {action, agent?} dicts for work that needs doing.
        continuation: Action description for the step that resumes your work
                     after subtasks complete.

    Returns: IDs of created subtasks and the continuation step.

    Note: This tool is hidden at max depth — you must complete atomically.
    """
    db = await get_db()
    result = await db.spawn_and_continue(
        step_id, partial_output, subtasks, continuation
    )
    return {
        "status": "spliced",
        "subtask_ids": result["subtask_ids"],
        "continuation_id": result["continuation_id"],
    }


@worker_tool()
async def abort_branch(
    target_id: int,
    output: str,
    reason: str,
    step_id: int | None = None,
) -> dict[str, Any]:
    """Early return from an ancestor step, skipping the rest of its branch.

    Use when you discover that an entire branch of reasoning is pointless —
    not just your step, but the ancestor that initiated it.

    The target step completes with your output (it doesn't fail). The target's
    parent (the requestor) sees the result and decides what to do next.

    Args:
        target_id: The ancestor step to return from.
        output: The result for the target step (e.g. "Approach debunked by X").
        reason: Why this branch is untenable. Saved as a campaign note.
        step_id: Your step ID. Inferred from session if omitted.

    Returns: List of skipped step IDs.
    """
    db = await get_db()
    if step_id is None:
        return {"error": "step_id is required (session inference not yet implemented)"}

    result = await db.abort_branch(step_id, target_id, output, reason)
    return {
        "status": "aborted",
        "target_id": result["target_id"],
        "skipped_count": len(result["skipped_ids"]),
        "skipped_ids": result["skipped_ids"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server (stdio transport)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
