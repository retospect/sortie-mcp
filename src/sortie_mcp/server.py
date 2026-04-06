"""sortie-mcp MCP server — campaign orchestration tools for AI agents.

Three perspectives on one server:
- Coordinator (Asa): create, list, get, steer, pause/resume/cancel campaigns
- Worker (agents): get_my_context, add_note, search_notes, complete_step, etc.
- Executor tools are Python internal API (see db.py), not MCP-exposed.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from .db import DB
from .models import (
    CampaignStatus,
    Priority,
    StepStatus,
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
# Coordinator tools (Asa)
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_campaign(
    goal: str,
    name: str | None = None,
    channel: str | None = None,
    priority: str = "normal",
    max_depth: int = 4,
    token_budget: int | None = None,
    dry_run: bool = False,
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
    )
    return {
        "id": str(campaign.id),
        "name": campaign.name,
        "status": campaign.status.value,
        "priority": campaign.priority.value,
        "next_action_at": str(campaign.next_action_at),
    }


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def get_updates(id: str | None = None) -> dict[str, Any]:
    """Get delta since last report: completed steps, failures, new notes.

    Args:
        id: Campaign UUID. Omit for updates across all active campaigns.

    Returns: Recent completions, failures, and notes.
    """
    db = await get_db()
    if id:
        campaigns = [await db.get_campaign(UUID(id))]
        campaigns = [c for c in campaigns if c]
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


@mcp.tool()
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


@mcp.tool()
async def pause_campaign(id: str) -> dict[str, str]:
    """Pause a campaign. Running steps finish but no new ones start.

    Args:
        id: Campaign UUID.
    """
    db = await get_db()
    await db.update_campaign(UUID(id), status=CampaignStatus.PAUSED)
    return {"id": id, "status": "paused"}


@mcp.tool()
async def resume_campaign(id: str) -> dict[str, str]:
    """Resume a paused campaign.

    Args:
        id: Campaign UUID.
    """
    db = await get_db()
    await db.update_campaign(UUID(id), status=CampaignStatus.ACTIVE)
    return {"id": id, "status": "active"}


@mcp.tool()
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


@mcp.tool()
async def get_my_context(step_id: int) -> dict[str, Any]:
    """Get campaign context for the step you're executing.

    Args:
        step_id: Your step ID (provided in your prompt).

    Returns: Campaign goal, your task, upstream outputs, relevant notes.

    Next: Do your work, then call `complete_step(step_id, summary)` or `fail_step(step_id, error)`.
    """
    db = await get_db()
    step = await db.get_step(step_id)
    if not step:
        return {"error": f"Step {step_id} not found"}

    campaign = await db.get_campaign(step.campaign_id)
    if not campaign:
        return {"error": "Campaign not found"}

    # Get upstream outputs
    upstream = []
    if step.depends_on:
        for dep_id in step.depends_on:
            dep = await db.get_step(dep_id)
            if dep and dep.output:
                upstream.append(
                    {
                        "step_id": dep.id,
                        "action": dep.action,
                        "output": dep.output,
                    }
                )

    # Get relevant notes (most recent)
    notes = await db.get_notes(step.campaign_id)
    recent_notes = notes[:10]

    return {
        "campaign_name": campaign.name,
        "campaign_goal": campaign.goal,
        "your_task": step.action,
        "your_step_id": step.id,
        "depth": step.depth,
        "max_depth": campaign.max_depth,
        "upstream_context": upstream,
        "relevant_notes": [
            {"content": n.content, "tags": n.tags, "agent": n.agent}
            for n in recent_notes
        ],
    }


@mcp.tool()
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
    # TODO: generate embedding via LiteLLM embedding endpoint
    note = await db.add_note(cid, content, tags=tags)
    return {
        "note_id": note.id,
        "recorded": True,
    }


@mcp.tool()
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

    Returns: Ranked results with content and tags.
    """
    # TODO: generate query embedding via LiteLLM embedding endpoint
    # For now, return notes by recency as fallback
    db = await get_db()
    if campaign_id:
        notes = await db.get_notes(UUID(campaign_id))
    else:
        notes = []
    return [
        {"id": n.id, "content": n.content, "tags": n.tags, "agent": n.agent}
        for n in notes[:top_k]
    ]


@mcp.tool()
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


@mcp.tool()
async def complete_step(step_id: int, summary: str) -> dict[str, Any]:
    """Mark your step as done with a summary of what you accomplished.

    Args:
        step_id: Your step ID.
        summary: What you did and what you found. This becomes the step output
                 visible to downstream steps.

    Returns: Confirmation. If the step was already skipped (branch abort),
             returns {status: "skipped"} — your output is recorded for audit.
    """
    db = await get_db()
    step = await db.complete_step(step_id, summary)
    if not step:
        return {"error": f"Step {step_id} not found"}

    if step.status == StepStatus.SKIPPED:
        return {
            "status": "skipped",
            "reason": step.error or "Branch aborted — output recorded for audit",
            "output_recorded": True,
        }

    return {"status": "done", "step_id": step_id}


@mcp.tool()
async def fail_step(step_id: int, error: str) -> dict[str, Any]:
    """Report that you cannot complete your step.

    Args:
        step_id: Your step ID.
        error: What went wrong and why you can't continue.

    Returns: Whether the step can be retried or has failed permanently.
    """
    db = await get_db()
    step = await db.fail_step(step_id, error)
    if not step:
        return {"error": f"Step {step_id} not found"}

    return {
        "status": step.status.value,
        "retry_count": step.retry_count,
        "max_retries": step.max_retries,
        "can_retry": step.retry_count < step.max_retries,
    }


@mcp.tool()
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


@mcp.tool()
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
