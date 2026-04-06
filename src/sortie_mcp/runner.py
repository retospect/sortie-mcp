"""Campaign runner — cron entry point for sortie-mcp.

Runs as: ``sortie-runner`` (or ``python -m sortie_mcp.runner``)
Cron: ``*/15 * * * * /path/to/venv/bin/sortie-runner``

The runner is a watchdog, not an eager spawner:
1. Reset zombies (steps stuck in running past timeout)
2. Check capacity (are we at max_concurrent_steps?)
3. If slots free: find ready steps, dispatch up to capacity
4. If at capacity: just housekeeping, exit
5. Deliver pending notifications
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

import httpx

from .db import DB
from .models import (
    PRIORITY_ORDER,
    Campaign,
    CampaignStatus,
    FailurePolicy,
    NotificationLevel,
    Priority,
    Step,
    StepStatus,
    StepType,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT_STEPS = int(os.environ.get("SORTIE_MAX_CONCURRENT", "4"))
ZOMBIE_TIMEOUT_MINUTES = int(os.environ.get("SORTIE_ZOMBIE_TIMEOUT", "30"))
STUCK_CYCLES = int(os.environ.get("SORTIE_STUCK_CYCLES", "8"))
OPENCLAW_RUNTIME_URL = os.environ.get("OPENCLAW_RUNTIME_URL", "http://localhost:3000")
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "")
PLANNER_MODEL = os.environ.get("SORTIE_PLANNER_MODEL", "qwen3.5:9b")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    """Campaign runner — capacity-aware watchdog."""

    def __init__(self, db: DB) -> None:
        self.db = db
        self.http = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self.http.aclose()

    async def tick(self) -> None:
        """One cron tick. Called every 15 minutes."""
        log.info("Runner tick starting")

        # 1. Reset zombies
        zombies = await self.db.reset_zombies(ZOMBIE_TIMEOUT_MINUTES)
        if zombies:
            log.info("Reset %d zombie steps", zombies)

        # 2. Check capacity
        running = await self.db.count_running()
        available = MAX_CONCURRENT_STEPS - running
        log.info(
            "Capacity: %d/%d running, %d slots free",
            running,
            MAX_CONCURRENT_STEPS,
            available,
        )

        if available <= 0:
            # At capacity — just check for stuck campaigns
            await self._check_stuck_campaigns()
            await self._deliver_notifications()
            log.info("At capacity, tick done (housekeeping only)")
            return

        # 3. Get due campaigns sorted by priority
        campaigns = await self.db.get_due_campaigns()
        if not campaigns:
            log.info("No due campaigns, tick done")
            await self._deliver_notifications()
            return

        # 4. Allocate slots by priority tier (round-robin within tier)
        slots_remaining = available
        for priority in PRIORITY_ORDER:
            if slots_remaining <= 0:
                break
            tier_campaigns = [c for c in campaigns if c.priority == priority]
            if not tier_campaigns:
                continue

            # Tier allocation: urgent gets all, high gets 3/4, etc.
            tier_fraction = {
                Priority.URGENT: 1.0,
                Priority.HIGH: 0.75,
                Priority.NORMAL: 0.5,
                Priority.LOW: 0.25,
                Priority.BACKGROUND: 0.25,
            }[priority]
            tier_slots = max(1, math.ceil(slots_remaining * tier_fraction))
            per_campaign = max(1, math.ceil(tier_slots / len(tier_campaigns)))

            for campaign in tier_campaigns:
                if slots_remaining <= 0:
                    break
                dispatched = await self._dispatch_campaign(
                    campaign, min(per_campaign, slots_remaining)
                )
                slots_remaining -= dispatched

        # 5. Deliver notifications
        await self._deliver_notifications()
        log.info("Runner tick complete")

    async def _dispatch_campaign(self, campaign: Campaign, max_slots: int) -> int:
        """Dispatch ready steps for a campaign. Returns number dispatched."""
        ready = await self.db.get_ready_steps(campaign.id)

        if not ready:
            # No ready steps — consult planner for new steps
            log.info(
                "Campaign %s (%s): no ready steps, consulting planner",
                campaign.name,
                campaign.id,
            )
            await self._consult_planner(campaign)
            # Re-check after planner adds steps
            ready = await self.db.get_ready_steps(campaign.id)

        dispatched = 0
        for step in ready[:max_slots]:
            claimed = await self.db.claim_step(step.id)
            if claimed:
                log.info(
                    "Dispatching step %d: %s (agent=%s)",
                    step.id,
                    step.action[:80],
                    step.agent,
                )
                # Fire-and-forget dispatch to OpenClaw runtime
                task = asyncio.create_task(self._execute_step(claimed, campaign))
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
                dispatched += 1

        return dispatched

    async def _execute_step(self, step: Step, campaign: Campaign) -> None:
        """Execute a step via the OpenClaw runtime API."""
        try:
            # Build context envelope for the agent
            context = await self._build_step_context(step, campaign)

            # Dispatch to OpenClaw runtime
            response = await self.http.post(
                f"{OPENCLAW_RUNTIME_URL}/api/agent/execute",
                json={
                    "agent": step.agent or "research",
                    "prompt": context,
                    "step_id": step.id,
                    "campaign_id": str(campaign.id),
                    "tools": self._get_tools_for_depth(step.depth, campaign.max_depth),
                },
            )

            if response.status_code == 200:
                result = response.json()
                # Agent should call complete_step/fail_step via MCP,
                # but if it returns directly, handle it here
                if result.get("output") and not result.get("handled_by_mcp"):
                    await self.db.complete_step(
                        step.id,
                        result["output"],
                        tokens_used=result.get("tokens_used"),
                        duration_ms=result.get("duration_ms"),
                    )
            else:
                log.error(
                    "Runtime returned %d for step %d", response.status_code, step.id
                )
                await self.db.fail_step(
                    step.id, f"Runtime error: HTTP {response.status_code}"
                )

        except Exception as e:
            log.exception("Failed to execute step %d", step.id)
            await self.db.fail_step(step.id, f"Execution error: {e}")

    async def _build_step_context(self, step: Step, campaign: Campaign) -> str:
        """Build the context envelope prepended to the agent's system prompt."""
        # Get upstream outputs
        upstream_text = ""
        if step.depends_on:
            for dep_id in step.depends_on:
                dep = await self.db.get_step(dep_id)
                if dep and dep.output:
                    upstream_text += f"\n**{dep.action}**: {dep.output}\n"

        # Get relevant notes
        notes = await self.db.get_notes(campaign.id)
        notes_text = "\n".join(
            f"- [{', '.join(n.tags)}] {n.content}" for n in notes[:10]
        )

        depth_warning = ""
        if step.depth >= campaign.max_depth:
            depth_warning = (
                f"\n\n**You are at max depth ({step.depth}/{campaign.max_depth}). "
                "You must complete this task atomically — no subtasks available.**"
            )

        return f"""## Campaign context (sortie)
**Campaign**: {campaign.name or "Unnamed"}
**Goal**: {campaign.goal}
**Your task**: {step.action}

### Upstream context
{upstream_text or "No upstream outputs yet."}

### Relevant notes from other steps
{notes_text or "No notes yet."}

### Instructions
1. Do the work using your own tools (perplexity, precis, etc.)
2. Call add_note(content, tags) when you discover something noteworthy
3. Call complete_step({step.id}, summary) when done
4. Call fail_step({step.id}, error) if you can't complete the task
{self._spawn_instruction(step.depth, campaign.max_depth, step.id)}
5. Call abort_branch(target_id, output, reason) if you discover that
   an ancestor branch is pointless — target completes with your
   output, remaining siblings are skipped, and the requestor (one
   level above target) decides what to do next{depth_warning}
"""

    def _spawn_instruction(self, depth: int, max_depth: int, step_id: int) -> str:
        if depth < max_depth:
            return (
                f"5. Call spawn_and_continue({step_id}, partial_output, subtasks, continuation)\n"
                "   if you need additional work done before you can finish — the DAG\n"
                "   will rewire automatically and a continuation step will resume\n"
                "   your work once subtasks complete\n"
            )
        return ""

    def _get_tools_for_depth(self, depth: int, max_depth: int) -> list[str]:
        """Return tool names available at this depth. spawn_and_continue hidden at max depth."""
        tools = [
            "get_my_context",
            "add_note",
            "search_notes",
            "get_notes",
            "complete_step",
            "fail_step",
            "abort_branch",
        ]
        if depth < max_depth:
            tools.append("spawn_and_continue")
        return tools

    async def _consult_planner(self, campaign: Campaign) -> None:
        """Ask the planner LLM what to do next."""
        steps = await self.db.get_steps(campaign.id)
        notes = await self.db.get_notes(campaign.id)

        # Build planner prompt
        completed = [s for s in steps if s.status == StepStatus.DONE]
        failed = [s for s in steps if s.status == StepStatus.FAILED]
        pending = [s for s in steps if s.status == StepStatus.PENDING]

        prompt = f"""You are the campaign planner for sortie. Given a campaign's current
state, you decide what to do next.

## Campaign
**Goal**: {campaign.goal}
**Strategy**: {campaign.strategy or "Not yet defined"}
**Progress**: {campaign.progress or "Just started"}
**Max depth**: {campaign.max_depth}
**Tokens used**: {campaign.tokens_used}/{campaign.token_budget or "unlimited"}

## Recently completed steps
{self._format_steps(completed[-10:])}

## Failed steps
{self._format_steps(failed[-5:])}

## Pending steps
{self._format_steps(pending[:10])}

## Recent notes
{chr(10).join("- [" + ", ".join(n.tags) + "] " + n.content for n in notes[-10:])}

## Output (JSON)
Respond with ONLY a JSON object:
{{
  "new_steps": [
    {{"action": "...", "agent": "...", "depends_on": [], "step_type": "atomic"}},
    {{"action": "...", "step_type": "sequence", "steps": [...]}},
    {{"action": "...", "step_type": "parallel_group", "children": [...]}},
    {{"action": "...", "step_type": "for_each", "items": [...], "template": {{...}}, "collect": {{...}}}}
  ],
  "strategy_update": "revised strategy if needed",
  "progress_update": "updated summary",
  "notify": {{"message": "...", "level": "milestone|done|error|none"}},
  "done": false,
  "next_delay_minutes": 15
}}

## Rules
- Decompose into concrete, actionable steps.
- Max depth: {campaign.max_depth}. At max depth, do the work atomically.
- Prefer parallel when independent (5 papers = 5 parallel steps).
- Use sequences for pipelines (find → write → validate → keep/toss).
- For failures: retry, alternative, skip, or escalate. You decide.
- Set notify.level = "milestone" when a phase completes.
- Set notify.level = "error" when stuck and needs human attention.
- Token budget: {campaign.token_budget or "unlimited"} ({campaign.tokens_used} used). Wrap up if low.
"""

        try:
            response = await self.http.post(
                f"{LITELLM_URL}/v1/chat/completions",
                json={
                    "model": PLANNER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {LITELLM_KEY}"}
                if LITELLM_KEY
                else {},
            )

            if response.status_code != 200:
                log.error("Planner LLM returned %d", response.status_code)
                return

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            import json

            plan = json.loads(content)

            # Apply planner decisions
            if plan.get("strategy_update"):
                await self.db.update_campaign(
                    campaign.id, strategy=plan["strategy_update"]
                )
            if plan.get("progress_update"):
                await self.db.update_campaign(
                    campaign.id, progress=plan["progress_update"]
                )
            if plan.get("done"):
                await self.db.update_campaign(
                    campaign.id,
                    status=CampaignStatus.DONE,
                )
                if campaign.channel:
                    await self.db.notify(
                        campaign.id,
                        campaign.channel,
                        f"Campaign complete: {campaign.name or campaign.goal[:100]}",
                        level=NotificationLevel.DONE,
                    )
                return

            # Add new steps
            for step_plan in plan.get("new_steps", []):
                await self._add_planned_step(campaign, step_plan)

            # Handle notifications
            notify = plan.get("notify", {})
            if (
                notify.get("message")
                and notify.get("level", "none") != "none"
                and campaign.channel
            ):
                await self.db.notify(
                    campaign.id,
                    campaign.channel,
                    notify["message"],
                    level=NotificationLevel(notify["level"]),
                )

            # Schedule next action
            delay = plan.get("next_delay_minutes", 15)
            await self.db.pool.execute(
                f"UPDATE {self.db._t('campaigns')} SET next_action_at = now() + ($2 || ' minutes')::interval WHERE id = $1",
                campaign.id,
                str(delay),
            )

        except Exception:
            log.exception("Planner consultation failed for campaign %s", campaign.id)

    async def _add_planned_step(
        self,
        campaign: Campaign,
        plan: dict[str, Any],
        *,
        parent_id: int | None = None,
        depth: int = 0,
    ) -> int | None:
        """Recursively add a planned step and its children."""
        step_type = plan.get("step_type", "atomic")

        step = await self.db.add_step(
            campaign.id,
            plan["action"],
            agent=plan.get("agent"),
            step_type=StepType(step_type),
            parent_step_id=parent_id,
            depth=depth,
            depends_on=plan.get("depends_on", []),
            failure_policy=FailurePolicy(plan.get("failure_policy", "continue")),
            completion_threshold=plan.get("completion_threshold"),
        )

        # Expand children for compound types
        if step_type == "sequence":
            prev_id = None
            for child_plan in plan.get("steps", []):
                child_plan = dict(child_plan)
                if prev_id is not None:
                    child_plan.setdefault("depends_on", []).append(prev_id)
                child_id = await self._add_planned_step(
                    campaign, child_plan, parent_id=step.id, depth=depth + 1
                )
                prev_id = child_id

        elif step_type == "parallel_group":
            for child_plan in plan.get("children", []):
                await self._add_planned_step(
                    campaign, child_plan, parent_id=step.id, depth=depth + 1
                )

        elif step_type == "for_each":
            items = plan.get("items", [])
            template = plan.get("template", {})
            collect = plan.get("collect")
            child_ids = []

            for item in items:
                # Expand template for this item
                expanded = self._expand_template(template, item)
                child_id = await self._add_planned_step(
                    campaign, expanded, parent_id=step.id, depth=depth + 1
                )
                if child_id:
                    child_ids.append(child_id)

            # Add collect step if specified
            if collect and child_ids:
                collect = dict(collect)
                collect["depends_on"] = child_ids
                await self._add_planned_step(
                    campaign, collect, parent_id=step.id, depth=depth + 1
                )

        return step.id

    def _expand_template(
        self, template: dict[str, Any], item: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace {item.*} placeholders in a template with actual item values."""
        import json

        text = json.dumps(template)
        for key, value in item.items():
            text = text.replace(f"{{item.{key}}}", str(value))
        return json.loads(text)

    def _format_steps(self, steps: list[Step]) -> str:
        if not steps:
            return "None"
        return "\n".join(
            f"- Step {s.id} [{s.status.value}] ({s.agent}): {s.action[:120]}"
            + (f"\n  Output: {s.output[:200]}" if s.output else "")
            + (f"\n  Error: {s.error}" if s.error else "")
            for s in steps
        )

    async def _check_stuck_campaigns(self) -> None:
        """Check for campaigns that haven't progressed in too long."""
        # TODO: track progress cycles and alert after STUCK_CYCLES
        pass

    async def _deliver_notifications(self) -> None:
        """Deliver pending notifications."""
        notifications = await self.db.get_undelivered_notifications()
        if not notifications:
            return

        delivered_ids = []
        for notif in notifications:
            log.info(
                "Notification [%s] campaign=%s: %s",
                notif.level.value,
                notif.campaign_id,
                notif.message[:100],
            )
            # TODO: deliver via PG NOTIFY → gateway → Discord
            # For now, just mark as delivered
            delivered_ids.append(notif.id)

        if delivered_ids:
            await self.db.mark_delivered(delivered_ids)
            log.info("Delivered %d notifications", len(delivered_ids))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run() -> None:
    dsn = os.environ.get("DATABASE_URL", "postgresql://localhost/sortie")
    schema = os.environ.get("SORTIE_SCHEMA", "sortie")

    db = DB(dsn, schema=schema)
    await db.connect()
    await db.migrate()

    runner = Runner(db)
    try:
        await runner.tick()
    finally:
        await runner.close()
        await db.close()


def main() -> None:
    """Run one tick of the campaign runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
