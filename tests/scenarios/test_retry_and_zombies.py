"""Scenarios 11 & 12: Retry logic and zombie reset.

Tests that retry_count increments correctly, failed steps are excluded
from ready queries, and zombie steps (stuck in running) are reset.
"""

from __future__ import annotations

import pytest

from sortie_mcp.db import DB
from sortie_mcp.models import FailurePolicy, StepStatus

pytestmark = pytest.mark.postgres


class TestRetryLogic:
    """Steps retry up to max_retries, then fail permanently."""

    async def test_retry_cycle(self, db: DB) -> None:
        """Fail a step 3 times: first 2 return to pending, 3rd is permanent."""
        campaign = await db.create_campaign("Retry test", name="Retry")
        step = await db.add_step(campaign.id, "Flaky step", agent="research")

        # Fail #1 — retry_count=1, status=pending
        await db.claim_step(step.id)
        s = await db.fail_step(step.id, "Timeout #1")
        assert s.status == StepStatus.PENDING
        assert s.retry_count == 1

        # Step is still ready (pending, retry_count < max_retries=3)
        ready = await db.get_ready_steps(campaign.id)
        assert step.id in [r.id for r in ready]

        # Fail #2 — retry_count=2, status=pending
        await db.claim_step(step.id)
        s = await db.fail_step(step.id, "Timeout #2")
        assert s.status == StepStatus.PENDING
        assert s.retry_count == 2

        # Fail #3 — retry_count=3 >= max_retries=3 → permanent failure
        await db.claim_step(step.id)
        s = await db.fail_step(step.id, "Timeout #3")
        assert s.status == StepStatus.FAILED
        assert s.retry_count == 3

        # Failed step is NOT in ready list
        ready = await db.get_ready_steps(campaign.id)
        assert step.id not in [r.id for r in ready]

    async def test_fail_fast_cascades_on_permanent_failure(self, db: DB) -> None:
        """A fail_fast step that exhausts retries cascades failure."""
        campaign = await db.create_campaign("Fail fast", name="FailFast")
        a = await db.add_step(
            campaign.id,
            "Critical step",
            agent="research",
            failure_policy=FailurePolicy.FAIL_FAST,
        )
        b = await db.add_step(
            campaign.id, "Depends on A", depends_on=[a.id], agent="writing"
        )

        # Exhaust retries
        for i in range(3):
            await db.claim_step(a.id)
            await db.fail_step(a.id, f"Error #{i + 1}")

        a_step = await db.get_step(a.id)
        assert a_step.status == StepStatus.FAILED

        # Campaign should be failed (fail_fast on top-level step)
        c = await db.get_campaign(campaign.id)
        from sortie_mcp.models import CampaignStatus

        assert c.status == CampaignStatus.FAILED


class TestZombieReset:
    """Steps stuck in running past timeout are reset to pending."""

    async def test_zombie_reset(self, db: DB) -> None:
        """Backdate started_at, then reset_zombies reclaims the step."""
        campaign = await db.create_campaign("Zombie test", name="Zombie")
        step = await db.add_step(campaign.id, "Stuck step", agent="research")

        # Claim it
        await db.claim_step(step.id)
        s = await db.get_step(step.id)
        assert s.status == StepStatus.RUNNING

        # Backdate heartbeat_at (the staleness signal since migration 0002)
        # and started_at together by 31 minutes.
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET started_at = now() - interval '31 minutes', "
            f"    heartbeat_at = now() - interval '31 minutes' "
            f"WHERE id = $1",
            step.id,
        )

        # Reset zombies with 30-minute timeout
        count = await db.reset_zombies(30)
        assert count == 1

        # Step is back to pending
        s = await db.get_step(step.id)
        assert s.status == StepStatus.PENDING
        assert s.started_at is None

        # Step is ready again
        ready = await db.get_ready_steps(campaign.id)
        assert step.id in [r.id for r in ready]

    async def test_zombie_reset_ignores_fresh_running(self, db: DB) -> None:
        """Steps that started recently are NOT zombies."""
        campaign = await db.create_campaign("Fresh test", name="Fresh")
        step = await db.add_step(campaign.id, "Fresh step", agent="research")
        await db.claim_step(step.id)

        count = await db.reset_zombies(30)
        assert count == 0

        s = await db.get_step(step.id)
        assert s.status == StepStatus.RUNNING

    async def test_zombie_reset_ignores_waiting_input(self, db: DB) -> None:
        """Steps in waiting_input are NOT reset as zombies."""
        campaign = await db.create_campaign("Waiting test", name="Waiting")
        step = await db.add_step(campaign.id, "Needs input", agent="research")
        await db.claim_step(step.id)
        await db.request_input(step.id, "Which approach?")

        # Backdate started_at
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} SET started_at = now() - interval '60 minutes' WHERE id = $1",
            step.id,
        )

        # Zombie reset should NOT touch it
        count = await db.reset_zombies(30)
        assert count == 0

        s = await db.get_step(step.id)
        assert s.status == StepStatus.WAITING_INPUT
