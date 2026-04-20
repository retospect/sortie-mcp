"""Scenario 1: Linear campaign happy path.

Create campaign → add sequential steps → claim/complete through the chain →
verify readiness propagates correctly through the dependency graph.
"""

from __future__ import annotations

import pytest

from sortie_mcp.db import DB
from sortie_mcp.models import (
    StepStatus,
    StepType,
)

pytestmark = pytest.mark.postgres


class TestLinearCampaignHappyPath:
    """A → B → C linear chain: only the head is ready at any time."""

    async def test_full_chain(self, db: DB) -> None:
        # --- Setup ---
        campaign = await db.create_campaign("Test linear chain", name="Linear")
        a = await db.add_step(campaign.id, "Step A", agent="research")
        b = await db.add_step(campaign.id, "Step B", agent="writing", depends_on=[a.id])
        c = await db.add_step(
            campaign.id, "Step C", agent="research", depends_on=[b.id]
        )

        # --- Act 1: Only A is ready ---
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [a.id]

        # --- Act 2: Claim and complete A ---
        claimed = await db.claim_step(a.id)
        assert claimed is not None
        assert claimed.status == StepStatus.RUNNING

        # B still not ready (A is running, not done)
        ready = await db.get_ready_steps(campaign.id)
        assert ready == []

        await db.complete_step(a.id, "A done")

        # --- Act 3: B is now ready ---
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [b.id]

        # C is not ready
        c_step = await db.get_step(c.id)
        assert c_step.status == StepStatus.PENDING

        # --- Act 4: Complete B, C becomes ready ---
        await db.claim_step(b.id)
        await db.complete_step(b.id, "B done")
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [c.id]

        # --- Act 5: Complete C ---
        await db.claim_step(c.id)
        await db.complete_step(c.id, "C done")

        # All done
        ready = await db.get_ready_steps(campaign.id)
        assert ready == []

        # Verify final states
        for step_id, expected_output in [
            (a.id, "A done"),
            (b.id, "B done"),
            (c.id, "C done"),
        ]:
            step = await db.get_step(step_id)
            assert step.status == StepStatus.DONE
            assert step.output == expected_output

    async def test_parallel_independence(self, db: DB) -> None:
        """Two steps with no deps are both immediately ready."""
        campaign = await db.create_campaign("Parallel test", name="Parallel")
        a = await db.add_step(campaign.id, "Step A", agent="research")
        b = await db.add_step(campaign.id, "Step B", agent="writing")

        ready = await db.get_ready_steps(campaign.id)
        ready_ids = {s.id for s in ready}
        assert ready_ids == {a.id, b.id}

    async def test_diamond_dependency(self, db: DB) -> None:
        """Diamond: A → B, A → C, B+C → D. D waits for both."""
        campaign = await db.create_campaign("Diamond", name="Diamond")
        a = await db.add_step(campaign.id, "A")
        b = await db.add_step(campaign.id, "B", depends_on=[a.id])
        c = await db.add_step(campaign.id, "C", depends_on=[a.id])
        d = await db.add_step(campaign.id, "D", depends_on=[b.id, c.id])

        # Only A ready
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [a.id]

        # Complete A → B and C ready
        await db.claim_step(a.id)
        await db.complete_step(a.id, "A")
        ready = await db.get_ready_steps(campaign.id)
        assert {s.id for s in ready} == {b.id, c.id}

        # Complete B only → D not ready yet
        await db.claim_step(b.id)
        await db.complete_step(b.id, "B")
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [c.id]  # only C

        # Complete C → D ready
        await db.claim_step(c.id)
        await db.complete_step(c.id, "C")
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [d.id]


class TestParallelGroupCompletion:
    """Parent auto-completes when children meet the threshold."""

    async def test_threshold_completion(self, db: DB) -> None:
        """Parent with threshold=2 completes after 2 of 3 children."""
        campaign = await db.create_campaign("Threshold", name="Threshold")
        parent = await db.add_step(
            campaign.id,
            "Parent group",
            step_type=StepType.PARALLEL_GROUP,
            completion_threshold=2,
        )
        c1 = await db.add_step(campaign.id, "Child 1", parent_step_id=parent.id)
        c2 = await db.add_step(campaign.id, "Child 2", parent_step_id=parent.id)
        c3 = await db.add_step(campaign.id, "Child 3", parent_step_id=parent.id)

        # Complete c1 — parent still pending
        await db.claim_step(c1.id)
        await db.complete_step(c1.id, "Child 1 done")
        p = await db.get_step(parent.id)
        assert p.status == StepStatus.PENDING

        # Complete c2 — threshold met, parent auto-completes
        await db.claim_step(c2.id)
        await db.complete_step(c2.id, "Child 2 done")
        p = await db.get_step(parent.id)
        assert p.status == StepStatus.DONE
        assert "Child 1 done" in p.output
        assert "Child 2 done" in p.output

    async def test_all_required_when_no_threshold(self, db: DB) -> None:
        """Without threshold, all children must complete."""
        campaign = await db.create_campaign("All required", name="AllReq")
        parent = await db.add_step(
            campaign.id,
            "Parent",
            step_type=StepType.PARALLEL_GROUP,
        )
        c1 = await db.add_step(campaign.id, "Child 1", parent_step_id=parent.id)
        c2 = await db.add_step(campaign.id, "Child 2", parent_step_id=parent.id)

        await db.claim_step(c1.id)
        await db.complete_step(c1.id, "Done 1")

        p = await db.get_step(parent.id)
        assert p.status == StepStatus.PENDING  # 1 of 2

        await db.claim_step(c2.id)
        await db.complete_step(c2.id, "Done 2")

        p = await db.get_step(parent.id)
        assert p.status == StepStatus.DONE
