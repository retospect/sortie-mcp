"""Scenario 8: Branch abort with skip cascade.

Tests that abort_branch correctly marks the target as done, skips
descendants, propagates through the depends_on graph, and respects
boundary conditions (siblings of target are unaffected).
"""

from __future__ import annotations

import pytest

from sortie_mcp.db import DB
from sortie_mcp.models import StepStatus, StepType

pytestmark = pytest.mark.postgres


class TestBranchAbort:
    """Abort a branch and verify cascade propagation."""

    async def test_abort_skips_descendants_and_cascades(self, db: DB) -> None:
        """
        Structure:
          Parent P (parallel_group)
            ├── A (parallel_group)
            │   ├── A1  ← discoverer (aborts targeting A)
            │   └── A2
            ├── B
            └── C

          D depends_on=[A2]  (downstream of a skipped step)

        After abort_branch(step_id=A1, target_id=A):
          - A1 → done (discoverer)
          - A  → done (target, with provided output)
          - A2 → skipped (descendant of target)
          - B, C → NOT skipped (siblings of target under P)
          - D  → skipped (depends on A2 which was skipped)
        """
        campaign = await db.create_campaign("Abort test", name="Abort")

        p = await db.add_step(
            campaign.id, "Parent", step_type=StepType.PARALLEL_GROUP
        )
        a = await db.add_step(
            campaign.id,
            "Branch A",
            parent_step_id=p.id,
            step_type=StepType.PARALLEL_GROUP,
        )
        a1 = await db.add_step(
            campaign.id, "A1 discoverer", parent_step_id=a.id, agent="research"
        )
        a2 = await db.add_step(
            campaign.id, "A2 sibling", parent_step_id=a.id, agent="research"
        )
        b = await db.add_step(
            campaign.id, "Branch B", parent_step_id=p.id, agent="writing"
        )
        c = await db.add_step(
            campaign.id, "Branch C", parent_step_id=p.id, agent="writing"
        )
        d = await db.add_step(
            campaign.id, "D downstream", depends_on=[a2.id], agent="research"
        )

        # Claim A1 so it's running (agents can only abort from running steps)
        await db.claim_step(a1.id)

        # Abort: A1 discovers that branch A is pointless
        result = await db.abort_branch(
            step_id=a1.id,
            target_id=a.id,
            output="Approach debunked by Miller 2001",
            reason="Miller 2001 disproves the hypothesis",
        )

        # Verify result
        assert result["target_id"] == a.id
        assert a2.id in result["skipped_ids"]

        # A1 (discoverer) → done
        a1_step = await db.get_step(a1.id)
        assert a1_step.status == StepStatus.DONE

        # A (target) → done with provided output
        a_step = await db.get_step(a.id)
        assert a_step.status == StepStatus.DONE
        assert "Approach debunked" in a_step.output

        # A2 → skipped
        a2_step = await db.get_step(a2.id)
        assert a2_step.status == StepStatus.SKIPPED

        # B, C → NOT skipped (siblings of target, not descendants)
        b_step = await db.get_step(b.id)
        c_step = await db.get_step(c.id)
        assert b_step.status == StepStatus.PENDING
        assert c_step.status == StepStatus.PENDING

        # D → skipped (depends on A2 which was skipped, transitive cascade)
        d_step = await db.get_step(d.id)
        assert d_step.status == StepStatus.SKIPPED

        # A note was created with the reason
        notes = await db.get_notes(campaign.id, tags=["abort"])
        assert len(notes) >= 1
        assert "Miller 2001" in notes[0].content


class TestAbortValidation:
    """Guard clauses prevent invalid abort operations."""

    async def test_cannot_abort_done_target(self, db: DB) -> None:
        """Targeting a done step raises ValueError."""
        campaign = await db.create_campaign("Guard test", name="Guard")
        parent = await db.add_step(campaign.id, "Parent")
        child = await db.add_step(
            campaign.id, "Child", parent_step_id=parent.id
        )

        # Complete the parent
        await db.claim_step(parent.id)
        await db.complete_step(parent.id, "Done")

        # Child tries to abort targeting done parent
        await db.claim_step(child.id)
        with pytest.raises(ValueError, match="already done"):
            await db.abort_branch(child.id, parent.id, "out", "reason")

    async def test_cannot_abort_cross_campaign(self, db: DB) -> None:
        """Step and target must be in the same campaign."""
        c1 = await db.create_campaign("C1")
        c2 = await db.create_campaign("C2")
        s1 = await db.add_step(c1.id, "Step in C1")
        s2 = await db.add_step(c2.id, "Step in C2")

        await db.claim_step(s1.id)
        with pytest.raises(ValueError, match="same campaign"):
            await db.abort_branch(s1.id, s2.id, "out", "reason")

    async def test_cannot_abort_non_ancestor(self, db: DB) -> None:
        """Step must be a descendant of target."""
        campaign = await db.create_campaign("Ancestry test")
        a = await db.add_step(campaign.id, "A")
        b = await db.add_step(campaign.id, "B")  # no parent relationship

        await db.claim_step(a.id)
        with pytest.raises(ValueError, match="not a descendant"):
            await db.abort_branch(a.id, b.id, "out", "reason")
