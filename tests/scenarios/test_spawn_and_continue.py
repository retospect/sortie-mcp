"""Scenario 5: Spawn-and-continue with DAG retargeting.

Tests the atomic DAG splice: a running step spawns subtasks + continuation,
downstream dependencies are retargeted to the continuation, and the chain
resolves correctly through canonical resolution.
"""

from __future__ import annotations

import pytest

from sortie_mcp.db import DB
from sortie_mcp.models import StepStatus

pytestmark = pytest.mark.postgres


class TestSpawnAndContinue:
    """Core DAG splice: spawn subtasks, create continuation, retarget deps."""

    async def test_basic_splice(self, db: DB) -> None:
        """A→B chain. A spawns S1,S2 + continuation. B retargeted to cont."""
        campaign = await db.create_campaign("Splice test", name="Splice")
        a = await db.add_step(campaign.id, "Step A", agent="research")
        b = await db.add_step(
            campaign.id, "Step B", agent="writing", depends_on=[a.id]
        )

        # Claim A so it's running
        await db.claim_step(a.id)

        # Spawn subtasks + continuation
        result = await db.spawn_and_continue(
            a.id,
            "Partial output from A",
            [
                {"action": "Subtask S1", "agent": "research"},
                {"action": "Subtask S2", "agent": "research"},
            ],
            "Continue A after subtasks",
        )

        s1_id, s2_id = result["subtask_ids"]
        cont_id = result["continuation_id"]

        # A is now done with partial output
        a_step = await db.get_step(a.id)
        assert a_step.status == StepStatus.DONE
        assert a_step.output == "Partial output from A"

        # Subtasks exist at depth+1
        s1 = await db.get_step(s1_id)
        s2 = await db.get_step(s2_id)
        assert s1.depth == a_step.depth + 1
        assert s2.depth == a_step.depth + 1
        assert s1.parent_step_id == a.id
        assert s2.parent_step_id == a.id
        assert s1.status == StepStatus.PENDING
        assert s2.status == StepStatus.PENDING

        # Continuation at same depth as A, continuation_of = A
        cont = await db.get_step(cont_id)
        assert cont.depth == a_step.depth
        assert cont.continuation_of == a.id
        assert cont.parent_step_id == a_step.parent_step_id
        assert set(cont.depends_on) == {s1_id, s2_id}

        # B now depends on continuation, not A
        b_step = await db.get_step(b.id)
        assert cont_id in b_step.depends_on
        assert a.id not in b_step.depends_on

        # Subtasks are ready (A is done)
        ready = await db.get_ready_steps(campaign.id)
        ready_ids = {s.id for s in ready}
        assert s1_id in ready_ids
        assert s2_id in ready_ids
        assert cont_id not in ready_ids  # cont waits for subtasks

    async def test_splice_chain_resolves(self, db: DB) -> None:
        """Complete the full chain: subtasks → continuation → B."""
        campaign = await db.create_campaign("Chain test", name="Chain")
        a = await db.add_step(campaign.id, "Step A", agent="research")
        b = await db.add_step(
            campaign.id, "Step B", agent="writing", depends_on=[a.id]
        )

        await db.claim_step(a.id)
        result = await db.spawn_and_continue(
            a.id,
            "Partial",
            [{"action": "S1"}, {"action": "S2"}],
            "Continue A",
        )

        s1_id, s2_id = result["subtask_ids"]
        cont_id = result["continuation_id"]

        # Complete subtasks
        await db.claim_step(s1_id)
        await db.complete_step(s1_id, "S1 output")
        await db.claim_step(s2_id)
        await db.complete_step(s2_id, "S2 output")

        # Continuation is now ready
        ready = await db.get_ready_steps(campaign.id)
        assert cont_id in [s.id for s in ready]

        # Complete continuation → B becomes ready
        await db.claim_step(cont_id)
        await db.complete_step(cont_id, "Continuation done with S1+S2 merged")
        ready = await db.get_ready_steps(campaign.id)
        assert [s.id for s in ready] == [b.id]

    async def test_depth_limit_prevents_spawn(self, db: DB) -> None:
        """Spawn at max depth raises ValueError."""
        campaign = await db.create_campaign(
            "Depth limit", name="DepthLimit", max_depth=1
        )
        a = await db.add_step(
            campaign.id, "Step A", agent="research", depth=1
        )
        await db.claim_step(a.id)

        with pytest.raises(ValueError, match="Depth limit"):
            await db.spawn_and_continue(
                a.id,
                "Partial",
                [{"action": "Sub"}],
                "Continue",
            )


class TestCanonicalResolution:
    """Dependency resolution through continuation chains."""

    async def test_resolve_through_continuation(self, db: DB) -> None:
        """A spawns → C1. New step D depends_on=[A] → resolved to C1."""
        campaign = await db.create_campaign("Resolution", name="Resolve")
        a = await db.add_step(campaign.id, "A", agent="research")

        await db.claim_step(a.id)
        result = await db.spawn_and_continue(
            a.id,
            "Partial",
            [{"action": "S1"}],
            "Continuation of A",
        )
        cont_id = result["continuation_id"]

        # Add new step that depends on the original A
        d = await db.add_step(
            campaign.id, "D depends on A", depends_on=[a.id]
        )

        # D's depends_on should be resolved to the continuation
        d_step = await db.get_step(d.id)
        assert cont_id in d_step.depends_on
