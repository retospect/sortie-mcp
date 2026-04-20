"""Scenario 18: User input request/provide lifecycle.

Tests the full flow: agent requests input → step pauses → coordinator
provides answer → step resumes → agent sees the answer in context.
"""

from __future__ import annotations

import pytest

from sortie_mcp.db import DB
from sortie_mcp.models import StepStatus

pytestmark = pytest.mark.postgres


class TestUserInputLifecycle:
    """Full request → provide → resume cycle."""

    async def test_request_pauses_step(self, db: DB) -> None:
        """Agent requests input, step transitions to waiting_input."""
        campaign = await db.create_campaign("Input test", name="Input")
        step = await db.add_step(campaign.id, "Needs decision", agent="research")

        await db.claim_step(step.id)
        result = await db.request_input(
            step.id,
            "Should I use approach A or B?",
            partial_output="Found 3 relevant papers so far.",
        )

        assert result is not None
        assert result.status == StepStatus.WAITING_INPUT
        assert "approach A or B" in result.output
        assert "Found 3 relevant papers" in result.output

    async def test_waiting_step_not_in_ready_list(self, db: DB) -> None:
        """A waiting_input step is not returned by get_ready_steps."""
        campaign = await db.create_campaign("Ready test", name="Ready")
        step = await db.add_step(campaign.id, "Needs input", agent="research")

        await db.claim_step(step.id)
        await db.request_input(step.id, "Which approach?")

        ready = await db.get_ready_steps(campaign.id)
        assert step.id not in [s.id for s in ready]

    async def test_provide_input_resumes_step(self, db: DB) -> None:
        """Coordinator provides input, step returns to pending."""
        campaign = await db.create_campaign("Resume test", name="Resume")
        step = await db.add_step(campaign.id, "Needs input", agent="research")

        await db.claim_step(step.id)
        await db.request_input(step.id, "Which approach?")

        result = await db.provide_input(step.id, "Use approach B")
        assert result is not None
        assert result.status == StepStatus.PENDING
        assert result.input == "Use approach B"
        assert result.started_at is None  # reset for re-claim

    async def test_resumed_step_is_ready(self, db: DB) -> None:
        """After provide_input, the step appears in get_ready_steps."""
        campaign = await db.create_campaign("Requeue test", name="Requeue")
        step = await db.add_step(campaign.id, "Needs input", agent="research")

        await db.claim_step(step.id)
        await db.request_input(step.id, "Which approach?")
        await db.provide_input(step.id, "Use approach B")

        ready = await db.get_ready_steps(campaign.id)
        assert step.id in [s.id for s in ready]

    async def test_full_cycle_claim_complete(self, db: DB) -> None:
        """Full cycle: claim → request → provide → re-claim → complete."""
        campaign = await db.create_campaign("Full cycle", name="FullCycle")
        step = await db.add_step(campaign.id, "Research task", agent="research")

        # First dispatch: agent hits a fork
        await db.claim_step(step.id)
        await db.request_input(
            step.id,
            "Found two competing theories. Which to pursue?",
            partial_output="Literature review complete.",
        )

        # Coordinator answers
        await db.provide_input(step.id, "Pursue theory A — more recent evidence")

        # Second dispatch: agent sees the answer and completes
        claimed = await db.claim_step(step.id)
        assert claimed is not None
        assert claimed.status == StepStatus.RUNNING
        assert claimed.input == "Pursue theory A — more recent evidence"

        await db.complete_step(step.id, "Completed using theory A")
        final = await db.get_step(step.id)
        assert final.status == StepStatus.DONE
        assert final.output == "Completed using theory A"

    async def test_provide_wrong_status_returns_none(self, db: DB) -> None:
        """provide_input on a non-waiting step returns None."""
        campaign = await db.create_campaign("Wrong status", name="WrongStatus")
        step = await db.add_step(campaign.id, "Normal step", agent="research")

        # Step is pending, not waiting_input
        result = await db.provide_input(step.id, "Unsolicited answer")
        assert result is None

    async def test_request_wrong_status_returns_none(self, db: DB) -> None:
        """request_input on a non-running step returns None."""
        campaign = await db.create_campaign("Not running", name="NotRunning")
        step = await db.add_step(campaign.id, "Pending step", agent="research")

        # Step is pending, not running
        result = await db.request_input(step.id, "Question?")
        assert result is None

    async def test_waiting_step_visible_in_campaign_state(self, db: DB) -> None:
        """get_steps shows the waiting_input status for visibility."""
        campaign = await db.create_campaign("Visible test", name="Visible")
        step = await db.add_step(campaign.id, "Needs input", agent="research")

        await db.claim_step(step.id)
        await db.request_input(step.id, "Need guidance on scope")

        steps = await db.get_steps(campaign.id)
        waiting = [s for s in steps if s.status == StepStatus.WAITING_INPUT]
        assert len(waiting) == 1
        assert waiting[0].id == step.id
        assert "guidance on scope" in waiting[0].output

    async def test_input_with_dependencies(self, db: DB) -> None:
        """Step with deps: request input doesn't break dependency chain."""
        campaign = await db.create_campaign("Dep input", name="DepInput")
        a = await db.add_step(campaign.id, "Step A", agent="research")
        b = await db.add_step(campaign.id, "Step B", agent="writing", depends_on=[a.id])

        # Complete A normally
        await db.claim_step(a.id)
        await db.complete_step(a.id, "A output")

        # B is ready, claim it, then request input
        await db.claim_step(b.id)
        await db.request_input(b.id, "How to incorporate A's findings?")

        # B is waiting — not ready, not blocking anything
        ready = await db.get_ready_steps(campaign.id)
        assert b.id not in [s.id for s in ready]

        # Provide input → B is ready again
        await db.provide_input(b.id, "Summarize and cite")
        ready = await db.get_ready_steps(campaign.id)
        assert b.id in [s.id for s in ready]
