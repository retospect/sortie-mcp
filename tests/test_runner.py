"""Tests for sortie_mcp.runner — campaign runner logic with mocked DB."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from sortie_mcp.models import (
    Campaign,
    CampaignStatus,
    FailurePolicy,
    Note,
    Notification,
    NotificationLevel,
    Priority,
    Step,
    StepStatus,
    StepType,
    PRIORITY_ORDER,
)
from sortie_mcp.runner import Runner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_campaign(**overrides) -> Campaign:
    defaults = dict(
        id=uuid4(),
        name="Test Campaign",
        goal="Test goal",
        status=CampaignStatus.ACTIVE,
        priority=Priority.NORMAL,
        max_depth=4,
        token_budget=None,
        tokens_used=0,
        failure_policy=FailurePolicy.CONTINUE,
        channel="research",
    )
    defaults.update(overrides)
    return Campaign(**defaults)


def make_step(campaign_id=None, **overrides) -> Step:
    defaults = dict(
        id=1,
        campaign_id=campaign_id or uuid4(),
        action="Test step",
        step_type=StepType.ATOMIC,
        status=StepStatus.PENDING,
        depth=0,
        agent="research",
        depends_on=[],
        failure_policy=FailurePolicy.CONTINUE,
        retry_count=0,
        max_retries=3,
    )
    defaults.update(overrides)
    return Step(**defaults)


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.schema = "sortie"
    db._t = MagicMock(side_effect=lambda t: f"sortie.{t}")
    db.pool = AsyncMock()
    return db


@pytest.fixture
def runner(mock_db):
    r = Runner(mock_db)
    r.http = AsyncMock()
    return r


# ---------------------------------------------------------------------------
# Runner.tick — watchdog behaviour
# ---------------------------------------------------------------------------


class TestRunnerTick:
    async def test_resets_zombies_first(self, runner, mock_db) -> None:
        mock_db.reset_zombies.return_value = 2
        mock_db.count_running.return_value = 10  # At capacity
        mock_db.get_undelivered_notifications.return_value = []
        await runner.tick()
        mock_db.reset_zombies.assert_awaited_once()

    async def test_at_capacity_skips_dispatch(self, runner, mock_db) -> None:
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 4  # At default capacity
        mock_db.get_undelivered_notifications.return_value = []
        with patch.object(runner, "_dispatch_campaign") as dispatch:
            await runner.tick()
            dispatch.assert_not_awaited()

    async def test_no_due_campaigns_exits_early(self, runner, mock_db) -> None:
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 0
        mock_db.get_due_campaigns.return_value = []
        mock_db.get_undelivered_notifications.return_value = []
        await runner.tick()
        mock_db.get_due_campaigns.assert_awaited_once()

    async def test_dispatches_when_slots_free(self, runner, mock_db) -> None:
        campaign = make_campaign()
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 0
        mock_db.get_due_campaigns.return_value = [campaign]
        mock_db.get_undelivered_notifications.return_value = []
        with patch.object(runner, "_dispatch_campaign", new_callable=AsyncMock, return_value=1):
            await runner.tick()
            runner._dispatch_campaign.assert_awaited()

    async def test_delivers_notifications(self, runner, mock_db) -> None:
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 4  # At capacity
        notif = Notification(
            id=1, campaign_id=uuid4(), channel="research",
            message="Phase 1 done", level=NotificationLevel.MILESTONE,
        )
        mock_db.get_undelivered_notifications.return_value = [notif]
        mock_db.mark_delivered = AsyncMock()
        await runner.tick()
        mock_db.mark_delivered.assert_awaited_once_with([1])


# ---------------------------------------------------------------------------
# Priority-aware scheduling
# ---------------------------------------------------------------------------


class TestPriorityScheduling:
    async def test_urgent_gets_all_slots(self, runner, mock_db) -> None:
        urgent = make_campaign(priority=Priority.URGENT)
        normal = make_campaign(priority=Priority.NORMAL)
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 0
        mock_db.get_due_campaigns.return_value = [urgent, normal]
        mock_db.get_undelivered_notifications.return_value = []

        dispatched = {}

        async def track_dispatch(campaign, max_slots):
            dispatched[campaign.priority] = max_slots
            return min(max_slots, 2)  # Pretend we dispatched 2

        with patch.object(runner, "_dispatch_campaign", side_effect=track_dispatch):
            await runner.tick()

        # Urgent should get slots first
        assert Priority.URGENT in dispatched
        assert dispatched[Priority.URGENT] >= 1

    async def test_multiple_campaigns_same_tier_share_slots(self, runner, mock_db) -> None:
        c1 = make_campaign(priority=Priority.NORMAL, name="C1")
        c2 = make_campaign(priority=Priority.NORMAL, name="C2")
        mock_db.reset_zombies.return_value = 0
        mock_db.count_running.return_value = 0
        mock_db.get_due_campaigns.return_value = [c1, c2]
        mock_db.get_undelivered_notifications.return_value = []

        calls = []

        async def track_dispatch(campaign, max_slots):
            calls.append((campaign.name, max_slots))
            return 0

        with patch.object(runner, "_dispatch_campaign", side_effect=track_dispatch):
            await runner.tick()

        # Both should get called
        names = [c[0] for c in calls]
        assert "C1" in names
        assert "C2" in names


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------


class TestDispatchCampaign:
    async def test_dispatches_ready_steps(self, runner, mock_db) -> None:
        campaign = make_campaign()
        step = make_step(campaign_id=campaign.id, id=42)
        mock_db.get_ready_steps.return_value = [step]
        mock_db.claim_step.return_value = step._replace(status=StepStatus.RUNNING) if hasattr(step, '_replace') else step

        # claim_step returns the claimed step
        claimed = make_step(campaign_id=campaign.id, id=42, status=StepStatus.RUNNING)
        mock_db.claim_step.return_value = claimed

        with patch.object(runner, "_execute_step", new_callable=AsyncMock):
            dispatched = await runner._dispatch_campaign(campaign, 4)
        assert dispatched == 1
        mock_db.claim_step.assert_awaited_once_with(42)

    async def test_consults_planner_when_no_ready_steps(self, runner, mock_db) -> None:
        campaign = make_campaign()
        mock_db.get_ready_steps.return_value = []

        with patch.object(runner, "_consult_planner", new_callable=AsyncMock):
            await runner._dispatch_campaign(campaign, 4)
            runner._consult_planner.assert_awaited_once_with(campaign)

    async def test_respects_max_slots(self, runner, mock_db) -> None:
        campaign = make_campaign()
        steps = [make_step(campaign_id=campaign.id, id=i) for i in range(10)]
        mock_db.get_ready_steps.return_value = steps
        mock_db.claim_step.return_value = make_step(status=StepStatus.RUNNING)

        with patch.object(runner, "_execute_step", new_callable=AsyncMock):
            dispatched = await runner._dispatch_campaign(campaign, 2)
        assert dispatched == 2


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


class TestBuildStepContext:
    async def test_includes_campaign_goal(self, runner, mock_db) -> None:
        campaign = make_campaign(goal="Research tRNA engineering")
        step = make_step(campaign_id=campaign.id)
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "Research tRNA engineering" in ctx

    async def test_includes_step_action(self, runner, mock_db) -> None:
        campaign = make_campaign()
        step = make_step(campaign_id=campaign.id, action="Find 2024 papers on MOFs")
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "Find 2024 papers on MOFs" in ctx

    async def test_includes_upstream_outputs(self, runner, mock_db) -> None:
        campaign = make_campaign()
        dep = make_step(campaign_id=campaign.id, id=10, output="Found 5 papers")
        step = make_step(campaign_id=campaign.id, depends_on=[10])
        mock_db.get_step.return_value = dep
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "Found 5 papers" in ctx

    async def test_depth_warning_at_max(self, runner, mock_db) -> None:
        campaign = make_campaign(max_depth=3)
        step = make_step(campaign_id=campaign.id, depth=3)
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "max depth" in ctx.lower()
        assert "atomically" in ctx.lower()

    async def test_no_depth_warning_below_max(self, runner, mock_db) -> None:
        campaign = make_campaign(max_depth=4)
        step = make_step(campaign_id=campaign.id, depth=2)
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "max depth" not in ctx.lower()

    async def test_spawn_instruction_hidden_at_max_depth(self, runner, mock_db) -> None:
        campaign = make_campaign(max_depth=3)
        step = make_step(campaign_id=campaign.id, depth=3)
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "spawn_and_continue" not in ctx

    async def test_spawn_instruction_shown_below_max_depth(self, runner, mock_db) -> None:
        campaign = make_campaign(max_depth=4)
        step = make_step(campaign_id=campaign.id, depth=2)
        mock_db.get_notes.return_value = []
        ctx = await runner._build_step_context(step, campaign)
        assert "spawn_and_continue" in ctx


# ---------------------------------------------------------------------------
# Tool visibility
# ---------------------------------------------------------------------------


class TestToolVisibility:
    def test_spawn_hidden_at_max_depth(self, runner) -> None:
        tools = runner._get_tools_for_depth(depth=4, max_depth=4)
        assert "spawn_and_continue" not in tools
        assert "complete_step" in tools
        assert "abort_branch" in tools

    def test_spawn_visible_below_max_depth(self, runner) -> None:
        tools = runner._get_tools_for_depth(depth=2, max_depth=4)
        assert "spawn_and_continue" in tools

    def test_spawn_visible_at_depth_zero(self, runner) -> None:
        tools = runner._get_tools_for_depth(depth=0, max_depth=4)
        assert "spawn_and_continue" in tools

    def test_all_base_tools_always_present(self, runner) -> None:
        base = {"get_my_context", "add_note", "search_notes", "get_notes",
                "complete_step", "fail_step", "abort_branch"}
        for depth in range(5):
            tools = set(runner._get_tools_for_depth(depth, max_depth=4))
            assert base.issubset(tools)


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------


class TestTemplateExpansion:
    def test_simple_substitution(self, runner) -> None:
        template = {"action": "Improve: {item.text}", "agent": "writing"}
        item = {"text": "Paragraph about MOFs"}
        result = runner._expand_template(template, item)
        assert result["action"] == "Improve: Paragraph about MOFs"
        assert result["agent"] == "writing"

    def test_multiple_placeholders(self, runner) -> None:
        template = {"action": "Review {item.id}: {item.title}"}
        item = {"id": "para_20", "title": "Synthesis methods"}
        result = runner._expand_template(template, item)
        assert result["action"] == "Review para_20: Synthesis methods"

    def test_nested_template(self, runner) -> None:
        template = {
            "action": "Pipeline for {item.id}",
            "steps": [
                {"action": "Find citations for {item.context}"},
                {"action": "Write improved {item.id}"},
            ],
        }
        item = {"id": "para_20", "context": "Current synthesis via solvothermal"}
        result = runner._expand_template(template, item)
        assert result["steps"][0]["action"] == "Find citations for Current synthesis via solvothermal"
        assert result["steps"][1]["action"] == "Write improved para_20"

    def test_no_placeholders_passthrough(self, runner) -> None:
        template = {"action": "Static action", "agent": "research"}
        item = {"id": "x"}
        result = runner._expand_template(template, item)
        assert result == template


# ---------------------------------------------------------------------------
# Step formatting
# ---------------------------------------------------------------------------


class TestFormatSteps:
    def test_empty_returns_none_text(self, runner) -> None:
        assert runner._format_steps([]) == "None"

    def test_includes_step_id_and_action(self, runner) -> None:
        step = make_step(id=42, action="Search for papers")
        text = runner._format_steps([step])
        assert "42" in text
        assert "Search for papers" in text

    def test_includes_output_when_present(self, runner) -> None:
        step = make_step(id=1, output="Found 5 papers on MOFs")
        text = runner._format_steps([step])
        assert "Found 5 papers" in text

    def test_includes_error_when_present(self, runner) -> None:
        step = make_step(id=1, error="Timeout connecting to API")
        text = runner._format_steps([step])
        assert "Timeout" in text
