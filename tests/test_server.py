"""Tests for sortie_mcp.server — MCP tool definitions, coordinator tools, worker tools."""

from __future__ import annotations

from datetime import datetime, timezone
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
)
from sortie_mcp.server import mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_campaign(**overrides) -> Campaign:
    defaults = dict(
        id=uuid4(), name="Test", goal="Test goal",
        status=CampaignStatus.ACTIVE, priority=Priority.NORMAL,
        max_depth=4, token_budget=None, tokens_used=0,
        failure_policy=FailurePolicy.CONTINUE, channel="research",
        strategy=None, progress=None, user_id=None,
        next_action_at=_now(), last_reported_at=None,
        created_at=_now(), updated_at=_now(), completed_at=None,
    )
    defaults.update(overrides)
    return Campaign(**defaults)


def make_step(campaign_id=None, **overrides) -> Step:
    defaults = dict(
        id=1, campaign_id=campaign_id or uuid4(), action="Test step",
        step_type=StepType.ATOMIC, status=StepStatus.PENDING, depth=0,
        agent="research", depends_on=[], failure_policy=FailurePolicy.CONTINUE,
        parent_step_id=None, input=None, output=None, error=None,
        fingerprint="abc123", continuation_of=None, completion_threshold=None,
        retry_count=0, max_retries=3, tokens_used=None, duration_ms=None,
        created_at=_now(), started_at=None, completed_at=None,
    )
    defaults.update(overrides)
    return Step(**defaults)


def make_note(campaign_id=None, **overrides) -> Note:
    defaults = dict(
        id=1, campaign_id=campaign_id or uuid4(), content="A finding",
        step_id=None, agent="research", tags=["finding"],
        embedding=None, created_at=_now(),
    )
    defaults.update(overrides)
    return Note(**defaults)


def mock_db():
    db = AsyncMock()
    db.schema = "sortie"
    return db


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def _tool_names(self) -> set[str]:
        if hasattr(mcp, "_tool_manager"):
            tm = mcp._tool_manager
            if hasattr(tm, "_tools"):
                return set(tm._tools.keys())
            if hasattr(tm, "tools"):
                return set(tm.tools.keys())
        if hasattr(mcp, "_tools"):
            return set(mcp._tools.keys())
        return set()

    def test_coordinator_tools_registered(self) -> None:
        names = self._tool_names()
        for tool in ("create_campaign", "list_campaigns", "get_campaign",
                     "get_updates", "steer_campaign", "pause_campaign",
                     "resume_campaign", "cancel_campaign"):
            assert tool in names, f"Missing coordinator tool: {tool}"

    def test_worker_tools_registered(self) -> None:
        names = self._tool_names()
        for tool in ("get_my_context", "add_note", "search_notes", "get_notes",
                     "complete_step", "fail_step", "spawn_and_continue",
                     "abort_branch"):
            assert tool in names, f"Missing worker tool: {tool}"

    def test_executor_tools_not_exposed(self) -> None:
        names = self._tool_names()
        for tool in ("get_due_campaigns", "get_ready_steps", "claim_step",
                     "reset_zombies", "count_running"):
            assert tool not in names, f"Internal tool exposed: {tool}"

    def test_total_tool_count(self) -> None:
        names = self._tool_names()
        # 8 coordinator + 8 worker = 16
        assert len(names) == 16, f"Expected 16 tools, got {len(names)}: {names}"


# ---------------------------------------------------------------------------
# Coordinator tools
# ---------------------------------------------------------------------------


class TestCreateCampaign:
    async def test_creates_active_campaign(self) -> None:
        from sortie_mcp.server import create_campaign
        db = mock_db()
        campaign = make_campaign(name="tRNA Review")
        db.create_campaign.return_value = campaign
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await create_campaign(goal="Research tRNA", name="tRNA Review")
        assert result["name"] == "tRNA Review"
        assert result["status"] == "active"
        db.create_campaign.assert_awaited_once()

    async def test_dry_run_creates_paused(self) -> None:
        from sortie_mcp.server import create_campaign
        db = mock_db()
        campaign = make_campaign(status=CampaignStatus.PAUSED)
        db.create_campaign.return_value = campaign
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await create_campaign(goal="Test", dry_run=True)
        assert result["status"] == "paused"

    async def test_priority_passed_through(self) -> None:
        from sortie_mcp.server import create_campaign
        db = mock_db()
        campaign = make_campaign(priority=Priority.URGENT)
        db.create_campaign.return_value = campaign
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await create_campaign(goal="Urgent work", priority="urgent")
        assert result["priority"] == "urgent"


class TestListCampaigns:
    async def test_returns_all_campaigns(self) -> None:
        from sortie_mcp.server import list_campaigns
        db = mock_db()
        db.list_campaigns.return_value = [
            make_campaign(name="C1"), make_campaign(name="C2"),
        ]
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await list_campaigns()
        assert len(result) == 2

    async def test_filters_by_status(self) -> None:
        from sortie_mcp.server import list_campaigns
        db = mock_db()
        db.list_campaigns.return_value = [make_campaign()]
        with patch("sortie_mcp.server.get_db", return_value=db):
            await list_campaigns(status="active")
        db.list_campaigns.assert_awaited_once_with(status=CampaignStatus.ACTIVE)


class TestGetCampaign:
    async def test_returns_full_state(self) -> None:
        from sortie_mcp.server import get_campaign
        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, goal="Full state test")
        step = make_step(campaign_id=cid, id=1, action="Step A", output="Done")
        note = make_note(campaign_id=cid, content="Found X", created_at=_now())
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = [step]
        db.get_notes.return_value = [note]
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_campaign(str(cid))
        assert result["goal"] == "Full state test"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["action"] == "Step A"
        db.set_last_reported.assert_awaited_once_with(cid)

    async def test_not_found_returns_error(self) -> None:
        from sortie_mcp.server import get_campaign
        db = mock_db()
        db.get_campaign.return_value = None
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_campaign(str(uuid4()))
        assert "error" in result


class TestSteerCampaign:
    async def test_appends_guidance_to_strategy(self) -> None:
        from sortie_mcp.server import steer_campaign
        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, strategy="Original strategy")
        db.get_campaign.return_value = campaign
        db.update_campaign.return_value = make_campaign(
            strategy="Original strategy\n\n[User guidance]: Focus on MOFs"
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await steer_campaign(str(cid), "Focus on MOFs")
        assert result["status"] == "updated"
        assert "MOFs" in (result.get("strategy") or "")


class TestPauseResumeCancelCampaign:
    async def test_pause(self) -> None:
        from sortie_mcp.server import pause_campaign
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await pause_campaign(str(uuid4()))
        assert result["status"] == "paused"

    async def test_resume(self) -> None:
        from sortie_mcp.server import resume_campaign
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await resume_campaign(str(uuid4()))
        assert result["status"] == "active"

    async def test_cancel_skips_pending_steps(self) -> None:
        from sortie_mcp.server import cancel_campaign
        db = mock_db()
        cid = uuid4()
        pending = [make_step(campaign_id=cid, id=i) for i in range(3)]
        db.get_steps.return_value = pending
        mock_conn = AsyncMock()
        acm = MagicMock()
        acm.__aenter__ = AsyncMock(return_value=mock_conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        db.pool.acquire = MagicMock(return_value=acm)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await cancel_campaign(str(cid))
        assert result["status"] == "cancelled"
        assert result["steps_skipped"] == 3


# ---------------------------------------------------------------------------
# Worker tools
# ---------------------------------------------------------------------------


class TestGetMyContext:
    async def test_returns_campaign_and_step_context(self) -> None:
        from sortie_mcp.server import get_my_context
        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, goal="MOF synthesis review")
        step = make_step(campaign_id=cid, id=42, action="Find papers")
        db.get_step.return_value = step
        db.get_campaign.return_value = campaign
        db.get_notes.return_value = []
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context(42)
        assert result["campaign_goal"] == "MOF synthesis review"
        assert result["your_task"] == "Find papers"
        assert result["your_step_id"] == 42

    async def test_includes_upstream_outputs(self) -> None:
        from sortie_mcp.server import get_my_context
        db = mock_db()
        cid = uuid4()
        dep = make_step(campaign_id=cid, id=10, action="Search", output="Found 5 papers")
        step = make_step(campaign_id=cid, id=42, depends_on=[10])
        campaign = make_campaign(id=cid)
        db.get_step.side_effect = lambda sid: step if sid == 42 else dep
        db.get_campaign.return_value = campaign
        db.get_notes.return_value = []
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context(42)
        assert any("Found 5 papers" in u["output"] for u in result["upstream_context"])

    async def test_not_found(self) -> None:
        from sortie_mcp.server import get_my_context
        db = mock_db()
        db.get_step.return_value = None
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context(999)
        assert "error" in result


class TestCompleteStep:
    async def test_marks_done(self) -> None:
        from sortie_mcp.server import complete_step
        db = mock_db()
        db.complete_step.return_value = make_step(status=StepStatus.DONE)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await complete_step(42, "Found 5 papers")
        assert result["status"] == "done"

    async def test_already_skipped_returns_skipped(self) -> None:
        from sortie_mcp.server import complete_step
        db = mock_db()
        db.complete_step.return_value = make_step(
            status=StepStatus.SKIPPED, error="Branch aborted"
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await complete_step(42, "My output")
        assert result["status"] == "skipped"
        assert result["output_recorded"] is True

    async def test_not_found(self) -> None:
        from sortie_mcp.server import complete_step
        db = mock_db()
        db.complete_step.return_value = None
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await complete_step(999, "output")
        assert "error" in result


class TestFailStep:
    async def test_retriable_failure(self) -> None:
        from sortie_mcp.server import fail_step
        db = mock_db()
        db.fail_step.return_value = make_step(
            status=StepStatus.PENDING, retry_count=1, max_retries=3
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await fail_step(42, "Timeout")
        assert result["can_retry"] is True
        assert result["status"] == "pending"

    async def test_permanent_failure(self) -> None:
        from sortie_mcp.server import fail_step
        db = mock_db()
        db.fail_step.return_value = make_step(
            status=StepStatus.FAILED, retry_count=3, max_retries=3
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await fail_step(42, "Unrecoverable")
        assert result["can_retry"] is False
        assert result["status"] == "failed"


class TestSpawnAndContinue:
    async def test_returns_splice_ids(self) -> None:
        from sortie_mcp.server import spawn_and_continue
        db = mock_db()
        db.spawn_and_continue.return_value = {
            "subtask_ids": [100, 101],
            "continuation_id": 102,
        }
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await spawn_and_continue(
                42, "Partial output",
                [{"action": "Sub A"}, {"action": "Sub B"}],
                "Continue after subtasks",
            )
        assert result["status"] == "spliced"
        assert result["subtask_ids"] == [100, 101]
        assert result["continuation_id"] == 102


class TestAbortBranch:
    async def test_returns_skipped_ids(self) -> None:
        from sortie_mcp.server import abort_branch
        db = mock_db()
        db.abort_branch.return_value = {
            "skipped_ids": [10, 11, 12],
            "target_id": 5,
        }
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await abort_branch(
                target_id=5, output="Approach debunked",
                reason="Miller 2001", step_id=42,
            )
        assert result["status"] == "aborted"
        assert result["skipped_count"] == 3

    async def test_requires_step_id(self) -> None:
        from sortie_mcp.server import abort_branch
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await abort_branch(
                target_id=5, output="x", reason="y", step_id=None,
            )
        assert "error" in result


class TestAddNote:
    async def test_creates_note(self) -> None:
        from sortie_mcp.server import add_note
        db = mock_db()
        cid = uuid4()
        db.add_note.return_value = make_note(campaign_id=cid, id=7)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await add_note(str(cid), "Important finding", tags=["finding"])
        assert result["note_id"] == 7
        assert result["recorded"] is True


class TestGetNotes:
    async def test_returns_filtered_notes(self) -> None:
        from sortie_mcp.server import get_notes
        db = mock_db()
        cid = uuid4()
        db.get_notes.return_value = [
            make_note(campaign_id=cid, content="Note A", tags=["finding"]),
        ]
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_notes(str(cid), tags=["finding"])
        assert len(result) == 1
        assert result[0]["content"] == "Note A"
