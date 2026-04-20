"""Tests for sortie_mcp.server — MCP tool definitions, coordinator tools, worker tools."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from sortie_mcp.models import (
    Campaign,
    CampaignStatus,
    FailurePolicy,
    Note,
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
    return datetime.now(UTC)


def make_campaign(**overrides) -> Campaign:
    """Build a test campaign; pins ``weight`` from ``priority`` like
    :meth:`sortie_mcp.db.DB.create_campaign` does in production."""
    from sortie_mcp.models import priority_weight

    priority = overrides.get("priority", Priority.NORMAL)
    defaults = dict(
        id=uuid4(),
        name="Test",
        goal="Test goal",
        status=CampaignStatus.ACTIVE,
        priority=priority,
        weight=priority_weight(priority),
        max_depth=4,
        token_budget=None,
        tokens_used=0,
        failure_policy=FailurePolicy.CONTINUE,
        channel="research",
        strategy=None,
        progress=None,
        user_id=None,
        next_action_at=_now(),
        last_reported_at=None,
        created_at=_now(),
        updated_at=_now(),
        completed_at=None,
    )
    defaults.update(overrides)
    return Campaign(**defaults)  # type: ignore[arg-type]


def make_step(campaign_id=None, **overrides: object) -> Step:
    defaults: dict[str, object] = dict(
        id=1,
        campaign_id=campaign_id or uuid4(),
        action="Test step",
        step_type=StepType.ATOMIC,
        status=StepStatus.PENDING,
        depth=0,
        agent="research",
        depends_on=[],
        failure_policy=FailurePolicy.CONTINUE,
        parent_step_id=None,
        input=None,
        output=None,
        error=None,
        fingerprint="abc123",
        continuation_of=None,
        completion_threshold=None,
        retry_count=0,
        max_retries=3,
        tokens_used=None,
        duration_ms=None,
        created_at=_now(),
        started_at=None,
        completed_at=None,
    )
    defaults.update(overrides)
    return Step(**defaults)  # type: ignore[arg-type]


def make_note(campaign_id=None, **overrides) -> Note:
    defaults = dict(
        id=1,
        campaign_id=campaign_id or uuid4(),
        content="A finding",
        step_id=None,
        agent="research",
        tags=["finding"],
        embedding=None,
        created_at=_now(),
    )
    defaults.update(overrides)
    return Note(**defaults)  # type: ignore[arg-type]


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
        for tool in (
            "create_campaign",
            "list_campaigns",
            "get_campaign",
            "get_updates",
            "steer_campaign",
            "pause_campaign",
            "resume_campaign",
            "cancel_campaign",
            "provide_input",
            "check_success",  # new in v0.3.0 — typed success contract
        ):
            assert tool in names, f"Missing coordinator tool: {tool}"

    def test_worker_tools_registered(self) -> None:
        names = self._tool_names()
        for tool in (
            "get_my_context",
            "add_note",
            "search_notes",
            "get_notes",
            "complete_step",
            "fail_step",
            "spawn_and_continue",
            "abort_branch",
            "request_input",
            "read_step_output",  # new in v0.2.0 — preview-plus-seek counterpart
            "heartbeat",  # new in v0.2.0 — keeps claim + leases fresh
        ):
            assert tool in names, f"Missing worker tool: {tool}"

    def test_executor_tools_not_exposed(self) -> None:
        names = self._tool_names()
        for tool in (
            "get_due_campaigns",
            "get_ready_steps",
            "claim_step",
            "reset_zombies",
            "count_running",
        ):
            assert tool not in names, f"Internal tool exposed: {tool}"

    def test_total_tool_count(self) -> None:
        names = self._tool_names()
        # 10 coordinator + 11 worker = 21.
        # v0.2.0: +2 worker (read_step_output, heartbeat).
        # v0.3.0: +1 coordinator (check_success).
        assert len(names) == 21, f"Expected 21 tools, got {len(names)}: {names}"


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
            make_campaign(name="C1"),
            make_campaign(name="C2"),
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
        dep = make_step(
            campaign_id=cid, id=10, action="Search", output="Found 5 papers"
        )
        step = make_step(campaign_id=cid, id=42, depends_on=[10])
        campaign = make_campaign(id=cid)
        db.get_step.side_effect = lambda sid: step if sid == 42 else dep
        db.get_campaign.return_value = campaign
        db.get_notes.return_value = []
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context(42)
        # v0.2.0: upstream outputs are preview-plus-seek — each entry has a
        # ``preview`` string (short outputs fit whole; long ones are elided).
        assert any("Found 5 papers" in u["preview"] for u in result["upstream_context"])

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
                42,
                "Partial output",
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
                target_id=5,
                output="Approach debunked",
                reason="Miller 2001",
                step_id=42,
            )
        assert result["status"] == "aborted"
        assert result["skipped_count"] == 3

    async def test_requires_step_id(self) -> None:
        from sortie_mcp.server import abort_branch

        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await abort_branch(
                target_id=5,
                output="x",
                reason="y",
                step_id=None,
            )
        assert "error" in result


class TestAddNote:
    async def test_creates_note_without_embedding_when_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import add_note

        monkeypatch.delenv("SORTIE_EMBEDDINGS_ENABLED", raising=False)
        db = mock_db()
        cid = uuid4()
        db.add_note.return_value = make_note(campaign_id=cid, id=7)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await add_note(str(cid), "Important finding", tags=["finding"])
        assert result["note_id"] == 7
        assert result["recorded"] is True
        assert result["embedded"] is False
        # embedding=None should be passed through so the pgvector column is NULL.
        _, kwargs = db.add_note.call_args
        assert kwargs["embedding"] is None

    async def test_creates_note_with_embedding_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When embeddings are enabled, add_note must ask embed_text for a
        vector and forward it to the DB insert."""
        from sortie_mcp.server import add_note

        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", "1")
        db = mock_db()
        cid = uuid4()
        db.add_note.return_value = make_note(campaign_id=cid, id=7)
        vec = [0.1] * 384
        with (
            patch("sortie_mcp.server.get_db", return_value=db),
            patch(
                "sortie_mcp.server.embed_text",
                new=AsyncMock(return_value=vec),
            ) as mock_embed,
        ):
            result = await add_note(str(cid), "Important finding")
        assert result["embedded"] is True
        mock_embed.assert_awaited_once_with("Important finding")
        _, kwargs = db.add_note.call_args
        assert kwargs["embedding"] == vec

    async def test_still_records_when_embedding_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail-open contract: a LiteLLM outage must not stop notes from
        being saved. The DB insert goes through with ``embedding=None``."""
        from sortie_mcp.server import add_note

        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", "1")
        db = mock_db()
        db.add_note.return_value = make_note(id=7, campaign_id=uuid4())
        with (
            patch("sortie_mcp.server.get_db", return_value=db),
            patch(
                "sortie_mcp.server.embed_text",
                new=AsyncMock(return_value=None),  # LiteLLM bounced
            ),
        ):
            result = await add_note(str(uuid4()), "Finding")
        assert result["recorded"] is True
        assert result["embedded"] is False


class TestSearchNotes:
    async def test_semantic_mode_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import search_notes

        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", "1")
        db = mock_db()
        cid = uuid4()
        db.search_notes.return_value = [
            make_note(campaign_id=cid, id=1, content="top hit", tags=["a"]),
        ]
        vec = [0.2] * 384
        with (
            patch("sortie_mcp.server.get_db", return_value=db),
            patch("sortie_mcp.server.embed_text", new=AsyncMock(return_value=vec)),
        ):
            result = await search_notes("my query", campaign_id=str(cid), top_k=3)
        assert len(result) == 1
        assert result[0]["content"] == "top hit"
        assert result[0]["mode"] == "semantic"
        db.search_notes.assert_awaited_once_with(vec, campaign_id=cid, top_k=3)

    async def test_recency_fallback_when_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import search_notes

        monkeypatch.delenv("SORTIE_EMBEDDINGS_ENABLED", raising=False)
        db = mock_db()
        cid = uuid4()
        db.get_notes.return_value = [
            make_note(campaign_id=cid, id=i, content=f"note {i}") for i in range(10)
        ]
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await search_notes("query", campaign_id=str(cid), top_k=3)
        assert len(result) == 3
        assert all(r["mode"] == "recency" for r in result)
        # Must NOT have gone through the semantic path.
        db.search_notes.assert_not_awaited()

    async def test_recency_fallback_when_embed_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with the flag on, if LiteLLM fails the query embedding we
        degrade gracefully to recency listing rather than returning []."""
        from sortie_mcp.server import search_notes

        monkeypatch.setenv("SORTIE_EMBEDDINGS_ENABLED", "1")
        db = mock_db()
        cid = uuid4()
        db.get_notes.return_value = [make_note(campaign_id=cid, id=1)]
        with (
            patch("sortie_mcp.server.get_db", return_value=db),
            patch("sortie_mcp.server.embed_text", new=AsyncMock(return_value=None)),
        ):
            result = await search_notes("query", campaign_id=str(cid))
        assert len(result) == 1
        assert result[0]["mode"] == "recency"
        db.search_notes.assert_not_awaited()

    async def test_global_recency_search_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unscoped recency search must NOT scan every campaign — too
        expensive in a cluster. It returns empty and forces the agent to
        pick a campaign."""
        from sortie_mcp.server import search_notes

        monkeypatch.delenv("SORTIE_EMBEDDINGS_ENABLED", raising=False)
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await search_notes("query")  # no campaign_id
        assert result == []
        db.get_notes.assert_not_awaited()


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


# ---------------------------------------------------------------------------
# v0.2.0 additions — preview-plus-seek, read_step_output, heartbeat
# ---------------------------------------------------------------------------


class TestPreviewHelper:
    def test_short_text_fits_whole(self) -> None:
        from sortie_mcp.server import _preview

        out = _preview("hello world")
        assert out["preview"] == "hello world"
        assert out["total_chars"] == 11
        assert out["truncated"] is False

    def test_empty_text(self) -> None:
        from sortie_mcp.server import _preview

        assert _preview(None) == {"preview": "", "total_chars": 0, "truncated": False}
        assert _preview("") == {"preview": "", "total_chars": 0, "truncated": False}

    def test_long_text_is_elided(self) -> None:
        from sortie_mcp.server import (
            PREVIEW_HEAD_CHARS,
            PREVIEW_TAIL_CHARS,
            _preview,
        )

        text = ("A" * PREVIEW_HEAD_CHARS) + ("B" * 1000) + ("C" * PREVIEW_TAIL_CHARS)
        out = _preview(text)
        assert out["truncated"] is True
        assert out["total_chars"] == len(text)
        assert out["preview"].startswith("A" * 50)  # head preserved
        assert out["preview"].endswith("C" * 50)  # tail preserved
        assert "ELIDED 1000 chars" in out["preview"]


class TestGetMyContextPreview:
    async def test_long_upstream_output_is_previewed_with_hint(self) -> None:
        """Upstream outputs over the head+tail budget must include a
        ``read_step_output`` hint so the agent can fetch the rest."""
        from sortie_mcp.server import PREVIEW_HEAD_CHARS, get_my_context

        db = mock_db()
        cid = uuid4()
        long_output = "X" * (PREVIEW_HEAD_CHARS + 5000)
        dep = make_step(campaign_id=cid, id=10, action="Search", output=long_output)
        step = make_step(campaign_id=cid, id=42, depends_on=[10])
        campaign = make_campaign(id=cid)
        db.get_step.side_effect = lambda sid: step if sid == 42 else dep
        db.get_campaign.return_value = campaign
        db.get_notes.return_value = []
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context(42)
        (entry,) = result["upstream_context"]
        assert entry["truncated"] is True
        assert entry["total_chars"] == len(long_output)
        assert "read_step_output(10)" in entry["hint"]

    async def test_step_id_inferred_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import get_my_context

        db = mock_db()
        cid = uuid4()
        step = make_step(campaign_id=cid, id=42, action="Do the thing")
        db.get_step.return_value = step
        db.get_campaign.return_value = make_campaign(id=cid)
        db.get_notes.return_value = []
        monkeypatch.setenv("SORTIE_STEP_ID", "42")
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context()  # no step_id arg!
        assert result["your_step_id"] == 42
        db.get_step.assert_awaited_with(42)

    async def test_error_when_no_step_id_and_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import get_my_context

        monkeypatch.delenv("SORTIE_STEP_ID", raising=False)
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_my_context()
        assert "error" in result
        assert "SORTIE_STEP_ID" in result["error"]


class TestReadStepOutput:
    async def test_reads_full_output(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(
            id=10, output="Hello world", campaign_id=uuid4()
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10)
        assert result["content"] == "Hello world"
        assert result["total_chars"] == 11
        assert result["has_more"] is False

    async def test_slices_by_offset_and_limit(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        text = "abcdefghij"  # 10 chars
        db.get_step.return_value = make_step(id=10, output=text, campaign_id=uuid4())
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10, offset=3, limit=4)
        assert result["content"] == "defg"
        assert result["offset"] == 3
        assert result["limit"] == 4
        assert result["has_more"] is True

    async def test_limit_capped_at_32k(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(id=10, output="x", campaign_id=uuid4())
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10, limit=10**9)
        assert result["limit"] == 32_000

    async def test_negative_offset_clamped(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(id=10, output="hello", campaign_id=uuid4())
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10, offset=-5)
        assert result["offset"] == 0
        assert result["content"] == "hello"

    async def test_reads_input_field(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(
            id=10, input="my input", output=None, campaign_id=uuid4()
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10, field="input")
        assert result["content"] == "my input"
        assert result["field"] == "input"

    async def test_rejects_unknown_field(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(id=10, campaign_id=uuid4())
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10, field="garbage")
        assert "error" in result

    async def test_step_not_found(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = None
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(999)
        assert "error" in result

    async def test_null_output(self) -> None:
        from sortie_mcp.server import read_step_output

        db = mock_db()
        db.get_step.return_value = make_step(id=10, output=None, campaign_id=uuid4())
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await read_step_output(10)
        assert result["content"] == ""
        assert result["total_chars"] == 0
        assert result["has_more"] is False


class TestHeartbeatTool:
    async def test_requires_claim_token_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import heartbeat

        monkeypatch.delenv("SORTIE_CLAIM_TOKEN", raising=False)
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await heartbeat(step_id=42)
        assert "error" in result
        assert "SORTIE_CLAIM_TOKEN" in result["error"]

    async def test_requires_step_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sortie_mcp.server import heartbeat

        monkeypatch.delenv("SORTIE_STEP_ID", raising=False)
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(uuid4()))
        db = mock_db()
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await heartbeat()
        assert "error" in result
        assert "SORTIE_STEP_ID" in result["error"]

    async def test_ok_when_token_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sortie_mcp.server import heartbeat

        tok = uuid4()
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(tok))
        monkeypatch.setenv("SORTIE_STEP_ID", "42")
        db = mock_db()
        db.heartbeat.return_value = make_step(id=42, status=StepStatus.RUNNING)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await heartbeat()
        assert result["status"] == "ok"
        assert result["step_id"] == 42
        # Verify we passed the token through, not a synthetic one.
        args, _kwargs = db.heartbeat.call_args
        assert args[0] == 42
        assert args[1] == tok

    async def test_stale_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sortie_mcp.server import heartbeat

        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(uuid4()))
        db = mock_db()
        db.heartbeat.return_value = None  # DB rejected — zombie reset happened
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await heartbeat(step_id=42)
        assert result["status"] == "stale_claim"

    async def test_zero_extend_skips_lease_bump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import heartbeat

        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(uuid4()))
        db = mock_db()
        db.heartbeat.return_value = make_step(id=42, status=StepStatus.RUNNING)
        with patch("sortie_mcp.server.get_db", return_value=db):
            await heartbeat(step_id=42, extend_leases_sec=0)
        _, kwargs = db.heartbeat.call_args
        assert kwargs["extend_leases_sec"] is None


class TestClaimTokenEnforcement:
    """Session claim_token must flow into complete/fail/request_input."""

    async def test_complete_step_passes_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import complete_step

        tok = uuid4()
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(tok))
        db = mock_db()
        db.complete_step.return_value = make_step(status=StepStatus.DONE)
        with patch("sortie_mcp.server.get_db", return_value=db):
            await complete_step(42, "done")
        _, kwargs = db.complete_step.call_args
        assert kwargs["claim_token"] == tok

    async def test_complete_step_stale_when_token_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import complete_step

        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(uuid4()))
        db = mock_db()
        db.complete_step.return_value = None  # token didn't match
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await complete_step(42, "done")
        assert result["status"] == "stale_claim"

    async def test_fail_step_passes_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import fail_step

        tok = uuid4()
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(tok))
        db = mock_db()
        db.fail_step.return_value = make_step(
            status=StepStatus.PENDING, retry_count=1, max_retries=3
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            await fail_step(42, "boom")
        _, kwargs = db.fail_step.call_args
        assert kwargs["claim_token"] == tok

    async def test_request_input_passes_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sortie_mcp.server import request_input

        tok = uuid4()
        monkeypatch.setenv("SORTIE_CLAIM_TOKEN", str(tok))
        db = mock_db()
        db.request_input.return_value = make_step(status=StepStatus.WAITING_INPUT)
        with patch("sortie_mcp.server.get_db", return_value=db):
            await request_input(42, "what now?")
        _, kwargs = db.request_input.call_args
        assert kwargs["claim_token"] == tok

    async def test_no_token_means_no_token_enforcement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no session token, the DB is called with claim_token=None
        (the ``trusted in-process caller`` path) — preserves v0.1 behaviour."""
        from sortie_mcp.server import complete_step

        monkeypatch.delenv("SORTIE_CLAIM_TOKEN", raising=False)
        db = mock_db()
        db.complete_step.return_value = make_step(status=StepStatus.DONE)
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await complete_step(42, "done")
        _, kwargs = db.complete_step.call_args
        assert kwargs["claim_token"] is None
        assert result["status"] == "done"


# ---------------------------------------------------------------------------
# v0.3.0 — typed success contract (B4)
# ---------------------------------------------------------------------------


class TestExtractMetricValue:
    """``_extract_metric_value`` is forgiving by design — producers
    write natural prose, the parser finds a number."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("metric=0.83", 0.83),
            ("accuracy_at_1k = 0.742", 0.742),
            ("metric: 12.5", 12.5),
            ("Final: 99.9%", 99.9),
            ("result=-3.2e-2", -0.032),
            ("Score: 1.0", 1.0),
            ("the answer is 42", 42.0),  # trailing number fallback
            ("pass rate 98.5%", 98.5),
        ],
    )
    def test_parses_various_forms(self, text: str, expected: float) -> None:
        from sortie_mcp.server import _extract_metric_value

        got = _extract_metric_value(text)
        assert got is not None
        assert got == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "no numbers here",
            "completed",
            "   ",
        ],
    )
    def test_returns_none_for_non_numeric(self, text: str) -> None:
        from sortie_mcp.server import _extract_metric_value

        assert _extract_metric_value(text) is None


class TestCheckSuccess:
    async def test_campaign_without_contract_returns_not_met(self) -> None:
        """Free-form campaigns (no ``success_metric``) short-circuit to
        ``met=False, reason=no_success_metric_configured`` so the
        planner can tell "this campaign wasn't designed for
        autochecking"."""
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, success_metric=None)
        db.get_campaign.return_value = campaign

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        assert result["met"] is False
        assert result["reason"] == "no_success_metric_configured"
        # Must NOT call downstream note/step lookups for this short-circuit.
        db.get_notes.assert_not_awaited()
        db.get_steps.assert_not_awaited()

    async def test_not_found_returns_error(self) -> None:
        from sortie_mcp.server import check_success

        db = mock_db()
        db.get_campaign.return_value = None
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(uuid4()))
        assert "error" in result

    async def test_met_when_metric_note_present(self) -> None:
        """A note tagged ``metric:<name>`` with a parseable value
        flips ``met`` to True."""
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(
            id=cid,
            success_metric="accuracy_at_1k",
            max_iterations=10,
        )
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = [
            make_step(id=i, status=StepStatus.DONE, campaign_id=cid) for i in range(3)
        ]
        db.get_notes.return_value = [
            make_note(
                id=99,
                campaign_id=cid,
                content="Benchmark result: accuracy_at_1k=0.92",
                tags=["metric:accuracy_at_1k"],
            )
        ]

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        assert result["met"] is True
        assert result["reason"] == "metric_recorded"
        assert result["metric_value"] == pytest.approx(0.92)
        assert result["iterations_used"] == 3
        assert result["max_iterations"] == 10
        assert result["last_metric_note_id"] == 99
        # The query must have been scoped to the metric tag.
        db.get_notes.assert_awaited_once_with(cid, tags=["metric:accuracy_at_1k"])

    async def test_iterations_count_only_done_steps(self) -> None:
        """``iterations_used`` must reflect DONE steps only — pending
        or failed steps don't count as completed iterations."""
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, success_metric="m", max_iterations=5)
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = [
            make_step(id=1, status=StepStatus.DONE, campaign_id=cid),
            make_step(id=2, status=StepStatus.DONE, campaign_id=cid),
        ]
        db.get_notes.return_value = []

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        # The mock must be filtered by status=DONE — assert the call shape.
        db.get_steps.assert_awaited_once_with(cid, status=StepStatus.DONE)
        assert result["iterations_used"] == 2

    async def test_budget_exhausted_without_metric(self) -> None:
        """When ``max_iterations`` is reached and no metric note
        exists, the reason surfaces as
        ``max_iterations_reached_without_metric`` so the coordinator
        can decide to escalate."""
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, success_metric="m", max_iterations=3)
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = [
            make_step(id=i, status=StepStatus.DONE, campaign_id=cid) for i in range(3)
        ]
        db.get_notes.return_value = []

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        assert result["met"] is False
        assert result["reason"] == "max_iterations_reached_without_metric"
        assert result["iterations_used"] == 3

    async def test_still_running_when_neither_metric_nor_budget(self) -> None:
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, success_metric="m", max_iterations=10)
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = [
            make_step(id=1, status=StepStatus.DONE, campaign_id=cid)
        ]
        db.get_notes.return_value = []

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        assert result["met"] is False
        assert result["reason"] == "still_running"

    async def test_unparseable_notes_do_not_count(self) -> None:
        """A metric-tagged note that doesn't contain a parseable
        number is ignored (``metric_value`` stays None). This prevents
        a chatty worker from accidentally declaring success with a
        prose-only update."""
        from sortie_mcp.server import check_success

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(id=cid, success_metric="m", max_iterations=10)
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = []
        db.get_notes.return_value = [
            make_note(
                id=1,
                campaign_id=cid,
                content="benchmark started, no result yet",
                tags=["metric:m"],
            )
        ]

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await check_success(str(cid))

        assert result["met"] is False
        assert result["reason"] == "still_running"
        assert result["notes_checked"] == 1


class TestCreateCampaignWithSuccessContract:
    """B4: create_campaign forwards the success-contract args to DB."""

    async def test_forwards_contract_fields(self) -> None:
        from sortie_mcp.server import create_campaign

        db = mock_db()
        db.create_campaign.return_value = make_campaign(
            success_metric="m",
            benchmark_command="run bench",
            scope="chapter-1",
            max_iterations=20,
        )
        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await create_campaign(
                goal="Optimize",
                success_metric="m",
                benchmark_command="run bench",
                scope="chapter-1",
                max_iterations=20,
            )

        _, kwargs = db.create_campaign.call_args
        assert kwargs["success_metric"] == "m"
        assert kwargs["benchmark_command"] == "run bench"
        assert kwargs["scope"] == "chapter-1"
        assert kwargs["max_iterations"] == 20
        # Response still has the campaign identity fields.
        assert "id" in result
        assert result["status"] == "active"

    async def test_omitted_contract_fields_default_to_none(self) -> None:
        """Free-form campaigns pass None so the column stays NULL."""
        from sortie_mcp.server import create_campaign

        db = mock_db()
        db.create_campaign.return_value = make_campaign()
        with patch("sortie_mcp.server.get_db", return_value=db):
            await create_campaign(goal="Research")

        _, kwargs = db.create_campaign.call_args
        assert kwargs["success_metric"] is None
        assert kwargs["benchmark_command"] is None
        assert kwargs["scope"] is None
        assert kwargs["max_iterations"] is None


class TestGetCampaignSurface:
    """``get_campaign`` must surface the v0.3.0 additions so operators
    and the planner can read the full state in one call."""

    async def test_exposes_fair_share_and_success_contract(self) -> None:
        from sortie_mcp.server import get_campaign

        db = mock_db()
        cid = uuid4()
        campaign = make_campaign(
            id=cid,
            slot_seconds_used=42.5,
            weight=8.0,
            success_metric="metric_a",
            benchmark_command="run",
            scope="s",
            max_iterations=50,
        )
        db.get_campaign.return_value = campaign
        db.get_steps.return_value = []
        db.get_notes.return_value = []

        with patch("sortie_mcp.server.get_db", return_value=db):
            result = await get_campaign(str(cid))

        # Fair-share ledger
        assert result["slot_seconds_used"] == pytest.approx(42.5)
        assert result["weight"] == 8.0
        assert result["virtual_time"] == pytest.approx(42.5 / 8.0)
        # Success contract
        assert result["success_metric"] == "metric_a"
        assert result["benchmark_command"] == "run"
        assert result["scope"] == "s"
        assert result["max_iterations"] == 50
