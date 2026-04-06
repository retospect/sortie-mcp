"""Tests for sortie_mcp.models — enums, dataclasses, helpers."""

from __future__ import annotations

from dataclasses import fields
from uuid import uuid4

from sortie_mcp.models import (
    PRIORITY_ORDER,
    SEP,
    Campaign,
    CampaignStatus,
    FailurePolicy,
    Note,
    Notification,
    NotificationLevel,
    Priority,
    Step,
    StepPlan,
    StepStatus,
    StepType,
    compute_fingerprint,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestCampaignStatus:
    def test_values(self) -> None:
        assert CampaignStatus.ACTIVE == "active"
        assert CampaignStatus.PAUSED == "paused"
        assert CampaignStatus.DONE == "done"
        assert CampaignStatus.FAILED == "failed"
        assert CampaignStatus.CANCELLED == "cancelled"

    def test_all_members_are_lowercase(self) -> None:
        for member in CampaignStatus:
            assert member.value == member.value.lower()

    def test_constructible_from_string(self) -> None:
        assert CampaignStatus("active") == CampaignStatus.ACTIVE


class TestStepStatus:
    def test_values(self) -> None:
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.DONE == "done"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"

    def test_terminal_states(self) -> None:
        terminal = {StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED}
        non_terminal = {StepStatus.PENDING, StepStatus.RUNNING}
        assert terminal & non_terminal == set()


class TestStepType:
    def test_values(self) -> None:
        assert StepType.ATOMIC == "atomic"
        assert StepType.PARALLEL_GROUP == "parallel_group"
        assert StepType.SEQUENCE == "sequence"
        assert StepType.FOR_EACH == "for_each"

    def test_compound_types(self) -> None:
        compound = {StepType.PARALLEL_GROUP, StepType.SEQUENCE, StepType.FOR_EACH}
        assert StepType.ATOMIC not in compound
        assert len(compound) == 3


class TestFailurePolicy:
    def test_values(self) -> None:
        assert FailurePolicy.CONTINUE == "continue"
        assert FailurePolicy.FAIL_FAST == "fail_fast"


class TestPriority:
    def test_values(self) -> None:
        assert Priority.URGENT == "urgent"
        assert Priority.HIGH == "high"
        assert Priority.NORMAL == "normal"
        assert Priority.LOW == "low"
        assert Priority.BACKGROUND == "background"

    def test_priority_order_complete(self) -> None:
        assert set(PRIORITY_ORDER) == set(Priority)

    def test_priority_order_urgent_first(self) -> None:
        assert PRIORITY_ORDER[0] == Priority.URGENT
        assert PRIORITY_ORDER[-1] == Priority.BACKGROUND

    def test_priority_order_no_duplicates(self) -> None:
        assert len(PRIORITY_ORDER) == len(set(PRIORITY_ORDER))


class TestNotificationLevel:
    def test_values(self) -> None:
        assert NotificationLevel.MILESTONE == "milestone"
        assert NotificationLevel.DONE == "done"
        assert NotificationLevel.ERROR == "error"
        assert NotificationLevel.INFO == "info"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_deterministic(self) -> None:
        fp1 = compute_fingerprint("search papers", "research", "tRNA")
        fp2 = compute_fingerprint("search papers", "research", "tRNA")
        assert fp1 == fp2

    def test_different_action_yields_different_fingerprint(self) -> None:
        fp1 = compute_fingerprint("search papers", "research", "tRNA")
        fp2 = compute_fingerprint("read papers", "research", "tRNA")
        assert fp1 != fp2

    def test_different_agent_yields_different_fingerprint(self) -> None:
        fp1 = compute_fingerprint("search papers", "research", "tRNA")
        fp2 = compute_fingerprint("search papers", "writing", "tRNA")
        assert fp1 != fp2

    def test_different_input_yields_different_fingerprint(self) -> None:
        fp1 = compute_fingerprint("search papers", "research", "tRNA")
        fp2 = compute_fingerprint("search papers", "research", "MOF")
        assert fp1 != fp2

    def test_none_agent_is_stable(self) -> None:
        fp1 = compute_fingerprint("search", None, "tRNA")
        fp2 = compute_fingerprint("search", None, "tRNA")
        assert fp1 == fp2

    def test_none_input_is_stable(self) -> None:
        fp1 = compute_fingerprint("search", "research", None)
        fp2 = compute_fingerprint("search", "research", None)
        assert fp1 == fp2

    def test_none_agent_differs_from_empty(self) -> None:
        fp1 = compute_fingerprint("search", None, "x")
        fp2 = compute_fingerprint("search", "", "x")
        assert fp1 == fp2  # Both normalize to empty string

    def test_is_valid_sha256_hex(self) -> None:
        fp = compute_fingerprint("action", "agent", "input")
        assert len(fp) == 64
        int(fp, 16)  # Should not raise

    def test_empty_strings(self) -> None:
        fp = compute_fingerprint("", "", "")
        assert len(fp) == 64


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestCampaignDataclass:
    def test_defaults(self) -> None:
        c = Campaign(id=uuid4(), name=None, goal="test", status=CampaignStatus.ACTIVE)
        assert c.max_depth == 4
        assert c.tokens_used == 0
        assert c.failure_policy == FailurePolicy.CONTINUE
        assert c.priority == Priority.NORMAL
        assert c.token_budget is None

    def test_all_fields_present(self) -> None:
        field_names = {f.name for f in fields(Campaign)}
        expected = {
            "id",
            "name",
            "goal",
            "status",
            "strategy",
            "progress",
            "channel",
            "user_id",
            "max_depth",
            "token_budget",
            "tokens_used",
            "failure_policy",
            "priority",
            "next_action_at",
            "last_reported_at",
            "created_at",
            "updated_at",
            "completed_at",
        }
        assert field_names == expected


class TestStepDataclass:
    def test_defaults(self) -> None:
        s = Step(id=1, campaign_id=uuid4(), action="test")
        assert s.step_type == StepType.ATOMIC
        assert s.status == StepStatus.PENDING
        assert s.depth == 0
        assert s.depends_on == []
        assert s.retry_count == 0
        assert s.max_retries == 3
        assert s.continuation_of is None
        assert s.completion_threshold is None

    def test_depends_on_is_independent_list(self) -> None:
        s1 = Step(id=1, campaign_id=uuid4(), action="a")
        s2 = Step(id=2, campaign_id=uuid4(), action="b")
        s1.depends_on.append(99)
        assert s2.depends_on == []  # Verify no shared mutable default


class TestNoteDataclass:
    def test_defaults(self) -> None:
        n = Note(id=1, campaign_id=uuid4(), content="finding")
        assert n.tags == []
        assert n.embedding is None
        assert n.step_id is None

    def test_tags_are_independent(self) -> None:
        n1 = Note(id=1, campaign_id=uuid4(), content="a")
        n2 = Note(id=2, campaign_id=uuid4(), content="b")
        n1.tags.append("x")
        assert n2.tags == []


class TestNotificationDataclass:
    def test_defaults(self) -> None:
        n = Notification(id=1, campaign_id=None, channel="c", message="m")
        assert n.level == NotificationLevel.INFO
        assert n.delivered is False


# ---------------------------------------------------------------------------
# StepPlan
# ---------------------------------------------------------------------------


class TestStepPlan:
    def test_atomic_defaults(self) -> None:
        plan = StepPlan(action="Do something")
        assert plan.step_type == StepType.ATOMIC
        assert plan.depends_on == []
        assert plan.agent is None
        assert plan.items is None
        assert plan.parallel is True
        assert plan.failure_policy == FailurePolicy.CONTINUE

    def test_for_each_fields(self, sample_for_each_plan: StepPlan) -> None:
        plan = sample_for_each_plan
        assert plan.step_type == StepType.FOR_EACH
        assert plan.items is not None
        assert len(plan.items) == 2
        assert plan.template is not None
        assert plan.collect is not None

    def test_sequence_fields(self, sample_sequence_plan: StepPlan) -> None:
        plan = sample_sequence_plan
        assert plan.step_type == StepType.SEQUENCE
        assert plan.steps is not None
        assert len(plan.steps) == 4
        for child in plan.steps:
            assert isinstance(child, StepPlan)

    def test_for_each_template_has_placeholders(
        self, sample_for_each_plan: StepPlan
    ) -> None:
        import json

        template_json = json.dumps(sample_for_each_plan.template)
        assert "{item." in template_json

    def test_depends_on_mutable_default_safe(self) -> None:
        p1 = StepPlan(action="a")
        p2 = StepPlan(action="b")
        p1.depends_on.append(42)
        assert p2.depends_on == []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_sep_is_single_char(self) -> None:
        assert len(SEP) == 1
        assert SEP == "›"
