"""Data models for sortie-mcp campaigns, steps, notes, and notifications."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CampaignStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(StrEnum):
    ATOMIC = "atomic"
    PARALLEL_GROUP = "parallel_group"
    SEQUENCE = "sequence"
    FOR_EACH = "for_each"


class FailurePolicy(StrEnum):
    CONTINUE = "continue"
    FAIL_FAST = "fail_fast"


class Priority(StrEnum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class NotificationLevel(StrEnum):
    MILESTONE = "milestone"
    DONE = "done"
    ERROR = "error"
    INFO = "info"


# Priority tier ordering for the runner scheduler
PRIORITY_ORDER: list[Priority] = [
    Priority.URGENT,
    Priority.HIGH,
    Priority.NORMAL,
    Priority.LOW,
    Priority.BACKGROUND,
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Campaign:
    id: UUID
    name: str | None
    goal: str
    status: CampaignStatus
    strategy: str | None = None
    progress: str | None = None
    channel: str | None = None
    user_id: str | None = None
    max_depth: int = 4
    token_budget: int | None = None
    tokens_used: int = 0
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE
    priority: Priority = Priority.NORMAL
    next_action_at: datetime | None = None
    last_reported_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class Step:
    id: int
    campaign_id: UUID
    action: str
    step_type: StepType = StepType.ATOMIC
    status: StepStatus = StepStatus.PENDING
    parent_step_id: int | None = None
    depth: int = 0
    agent: str | None = None
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE
    depends_on: list[int] = field(default_factory=list)
    input: str | None = None
    output: str | None = None
    error: str | None = None
    fingerprint: str | None = None
    continuation_of: int | None = None
    completion_threshold: int | None = None
    retry_count: int = 0
    max_retries: int = 3
    tokens_used: int | None = None
    duration_ms: int | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class Note:
    id: int
    campaign_id: UUID
    content: str
    step_id: int | None = None
    agent: str | None = None
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime | None = None


@dataclass
class Notification:
    id: int
    campaign_id: UUID | None
    channel: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    delivered: bool = False
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Step plan — what the planner or agent submits (before DB insertion)
# ---------------------------------------------------------------------------


@dataclass
class StepPlan:
    """A step definition before it gets an ID. Used by planner output and
    spawn_and_continue subtask lists."""

    action: str
    agent: str | None = None
    step_type: StepType = StepType.ATOMIC
    depends_on: list[int] = field(default_factory=list)
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE
    input: str | None = None
    completion_threshold: int | None = None
    # for_each fields
    items: list[dict[str, Any]] | None = None
    items_from: str | None = None
    template: dict[str, Any] | None = None
    collect: dict[str, Any] | None = None
    parallel: bool = True
    # sequence / parallel_group children
    steps: list[StepPlan] | None = None
    children: list[StepPlan] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_fingerprint(action: str, agent: str | None, input_text: str | None) -> str:
    """SHA-256 fingerprint for advisory dedup."""
    parts = [action, agent or "", input_text or ""]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# Separator used in structured output hints
SEP = "›"
