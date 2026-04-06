"""Shared fixtures for sortie-mcp tests."""

from __future__ import annotations

import pytest

from sortie_mcp.models import (
    FailurePolicy,
    Priority,
    StepPlan,
    StepType,
)


@pytest.fixture
def sample_campaign_kwargs() -> dict:
    """Kwargs for creating a test campaign."""
    return {
        "goal": "Research tRNA engineering delivery mechanisms",
        "name": "tRNA Engineering Review",
        "channel": "research",
        "max_depth": 4,
        "priority": Priority.NORMAL,
        "failure_policy": FailurePolicy.CONTINUE,
    }


@pytest.fixture
def sample_step_plan() -> StepPlan:
    """A simple atomic step plan."""
    return StepPlan(
        action="Search perplexity for tRNA delivery papers 2024-2026",
        agent="research",
        step_type=StepType.ATOMIC,
    )


@pytest.fixture
def sample_sequence_plan() -> StepPlan:
    """A sequence step plan with pipeline stages."""
    return StepPlan(
        action="Improve MOF synthesis section",
        step_type=StepType.SEQUENCE,
        steps=[
            StepPlan(action="Find relevant citations", agent="research"),
            StepPlan(action="Write paragraph based on citations", agent="writing"),
            StepPlan(action="Validate citations are justifiable", agent="research"),
            StepPlan(action="Keep or toss?", agent="writing"),
        ],
    )


@pytest.fixture
def sample_for_each_plan() -> StepPlan:
    """A for_each step plan."""
    return StepPlan(
        action="Improve paragraphs 20-22",
        step_type=StepType.FOR_EACH,
        items=[
            {"id": "para_20", "context": "Paragraph 20: Current synthesis via..."},
            {"id": "para_21", "context": "Paragraph 21: Solvothermal methods..."},
        ],
        template={
            "action": "Improve: {item.context}",
            "step_type": "sequence",
            "steps": [
                {"action": "Find citations for: {item.context}", "agent": "research"},
                {"action": "Write improved paragraph", "agent": "writing"},
            ],
        },
        collect={"action": "Review all improvements", "agent": "writing"},
    )
