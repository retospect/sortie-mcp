"""Shared fixtures for sortie-mcp tests."""

from __future__ import annotations

import contextlib
import os
import subprocess

import pytest

from sortie_mcp.models import (
    FailurePolicy,
    Priority,
    StepPlan,
    StepType,
)

# ---------------------------------------------------------------------------
# Shared PostgreSQL test database — used by every test file that needs a
# real DB connection (test_db.py, test_claim.py, scenario suites that
# don't provide their own DSN).
# ---------------------------------------------------------------------------

TEST_DB_NAME = "sortie_test"
PG_USER = "bots"
DEFAULT_DSN = f"postgresql://{PG_USER}@localhost/{TEST_DB_NAME}"


def _createdb() -> bool:
    """Create the test database. Returns True if created or already exists."""
    try:
        subprocess.run(
            ["createdb", "-U", PG_USER, TEST_DB_NAME],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        return b"already exists" in e.stderr
    except FileNotFoundError:
        return False


def _dropdb() -> None:
    """Drop the test database, ignoring errors."""
    with contextlib.suppress(FileNotFoundError):
        subprocess.run(
            ["dropdb", "-U", PG_USER, "--if-exists", TEST_DB_NAME],
            capture_output=True,
            check=False,
        )


@pytest.fixture(scope="session", autouse=True)
def _shared_test_database():
    """Create ``sortie_test`` once per session, install pgvector, drop on exit.

    Tests opt into using it by depending on ``DATABASE_URL`` (default
    ``DEFAULT_DSN``) and constructing their own ``DB(dsn, schema=...)``
    with a unique schema name.
    """
    # Honour an externally-provided DATABASE_URL (CI or shared test DB).
    if os.environ.get("DATABASE_URL"):
        yield
        return
    if not _createdb():
        pytest.skip("PostgreSQL not reachable — skipping DB integration tests")
    subprocess.run(
        [
            "psql",
            "-U",
            PG_USER,
            "-d",
            TEST_DB_NAME,
            "-c",
            "CREATE EXTENSION IF NOT EXISTS vector;",
        ],
        capture_output=True,
        check=False,
    )
    yield
    _dropdb()


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
