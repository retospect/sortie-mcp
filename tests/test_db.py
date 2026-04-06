"""Tests for sortie_mcp.db — requires PostgreSQL with pgvector.

These tests are integration tests. Run with:
    uv run pytest -m postgres

Automatically creates/drops a ``sortie_test`` database using the local
``bots`` account (override with DATABASE_URL env var).
"""

from __future__ import annotations

import os
import subprocess

import pytest

TEST_DB_NAME = "sortie_test"
DEFAULT_DSN = f"postgresql://bots@localhost/{TEST_DB_NAME}"
PG_USER = "bots"


def _pg_is_reachable() -> bool:
    """Quick check: can we reach local PG or the one in DATABASE_URL?"""
    try:
        import asyncpg  # noqa: F811
        return True
    except ImportError:
        return False


def _createdb() -> bool:
    """Create the test database. Returns True if created or already exists."""
    try:
        subprocess.run(
            ["createdb", "-U", PG_USER, TEST_DB_NAME],
            capture_output=True, check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        if b"already exists" in e.stderr:
            return True
        return False
    except FileNotFoundError:
        return False


def _dropdb() -> None:
    """Drop the test database, ignoring errors."""
    try:
        subprocess.run(
            ["dropdb", "-U", PG_USER, "--if-exists", TEST_DB_NAME],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        pass


# Skip the whole module if we can't reach PG
_has_db = os.environ.get("DATABASE_URL") or _createdb()
if _has_db and not os.environ.get("DATABASE_URL"):
    _dropdb()  # clean up probe; real create happens in fixture

pytestmark = pytest.mark.skipif(
    not _has_db,
    reason="PostgreSQL not reachable — skipping DB integration tests",
)


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    """Session fixture: createdb before tests, dropdb after."""
    _createdb()
    # Install pgvector extension
    subprocess.run(
        ["psql", "-U", PG_USER, "-d", TEST_DB_NAME,
         "-c", "CREATE EXTENSION IF NOT EXISTS vector;"],
        capture_output=True, check=False,
    )
    yield
    _dropdb()


@pytest.fixture
async def db(_test_database):
    """Per-test fixture: fresh schema via migrate(), torn down after."""
    from sortie_mcp.db import DB

    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    schema = "sortie_test"
    instance = DB(dsn, schema=schema)
    await instance.connect()
    # Clean slate
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.migrate()
    yield instance
    # Teardown
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.close()


@pytest.mark.postgres
class TestCampaignCRUD:
    async def test_create_and_get(self, db) -> None:
        campaign = await db.create_campaign(
            "Research tRNA engineering",
            name="tRNA Review",
            channel="research",
        )
        assert campaign.id is not None
        assert campaign.goal == "Research tRNA engineering"
        assert campaign.name == "tRNA Review"
        assert campaign.status.value == "active"
        assert campaign.priority.value == "normal"

        fetched = await db.get_campaign(campaign.id)
        assert fetched is not None
        assert fetched.id == campaign.id

    async def test_list_campaigns(self, db) -> None:
        await db.create_campaign("Goal A")
        await db.create_campaign("Goal B")
        campaigns = await db.list_campaigns()
        assert len(campaigns) >= 2

    async def test_list_by_status(self, db) -> None:
        from sortie_mcp.models import CampaignStatus

        await db.create_campaign("Active goal")
        await db.create_campaign("Paused goal", status=CampaignStatus.PAUSED)
        active = await db.list_campaigns(status=CampaignStatus.ACTIVE)
        paused = await db.list_campaigns(status=CampaignStatus.PAUSED)
        assert all(c.status.value == "active" for c in active)
        assert all(c.status.value == "paused" for c in paused)

    async def test_update_campaign(self, db) -> None:
        from sortie_mcp.models import CampaignStatus

        campaign = await db.create_campaign("Goal")
        updated = await db.update_campaign(
            campaign.id, status=CampaignStatus.PAUSED, strategy="New strategy"
        )
        assert updated is not None
        assert updated.status.value == "paused"
        assert updated.strategy == "New strategy"


@pytest.mark.postgres
class TestStepCRUD:
    async def test_add_and_get_step(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(campaign.id, "Search papers", agent="research")
        assert step.id is not None
        assert step.action == "Search papers"
        assert step.status.value == "pending"

        fetched = await db.get_step(step.id)
        assert fetched is not None
        assert fetched.id == step.id

    async def test_get_ready_steps(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        s1 = await db.add_step(campaign.id, "Step 1", agent="research")
        s2 = await db.add_step(
            campaign.id, "Step 2", agent="research", depends_on=[s1.id]
        )

        # Only s1 should be ready (no deps)
        ready = await db.get_ready_steps(campaign.id)
        assert len(ready) == 1
        assert ready[0].id == s1.id

        # Complete s1, now s2 should be ready
        await db.complete_step(s1.id, "Done searching")
        ready = await db.get_ready_steps(campaign.id)
        assert len(ready) == 1
        assert ready[0].id == s2.id

    async def test_claim_step(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(campaign.id, "Step 1", agent="research")

        claimed = await db.claim_step(step.id)
        assert claimed is not None
        assert claimed.status.value == "running"
        assert claimed.started_at is not None

        # Can't claim again
        again = await db.claim_step(step.id)
        assert again is None

    async def test_complete_step(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(campaign.id, "Step 1", agent="research")
        await db.claim_step(step.id)

        completed = await db.complete_step(step.id, "Found 5 papers")
        assert completed is not None
        assert completed.status.value == "done"
        assert completed.output == "Found 5 papers"

    async def test_fail_step_with_retry(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(campaign.id, "Step 1", agent="research")
        await db.claim_step(step.id)

        failed = await db.fail_step(step.id, "Timeout")
        assert failed is not None
        assert failed.retry_count == 1
        assert failed.status.value == "pending"  # Can retry

    async def test_fail_step_exhausted(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(
            campaign.id, "Step 1", agent="research"
        )
        # Exhaust retries
        for i in range(3):
            await db.claim_step(step.id)
            await db.fail_step(step.id, f"Error {i}")
            if i < 2:
                # Reset to pending for next retry attempt
                async with db.pool.acquire() as conn:
                    await conn.execute(
                        f"UPDATE {db._t('campaign_steps')} SET status = 'running' WHERE id = $1",
                        step.id,
                    )

        final = await db.get_step(step.id)
        assert final is not None
        assert final.status.value == "failed"

    async def test_fingerprint_dedup(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        s1 = await db.add_step(campaign.id, "Search papers", agent="research")
        await db.claim_step(s1.id)
        await db.complete_step(s1.id, "Found papers")

        # Check for duplicate
        dup = await db.find_duplicate(campaign.id, s1.fingerprint)
        assert dup is not None
        assert dup.id == s1.id

    async def test_reset_zombies(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        step = await db.add_step(campaign.id, "Step 1", agent="research")
        await db.claim_step(step.id)

        # Simulate old started_at
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {db._t('campaign_steps')} SET started_at = now() - interval '2 hours' WHERE id = $1",
                step.id,
            )

        count = await db.reset_zombies(timeout_minutes=30)
        assert count == 1

        refreshed = await db.get_step(step.id)
        assert refreshed is not None
        assert refreshed.status.value == "pending"


@pytest.mark.postgres
class TestSpawnAndContinue:
    async def test_basic_splice(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        s1 = await db.add_step(campaign.id, "Step 1", agent="research")
        s2 = await db.add_step(
            campaign.id, "Step 2", agent="writing", depends_on=[s1.id]
        )
        await db.claim_step(s1.id)

        result = await db.spawn_and_continue(
            s1.id,
            partial_output="Read main text. Need supplementary data.",
            subtasks=[
                {"action": "Fetch supplementary data", "agent": "research"},
            ],
            continuation_action="Complete analysis with supplementary data",
        )

        assert len(result["subtask_ids"]) == 1
        assert result["continuation_id"] is not None

        # Original step should be done with partial output
        s1_after = await db.get_step(s1.id)
        assert s1_after.status.value == "done"
        assert s1_after.output == "Read main text. Need supplementary data."

        # Continuation should inherit parent's parent_step_id
        cont = await db.get_step(result["continuation_id"])
        assert cont.continuation_of == s1.id
        assert cont.depth == s1_after.depth  # Same depth

        # s2 should now depend on continuation, not s1
        s2_after = await db.get_step(s2.id)
        assert result["continuation_id"] in s2_after.depends_on
        assert s1.id not in s2_after.depends_on

    async def test_depth_limit_prevents_spawn(self, db) -> None:
        campaign = await db.create_campaign("Goal", max_depth=1)
        step = await db.add_step(
            campaign.id, "Step at depth 1", agent="research", depth=1
        )
        await db.claim_step(step.id)

        with pytest.raises(ValueError, match="Depth limit reached"):
            await db.spawn_and_continue(
                step.id,
                partial_output="partial",
                subtasks=[{"action": "subtask"}],
                continuation_action="continue",
            )


@pytest.mark.postgres
class TestAbortBranch:
    async def test_basic_abort(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        # Create a sequence: parent → A → B → C
        parent = await db.add_step(
            campaign.id, "Exosome approach", step_type="sequence"
        )
        step_a = await db.add_step(
            campaign.id, "Find papers", agent="research",
            parent_step_id=parent.id, depth=1,
        )
        step_b = await db.add_step(
            campaign.id, "Read papers", agent="research",
            parent_step_id=parent.id, depth=1,
            depends_on=[step_a.id],
        )
        step_c = await db.add_step(
            campaign.id, "Write summary", agent="writing",
            parent_step_id=parent.id, depth=1,
            depends_on=[step_b.id],
        )

        # Complete A, claim B
        await db.claim_step(step_a.id)
        await db.complete_step(step_a.id, "Found papers")
        await db.claim_step(step_b.id)

        # B discovers the approach is pointless, aborts parent
        result = await db.abort_branch(
            step_b.id,
            parent.id,
            output="Exosome approach debunked",
            reason="Miller 2001 disproves this",
        )

        # B should be done (discoverer)
        b_after = await db.get_step(step_b.id)
        assert b_after.status.value == "done"

        # C should be skipped
        c_after = await db.get_step(step_c.id)
        assert c_after.status.value == "skipped"

        # Parent should be done with abort output
        parent_after = await db.get_step(parent.id)
        assert parent_after.status.value == "done"
        assert parent_after.output == "Exosome approach debunked"

    async def test_abort_rejects_non_descendant(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        s1 = await db.add_step(campaign.id, "Step 1", agent="research")
        s2 = await db.add_step(campaign.id, "Step 2", agent="research")
        await db.claim_step(s1.id)

        with pytest.raises(ValueError, match="not a descendant"):
            await db.abort_branch(s1.id, s2.id, "output", "reason")

    async def test_abort_rejects_done_target(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        parent = await db.add_step(campaign.id, "Parent")
        child = await db.add_step(
            campaign.id, "Child", parent_step_id=parent.id, depth=1
        )
        await db.claim_step(parent.id)
        await db.complete_step(parent.id, "Already done")
        await db.claim_step(child.id)

        with pytest.raises(ValueError, match="already done"):
            await db.abort_branch(child.id, parent.id, "output", "reason")


@pytest.mark.postgres
class TestNotes:
    async def test_add_and_get_notes(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        note = await db.add_note(
            campaign.id, "Important finding about tRNA", tags=["finding"]
        )
        assert note.id is not None

        notes = await db.get_notes(campaign.id)
        assert len(notes) == 1
        assert notes[0].content == "Important finding about tRNA"

    async def test_filter_by_tags(self, db) -> None:
        campaign = await db.create_campaign("Goal")
        await db.add_note(campaign.id, "Finding A", tags=["finding"])
        await db.add_note(campaign.id, "Citation B", tags=["citation"])

        findings = await db.get_notes(campaign.id, tags=["finding"])
        assert len(findings) == 1
        assert findings[0].content == "Finding A"


@pytest.mark.postgres
class TestNotifications:
    async def test_notify_and_deliver(self, db) -> None:
        from sortie_mcp.models import NotificationLevel

        campaign = await db.create_campaign("Goal", channel="research")
        notif = await db.notify(
            campaign.id, "research", "Phase 1 complete", NotificationLevel.MILESTONE
        )
        assert notif.id is not None
        assert notif.delivered is False

        undelivered = await db.get_undelivered_notifications()
        assert len(undelivered) == 1

        await db.mark_delivered([notif.id])
        undelivered = await db.get_undelivered_notifications()
        assert len(undelivered) == 0
