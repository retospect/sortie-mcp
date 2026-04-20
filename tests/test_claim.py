"""Tests for multi-runner claim ownership (migration 0002).

Covers:
- ``claim_step`` stamps owner / token / heartbeat
- Concurrent claims race safely (only one wins)
- ``heartbeat`` keeps a long-running step alive
- ``reset_zombies`` uses heartbeat (not started_at) as staleness signal
- ``complete_step`` / ``fail_step`` / ``request_input`` reject mismatched
  claim_token (zombie-resurrection protection)
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from sortie_mcp.db import DB
from sortie_mcp.locks import default_owner
from sortie_mcp.models import StepStatus

from .conftest import DEFAULT_DSN

pytestmark = pytest.mark.postgres


@pytest.fixture
async def db():
    """Schema-isolated DB for claim tests."""
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    schema = "sortie_claim_test"
    instance = DB(dsn, schema=schema)
    await instance.connect()
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.migrate()
    yield instance
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.close()


class TestClaimOwnership:
    async def test_claim_stamps_owner_token_heartbeat(self, db: DB) -> None:
        c = await db.create_campaign("Claim test")
        s = await db.add_step(c.id, "do thing")

        claimed = await db.claim_step(s.id, owner="balthazar/runner-pid-99")
        assert claimed is not None
        assert claimed.status == StepStatus.RUNNING
        assert claimed.claim_owner == "balthazar/runner-pid-99"
        assert claimed.claim_token is not None
        assert claimed.heartbeat_at is not None
        assert claimed.started_at is not None

    async def test_claim_default_owner_includes_pid(self, db: DB) -> None:
        c = await db.create_campaign("Default owner")
        s = await db.add_step(c.id, "do thing")
        claimed = await db.claim_step(s.id)
        assert claimed is not None
        assert claimed.claim_owner == default_owner()
        assert str(os.getpid()) in claimed.claim_owner

    async def test_concurrent_claims_only_one_wins(self, db: DB) -> None:
        """Two runners racing to claim the same step → exactly one succeeds."""
        c = await db.create_campaign("Race test")
        s = await db.add_step(c.id, "contested step")

        # Issue 5 concurrent claims — only the first should win.
        results = await asyncio.gather(
            *(db.claim_step(s.id, owner=f"runner-{i}") for i in range(5))
        )
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        # Only the winner has a token
        assert winners[0].claim_token is not None
        # Final state matches the winner
        final = await db.get_step(s.id)
        assert final.claim_owner == winners[0].claim_owner
        assert final.claim_token == winners[0].claim_token


class TestHeartbeat:
    async def test_heartbeat_advances_timestamp(self, db: DB) -> None:
        c = await db.create_campaign("Heartbeat test")
        s = await db.add_step(c.id, "long step")
        claimed = await db.claim_step(s.id)

        first_hb = claimed.heartbeat_at
        await asyncio.sleep(0.05)
        beat = await db.heartbeat(s.id, claimed.claim_token)
        assert beat is not None
        assert beat.heartbeat_at > first_hb

    async def test_heartbeat_rejects_wrong_token(self, db: DB) -> None:
        c = await db.create_campaign("Wrong token")
        s = await db.add_step(c.id, "thing")
        await db.claim_step(s.id)
        wrong = uuid.uuid4()
        assert await db.heartbeat(s.id, wrong) is None

    async def test_heartbeat_rejects_after_zombie_reset(self, db: DB) -> None:
        c = await db.create_campaign("Resurrection test")
        s = await db.add_step(c.id, "thing")
        claimed = await db.claim_step(s.id)

        # Backdate heartbeat to make it a zombie.
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET heartbeat_at = now() - interval '1 hour' WHERE id = $1",
            s.id,
        )
        await db.reset_zombies(timeout_minutes=30)

        # Old owner cannot heartbeat — token is gone, status is pending.
        assert await db.heartbeat(s.id, claimed.claim_token) is None


class TestZombieResetUsesHeartbeat:
    async def test_fresh_heartbeat_is_not_zombie_even_with_old_started_at(
        self, db: DB
    ) -> None:
        """A long-running but heart-beating step is NOT a zombie."""
        c = await db.create_campaign("Long step")
        s = await db.add_step(c.id, "heavy compute")
        await db.claim_step(s.id)

        # started_at is ancient, but heartbeat is fresh.
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET started_at = now() - interval '4 hours' WHERE id = $1",
            s.id,
        )
        count = await db.reset_zombies(timeout_minutes=30)
        assert count == 0

    async def test_stale_heartbeat_is_zombie_even_with_fresh_started_at(
        self, db: DB
    ) -> None:
        """No-heartbeat-recently → zombie, regardless of started_at."""
        c = await db.create_campaign("Dead step")
        s = await db.add_step(c.id, "stuck")
        await db.claim_step(s.id)

        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET heartbeat_at = now() - interval '1 hour' WHERE id = $1",
            s.id,
        )
        count = await db.reset_zombies(timeout_minutes=30)
        assert count == 1

    async def test_zombie_reset_clears_claim_metadata(self, db: DB) -> None:
        c = await db.create_campaign("Cleanup")
        s = await db.add_step(c.id, "a step")
        await db.claim_step(s.id, owner="dead-runner")

        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET heartbeat_at = now() - interval '1 hour' WHERE id = $1",
            s.id,
        )
        await db.reset_zombies(timeout_minutes=30)

        refreshed = await db.get_step(s.id)
        assert refreshed.status == StepStatus.PENDING
        assert refreshed.claim_owner is None
        assert refreshed.claim_token is None
        assert refreshed.heartbeat_at is None


class TestClaimTokenEnforcement:
    async def test_complete_step_with_correct_token_succeeds(self, db: DB) -> None:
        c = await db.create_campaign("Token check")
        s = await db.add_step(c.id, "thing")
        claimed = await db.claim_step(s.id)

        result = await db.complete_step(s.id, "done", claim_token=claimed.claim_token)
        assert result is not None
        assert result.status == StepStatus.DONE

    async def test_complete_step_with_wrong_token_is_noop(self, db: DB) -> None:
        c = await db.create_campaign("Stale claim")
        s = await db.add_step(c.id, "thing")
        await db.claim_step(s.id)

        wrong = uuid.uuid4()
        result = await db.complete_step(s.id, "stolen", claim_token=wrong)
        assert result is None  # caller treats as STALE_CLAIM

        # Step is still running — the original claim is intact
        refreshed = await db.get_step(s.id)
        assert refreshed.status == StepStatus.RUNNING
        assert refreshed.output is None

    async def test_fail_step_with_wrong_token_is_noop(self, db: DB) -> None:
        c = await db.create_campaign("Stale fail")
        s = await db.add_step(c.id, "thing")
        await db.claim_step(s.id)

        wrong = uuid.uuid4()
        assert await db.fail_step(s.id, "boom", claim_token=wrong) is None

        refreshed = await db.get_step(s.id)
        assert refreshed.status == StepStatus.RUNNING
        assert refreshed.retry_count == 0

    async def test_request_input_with_wrong_token_is_noop(self, db: DB) -> None:
        c = await db.create_campaign("Stale waiting")
        s = await db.add_step(c.id, "thing")
        await db.claim_step(s.id)

        wrong = uuid.uuid4()
        assert await db.request_input(s.id, "what?", claim_token=wrong) is None

        refreshed = await db.get_step(s.id)
        assert refreshed.status == StepStatus.RUNNING

    async def test_no_token_passes_through_for_legacy_callers(self, db: DB) -> None:
        """``claim_token=None`` keeps pre-0.2 callers (runner direct path) working."""
        c = await db.create_campaign("Legacy")
        s = await db.add_step(c.id, "thing")
        await db.claim_step(s.id)

        # The runner's auto-complete from runtime response uses no token.
        result = await db.complete_step(s.id, "done from runner")
        assert result is not None
        assert result.status == StepStatus.DONE


class TestZombieResurrectionGuard:
    async def test_resurrected_zombie_cannot_overwrite_re_run(self, db: DB) -> None:
        """Plan §4.1: 'A resurrected zombie cannot overwrite a successful re-run.'

        Sequence:
          1. Runner A claims step → token T_A.
          2. Heartbeat goes stale → reset_zombies clears claim.
          3. Runner B re-claims → token T_B, completes successfully.
          4. Old runner A resurfaces and tries to complete with T_A → no-op.
        """
        c = await db.create_campaign("Resurrection")
        s = await db.add_step(c.id, "important step")

        # Runner A claims
        a_claim = await db.claim_step(s.id, owner="runner-A")
        a_token = a_claim.claim_token

        # Become zombie
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET heartbeat_at = now() - interval '1 hour' WHERE id = $1",
            s.id,
        )
        await db.reset_zombies(timeout_minutes=30)

        # Runner B re-claims and finishes properly
        b_claim = await db.claim_step(s.id, owner="runner-B")
        assert b_claim is not None
        await db.complete_step(s.id, "B's good output", claim_token=b_claim.claim_token)

        # Runner A wakes up and tries to overwrite with stale token
        result = await db.complete_step(
            s.id, "A's stale output (must be rejected)", claim_token=a_token
        )
        assert result is None

        # Final state is B's
        final = await db.get_step(s.id)
        assert final.status == StepStatus.DONE
        assert final.output == "B's good output"
