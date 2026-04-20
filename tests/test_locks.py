"""Tests for the resource-leases layer (migration 0003).

Covers:
- ``try_claim_with_locks`` atomic acquire (all-or-nothing)
- Hierarchical conflict matrix (EXCL vs SHARED, parent vs child keys)
- TTL expiry + ``reap_expired_leases``
- Lease release on complete / fail / request_input / reset_zombies
- ``heartbeat`` extends lease expires_at
- Concurrent racing claimants
- ``make_lock_key`` / ``key_*`` helpers
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from sortie_mcp.db import DB
from sortie_mcp.locks import (
    LockMode,
    key_ancestors,
    key_is_descendant_of,
    key_parent,
    lease_conflicts,
    make_lock_key,
)
from sortie_mcp.models import StepStatus

from .conftest import DEFAULT_DSN

pytestmark = pytest.mark.postgres


@pytest.fixture
async def db():
    """Schema-isolated DB for lease tests."""
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    schema = "sortie_locks_test"
    instance = DB(dsn, schema=schema)
    await instance.connect()
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.migrate()
    yield instance
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.close()


# ===========================================================================
# Pure helpers — no DB needed
# ===========================================================================


class TestLockKeyHelpers:
    def test_make_lock_key_no_slug(self) -> None:
        assert make_lock_key("file", "a/b.tex") == "file:a/b.tex"

    def test_make_lock_key_with_slug(self) -> None:
        assert make_lock_key("file", "a/b.tex", "PLXDX") == "file:a/b.tex§PLXDX"

    def test_make_lock_key_rejects_separator_in_kind(self) -> None:
        with pytest.raises(ValueError):
            make_lock_key("kind§bad", "x")
        with pytest.raises(ValueError):
            make_lock_key("kind:bad", "x")

    def test_key_parent(self) -> None:
        assert key_parent("file:a.tex§PLXDX") == "file:a.tex"
        assert key_parent("file:a.tex§sec1§PLXDX") == "file:a.tex§sec1"
        assert key_parent("file:a.tex") is None

    def test_key_ancestors_root_first(self) -> None:
        assert key_ancestors("file:a.tex§sec1§PLXDX") == [
            "file:a.tex",
            "file:a.tex§sec1",
        ]
        assert key_ancestors("file:a.tex") == []

    def test_key_is_descendant_of(self) -> None:
        assert key_is_descendant_of("file:a.tex§PLXDX", "file:a.tex")
        assert not key_is_descendant_of("file:a.tex", "file:a.tex§PLXDX")
        assert not key_is_descendant_of("file:a.tex", "file:a.tex")  # not strict

    def test_lease_conflicts_matrix(self) -> None:
        assert lease_conflicts(LockMode.EXCLUSIVE, LockMode.EXCLUSIVE)
        assert lease_conflicts(LockMode.EXCLUSIVE, LockMode.SHARED)
        assert lease_conflicts(LockMode.SHARED, LockMode.EXCLUSIVE)
        assert not lease_conflicts(LockMode.SHARED, LockMode.SHARED)


# ===========================================================================
# Atomic claim + lease acquire
# ===========================================================================


class TestTryClaimWithLocks:
    async def test_acquires_step_and_leases_atomically(self, db: DB) -> None:
        c = await db.create_campaign("Atomic acquire")
        s = await db.add_step(c.id, "edit ch01")
        keys = [
            make_lock_key("file", "ch01.tex"),
            make_lock_key("file", "ch01.tex", "PLXDX"),
        ]
        claimed = await db.try_claim_with_locks(s.id, keys, owner="runner-1")
        assert claimed is not None
        assert claimed.status == StepStatus.RUNNING
        assert claimed.claim_owner == "runner-1"
        assert claimed.claim_token is not None

        leases = await db.get_leases(s.id)
        assert {lease.resource_key for lease in leases} == set(keys)
        assert all(lease.mode == "exclusive" for lease in leases)

    async def test_no_keys_acquires_step_only(self, db: DB) -> None:
        c = await db.create_campaign("No-locks step")
        s = await db.add_step(c.id, "thinking")
        claimed = await db.try_claim_with_locks(s.id, [])
        assert claimed is not None
        assert claimed.status == StepStatus.RUNNING
        assert await db.get_leases(s.id) == []

    async def test_lost_race_returns_none_no_leases_taken(self, db: DB) -> None:
        c = await db.create_campaign("Race")
        s = await db.add_step(c.id, "thing")
        # Pre-claim by another path
        await db.claim_step(s.id, owner="other")
        result = await db.try_claim_with_locks(
            s.id, [make_lock_key("file", "x.tex")], owner="me"
        )
        assert result is None
        # No leases were inserted on the lost-race path
        assert await db.get_leases(s.id) == []


# ===========================================================================
# Conflict matrix — same key, parent, child
# ===========================================================================


class TestLeaseConflictMatrix:
    async def _hold(
        self,
        db: DB,
        campaign,
        key: str,
        mode: LockMode = LockMode.EXCLUSIVE,
        owner: str = "holder",
    ):
        """Helper: claim a fresh step and acquire one lease on ``key``."""
        s = await db.add_step(campaign.id, f"hold {key}")
        claimed = await db.try_claim_with_locks(s.id, [(key, mode)], owner=owner)
        assert claimed is not None, f"holder failed to acquire {key} {mode}"
        return s

    async def test_excl_blocks_excl_same_key(self, db: DB) -> None:
        c = await db.create_campaign("excl×excl same")
        await self._hold(db, c, "file:ch01.tex", LockMode.EXCLUSIVE)
        s2 = await db.add_step(c.id, "competitor")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.EXCLUSIVE)], owner="other"
        )
        assert result is None
        # Step row was rolled back to pending
        refreshed = await db.get_step(s2.id)
        assert refreshed.status == StepStatus.PENDING

    async def test_excl_blocks_shared_same_key(self, db: DB) -> None:
        c = await db.create_campaign("excl×shared same")
        await self._hold(db, c, "file:ch01.tex", LockMode.EXCLUSIVE)
        s2 = await db.add_step(c.id, "shared reader")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.SHARED)], owner="reader"
        )
        assert result is None

    async def test_shared_allows_shared_same_key(self, db: DB) -> None:
        c = await db.create_campaign("shared×shared")
        await self._hold(db, c, "file:ch01.tex", LockMode.SHARED)
        s2 = await db.add_step(c.id, "second reader")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.SHARED)], owner="r2"
        )
        assert result is not None
        # Both leases coexist
        held = await db.pool.fetchval(
            f"SELECT count(*) FROM {db._t('resource_leases')} "
            f"WHERE resource_key = 'file:ch01.tex'"
        )
        assert held == 2

    async def test_shared_blocks_excl_same_key(self, db: DB) -> None:
        c = await db.create_campaign("shared×excl")
        await self._hold(db, c, "file:ch01.tex", LockMode.SHARED)
        s2 = await db.add_step(c.id, "writer")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.EXCLUSIVE)], owner="w"
        )
        assert result is None

    async def test_parent_excl_blocks_child_lease(self, db: DB) -> None:
        """Holding the whole file exclusive blocks any paragraph lease."""
        c = await db.create_campaign("parent excl")
        await self._hold(db, c, "file:ch01.tex", LockMode.EXCLUSIVE)
        s2 = await db.add_step(c.id, "para edit")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex§PLXDX", LockMode.EXCLUSIVE)], owner="p"
        )
        assert result is None

    async def test_child_excl_blocks_parent_excl(self, db: DB) -> None:
        """Holding any paragraph exclusively blocks a whole-file reformat."""
        c = await db.create_campaign("child excl")
        await self._hold(db, c, "file:ch01.tex§PLXDX", LockMode.EXCLUSIVE)
        s2 = await db.add_step(c.id, "reformat")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.EXCLUSIVE)], owner="r"
        )
        assert result is None

    async def test_child_shared_allows_parent_shared(self, db: DB) -> None:
        """Two readers — one of paragraph, one of file — are compatible."""
        c = await db.create_campaign("child shared × parent shared")
        await self._hold(db, c, "file:ch01.tex§PLXDX", LockMode.SHARED)
        s2 = await db.add_step(c.id, "file reader")
        result = await db.try_claim_with_locks(
            s2.id, [("file:ch01.tex", LockMode.SHARED)], owner="r"
        )
        assert result is not None

    async def test_sibling_paragraphs_can_be_edited_concurrently(self, db: DB) -> None:
        """The whole point: two agents on different paragraphs of one file."""
        c = await db.create_campaign("siblings")
        await self._hold(
            db, c, "file:ch01.tex§PLXDX", LockMode.EXCLUSIVE, owner="agent-A"
        )
        s2 = await db.add_step(c.id, "edit other para")
        result = await db.try_claim_with_locks(
            s2.id,
            [("file:ch01.tex§ZZAAB", LockMode.EXCLUSIVE)],
            owner="agent-B",
        )
        assert result is not None

    async def test_partial_conflict_rolls_back_all_keys(self, db: DB) -> None:
        """If even one of N keys conflicts, the whole acquire is rolled back."""
        c = await db.create_campaign("all-or-nothing")
        await self._hold(db, c, "file:b.tex", LockMode.EXCLUSIVE)

        s = await db.add_step(c.id, "multi-file edit")
        result = await db.try_claim_with_locks(
            s.id,
            [
                "file:a.tex",  # available
                "file:b.tex",  # blocked
                "file:c.tex",  # available
            ],
            owner="me",
        )
        assert result is None

        # No partial leases — neither a.tex nor c.tex is held by s.
        leases = await db.get_leases(s.id)
        assert leases == []
        # And step is still pending.
        refreshed = await db.get_step(s.id)
        assert refreshed.status == StepStatus.PENDING


# ===========================================================================
# Lease lifecycle — release on complete / fail / request_input / reset_zombies
# ===========================================================================


class TestLeaseLifecycle:
    async def test_complete_step_releases_leases(self, db: DB) -> None:
        c = await db.create_campaign("release on complete")
        s = await db.add_step(c.id, "thing")
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"])
        assert await db.get_leases(s.id)

        await db.complete_step(s.id, "ok", claim_token=claimed.claim_token)
        assert await db.get_leases(s.id) == []

    async def test_fail_step_releases_leases_even_on_retry(self, db: DB) -> None:
        c = await db.create_campaign("release on fail-retry")
        s = await db.add_step(c.id, "flaky")
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"])

        # First failure → retry-pending (not final-failed)
        result = await db.fail_step(s.id, "transient", claim_token=claimed.claim_token)
        assert result.status == StepStatus.PENDING
        assert await db.get_leases(s.id) == []  # released for next attempt

    async def test_fail_step_releases_leases_on_final_fail(self, db: DB) -> None:
        c = await db.create_campaign("release on final-fail")
        s = await db.add_step(c.id, "bad")
        # Force max_retries=1 so a single fail goes straight to FAILED.
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} SET max_retries = 1 WHERE id = $1",
            s.id,
        )
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"])
        result = await db.fail_step(s.id, "boom", claim_token=claimed.claim_token)
        assert result.status == StepStatus.FAILED
        assert await db.get_leases(s.id) == []

    async def test_request_input_releases_leases(self, db: DB) -> None:
        c = await db.create_campaign("release on pause")
        s = await db.add_step(c.id, "needs help")
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"])
        await db.request_input(s.id, "what now?", claim_token=claimed.claim_token)
        assert await db.get_leases(s.id) == []

        # And another runner can now grab the same key
        s2 = await db.add_step(c.id, "competitor")
        result = await db.try_claim_with_locks(s2.id, ["file:a.tex"])
        assert result is not None

    async def test_reset_zombies_releases_leases(self, db: DB) -> None:
        c = await db.create_campaign("zombie + lease")
        s = await db.add_step(c.id, "stuck")
        await db.try_claim_with_locks(s.id, ["file:a.tex"], owner="dead")
        # Backdate heartbeat so the step is a zombie.
        await db.pool.execute(
            f"UPDATE {db._t('campaign_steps')} "
            f"SET heartbeat_at = now() - interval '1 hour' WHERE id = $1",
            s.id,
        )
        n = await db.reset_zombies(timeout_minutes=30)
        assert n == 1
        assert await db.get_leases(s.id) == []


# ===========================================================================
# TTL + reaper + heartbeat extension
# ===========================================================================


class TestLeaseTTL:
    async def test_expired_lease_does_not_block_new_acquire(self, db: DB) -> None:
        c = await db.create_campaign("expired")
        s1 = await db.add_step(c.id, "stale holder")
        await db.try_claim_with_locks(s1.id, ["file:a.tex"])
        # Force expiry on the held lease (without going through reaper).
        await db.pool.execute(
            f"UPDATE {db._t('resource_leases')} "
            f"SET expires_at = now() - interval '1 minute' WHERE step_id = $1",
            s1.id,
        )

        # A new claim on the same key must succeed despite the row existing.
        s2 = await db.add_step(c.id, "new holder")
        result = await db.try_claim_with_locks(s2.id, ["file:a.tex"])
        assert result is not None

    async def test_reap_expired_leases_deletes_stale_rows(self, db: DB) -> None:
        c = await db.create_campaign("reaper")
        s = await db.add_step(c.id, "x")
        await db.try_claim_with_locks(s.id, ["file:a.tex"])
        await db.pool.execute(
            f"UPDATE {db._t('resource_leases')} "
            f"SET expires_at = now() - interval '1 minute'"
        )

        n = await db.reap_expired_leases()
        assert n == 1

        remaining = await db.pool.fetchval(
            f"SELECT count(*) FROM {db._t('resource_leases')}"
        )
        assert remaining == 0

    async def test_heartbeat_extends_lease_expires_at(self, db: DB) -> None:
        c = await db.create_campaign("extend")
        s = await db.add_step(c.id, "long")
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"], ttl_sec=60)
        first = (await db.get_leases(s.id))[0].expires_at

        await asyncio.sleep(0.05)
        await db.heartbeat(s.id, claimed.claim_token, extend_leases_sec=300)
        second = (await db.get_leases(s.id))[0].expires_at
        assert second > first
        # ~300s in the future.
        delta = second - datetime.now(tz=UTC)
        assert timedelta(seconds=290) < delta < timedelta(seconds=310)

    async def test_heartbeat_with_extend_none_leaves_lease_alone(self, db: DB) -> None:
        c = await db.create_campaign("no-extend")
        s = await db.add_step(c.id, "x")
        claimed = await db.try_claim_with_locks(s.id, ["file:a.tex"], ttl_sec=60)
        first = (await db.get_leases(s.id))[0].expires_at

        await asyncio.sleep(0.05)
        await db.heartbeat(s.id, claimed.claim_token, extend_leases_sec=None)
        second = (await db.get_leases(s.id))[0].expires_at
        assert second == first


# ===========================================================================
# Concurrency
# ===========================================================================


class TestConcurrentLeaseClaims:
    async def test_only_one_of_many_concurrent_claims_wins_excl(self, db: DB) -> None:
        """N runners race for the same step + same exclusive key."""
        c = await db.create_campaign("excl race")
        s = await db.add_step(c.id, "contested")

        # We can't acquire the same step from N runners (claim race serializes
        # them naturally). The interesting race is N different steps competing
        # for one exclusive key.
        steps = [await db.add_step(c.id, f"runner-{i}") for i in range(5)]
        results = await asyncio.gather(
            *(
                db.try_claim_with_locks(st.id, ["file:hot.tex"], owner=f"r{i}")
                for i, st in enumerate(steps)
            ),
            return_exceptions=True,
        )

        # Filter out any retried-out SerializationErrors that exhaust retries
        # under heavy contention — they're a legitimate outcome and the
        # caller would re-tick. Surfaced as exceptions here.
        winners = [
            r for r in results if not isinstance(r, BaseException) and r is not None
        ]
        losers = [r for r in results if r is None]
        assert len(winners) == 1, f"{len(winners)} winners, {len(losers)} losers"

        # Exactly one lease in the table.
        n_leases = await db.pool.fetchval(
            f"SELECT count(*) FROM {db._t('resource_leases')} "
            f"WHERE resource_key = 'file:hot.tex'"
        )
        assert n_leases == 1
