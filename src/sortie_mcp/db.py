"""Database layer for sortie-mcp — schema migration, queries, and transactions.

All SQL uses a configurable schema name (default: ``sortie``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from .locks import KEY_SEP, LockMode
from .models import (
    Campaign,
    CampaignStatus,
    FailurePolicy,
    Note,
    Notification,
    NotificationLevel,
    Priority,
    ResourceLease,
    Step,
    StepStatus,
    StepType,
    compute_fingerprint,
)

log = logging.getLogger(__name__)


class _LockConflict(Exception):
    """Internal sentinel raised inside :meth:`DB._try_claim_with_locks_once`
    to roll back the SERIALIZABLE transaction when a requested lease
    conflicts with an existing one. Caller translates to ``None``."""


class DB:
    """Async database interface for sortie-mcp.

    Usage::

        db = DB("postgresql://...", schema="sortie")
        await db.connect()
        await db.migrate()
        ...
        await db.close()
    """

    def __init__(self, dsn: str, *, schema: str = "sortie") -> None:
        self.dsn = dsn
        self.schema = schema
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        log.info("Connected to database (schema=%s)", self.schema)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("DB not connected — call await db.connect() first")
        return self._pool

    async def migrate(self) -> None:
        """Apply any pending migrations via yoyo.

        Safe to call on every startup: yoyo takes an advisory lock and
        only applies migrations whose ids are not already in the
        ``_sortie_migrations__<schema>`` tracking table.
        """
        async with self.pool.acquire() as conn:
            # pgvector extension (may require superuser; skip gracefully).
            # The tracking table is schema-public, so this must succeed
            # before yoyo imports the migration files that reference
            # ``vector(384)``.
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except asyncpg.InsufficientPrivilegeError:
                log.warning(
                    "Cannot create vector extension — must be created by superuser"
                )

        # yoyo is synchronous; run it in a worker thread so we don't
        # block the event loop during startup.
        from ._migrate_impl import apply_migrations

        await asyncio.to_thread(apply_migrations, self.dsn, self.schema)
        log.info("Schema migration complete (schema=%s)", self.schema)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _t(self, table: str) -> str:
        """Qualify a table name with the schema."""
        return f"{self.schema}.{table}"

    def _row_to_campaign(self, row: asyncpg.Record) -> Campaign:
        return Campaign(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            status=CampaignStatus(row["status"]),
            strategy=row["strategy"],
            progress=row["progress"],
            channel=row["channel"],
            user_id=row["user_id"],
            max_depth=row["max_depth"],
            token_budget=row["token_budget"],
            tokens_used=row["tokens_used"],
            failure_policy=FailurePolicy(row["failure_policy"]),
            priority=Priority(row["priority"]),
            next_action_at=row["next_action_at"],
            last_reported_at=row["last_reported_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_step(self, row: asyncpg.Record) -> Step:
        return Step(
            id=row["id"],
            campaign_id=row["campaign_id"],
            action=row["action"],
            step_type=StepType(row["step_type"]),
            status=StepStatus(row["status"]),
            parent_step_id=row["parent_step_id"],
            depth=row["depth"],
            agent=row["agent"],
            failure_policy=FailurePolicy(row["failure_policy"]),
            depends_on=list(row["depends_on"] or []),
            input=row["input"],
            output=row["output"],
            error=row["error"],
            fingerprint=row["fingerprint"],
            continuation_of=row["continuation_of"],
            completion_threshold=row["completion_threshold"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            tokens_used=row["tokens_used"],
            duration_ms=row["duration_ms"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            claim_owner=row["claim_owner"],
            claim_token=row["claim_token"],
            heartbeat_at=row["heartbeat_at"],
            requires_locks=list(row["requires_locks"] or []),
        )

    def _row_to_note(self, row: asyncpg.Record) -> Note:
        return Note(
            id=row["id"],
            campaign_id=row["campaign_id"],
            content=row["content"],
            step_id=row["step_id"],
            agent=row["agent"],
            tags=list(row["tags"] or []),
            created_at=row["created_at"],
        )

    def _row_to_notification(self, row: asyncpg.Record) -> Notification:
        return Notification(
            id=row["id"],
            campaign_id=row["campaign_id"],
            channel=row["channel"],
            message=row["message"],
            level=NotificationLevel(row["level"]),
            delivered=row["delivered"],
            created_at=row["created_at"],
        )

    def _row_to_lease(self, row: asyncpg.Record) -> ResourceLease:
        return ResourceLease(
            resource_key=row["resource_key"],
            step_id=row["step_id"],
            owner=row["owner"],
            mode=row["mode"],
            acquired_at=row["acquired_at"],
            expires_at=row["expires_at"],
        )

    # ------------------------------------------------------------------
    # Campaign CRUD
    # ------------------------------------------------------------------

    async def create_campaign(
        self,
        goal: str,
        *,
        name: str | None = None,
        channel: str | None = None,
        user_id: str | None = None,
        max_depth: int = 4,
        token_budget: int | None = None,
        failure_policy: FailurePolicy = FailurePolicy.CONTINUE,
        priority: Priority = Priority.NORMAL,
        status: CampaignStatus = CampaignStatus.ACTIVE,
    ) -> Campaign:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self._t("campaigns")}
                (name, goal, status, channel, user_id, max_depth,
                 token_budget, failure_policy, priority)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            name,
            goal,
            status.value,
            channel,
            user_id,
            max_depth,
            token_budget,
            failure_policy.value,
            priority.value,
        )
        return self._row_to_campaign(row)

    async def get_campaign(self, campaign_id: UUID) -> Campaign | None:
        row = await self.pool.fetchrow(
            f"SELECT * FROM {self._t('campaigns')} WHERE id = $1",
            campaign_id,
        )
        return self._row_to_campaign(row) if row else None

    async def list_campaigns(
        self, *, status: CampaignStatus | None = None
    ) -> list[Campaign]:
        if status:
            rows = await self.pool.fetch(
                f"SELECT * FROM {self._t('campaigns')} WHERE status = $1 ORDER BY created_at DESC",
                status.value,
            )
        else:
            rows = await self.pool.fetch(
                f"SELECT * FROM {self._t('campaigns')} ORDER BY created_at DESC"
            )
        return [self._row_to_campaign(r) for r in rows]

    async def update_campaign(
        self, campaign_id: UUID, **fields: Any
    ) -> Campaign | None:
        if not fields:
            return await self.get_campaign(campaign_id)
        sets = []
        vals: list[Any] = []
        for i, (k, v) in enumerate(fields.items(), start=2):
            sets.append(f"{k} = ${i}")
            vals.append(
                v.value
                if isinstance(v, (CampaignStatus, FailurePolicy, Priority))
                else v
            )
        sets.append("updated_at = now()")
        row = await self.pool.fetchrow(
            f"UPDATE {self._t('campaigns')} SET {', '.join(sets)} WHERE id = $1 RETURNING *",
            campaign_id,
            *vals,
        )
        return self._row_to_campaign(row) if row else None

    async def set_last_reported(self, campaign_id: UUID) -> None:
        await self.pool.execute(
            f"UPDATE {self._t('campaigns')} SET last_reported_at = now() WHERE id = $1",
            campaign_id,
        )

    # ------------------------------------------------------------------
    # Step CRUD
    # ------------------------------------------------------------------

    async def add_step(
        self,
        campaign_id: UUID,
        action: str,
        *,
        agent: str | None = None,
        step_type: StepType = StepType.ATOMIC,
        parent_step_id: int | None = None,
        depth: int = 0,
        depends_on: list[int] | None = None,
        input_text: str | None = None,
        failure_policy: FailurePolicy = FailurePolicy.CONTINUE,
        continuation_of: int | None = None,
        completion_threshold: int | None = None,
        requires_locks: list[str] | None = None,
    ) -> Step:
        # Canonical resolution: resolve depends_on through continuation chains
        resolved_deps = await self._resolve_deps(campaign_id, depends_on or [])
        fp = compute_fingerprint(action, agent, input_text)
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self._t("campaign_steps")}
                (campaign_id, action, agent, step_type, parent_step_id,
                 depth, depends_on, input, failure_policy, fingerprint,
                 continuation_of, completion_threshold, requires_locks)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING *
            """,
            campaign_id,
            action,
            agent,
            step_type.value,
            parent_step_id,
            depth,
            resolved_deps or None,
            input_text,
            failure_policy.value,
            fp,
            continuation_of,
            completion_threshold,
            requires_locks or None,
        )
        return self._row_to_step(row)

    async def get_step(self, step_id: int) -> Step | None:
        row = await self.pool.fetchrow(
            f"SELECT * FROM {self._t('campaign_steps')} WHERE id = $1",
            step_id,
        )
        return self._row_to_step(row) if row else None

    async def get_steps(
        self, campaign_id: UUID, *, status: StepStatus | None = None
    ) -> list[Step]:
        if status:
            rows = await self.pool.fetch(
                f"SELECT * FROM {self._t('campaign_steps')} WHERE campaign_id = $1 AND status = $2 ORDER BY id",
                campaign_id,
                status.value,
            )
        else:
            rows = await self.pool.fetch(
                f"SELECT * FROM {self._t('campaign_steps')} WHERE campaign_id = $1 ORDER BY id",
                campaign_id,
            )
        return [self._row_to_step(r) for r in rows]

    async def get_ready_steps(self, campaign_id: UUID) -> list[Step]:
        """Get pending steps whose dependencies are all done."""
        rows = await self.pool.fetch(
            f"""
            SELECT s.* FROM {self._t("campaign_steps")} s
            WHERE s.campaign_id = $1
              AND s.status = 'pending'
              AND s.retry_count < s.max_retries
              AND NOT EXISTS (
                  SELECT 1
                  FROM unnest(s.depends_on) AS dep_id
                  JOIN {self._t("campaign_steps")} d ON d.id = dep_id
                  WHERE d.status != 'done'
              )
            ORDER BY s.id
            """,
            campaign_id,
        )
        return [self._row_to_step(r) for r in rows]

    async def claim_step(self, step_id: int, owner: str | None = None) -> Step | None:
        """Atomically claim a pending step for execution.

        Stamps ``claim_owner``, a fresh ``claim_token`` (UUID), and
        ``heartbeat_at = now()``. Subsequent ``complete_step`` /
        ``fail_step`` / ``request_input`` calls must present this same
        ``claim_token`` or they receive a ``stale_claim`` no-op
        (see :data:`sortie_mcp.locks.STALE_CLAIM`).

        ``owner`` defaults to :func:`sortie_mcp.locks.default_owner`
        (``<host>/runner-pid-<pid>``).
        """
        from .locks import default_owner

        owner = owner or default_owner()
        token = uuid4()
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status        = 'running',
                started_at    = now(),
                heartbeat_at  = now(),
                claim_owner   = $2,
                claim_token   = $3
            WHERE id = $1 AND status = 'pending'
            RETURNING *
            """,
            step_id,
            owner,
            token,
        )
        return self._row_to_step(row) if row else None

    async def heartbeat(
        self,
        step_id: int,
        claim_token: UUID | str,
        *,
        extend_leases_sec: int | None = 900,
    ) -> Step | None:
        """Refresh ``heartbeat_at`` to defer zombie reset.

        Also extends the ``expires_at`` of every resource lease held by
        this step (by ``extend_leases_sec`` from now), so a healthy
        long-running worker doesn't lose its locks to the reaper.
        Pass ``extend_leases_sec=None`` to skip lease extension.

        Returns the updated Step on success, or ``None`` if the step is
        no longer claimed by this token (zombie-reset, completion, etc.
        — caller should stop working).
        """
        token = UUID(claim_token) if isinstance(claim_token, str) else claim_token
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"""
                UPDATE {self._t("campaign_steps")}
                SET heartbeat_at = now()
                WHERE id = $1
                  AND status = 'running'
                  AND claim_token = $2
                RETURNING *
                """,
                step_id,
                token,
            )
            if not row:
                return None
            if extend_leases_sec is not None:
                await conn.execute(
                    f"""
                    UPDATE {self._t("resource_leases")}
                    SET expires_at = now() + ($2 || ' seconds')::interval
                    WHERE step_id = $1
                    """,
                    step_id,
                    str(extend_leases_sec),
                )
            return self._row_to_step(row)

    # ------------------------------------------------------------------
    # Resource leases (migration 0003)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_lock_keys(
        keys: list[str] | list[tuple[str, LockMode | str]],
        default_mode: LockMode,
    ) -> list[tuple[str, str]]:
        """Coerce a mixed input into a ``[(key, mode_str), ...]`` list."""
        out: list[tuple[str, str]] = []
        for item in keys:
            if isinstance(item, tuple):
                key, mode = item
                mode_str = mode.value if isinstance(mode, LockMode) else str(mode)
            else:
                key = item
                mode_str = default_mode.value
            if mode_str not in ("exclusive", "shared"):
                raise ValueError(f"invalid lock mode {mode_str!r} for key {key!r}")
            out.append((key, mode_str))
        return out

    async def try_claim_with_locks(
        self,
        step_id: int,
        keys: list[str] | list[tuple[str, LockMode | str]],
        *,
        owner: str | None = None,
        ttl_sec: int = 900,
        default_mode: LockMode = LockMode.EXCLUSIVE,
        max_retries: int = 3,
    ) -> Step | None:
        """Atomically claim a step *and* acquire all requested resource leases.

        Returns the claimed :class:`~sortie_mcp.models.Step` (with fresh
        ``claim_token``) on success, or ``None`` if either:

        - the step is no longer ``pending`` (lost the claim race), or
        - any of ``keys`` conflicts with an existing non-expired lease
          (hierarchical EXCL/SHARED conflict per :func:`sortie_mcp.locks.lease_conflicts`).

        Under contention the caller should treat ``None`` as "skip and
        try a different ready step" — leases are revisited on the next
        runner tick.

        ``keys`` accepts plain strings (all use ``default_mode``) or
        ``(key, mode)`` tuples for mixed-mode atomic acquire.
        """
        from .locks import default_owner as _default_owner

        owner = owner or _default_owner()
        normalized = self._normalize_lock_keys(keys, default_mode)

        for attempt in range(max_retries):
            try:
                return await self._try_claim_with_locks_once(
                    step_id, normalized, owner, ttl_sec
                )
            except _LockConflict:
                # A requested key conflicts with an existing lease.
                # Step claim was rolled back by the SERIALIZABLE
                # transaction, so the row is back to ``pending``.
                return None
            except asyncpg.SerializationError:
                # Two concurrent claimers raced through SERIALIZABLE; back
                # off briefly and retry. The second claimer will likely
                # see the first's row on retry and return None cleanly.
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01 * (attempt + 1))
        return None  # unreachable

    async def _try_claim_with_locks_once(
        self,
        step_id: int,
        normalized_keys: list[tuple[str, str]],
        owner: str,
        ttl_sec: int,
    ) -> Step | None:
        token = uuid4()
        async with (
            self.pool.acquire() as conn,
            conn.transaction(isolation="serializable"),
        ):
            # Stage 1 — claim the step row (only if still pending).
            row = await conn.fetchrow(
                f"""
                    UPDATE {self._t("campaign_steps")}
                    SET status        = 'running',
                        started_at    = now(),
                        heartbeat_at  = now(),
                        claim_owner   = $2,
                        claim_token   = $3
                    WHERE id = $1 AND status = 'pending'
                    RETURNING *
                    """,
                step_id,
                owner,
                token,
            )
            if not row:
                return None  # lost the race

            # Stage 2 — check every requested key for hierarchical conflicts.
            # Build the prefix-conflict predicate per requested key.
            # Held EXCL conflicts with anything; held SHARED conflicts
            # only with requested EXCL.
            for req_key, req_mode in normalized_keys:
                conflict = await conn.fetchval(
                    f"""
                        SELECT 1 FROM {self._t("resource_leases")}
                        WHERE expires_at > now()
                          AND step_id <> $1
                          AND (
                              resource_key = $2
                              OR $2 LIKE resource_key || $3 || '%'
                              OR resource_key LIKE $2 || $3 || '%'
                          )
                          AND ($4 = 'exclusive' OR mode = 'exclusive')
                        LIMIT 1
                        """,
                    step_id,
                    req_key,
                    KEY_SEP,
                    req_mode,
                )
                if conflict:
                    # Roll back the claim — caller will see None.
                    raise _LockConflict()

            # Stage 3 — insert all the leases atomically.
            if normalized_keys:
                await conn.executemany(
                    f"""
                        INSERT INTO {self._t("resource_leases")}
                            (resource_key, step_id, owner, mode, expires_at)
                        VALUES ($1, $2, $3, $4, now() + ($5 || ' seconds')::interval)
                        ON CONFLICT (resource_key, step_id) DO UPDATE
                            SET mode = EXCLUDED.mode,
                                owner = EXCLUDED.owner,
                                expires_at = EXCLUDED.expires_at
                        """,
                    [
                        (req_key, step_id, owner, req_mode, str(ttl_sec))
                        for req_key, req_mode in normalized_keys
                    ],
                )
            return self._row_to_step(row)

    async def release_leases(self, step_id: int) -> int:
        """Drop every lease held by ``step_id``. Returns rows deleted."""
        result = await self.pool.execute(
            f"DELETE FROM {self._t('resource_leases')} WHERE step_id = $1",
            step_id,
        )
        return int(result.split()[-1])

    async def get_leases(self, step_id: int) -> list[ResourceLease]:
        """List all (non-expired) leases held by a step."""
        rows = await self.pool.fetch(
            f"""
            SELECT * FROM {self._t("resource_leases")}
            WHERE step_id = $1 AND expires_at > now()
            ORDER BY resource_key
            """,
            step_id,
        )
        return [self._row_to_lease(r) for r in rows]

    async def reap_expired_leases(self) -> int:
        """Delete leases whose ``expires_at`` is in the past.

        The runner calls this each tick alongside :meth:`reset_zombies`.
        Healthy workers extend their leases via :meth:`heartbeat`.
        """
        result = await self.pool.execute(
            f"DELETE FROM {self._t('resource_leases')} WHERE expires_at <= now()"
        )
        count = int(result.split()[-1])
        if count:
            log.info("Reaped %d expired resource leases", count)
        return count

    async def complete_step(
        self,
        step_id: int,
        output: str,
        *,
        tokens_used: int | None = None,
        duration_ms: int | None = None,
        claim_token: UUID | str | None = None,
    ) -> Step | None:
        """Mark a step as done. Returns None if step was already skipped.

        If ``claim_token`` is provided, the update is gated on token match;
        a mismatch returns ``None`` (caller should treat as ``stale_claim``).
        ``claim_token=None`` preserves pre-0.2 behaviour for trusted
        in-process callers (e.g. the runner's auto-complete from runtime
        responses).
        """
        token = UUID(claim_token) if isinstance(claim_token, str) else claim_token
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status = CASE WHEN status = 'skipped' THEN 'skipped' ELSE 'done' END,
                output = $2,
                tokens_used = $3,
                duration_ms = $4,
                completed_at = now()
            WHERE id = $1
              AND ($5::uuid IS NULL OR claim_token = $5::uuid)
            RETURNING *
            """,
            step_id,
            output,
            tokens_used,
            duration_ms,
            token,
        )
        if not row:
            return None
        step = self._row_to_step(row)
        # Release any held resource leases — work is done.
        await self.release_leases(step_id)
        # If the step was already skipped (branch abort while running),
        # return it so the caller can see the status
        if step.status == StepStatus.SKIPPED:
            return step
        # Check if parent group is now satisfied
        if step.parent_step_id is not None:
            await self._check_parent_completion(step.parent_step_id)
        # Accumulate tokens on campaign
        if tokens_used:
            await self.pool.execute(
                f"UPDATE {self._t('campaigns')} SET tokens_used = tokens_used + $2 WHERE id = $1",
                step.campaign_id,
                tokens_used,
            )
        return step

    async def fail_step(
        self,
        step_id: int,
        error: str,
        *,
        claim_token: UUID | str | None = None,
    ) -> Step | None:
        """Increment retry count and record error. If max retries exceeded,
        mark as failed and optionally cascade via fail_fast.

        ``claim_token`` semantics match :meth:`complete_step` — when set,
        a mismatched token returns ``None`` (stale claim).
        """
        token = UUID(claim_token) if isinstance(claim_token, str) else claim_token
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET retry_count = retry_count + 1,
                error = $2,
                status = CASE
                    WHEN retry_count + 1 >= max_retries THEN 'failed'
                    ELSE 'pending'
                END,
                completed_at = CASE
                    WHEN retry_count + 1 >= max_retries THEN now()
                    ELSE NULL
                END,
                -- Clear claim on retry-pending OR final-failed so the
                -- next claimer gets a clean token slot.
                claim_owner  = NULL,
                claim_token  = NULL,
                heartbeat_at = NULL
            WHERE id = $1
              AND ($3::uuid IS NULL OR claim_token = $3::uuid)
            RETURNING *
            """,
            step_id,
            error,
            token,
        )
        if not row:
            return None
        step = self._row_to_step(row)
        # Release any held resource leases — whether retrying or final-failed,
        # the leases must be dropped so other work / the next attempt can
        # re-acquire them.
        await self.release_leases(step_id)
        if (
            step.status == StepStatus.FAILED
            and step.failure_policy == FailurePolicy.FAIL_FAST
        ):
            await self._cascade_fail_fast(step)
        return step

    async def request_input(
        self,
        step_id: int,
        question: str,
        *,
        partial_output: str | None = None,
        claim_token: UUID | str | None = None,
    ) -> Step | None:
        """Mark a running step as waiting for human/coordinator input.

        The agent calls this when it cannot proceed without a decision.
        The question is stored in ``output`` (alongside any partial work)
        and the step pauses until ``provide_input`` resumes it.

        ``claim_token`` semantics match :meth:`complete_step`.
        """
        combined = question
        if partial_output:
            combined = f"{partial_output}\n\n[WAITING FOR INPUT]: {question}"
        token = UUID(claim_token) if isinstance(claim_token, str) else claim_token
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status = 'waiting_input',
                output = $2
            WHERE id = $1
              AND status = 'running'
              AND ($3::uuid IS NULL OR claim_token = $3::uuid)
            RETURNING *
            """,
            step_id,
            combined,
            token,
        )
        if not row:
            return None
        # Step is paused — release leases so unrelated work can proceed.
        # ``provide_input`` will return the step to ``pending`` and the
        # next claimer must re-acquire its leases.
        await self.release_leases(step_id)
        return self._row_to_step(row)

    async def provide_input(
        self,
        step_id: int,
        answer: str,
    ) -> Step | None:
        """Provide input to a waiting step, returning it to pending for re-dispatch.

        The answer is stored in ``input`` so the agent sees it on the next run.
        Claim metadata is cleared so the step can be re-claimed by any runner.
        """
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status       = 'pending',
                input        = $2,
                started_at   = NULL,
                heartbeat_at = NULL,
                claim_owner  = NULL,
                claim_token  = NULL
            WHERE id = $1 AND status = 'waiting_input'
            RETURNING *
            """,
            step_id,
            answer,
        )
        return self._row_to_step(row) if row else None

    async def reset_zombies(self, timeout_minutes: int = 30) -> int:
        """Reset steps stuck in 'running' past timeout back to 'pending'.

        Staleness is measured by ``heartbeat_at`` (kept fresh by
        long-running but healthy steps via :meth:`heartbeat`), falling
        back to ``started_at`` for legacy rows that pre-date migration
        0002. Clears claim metadata so the step can be re-claimed and
        invalidates the previous owner's token.

        Does NOT reset ``waiting_input`` steps — those are legitimately paused.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            # Reset rows and capture their IDs so we can drop the
            # corresponding leases in the same transaction.
            rows = await conn.fetch(
                f"""
                UPDATE {self._t("campaign_steps")}
                SET status       = 'pending',
                    started_at   = NULL,
                    heartbeat_at = NULL,
                    claim_owner  = NULL,
                    claim_token  = NULL
                WHERE status = 'running'
                  AND COALESCE(heartbeat_at, started_at)
                      < now() - ($1 || ' minutes')::interval
                RETURNING id
                """,
                str(timeout_minutes),
            )
            count = len(rows)
            if count:
                step_ids = [r["id"] for r in rows]
                await conn.execute(
                    f"DELETE FROM {self._t('resource_leases')} "
                    f"WHERE step_id = ANY($1::int[])",
                    step_ids,
                )
                log.info("Reset %d zombie steps (timeout=%dm)", count, timeout_minutes)
            return count

    async def count_running(self) -> int:
        """Count all currently running steps across all campaigns."""
        row = await self.pool.fetchrow(
            f"SELECT COUNT(*) AS n FROM {self._t('campaign_steps')} WHERE status = 'running'"
        )
        return row["n"] if row else 0

    # ------------------------------------------------------------------
    # DAG Splice (spawn_and_continue)
    # ------------------------------------------------------------------

    async def spawn_and_continue(
        self,
        step_id: int,
        partial_output: str,
        subtasks: list[dict[str, Any]],
        continuation_action: str,
    ) -> dict[str, Any]:
        """Atomic DAG splice. See spec: DAG Splice section."""
        step = await self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        campaign = await self.get_campaign(step.campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {step.campaign_id} not found")

        # Check depth limit for subtasks
        if step.depth + 1 > campaign.max_depth:
            raise ValueError(
                f"Depth limit reached ({step.depth + 1} > {campaign.max_depth}). "
                "Cannot spawn subtasks."
            )

        async with self.pool.acquire() as conn, conn.transaction():
            # 1. Mark current step done with partial output
            await conn.execute(
                f"""
                    UPDATE {self._t("campaign_steps")}
                    SET status = 'done', output = $2, completed_at = now()
                    WHERE id = $1
                    """,
                step_id,
                partial_output,
            )

            # 2. Create subtask steps
            subtask_ids = []
            for st in subtasks:
                fp = compute_fingerprint(st["action"], st.get("agent"), st.get("input"))
                row = await conn.fetchrow(
                    f"""
                        INSERT INTO {self._t("campaign_steps")}
                            (campaign_id, action, agent, step_type, parent_step_id,
                             depth, depends_on, input, fingerprint)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                    step.campaign_id,
                    st["action"],
                    st.get("agent", step.agent),
                    st.get("step_type", "atomic"),
                    step_id,
                    step.depth + 1,
                    [step_id],
                    st.get("input"),
                    fp,
                )
                subtask_ids.append(row["id"])

            # 3. Create continuation step (inherits parent's parent_step_id
            # and depth — same logical step split in time)
            cont_fp = compute_fingerprint(continuation_action, step.agent, None)
            cont_row = await conn.fetchrow(
                f"""
                    INSERT INTO {self._t("campaign_steps")}
                        (campaign_id, action, agent, step_type, parent_step_id,
                         depth, depends_on, continuation_of, fingerprint)
                    VALUES ($1, $2, $3, 'atomic', $4, $5, $6, $7, $8)
                    RETURNING id
                    """,
                step.campaign_id,
                continuation_action,
                step.agent,
                step.parent_step_id,  # inherits parent's parent
                step.depth,  # same depth
                subtask_ids,
                step_id,
                cont_fp,
            )
            cont_id = cont_row["id"]

            # 4. Retarget downstream deps: anything that depended on
            # this step now depends on the continuation
            await conn.execute(
                f"""
                    UPDATE {self._t("campaign_steps")}
                    SET depends_on = array_replace(depends_on, $1, $2)
                    WHERE campaign_id = $3
                      AND $1 = ANY(depends_on)
                      AND id != ALL($4::int[])
                    """,
                step_id,
                cont_id,
                step.campaign_id,
                [*subtask_ids, cont_id],
            )

        return {"subtask_ids": subtask_ids, "continuation_id": cont_id}

    # ------------------------------------------------------------------
    # Branch Abort
    # ------------------------------------------------------------------

    async def abort_branch(
        self,
        step_id: int,
        target_id: int,
        output: str,
        reason: str,
    ) -> dict[str, Any]:
        """Scoped early return from an ancestor step. See spec: Branch Abort."""
        step = await self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        target = await self.get_step(target_id)
        if not target:
            raise ValueError(f"Target step {target_id} not found")
        if target.status == StepStatus.DONE:
            raise ValueError(f"Target step {target_id} is already done — cannot abort")
        if target.campaign_id != step.campaign_id:
            raise ValueError("Step and target must be in the same campaign")

        # Verify ancestry: step must be a descendant of target
        if not await self._is_descendant(step_id, target_id):
            raise ValueError(
                f"Step {step_id} is not a descendant of step {target_id}. Cannot abort."
            )

        async with self.pool.acquire() as conn, conn.transaction():
            # 1. Mark discoverer as done
            await conn.execute(
                f"""
                UPDATE {self._t("campaign_steps")}
                SET status = 'done', output = $2, completed_at = now()
                WHERE id = $1
                """,
                step_id,
                reason,
            )

            # 2. Skip all pending/running descendants of target (except discoverer)
            skipped = await conn.fetch(
                f"""
                WITH RECURSIVE descendants AS (
                    SELECT id FROM {self._t("campaign_steps")}
                    WHERE parent_step_id = $1 AND id != $2
                    UNION ALL
                    SELECT s.id FROM {self._t("campaign_steps")} s
                    JOIN descendants d ON s.parent_step_id = d.id
                    WHERE s.id != $2
                )
                UPDATE {self._t("campaign_steps")}
                SET status = 'skipped',
                    error = 'Branch aborted by step ' || $2
                WHERE id IN (SELECT id FROM descendants)
                  AND status IN ('pending', 'running')
                RETURNING id
                """,
                target_id,
                step_id,
            )
            skipped_ids = [r["id"] for r in skipped]

            # 3. Mark target as done with output
            await conn.execute(
                f"""
                UPDATE {self._t("campaign_steps")}
                SET status = 'done', output = $2, completed_at = now()
                WHERE id = $1
                """,
                target_id,
                output,
            )

            # 4. Transitive skip cascade through depends_on graph
            if skipped_ids:
                cascade_result = await conn.fetch(
                    f"""
                    WITH RECURSIVE cascade AS (
                        SELECT id FROM {self._t("campaign_steps")}
                        WHERE id = ANY($1::int[])
                        UNION ALL
                        SELECT s.id FROM {self._t("campaign_steps")} s
                        JOIN cascade c ON c.id = ANY(s.depends_on)
                        WHERE s.status = 'pending'
                          AND s.campaign_id = $2
                    )
                    UPDATE {self._t("campaign_steps")}
                    SET status = 'skipped',
                        error = 'Dependency skipped (cascade from step ' || $3::text || ')'
                    WHERE id IN (SELECT id FROM cascade)
                      AND status = 'pending'
                      AND id != ALL($1::int[])
                    RETURNING id
                    """,
                    skipped_ids,
                    step.campaign_id,
                    str(step_id),
                )
                skipped_ids.extend(r["id"] for r in cascade_result)

            # 5. Add a note with the reason
            await conn.execute(
                f"""
                INSERT INTO {self._t("campaign_notes")}
                    (campaign_id, step_id, agent, content, tags)
                VALUES ($1, $2, $3, $4, $5)
                """,
                step.campaign_id,
                step_id,
                step.agent,
                f"Branch abort: {reason}",
                ["abort", "finding"],
            )

        # Check if target's parent is now satisfied
        if target.parent_step_id is not None:
            await self._check_parent_completion(target.parent_step_id)

        return {"skipped_ids": skipped_ids, "target_id": target_id}

    # ------------------------------------------------------------------
    # Canonical resolution
    # ------------------------------------------------------------------

    async def _resolve_deps(self, campaign_id: UUID, dep_ids: list[int]) -> list[int]:
        """Resolve each dep ID to its latest continuation."""
        if not dep_ids:
            return []
        resolved = []
        for dep_id in dep_ids:
            row = await self.pool.fetchrow(
                f"""
                WITH RECURSIVE chain AS (
                    SELECT id FROM {self._t("campaign_steps")}
                    WHERE id = $1 AND campaign_id = $2
                    UNION ALL
                    SELECT s.id FROM {self._t("campaign_steps")} s
                    JOIN chain c ON s.continuation_of = c.id
                )
                SELECT id FROM chain ORDER BY id DESC LIMIT 1
                """,
                dep_id,
                campaign_id,
            )
            resolved.append(row["id"] if row else dep_id)
        return resolved

    # ------------------------------------------------------------------
    # Ancestry check
    # ------------------------------------------------------------------

    async def _is_descendant(self, step_id: int, ancestor_id: int) -> bool:
        """Check if step_id is a descendant of ancestor_id via parent_step_id."""
        row = await self.pool.fetchrow(
            f"""
            WITH RECURSIVE ancestors AS (
                SELECT parent_step_id FROM {self._t("campaign_steps")}
                WHERE id = $1
                UNION ALL
                SELECT s.parent_step_id FROM {self._t("campaign_steps")} s
                JOIN ancestors a ON s.id = a.parent_step_id
            )
            SELECT 1 FROM ancestors WHERE parent_step_id = $2 LIMIT 1
            """,
            step_id,
            ancestor_id,
        )
        return row is not None

    # ------------------------------------------------------------------
    # Parent completion check
    # ------------------------------------------------------------------

    async def _check_parent_completion(self, parent_id: int) -> None:
        """Check if a parent group/sequence/for_each is now satisfied."""
        parent = await self.get_step(parent_id)
        if not parent or parent.status != StepStatus.PENDING:
            return

        children = await self.pool.fetch(
            f"""
            SELECT status FROM {self._t("campaign_steps")}
            WHERE parent_step_id = $1
            """,
            parent_id,
        )
        if not children:
            return

        done_count = sum(1 for c in children if c["status"] in ("done", "skipped"))
        threshold = parent.completion_threshold or len(children)

        if done_count >= threshold:
            # Gather child outputs for the parent
            child_outputs = await self.pool.fetch(
                f"""
                SELECT action, output, status FROM {self._t("campaign_steps")}
                WHERE parent_step_id = $1 AND output IS NOT NULL
                ORDER BY id
                """,
                parent_id,
            )
            summary = "\n\n".join(
                f"[{c['status']}] {c['action']}: {c['output']}" for c in child_outputs
            )
            await self.complete_step(parent_id, summary)

    # ------------------------------------------------------------------
    # Fail-fast cascade
    # ------------------------------------------------------------------

    async def _cascade_fail_fast(self, failed_step: Step) -> None:
        """When a step with fail_fast policy fails, cascade to parent."""
        if failed_step.parent_step_id is None:
            # Top-level step: fail the campaign
            await self.update_campaign(
                failed_step.campaign_id, status=CampaignStatus.FAILED
            )
            return

        # Skip siblings and cascade
        async with self.pool.acquire() as conn, conn.transaction():
            skipped = await conn.fetch(
                f"""
                    UPDATE {self._t("campaign_steps")}
                    SET status = 'skipped',
                        error = 'Sibling failed with fail_fast (step ' || $2 || ')'
                    WHERE parent_step_id = $1
                      AND status IN ('pending', 'running')
                      AND id != $2
                    RETURNING id
                    """,
                failed_step.parent_step_id,
                failed_step.id,
            )
            skipped_ids = [r["id"] for r in skipped]

            # Mark parent as failed
            await conn.execute(
                f"""
                    UPDATE {self._t("campaign_steps")}
                    SET status = 'failed',
                        error = 'Child step ' || $2 || ' failed with fail_fast',
                        completed_at = now()
                    WHERE id = $1
                    """,
                failed_step.parent_step_id,
                failed_step.id,
            )

            # Transitive skip cascade
            if skipped_ids:
                await conn.execute(
                    f"""
                        WITH RECURSIVE cascade AS (
                            SELECT id FROM {self._t("campaign_steps")}
                            WHERE id = ANY($1::int[])
                            UNION ALL
                            SELECT s.id FROM {self._t("campaign_steps")} s
                            JOIN cascade c ON c.id = ANY(s.depends_on)
                            WHERE s.status = 'pending'
                              AND s.campaign_id = $2
                        )
                        UPDATE {self._t("campaign_steps")}
                        SET status = 'skipped',
                            error = 'Dependency skipped (fail_fast cascade)'
                        WHERE id IN (SELECT id FROM cascade)
                          AND status = 'pending'
                          AND id != ALL($1::int[])
                        """,
                    skipped_ids,
                    failed_step.campaign_id,
                )

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    async def add_note(
        self,
        campaign_id: UUID,
        content: str,
        *,
        step_id: int | None = None,
        agent: str | None = None,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Note:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self._t("campaign_notes")}
                (campaign_id, step_id, agent, content, tags, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            """,
            campaign_id,
            step_id,
            agent,
            content,
            tags or [],
            str(embedding) if embedding else None,
        )
        return self._row_to_note(row)

    async def search_notes(
        self,
        query_embedding: list[float],
        *,
        campaign_id: UUID | None = None,
        top_k: int = 5,
    ) -> list[Note]:
        if campaign_id:
            rows = await self.pool.fetch(
                f"""
                SELECT * FROM {self._t("campaign_notes")}
                WHERE campaign_id = $1 AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                campaign_id,
                str(query_embedding),
                top_k,
            )
        else:
            rows = await self.pool.fetch(
                f"""
                SELECT * FROM {self._t("campaign_notes")}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )
        return [self._row_to_note(r) for r in rows]

    async def get_notes(
        self,
        campaign_id: UUID,
        *,
        tags: list[str] | None = None,
        step_id: int | None = None,
    ) -> list[Note]:
        conditions = ["campaign_id = $1"]
        params: list[Any] = [campaign_id]
        if tags:
            conditions.append(f"tags && ${len(params) + 1}")
            params.append(tags)
        if step_id is not None:
            conditions.append(f"step_id = ${len(params) + 1}")
            params.append(step_id)
        rows = await self.pool.fetch(
            f"SELECT * FROM {self._t('campaign_notes')} WHERE {' AND '.join(conditions)} ORDER BY created_at DESC",
            *params,
        )
        return [self._row_to_note(r) for r in rows]

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def notify(
        self,
        campaign_id: UUID | None,
        channel: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> Notification:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self._t("notifications")}
                (campaign_id, channel, message, level)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            """,
            campaign_id,
            channel,
            message,
            level.value,
        )
        # PG NOTIFY for real-time delivery
        await self.pool.execute("SELECT pg_notify('sortie_update', $1)", str(row["id"]))
        return self._row_to_notification(row)

    async def get_undelivered_notifications(self) -> list[Notification]:
        rows = await self.pool.fetch(
            f"""
            SELECT * FROM {self._t("notifications")}
            WHERE delivered = false AND level != 'info'
            ORDER BY created_at
            """
        )
        return [self._row_to_notification(r) for r in rows]

    async def mark_delivered(self, notification_ids: list[int]) -> None:
        if notification_ids:
            await self.pool.execute(
                f"UPDATE {self._t('notifications')} SET delivered = true WHERE id = ANY($1::int[])",
                notification_ids,
            )

    # ------------------------------------------------------------------
    # Runner helpers
    # ------------------------------------------------------------------

    async def get_due_campaigns(self) -> list[Campaign]:
        """Get active campaigns due for processing, locked."""
        rows = await self.pool.fetch(
            f"""
            SELECT * FROM {self._t("campaigns")}
            WHERE status = 'active' AND next_action_at <= now()
            ORDER BY
                CASE priority
                    WHEN 'urgent' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'normal' THEN 2
                    WHEN 'low' THEN 3
                    WHEN 'background' THEN 4
                END,
                next_action_at
            FOR UPDATE SKIP LOCKED
            """
        )
        return [self._row_to_campaign(r) for r in rows]

    async def find_duplicate(self, campaign_id: UUID, fingerprint: str) -> Step | None:
        """Advisory dedup: find a completed step with the same fingerprint."""
        row = await self.pool.fetchrow(
            f"""
            SELECT * FROM {self._t("campaign_steps")}
            WHERE campaign_id = $1 AND fingerprint = $2 AND status = 'done'
            LIMIT 1
            """,
            campaign_id,
            fingerprint,
        )
        return self._row_to_step(row) if row else None
