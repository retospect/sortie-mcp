"""Database layer for sortie-mcp — schema migration, queries, and transactions.

All SQL uses a configurable schema name (default: ``sortie``).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import asyncpg

from .models import (
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
    compute_fingerprint,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.campaigns (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text,
    goal            text NOT NULL,
    status          text NOT NULL DEFAULT 'active',
    strategy        text,
    progress        text,
    channel         text,
    user_id         text,
    max_depth       smallint NOT NULL DEFAULT 4,
    token_budget    integer,
    tokens_used     integer NOT NULL DEFAULT 0,
    failure_policy  text NOT NULL DEFAULT 'continue',
    priority        text NOT NULL DEFAULT 'normal',
    next_action_at  timestamptz NOT NULL DEFAULT now(),
    last_reported_at timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    completed_at    timestamptz
);

CREATE INDEX IF NOT EXISTS idx_campaigns_due
    ON {schema}.campaigns (next_action_at)
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS {schema}.campaign_steps (
    id              serial PRIMARY KEY,
    campaign_id     uuid NOT NULL REFERENCES {schema}.campaigns(id),
    parent_step_id  integer REFERENCES {schema}.campaign_steps(id),
    depth           smallint NOT NULL DEFAULT 0,
    action          text NOT NULL,
    agent           text,
    step_type       text NOT NULL DEFAULT 'atomic',
    status          text NOT NULL DEFAULT 'pending',
    failure_policy  text NOT NULL DEFAULT 'continue',
    depends_on      integer[],
    input           text,
    output          text,
    error           text,
    fingerprint     text,
    continuation_of integer REFERENCES {schema}.campaign_steps(id),
    completion_threshold smallint,
    retry_count     smallint NOT NULL DEFAULT 0,
    max_retries     smallint NOT NULL DEFAULT 3,
    tokens_used     integer,
    duration_ms     integer,
    created_at      timestamptz NOT NULL DEFAULT now(),
    started_at      timestamptz,
    completed_at    timestamptz
);

CREATE INDEX IF NOT EXISTS idx_steps_campaign_status
    ON {schema}.campaign_steps (campaign_id, status)
    WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS idx_steps_fingerprint
    ON {schema}.campaign_steps (fingerprint);

CREATE TABLE IF NOT EXISTS {schema}.campaign_notes (
    id              serial PRIMARY KEY,
    campaign_id     uuid NOT NULL REFERENCES {schema}.campaigns(id),
    step_id         integer REFERENCES {schema}.campaign_steps(id),
    agent           text,
    content         text NOT NULL,
    tags            text[] DEFAULT '{{}}',
    embedding       vector(384),
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_notes_campaign
    ON {schema}.campaign_notes (campaign_id);

CREATE TABLE IF NOT EXISTS {schema}.notifications (
    id              serial PRIMARY KEY,
    campaign_id     uuid REFERENCES {schema}.campaigns(id),
    channel         text NOT NULL,
    message         text NOT NULL,
    level           text NOT NULL DEFAULT 'info',
    delivered       boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);
"""


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
        """Create schema and tables if they don't exist."""
        async with self.pool.acquire() as conn:
            # pgvector extension (may require superuser; skip gracefully)
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except asyncpg.InsufficientPrivilegeError:
                log.warning(
                    "Cannot create vector extension — must be created by superuser"
                )
            sql = SCHEMA_SQL.format(schema=self.schema)
            await conn.execute(sql)
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
    ) -> Step:
        # Canonical resolution: resolve depends_on through continuation chains
        resolved_deps = await self._resolve_deps(campaign_id, depends_on or [])
        fp = compute_fingerprint(action, agent, input_text)
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self._t("campaign_steps")}
                (campaign_id, action, agent, step_type, parent_step_id,
                 depth, depends_on, input, failure_policy, fingerprint,
                 continuation_of, completion_threshold)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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

    async def claim_step(self, step_id: int) -> Step | None:
        """Atomically claim a pending step for execution."""
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status = 'running', started_at = now()
            WHERE id = $1 AND status = 'pending'
            RETURNING *
            """,
            step_id,
        )
        return self._row_to_step(row) if row else None

    async def complete_step(
        self,
        step_id: int,
        output: str,
        *,
        tokens_used: int | None = None,
        duration_ms: int | None = None,
    ) -> Step | None:
        """Mark a step as done. Returns None if step was already skipped."""
        row = await self.pool.fetchrow(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status = CASE WHEN status = 'skipped' THEN 'skipped' ELSE 'done' END,
                output = $2,
                tokens_used = $3,
                duration_ms = $4,
                completed_at = now()
            WHERE id = $1
            RETURNING *
            """,
            step_id,
            output,
            tokens_used,
            duration_ms,
        )
        if not row:
            return None
        step = self._row_to_step(row)
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

    async def fail_step(self, step_id: int, error: str) -> Step | None:
        """Increment retry count and record error. If max retries exceeded,
        mark as failed and optionally cascade via fail_fast."""
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
                END
            WHERE id = $1
            RETURNING *
            """,
            step_id,
            error,
        )
        if not row:
            return None
        step = self._row_to_step(row)
        if (
            step.status == StepStatus.FAILED
            and step.failure_policy == FailurePolicy.FAIL_FAST
        ):
            await self._cascade_fail_fast(step)
        return step

    async def reset_zombies(self, timeout_minutes: int = 30) -> int:
        """Reset steps stuck in 'running' past timeout back to 'pending'."""
        result = await self.pool.execute(
            f"""
            UPDATE {self._t("campaign_steps")}
            SET status = 'pending', started_at = NULL
            WHERE status = 'running'
              AND started_at < now() - ($1 || ' minutes')::interval
            """,
            str(timeout_minutes),
        )
        count = int(result.split()[-1])
        if count:
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
                        error = 'Dependency skipped (cascade from step ' || $3 || ')'
                    WHERE id IN (SELECT id FROM cascade)
                      AND status = 'pending'
                      AND id != ALL($1::int[])
                    RETURNING id
                    """,
                    skipped_ids,
                    step.campaign_id,
                    step_id,
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
