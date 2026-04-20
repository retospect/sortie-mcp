-- 0004 — Fair-share scheduler accounting.
--
-- Replaces the per-tick priority-fraction allocator in ``runner.tick``
-- with a weighted deficit round-robin (WDRR) picker that persists
-- compute usage across ticks. See docs/sortie-mcp-plan.md §4.2.
--
-- Two new columns on ``campaigns``:
--
--   slot_seconds_used   monotonically-increasing running total of
--                       wall-clock step seconds charged to this
--                       campaign. Incremented by DB.complete_step and
--                       DB.fail_step using the step's duration_ms.
--
--   weight              priority-derived multiplier used as the
--                       divisor in virtual-time fairness:
--                           virtual_time = slot_seconds_used / weight
--                       The picker always serves the campaign with the
--                       lowest virtual time that has ready work.
--
-- Weight defaults come from the Priority → weight mapping in
-- :func:`sortie_mcp.models.priority_weight` and are pinned at INSERT
-- time in :meth:`sortie_mcp.db.DB.create_campaign`. Existing rows get
-- a uniform 1.0 via the ADD COLUMN default; the backfill UPDATE below
-- corrects them to match their priority.
--
-- ``tokens_used`` already exists on ``campaigns`` and is independent —
-- it tracks LLM spend, not slot occupancy.
-- depends: 0003.resource-leases

ALTER TABLE campaigns
    ADD COLUMN IF NOT EXISTS slot_seconds_used real NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS weight real NOT NULL DEFAULT 1.0;

-- Backfill weights on existing rows. The mapping mirrors
-- :func:`sortie_mcp.models.priority_weight`; keep them in sync.
UPDATE campaigns SET weight = CASE priority
    WHEN 'urgent'     THEN 8.0
    WHEN 'high'       THEN 4.0
    WHEN 'normal'     THEN 2.0
    WHEN 'low'        THEN 1.0
    WHEN 'background' THEN 0.5
    ELSE 1.0
END
WHERE weight = 1.0;  -- only touch untouched rows so test fixtures can override

-- Virtual-time index: the picker orders candidates by
-- ``slot_seconds_used / weight`` ASC. Since PostgreSQL can't directly
-- index an expression across two columns with equal precision without
-- IMMUTABLE wrappers, we index each input separately and let the
-- planner combine.
CREATE INDEX IF NOT EXISTS idx_campaigns_vtime_inputs
    ON campaigns (status, slot_seconds_used, weight)
    WHERE status = 'active';
