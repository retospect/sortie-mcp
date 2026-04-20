-- Rollback for 0004 — fair-share columns.

DROP INDEX IF EXISTS idx_campaigns_vtime_inputs;

ALTER TABLE campaigns
    DROP COLUMN IF EXISTS weight,
    DROP COLUMN IF EXISTS slot_seconds_used;
