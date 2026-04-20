-- Rollback for 0002.claim-ownership.sql.
-- depends:

DROP INDEX IF EXISTS idx_steps_heartbeat;
ALTER TABLE campaign_steps
    DROP COLUMN IF EXISTS requires_locks,
    DROP COLUMN IF EXISTS heartbeat_at,
    DROP COLUMN IF EXISTS claim_token,
    DROP COLUMN IF EXISTS claim_owner;
