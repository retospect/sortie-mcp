-- Rollback for 0005 — success contract.

DROP INDEX IF EXISTS idx_campaigns_max_iterations;

ALTER TABLE campaigns
    DROP COLUMN IF EXISTS max_iterations,
    DROP COLUMN IF EXISTS scope,
    DROP COLUMN IF EXISTS benchmark_command,
    DROP COLUMN IF EXISTS success_metric;
