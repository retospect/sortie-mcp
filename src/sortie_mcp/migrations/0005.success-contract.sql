-- 0005 — Typed success contract on campaigns.
--
-- Adds four columns that let templates (V2 — workflow registry in
-- migration-adjacent code) declare up-front what "done" looks like.
-- The planner loop calls :func:`check_success` between iterations and
-- decides whether to keep iterating or complete the campaign.
--
--   success_metric      Short human-readable name of the metric (e.g.
--                       'accuracy_at_1k' or 'citations_verified').
--                       The verifier / benchmark emits this key.
--
--   benchmark_command   Shell/Python invocation that produces a JSON
--                       line with the metric value, e.g.
--                         "python -m bench.ammonia --out /tmp/m.json"
--                       The runner does NOT execute this — it's
--                       metadata for workers to know how to measure.
--
--   scope               Freeform identifier narrowing the benchmark
--                       (e.g. 'chapter-01', 'test_subset_A').
--
--   max_iterations      Hard cap on the autoresearch loop. NULL means
--                       open-ended (planner decides). Templates that
--                       set this also set ``benchmark_command`` so the
--                       loop has a termination signal.
--
-- All four are nullable — free-form campaigns created via
-- :meth:`DB.create_campaign` never touch them. Templates (V2) set all
-- of them at once.
-- depends: 0004.fair-share

ALTER TABLE campaigns
    ADD COLUMN IF NOT EXISTS success_metric    text,
    ADD COLUMN IF NOT EXISTS benchmark_command text,
    ADD COLUMN IF NOT EXISTS scope             text,
    ADD COLUMN IF NOT EXISTS max_iterations    integer;

-- Lookup accelerator: "which active campaigns have a bounded budget?"
CREATE INDEX IF NOT EXISTS idx_campaigns_max_iterations
    ON campaigns (max_iterations)
    WHERE max_iterations IS NOT NULL AND status = 'active';
