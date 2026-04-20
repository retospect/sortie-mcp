-- 0002 — Multi-runner claim ownership.
--
-- Adds owner / token / heartbeat columns so two ``sortie-runner``
-- processes on different hosts can share a database without
-- double-dispatch or silently resurrecting a zombie. Also adds
-- ``requires_locks`` so the planner can declare resource needs ahead
-- of time (used by migration 0003 for ``resource_leases``).
--
-- Behaviour change: zombie reset now uses ``heartbeat_at`` (kept fresh
-- by long-running but healthy steps) instead of ``started_at``.
-- depends: 0001.initial-schema

ALTER TABLE campaign_steps
    ADD COLUMN claim_owner    text,
    ADD COLUMN claim_token    uuid,
    ADD COLUMN heartbeat_at   timestamptz,
    ADD COLUMN requires_locks text[];

CREATE INDEX idx_steps_heartbeat
    ON campaign_steps (heartbeat_at)
    WHERE status = 'running';
