-- 0003 — Resource leases for fine-grained shared-resource arbitration.
--
-- Companion to migration 0002 (claim_owner / claim_token / requires_locks).
-- A step's ``requires_locks`` text[] is matched against this table to
-- decide whether the step can be claimed: see
-- :meth:`sortie_mcp.db.DB.try_claim_with_locks`.
--
-- ``resource_key`` is opaque but hierarchical (``§`` separator).
-- Examples: ``file:content/books/nanobuds/ch01.tex``,
--           ``file:content/books/nanobuds/ch01.tex§PLXDX``,
--           ``campaign:<uuid>§strategy``.
--
-- Conflict matrix (held vs requested):
--     held \ requested  | EXCLUSIVE | SHARED
--     EXCLUSIVE         | block     | block
--     SHARED            | block     | OK
-- Hierarchical: a lease on ``K`` blocks any lease on ``K§child`` and
-- vice-versa, except when both are SHARED.
--
-- ``expires_at`` is a TTL safety net; healthy steps extend it via
-- :meth:`sortie_mcp.db.DB.heartbeat`. Expired leases are reaped by
-- :meth:`sortie_mcp.db.DB.reap_expired_leases`.
-- depends: 0002.claim-ownership

CREATE TABLE resource_leases (
    -- Composite PK: same key may be SHARED by multiple holders.
    resource_key  text NOT NULL,
    step_id       integer NOT NULL REFERENCES campaign_steps(id) ON DELETE CASCADE,
    owner         text NOT NULL,
    mode          text NOT NULL DEFAULT 'exclusive'
                  CHECK (mode IN ('exclusive', 'shared')),
    acquired_at   timestamptz NOT NULL DEFAULT now(),
    expires_at    timestamptz NOT NULL,
    PRIMARY KEY (resource_key, step_id)
);

-- Lookup leases held by a step (release on complete/fail).
CREATE INDEX idx_leases_step ON resource_leases (step_id);

-- Prefix lookup for hierarchical conflict checks.
CREATE INDEX idx_leases_key_prefix
    ON resource_leases (resource_key text_pattern_ops);

-- Reaper: expired leases.
CREATE INDEX idx_leases_expires
    ON resource_leases (expires_at)
    WHERE expires_at IS NOT NULL;
