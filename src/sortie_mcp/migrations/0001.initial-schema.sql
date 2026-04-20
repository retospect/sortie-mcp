-- 0001 — Initial sortie-mcp schema.
--
-- Tables use UNQUALIFIED names; the connection search_path (set via the
-- ``?schema=`` DSN parameter read by yoyo) puts them in the target
-- schema. This file is identical across dev, test, and prod.
-- depends:

CREATE TABLE campaigns (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name              text,
    goal              text NOT NULL,
    status            text NOT NULL DEFAULT 'active',
    strategy          text,
    progress          text,
    channel           text,
    user_id           text,
    max_depth         smallint NOT NULL DEFAULT 4,
    token_budget      integer,
    tokens_used       integer NOT NULL DEFAULT 0,
    failure_policy    text NOT NULL DEFAULT 'continue',
    priority          text NOT NULL DEFAULT 'normal',
    next_action_at    timestamptz NOT NULL DEFAULT now(),
    last_reported_at  timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    completed_at      timestamptz
);

CREATE INDEX idx_campaigns_due
    ON campaigns (next_action_at)
    WHERE status = 'active';

CREATE TABLE campaign_steps (
    id                    serial PRIMARY KEY,
    campaign_id           uuid NOT NULL REFERENCES campaigns(id),
    parent_step_id        integer REFERENCES campaign_steps(id),
    depth                 smallint NOT NULL DEFAULT 0,
    action                text NOT NULL,
    agent                 text,
    step_type             text NOT NULL DEFAULT 'atomic',
    status                text NOT NULL DEFAULT 'pending',
    failure_policy        text NOT NULL DEFAULT 'continue',
    depends_on            integer[],
    input                 text,
    output                text,
    error                 text,
    fingerprint           text,
    continuation_of       integer REFERENCES campaign_steps(id),
    completion_threshold  smallint,
    retry_count           smallint NOT NULL DEFAULT 0,
    max_retries           smallint NOT NULL DEFAULT 3,
    tokens_used           integer,
    duration_ms           integer,
    created_at            timestamptz NOT NULL DEFAULT now(),
    started_at            timestamptz,
    completed_at          timestamptz
);

CREATE INDEX idx_steps_campaign_status
    ON campaign_steps (campaign_id, status)
    WHERE status IN ('pending', 'running');

CREATE INDEX idx_steps_fingerprint ON campaign_steps (fingerprint);

CREATE TABLE campaign_notes (
    id           serial PRIMARY KEY,
    campaign_id  uuid NOT NULL REFERENCES campaigns(id),
    step_id      integer REFERENCES campaign_steps(id),
    agent        text,
    content      text NOT NULL,
    tags         text[] DEFAULT '{}',
    embedding    public.vector(384),
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_notes_campaign ON campaign_notes (campaign_id);

CREATE TABLE notifications (
    id           serial PRIMARY KEY,
    campaign_id  uuid REFERENCES campaigns(id),
    channel      text NOT NULL,
    message      text NOT NULL,
    level        text NOT NULL DEFAULT 'info',
    delivered    boolean NOT NULL DEFAULT false,
    created_at   timestamptz NOT NULL DEFAULT now()
);
