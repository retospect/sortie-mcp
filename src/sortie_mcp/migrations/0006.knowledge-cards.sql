-- 0006 — Knowledge cards table.
--
-- Structured, verifiable findings separate from free-form
-- ``campaign_notes``. A knowledge card carries:
--
--     claim        — one assertion, made explicit for the verifier
--     source_ref   — DOI, acatome slug, URL, file path — anything a
--                    verifier can resolve to the source text
--     quote        — optional verbatim snippet supporting the claim
--     confidence   — self-reported [0, 1]; verifier may overwrite
--     verified_status — 'verified' | 'unsupported' | 'dead_link' |
--                       'number_mismatch' | 'conflicting_source' | NULL
--     verified_at  — timestamp of last verifier visit
--     embedding    — pgvector(384) for cosine retrieval
--
-- Cards are the primary substrate for the ``writer`` role — it reads
-- them by semantic query and cites by ``source_ref``. Notes stay for
-- free-form observations ("search query didn't return anything
-- useful") that are not intended as writing source material.
--
-- Ownership: cards are owned by the campaign, optionally attributed
-- to a step. ``ON DELETE CASCADE`` so cleaning up a cancelled
-- campaign doesn't leave dangling cards.
-- depends: 0005.success-contract

CREATE TABLE knowledge_cards (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id     uuid NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    step_id         integer REFERENCES campaign_steps(id) ON DELETE SET NULL,
    agent           text,                    -- which specialist produced this
    claim           text NOT NULL,
    source_ref      text NOT NULL,           -- DOI / slug / URL / path
    quote           text,                    -- verbatim supporting snippet
    confidence      real NOT NULL DEFAULT 0.5
                    CHECK (confidence >= 0 AND confidence <= 1),
    verified_status text CHECK (verified_status IS NULL OR verified_status IN
                         ('verified', 'unsupported', 'dead_link',
                          'number_mismatch', 'conflicting_source')),
    verified_at     timestamptz,
    verifier_note   text,                    -- free-form verifier output
    embedding       public.vector(384),
    tags            text[] DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_knowledge_cards_campaign ON knowledge_cards (campaign_id);
CREATE INDEX idx_knowledge_cards_step     ON knowledge_cards (step_id);
CREATE INDEX idx_knowledge_cards_source   ON knowledge_cards (source_ref);

-- Verifier work queue: "which cards in this campaign still need a
-- verifier pass?". Partial index keeps it tiny.
CREATE INDEX idx_knowledge_cards_unverified
    ON knowledge_cards (campaign_id, created_at)
    WHERE verified_status IS NULL;
