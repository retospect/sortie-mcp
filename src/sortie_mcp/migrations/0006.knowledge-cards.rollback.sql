-- Rollback for 0006 — knowledge cards.

DROP INDEX IF EXISTS idx_knowledge_cards_unverified;
DROP INDEX IF EXISTS idx_knowledge_cards_source;
DROP INDEX IF EXISTS idx_knowledge_cards_step;
DROP INDEX IF EXISTS idx_knowledge_cards_campaign;
DROP TABLE IF EXISTS knowledge_cards;
