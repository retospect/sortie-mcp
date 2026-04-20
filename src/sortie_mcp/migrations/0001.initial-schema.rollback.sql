-- Rollback for 0001.initial-schema.sql.
-- Runs with the connection's search_path set to the target schema.
-- depends:

DROP TABLE IF EXISTS notifications CASCADE;
DROP TABLE IF EXISTS campaign_notes CASCADE;
DROP TABLE IF EXISTS campaign_steps CASCADE;
DROP TABLE IF EXISTS campaigns CASCADE;
