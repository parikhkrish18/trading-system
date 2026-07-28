-- Phase 1 (extended): tradeable equity universe (S&P 500), scraped by
-- data/ingest/universe.py. Drives which symbols the --universe flag on the
-- ingestion/feature-building CLIs resolves to.
CREATE TABLE IF NOT EXISTS universe (
    symbol      TEXT        PRIMARY KEY,
    name        TEXT        NOT NULL,
    gics_sector TEXT,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_universe_active ON universe (is_active);
