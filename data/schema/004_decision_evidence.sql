-- Per-pick model evidence: which features moved the forecast for one symbol,
-- and by how much (models/evidence.py). Written by models/screener.py at
-- screening time, read by the dashboard's "Why this pick?" panel.
--
-- Linked to `decisions` by (ts, symbol) rather than by decisions.id: the
-- screener writes a whole batch with pandas to_sql, which does not hand back
-- the generated BIGSERIAL ids, and every candidate in one screener run shares
-- one identical ts (see models.screener.log_candidates). So (ts, symbol) is
-- both available at write time and unique per batch — decisions.id would have
-- to be round-tripped back out of the database to be used here, for no gain.
--
-- Same TimescaleDB-optional treatment as the other tables (see 001_init.sql):
-- if the extension isn't installed this stays a plain Postgres table and
-- nothing downstream notices.
CREATE TABLE IF NOT EXISTS decision_evidence (
    id                BIGSERIAL,
    ts                TIMESTAMPTZ NOT NULL,  -- matches decisions.ts for the same batch
    symbol            TEXT        NOT NULL,
    feature_set_id    TEXT        NOT NULL,
    model_version     TEXT        NOT NULL,
    feature_name      TEXT        NOT NULL,
    feature_value     DOUBLE PRECISION,      -- the feature's own value; NULL when the symbol has no data for it
    contribution      DOUBLE PRECISION NOT NULL,  -- signed: + pushed the forecast up, - pushed it down
    contribution_rank INT         NOT NULL,  -- 1 = largest absolute contribution for this symbol
    base_value        DOUBLE PRECISION,      -- model's expected output before any feature moved it
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, ts)
);
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('decision_evidence', 'ts', if_not_exists => TRUE);
    END IF;
END $$;

-- The dashboard's access pattern: everything explaining one batch's picks.
CREATE INDEX IF NOT EXISTS idx_decision_evidence_ts_symbol ON decision_evidence (ts DESC, symbol);
CREATE INDEX IF NOT EXISTS idx_decision_evidence_symbol_ts ON decision_evidence (symbol, ts DESC);

-- The screener's own confidence in a pick: the share of ensemble members that
-- agreed on the direction, in [0.5, 1.0] (models/forecast/ensemble.py). It was
-- computed at decision time and then thrown away, which left the dashboard
-- unable to say how sure the system was about anything it proposed. Nullable
-- and added in place rather than as a new table — it is one scalar per
-- decision, so it belongs on the decision.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS direction_agreement DOUBLE PRECISION;
