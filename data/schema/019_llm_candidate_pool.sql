-- The full Claude-advised candidate pool from a concentrated-mode screen
-- (every symbol Claude scored that cycle, not just the top picks that
-- actually got opened) -- so a later flat-book reactivation
-- (execution/full_book_rebalance.py) can redeploy against this week's
-- already-computed analysis instead of paying for a fresh ensemble retrain
-- + rescreen every time it finds the book empty. See
-- models/candidate_pool_store.py. Run via `python -m data.schema.migrate`.
CREATE TABLE IF NOT EXISTS llm_candidate_pool (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL,
    feature_set_id    TEXT        NOT NULL,
    symbol            TEXT        NOT NULL,
    side              TEXT        NOT NULL,
    predicted_return  DOUBLE PRECISION,
    direction_agreement DOUBLE PRECISION,
    conviction_score  DOUBLE PRECISION,
    confidence        DOUBLE PRECISION NOT NULL,
    take_profit_pct   DOUBLE PRECISION,
    stop_loss_pct     DOUBLE PRECISION,
    reasoning         JSONB
);
CREATE INDEX IF NOT EXISTS idx_llm_candidate_pool_fsid_ts ON llm_candidate_pool (feature_set_id, ts DESC);
