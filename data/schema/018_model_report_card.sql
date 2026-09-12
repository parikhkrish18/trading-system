-- Cumulative, real-decisions-only performance history for the dashboard's
-- "True Report Card" panel (monitoring/model_report_card.py,
-- scripts/refresh_model_report_card.py). One row per calendar date the
-- refresh script has run, and every row is a CUMULATIVE snapshot -- "as of
-- this date, here is the full track record of every real decision ever
-- logged" -- not that day's delta. That makes the latest row always the
-- current true report card, and the whole table a growing history to chart.
--
-- Deliberately separate from walk_forward_folds (data/schema/017_walk_forward_folds.sql):
-- that table is backtest/walk-forward results on held-out history, refreshed
-- only when someone manually runs models.train. This table is the opposite
-- by design -- it must never contain a backtested or simulated number, only
-- what the live system actually decided and actually sized, via the
-- decisions table filtered to mode IN ('paper', 'live') (mode='backfill'
-- rows -- replayed history -- are never counted here; see
-- monitoring/model_report_card.py for why).
CREATE TABLE IF NOT EXISTS model_report_card_history (
    id                      BIGSERIAL PRIMARY KEY,
    as_of_date              DATE        NOT NULL UNIQUE,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Every real (paper/live) decision ever logged that actually sized a
    -- trade (a nonzero target_position) -- opens, not closes/holds/rejections.
    n_trades_taken          INT              NOT NULL,
    -- Of those, how many are old enough (>= TARGET_HORIZON_DAYS trading days)
    -- to grade against a later price bar, and how many correctly called
    -- direction -- cumulative across the system's entire real history, never
    -- reset, so this is the "how right has it actually been" track record.
    n_matured               INT              NOT NULL,
    n_hits                  INT              NOT NULL,
    hit_rate                DOUBLE PRECISION,
    -- The average realized return across every matured real trade -- the
    -- actual money-performance track record, not a backtested one.
    avg_realized_return     DOUBLE PRECISION,
    -- A snapshot, not cumulative: the latest known target_position per
    -- symbol (paper/live decisions only), summed -- what fraction of the
    -- book is deployed as of this run. Read across many days' rows this
    -- becomes a real history of capital deployment over time.
    capital_deployed_pct    DOUBLE PRECISION NOT NULL
);
