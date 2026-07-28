-- Phase 8: monitoring support tables (equity curve + circuit breaker log).
-- Run via `python -m data.schema.migrate`.

-- Portfolio equity snapshots, used for the dashboard drawdown chart and as
-- input to risk/circuit_breakers.py's max_drawdown_breaker.
CREATE TABLE IF NOT EXISTS equity_curve (
    ts           TIMESTAMPTZ NOT NULL,
    mode         TEXT        NOT NULL,  -- 'paper' | 'live'
    equity_value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ts, mode)
);
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('equity_curve', 'ts', if_not_exists => TRUE);
    END IF;
END $$;

-- Log of every circuit-breaker check (not just the ones that triggered), so
-- the dashboard can show "last checked at X, all clear" as well as history
-- of past trips.
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    id            BIGSERIAL,
    ts            TIMESTAMPTZ NOT NULL,
    breaker_name  TEXT        NOT NULL,
    triggered     BOOLEAN     NOT NULL,
    reason        TEXT,
    PRIMARY KEY (id, ts)
);
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('circuit_breaker_state', 'ts', if_not_exists => TRUE);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_breaker_state_ts ON circuit_breaker_state (ts DESC);
CREATE INDEX IF NOT EXISTS idx_breaker_state_name_ts ON circuit_breaker_state (breaker_name, ts DESC);
