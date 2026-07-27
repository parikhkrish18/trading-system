-- Phase 1: core hypertables. Run via `python -m data.schema.migrate`.
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Raw daily OHLCV bars per symbol.
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('prices', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON prices (symbol, ts DESC);

-- Fundamentals snapshots (sparse — e.g. quarterly).
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol      TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    metric      TEXT        NOT NULL,
    value       DOUBLE PRECISION,
    source      TEXT        NOT NULL DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts, metric)
);
SELECT create_hypertable('fundamentals', 'ts', if_not_exists => TRUE);

-- News / filing events with an attached sentiment/surprise score.
CREATE TABLE IF NOT EXISTS news_events (
    id            BIGSERIAL,
    symbol        TEXT        NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    headline      TEXT,
    source        TEXT        NOT NULL DEFAULT 'unknown',
    sentiment     DOUBLE PRECISION,  -- populated by features/qualitative/sentiment.py
    surprise      DOUBLE PRECISION,  -- e.g. earnings surprise magnitude, if applicable
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, ts)
);
SELECT create_hypertable('news_events', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_news_symbol_ts ON news_events (symbol, ts DESC);

-- Macro calendar (FOMC, CPI, jobs report, etc).
CREATE TABLE IF NOT EXISTS macro_calendar (
    event_name  TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    category    TEXT        NOT NULL,  -- 'FOMC' | 'CPI' | 'JOBS' | 'OTHER'
    notes       TEXT,
    PRIMARY KEY (event_name, ts)
);
SELECT create_hypertable('macro_calendar', 'ts', if_not_exists => TRUE);

-- Engineered feature store, versioned by feature_set_id so any historical
-- model run can be reproduced exactly (Phase 2, point 5 of the plan).
CREATE TABLE IF NOT EXISTS features (
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    feature_set_id  TEXT        NOT NULL,
    feature_name    TEXT        NOT NULL,
    value           DOUBLE PRECISION,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts, feature_set_id, feature_name)
);
SELECT create_hypertable('features', 'ts', if_not_exists => TRUE);

-- Every live/paper decision, logged at decision time (Phase 6, point 2).
CREATE TABLE IF NOT EXISTS decisions (
    id                BIGSERIAL,
    ts                TIMESTAMPTZ NOT NULL,
    symbol            TEXT        NOT NULL,
    feature_set_id    TEXT        NOT NULL,
    model_version     TEXT        NOT NULL,
    forecast          DOUBLE PRECISION,
    regime            TEXT,
    target_position   DOUBLE PRECISION,  -- fraction of portfolio, signed
    executed_position DOUBLE PRECISION,
    mode              TEXT        NOT NULL,  -- 'paper' | 'live'
    PRIMARY KEY (id, ts)
);
SELECT create_hypertable('decisions', 'ts', if_not_exists => TRUE);
