-- Dated S&P 500 membership snapshots, appended on every universe refresh
-- (data/ingest/universe.py::refresh_universe). The `universe` table only
-- knows who is in the index *today*, which means every backtest run against
-- it carries survivorship bias — the losers that got kicked out of the
-- index are exactly the names it can no longer see. This table records who
-- was in the index on each refresh date, so future evaluations can use
-- point-in-time membership instead of today's winners.
CREATE TABLE IF NOT EXISTS universe_snapshot (
    snapshot_date DATE NOT NULL,
    symbol        TEXT NOT NULL,
    name          TEXT,
    gics_sector   TEXT,
    PRIMARY KEY (snapshot_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_universe_snapshot_date ON universe_snapshot (snapshot_date);
