-- Client self-service risk controls -- unlike leverage_multiplier
-- (012_client_leverage.sql, operator-only), these are columns the CLIENT
-- sets on their own account through the portal (POST /api/portal/liquidate,
-- POST /api/portal/resume, POST /api/portal/risk_settings in
-- monitoring/dashboard/server.py). The "results only" read-only design for
-- the portal's EXISTING data (positions/account/trades) is unchanged --
-- these are new, narrowly-scoped actions layered on top, not a reversal of
-- that decision: a client can flatten their own book and set their own
-- stop/profit thresholds, but still can't see or influence the model's
-- reasoning, and still can't touch any other client's account.
--
-- trading_paused is the single kill switch execution/client_fanout.py's
-- load_active_clients() checks (alongside active) before including a
-- client in any fan-out pass -- set directly by a client's "Liquidate now"
-- click, or automatically by execution/client_risk_controls.py when a
-- configured threshold trips. pause_reason records which: 'client_liquidate'
-- (their own button), 'max_drawdown', or 'profit_target' (auto-triggered --
-- see that module's docstring for why those two auto-triggers resume
-- differently: a drawdown pause waits for the client to click Resume,
-- a profit-target pause auto-resumes at the start of the next window).
--
-- max_drawdown_pct / equity_peak: NULL max_drawdown_pct means the client
-- hasn't opted in -- this feature never runs for them. equity_peak is
-- maintained automatically (the running high-water mark since the client
-- turned the feature on, or since their last Resume) and is not something
-- the client sets directly.
--
-- profit_target_pct / profit_target_window_days / the two
-- profit_target_period_* columns: same opt-in-via-NULL pattern, plus the
-- rolling-window bookkeeping execution/client_risk_controls.py needs to
-- know when "5% this week" resets vs. keeps accumulating.
--
-- All four thresholds are nullable and independent -- a client can set
-- just a drawdown limit, just a profit target, both, or neither (the
-- default, and the ONLY behavior for every client that existed before this
-- migration).
ALTER TABLE clients ADD COLUMN IF NOT EXISTS trading_paused BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS pause_reason TEXT;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS max_drawdown_pct DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS equity_peak DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS profit_target_pct DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS profit_target_window_days INTEGER;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS profit_target_period_start_equity DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS profit_target_period_start_ts TIMESTAMPTZ;

DO $$
BEGIN
    -- 0 excluded on both ends deliberately: 0% would either trip
    -- immediately (drawdown) or never mean anything (profit target) --
    -- the API layer (monitoring/dashboard/server.py's _validate_risk_settings)
    -- enforces the same bounds before they ever reach the DB, this is the
    -- backstop.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_max_drawdown_pct_range') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_max_drawdown_pct_range
            CHECK (max_drawdown_pct IS NULL OR max_drawdown_pct BETWEEN 0.01 AND 0.5);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_profit_target_pct_range') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_profit_target_pct_range
            CHECK (profit_target_pct IS NULL OR profit_target_pct BETWEEN 0.01 AND 1.0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_profit_target_window_days_range') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_profit_target_window_days_range
            CHECK (profit_target_window_days IS NULL OR profit_target_window_days BETWEEN 1 AND 365);
    END IF;

    -- The two profit-target columns are a pair -- one set without the
    -- other is meaningless (a target % with no window, or a window with
    -- no target %), so the DB rejects that combination outright.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_profit_target_pair') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_profit_target_pair
            CHECK ((profit_target_pct IS NULL) = (profit_target_window_days IS NULL));
    END IF;
END $$;
