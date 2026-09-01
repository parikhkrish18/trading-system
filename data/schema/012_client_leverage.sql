-- Per-client leverage multiplier for execution/client_fanout.py's position
-- sizing (target_shares = target_pct * leverage_multiplier * client_equity
-- / price, capped -- see that module for the exact mechanics and why the
-- cap only engages once leverage_multiplier > 1).
--
-- Deliberately operator-set only, the same way margin_enabled already is
-- (see that column's own comment below) -- the client-facing portal stays
-- read-only/"results only", no input that changes what gets traded. A
-- client's own leverage decision goes through you, not their own login.
--
-- Capped at 3x (a deliberate choice made 2026-09-01, not derived from any
-- Alpaca account-type limit) as a hard backstop against a fat-fingered
-- input -- e.g. "20" meant as "2.0x" -- wiping out an account. Raising this
-- cap later needs a new migration (to move this CHECK) AND the matching
-- _MAX_LEVERAGE constant in monitoring/dashboard/server.py updated
-- together; they intentionally aren't wired to a single settings value so
-- that raising the cap is a deliberate two-place code change, not an env
-- var someone can bump by accident.
--
-- leverage_multiplier > 1 is meaningless (and dangerous) on a cash account,
-- so it's constrained to require margin_enabled -- the same reasoning
-- margin_enabled already uses for shorts applies doubly here.
ALTER TABLE clients ADD COLUMN IF NOT EXISTS leverage_multiplier INTEGER NOT NULL DEFAULT 1;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_leverage_multiplier_range') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_leverage_multiplier_range CHECK (leverage_multiplier BETWEEN 1 AND 3);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'clients_leverage_requires_margin') THEN
        ALTER TABLE clients
            ADD CONSTRAINT clients_leverage_requires_margin CHECK (leverage_multiplier = 1 OR margin_enabled);
    END IF;
END $$;
