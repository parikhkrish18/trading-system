-- Hold-rule state for the weekly cycle (execution/hold_rules.py): how many
-- consecutive weekly cycles each currently-held position has missed the
-- shortlist. Rewritten every cycle to exactly the held set, so positions
-- closed by any path (weekly exit, contradiction monitor, circuit breaker
-- flatten) fall out of the table on the next cycle automatically.
CREATE TABLE IF NOT EXISTS position_hold_state (
    symbol        TEXT        PRIMARY KEY,
    missed_cycles INTEGER     NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL
);
