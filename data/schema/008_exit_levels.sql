-- Per-pick take-profit and stop-loss levels.
--
-- Exits used to be decided against two global settings read at the moment
-- the hold rules ran, which meant a position could be judged against
-- different numbers from the ones a human saw when they approved it —
-- change HOLD_STOP_LOSS_PCT on a Tuesday and every open position silently
-- inherits the new stop. Recording the levels with the decision makes the
-- approval and the enforcement refer to the same thing.
--
-- On `decisions`: what this pick was proposed with, kept as history.
-- On `position_hold_state`: what the currently open position is actually
-- being enforced against, which is what the hold rules read each cycle.
--
-- Nullable on purpose: rows written before this existed have no levels,
-- and the hold rules fall back to the globals for them rather than
-- refusing to evaluate a position that predates the column.

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS take_profit_pct DOUBLE PRECISION;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS stop_loss_pct DOUBLE PRECISION;

ALTER TABLE position_hold_state ADD COLUMN IF NOT EXISTS take_profit_pct DOUBLE PRECISION;
ALTER TABLE position_hold_state ADD COLUMN IF NOT EXISTS stop_loss_pct DOUBLE PRECISION;
