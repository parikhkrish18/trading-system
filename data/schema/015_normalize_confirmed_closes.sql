-- A successful close can disappear from broker.get_positions() entirely.
-- execution/trading_loop.py historically logged that missing dictionary key
-- as executed_position = NULL, while the dashboard's closed-trade episode
-- reconstruction correctly looks for executed_position = 0. The result was
-- that real, successful closes vanished from Closed Trades history.
--
-- Normalize confirmed close decisions at the persistence boundary. Rejected
-- closes already carry an explicit non-approved approval_status and are not
-- touched. This also repairs historical rows produced by the bug.

UPDATE decisions
SET executed_position = 0.0
WHERE executed_position IS NULL
  AND target_position = 0.0
  AND forecast IS NULL
  AND approval_status IN ('approved', 'auto');

CREATE OR REPLACE FUNCTION normalize_confirmed_close_decision()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.executed_position IS NULL
       AND NEW.target_position = 0.0
       AND NEW.forecast IS NULL
       AND NEW.approval_status IN ('approved', 'auto') THEN
        NEW.executed_position := 0.0;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_normalize_confirmed_close_decision ON decisions;
CREATE TRIGGER trg_normalize_confirmed_close_decision
BEFORE INSERT OR UPDATE ON decisions
FOR EACH ROW
EXECUTE FUNCTION normalize_confirmed_close_decision();
