-- Unified loop: two nullable columns on decisions, safe on any DB state
-- (fresh, engine-branch, or approval-branch history).
--
-- direction_agreement: fraction of ensemble members agreeing with the
-- consensus direction for the pick — the confidence signal the what-if
-- sliders on the dashboard filter by. Nullable: rows logged before this
-- migration were never scored.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS direction_agreement DOUBLE PRECISION;

-- approval_status: what the human said (or didn't) about this proposal —
-- 'approved' | 'rejected' | 'timeout' | 'auto'. Nullable: pre-gate rows
-- predate human approval entirely.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS approval_status TEXT;
