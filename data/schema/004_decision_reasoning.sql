-- Phase 8: per-decision explanation. Nullable — decisions logged before
-- this migration have no reasoning captured, and the dashboard shows that
-- plainly rather than pretending. See models/screener.py::run_screen,
-- which populates this via LightGBM's pred_contrib (genuine per-prediction
-- Tree SHAP feature contributions, not just global feature importance).
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS reasoning JSONB;
