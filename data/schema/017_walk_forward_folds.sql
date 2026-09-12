-- Walk-forward fold results (models/train.py::run_walk_forward), read by the
-- dashboard's "Model Analysis" and "Model Report Card" panels
-- (monitoring/dashboard/report_card.py, monitoring/dashboard/server.py).
--
-- Replaces the MLflow-backed version of this data: MLflow was a separate
-- always-on Railway service that added a whole extra failure mode (it went
-- to sleep on idle, and train.py's runs died waiting on it -- see the
-- walk-forward-job crash this replaces) for numbers that fit fine in a
-- handful of rows in the database this app already talks to. train.py
-- deletes and rewrites a model's rows each run, so this table only ever
-- holds the latest walk-forward run per model_name, not a full history --
-- the report card has never shown more than that.
CREATE TABLE IF NOT EXISTS walk_forward_folds (
    id                                   BIGSERIAL PRIMARY KEY,
    model_name                           TEXT        NOT NULL,
    feature_set_id                       TEXT,
    fold_id                              INT         NOT NULL,
    train_start                          TIMESTAMPTZ,
    train_end                            TIMESTAMPTZ,
    test_start                           TIMESTAMPTZ,
    test_end                             TIMESTAMPTZ,
    mae                                  DOUBLE PRECISION,
    rmse                                 DOUBLE PRECISION,
    directional_accuracy                 DOUBLE PRECISION,
    directional_accuracy_when_confident  DOUBLE PRECISION,
    pct_rows_confident                   DOUBLE PRECISION,
    mean_ensemble_std                    DOUBLE PRECISION,
    run_ts                               TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_walk_forward_folds_model ON walk_forward_folds (model_name, fold_id);
