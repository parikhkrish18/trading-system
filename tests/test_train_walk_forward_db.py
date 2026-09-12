"""
Real-DB integration test for models/train.py::_write_fold_result -- the
walk-forward harness's replacement for per-fold mlflow.log_metrics, proven
against the actual walk_forward_folds schema
(data/schema/017_walk_forward_folds.sql), not a mock.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest.db import get_engine
from models.train import _write_fold_result


@dataclasses.dataclass
class _FakeFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@pytest.fixture
def _walk_forward_folds_cleanup():
    engine = get_engine()
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM walk_forward_folds WHERE model_name = 'zzztest_model'"))


def test_write_fold_result_persists_the_metrics_a_fold_computed(_walk_forward_folds_cleanup):
    engine = _walk_forward_folds_cleanup
    fold = _FakeFold(
        fold_id=0,
        train_start=pd.Timestamp("2025-01-01", tz="UTC"),
        train_end=pd.Timestamp("2025-06-01", tz="UTC"),
        test_start=pd.Timestamp("2025-06-01", tz="UTC"),
        test_end=pd.Timestamp("2025-07-01", tz="UTC"),
    )
    row = {
        "mae": 0.02,
        "rmse": 0.03,
        "directional_accuracy": 0.55,
        "directional_accuracy_when_confident": 0.61,
        "pct_rows_confident": 0.4,
        "mean_ensemble_std": 0.004,
    }

    _write_fold_result("zzztest_model", "v4", fold, row)

    with engine.connect() as conn:
        written = conn.execute(
            text("SELECT * FROM walk_forward_folds WHERE model_name = 'zzztest_model'")
        ).mappings().one()
    assert written["fold_id"] == 0
    assert written["feature_set_id"] == "v4"
    assert written["directional_accuracy"] == pytest.approx(0.55)
    assert written["directional_accuracy_when_confident"] == pytest.approx(0.61)
    assert written["mean_ensemble_std"] == pytest.approx(0.004)


def test_write_fold_result_tolerates_a_missing_mean_ensemble_std(_walk_forward_folds_cleanup):
    """
    mean_ensemble_std is only ever absent if a caller builds the row dict
    by hand (run_walk_forward always sets it) -- .get() must not crash a
    write over one optional field, same reasoning as sentiment.py's
    per-field .get() calls elsewhere in this codebase.
    """
    engine = _walk_forward_folds_cleanup
    fold = _FakeFold(
        fold_id=1,
        train_start=pd.Timestamp("2025-01-01", tz="UTC"),
        train_end=pd.Timestamp("2025-06-01", tz="UTC"),
        test_start=pd.Timestamp("2025-06-01", tz="UTC"),
        test_end=pd.Timestamp("2025-07-01", tz="UTC"),
    )
    row = {
        "mae": 0.02,
        "rmse": 0.03,
        "directional_accuracy": 0.5,
        "directional_accuracy_when_confident": 0.5,
        "pct_rows_confident": 0.0,
    }

    _write_fold_result("zzztest_model", "v4", fold, row)

    with engine.connect() as conn:
        written = conn.execute(
            text("SELECT mean_ensemble_std FROM walk_forward_folds WHERE model_name = 'zzztest_model'")
        ).mappings().one()
    assert written["mean_ensemble_std"] is None
