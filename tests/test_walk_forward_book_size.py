"""
Regression coverage for models/train.py's `book_top_k`: the concentrated
strategy holds up to settings.max_concentrated_positions names (2 or 3 with
the production defaults), never a hardcoded 2 — the walk-forward harness
has to measure the same book size the live screener actually trades.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from models import train


class _FakeEnsemble:
    """Avoids real LightGBM training — only .fit()/.predict() need to exist."""

    def __init__(self, n_models=5): ...

    def fit(self, X, y): ...

    def predict(self, X):
        n = len(X)
        return pd.DataFrame(
            {
                "mean_prediction": np.full(n, 0.05),
                "std_prediction": np.full(n, 0.01),
                "direction_agreement": np.full(n, 1.0),
            },
            index=X.index,
        )


class _FakeMlflow:
    """No-op stand-in — the real one needs a tracking server/local store this test shouldn't touch."""

    def set_tracking_uri(self, uri): ...

    def set_experiment(self, name): ...

    def log_params(self, params): ...

    def log_metrics(self, metrics): ...

    @contextmanager
    def start_run(self, run_name=None):
        yield None


def _synthetic_train_df():
    dates = pd.bdate_range("2026-01-01", periods=20)
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dates),
            "ts": dates,
            "close": 100.0 + np.arange(len(dates)) * 0.1,
            "fwd_return": 0.01,
            "target": 0.01,
            "f1": np.linspace(0, 1, len(dates)),
        }
    )


def _run_and_capture_book_top_k(monkeypatch, **run_kwargs):
    monkeypatch.setattr(train, "load_training_frame", lambda *a, **k: _synthetic_train_df())
    monkeypatch.setattr(train, "EnsembleForecastModel", _FakeEnsemble)
    monkeypatch.setattr(train, "mlflow", _FakeMlflow())

    captured = {}

    def _capture_production_book_mask(preds, dates, *, cost, allow_shorts, top_k):
        captured["top_k"] = top_k
        raise RuntimeError("stop here — only book_top_k is under test")

    monkeypatch.setattr(train, "production_book_mask", _capture_production_book_mask)

    with pytest.raises(RuntimeError, match="stop here"):
        train.run_walk_forward("v4", ["AAA"], n_folds=1, purge_days=0, target_mode="absolute", **run_kwargs)

    return captured["top_k"]


def test_concentrated_book_size_uses_max_concentrated_positions_not_a_hardcoded_two(monkeypatch):
    monkeypatch.setattr(train.settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(train.settings, "max_concentrated_positions", 3)

    assert _run_and_capture_book_top_k(monkeypatch) == 3


def test_concentrated_book_size_tracks_the_setting_when_it_changes(monkeypatch):
    """Not just '3 happens to be right' — changing the setting must change the measured book size."""
    monkeypatch.setattr(train.settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(train.settings, "max_concentrated_positions", 2)

    assert _run_and_capture_book_top_k(monkeypatch) == 2


def test_diversified_book_size_is_unaffected_and_still_uses_screener_top_k(monkeypatch):
    monkeypatch.setattr(train.settings, "strategy_mode", "diversified")
    monkeypatch.setattr(train.settings, "screener_top_k", 11)

    assert _run_and_capture_book_top_k(monkeypatch) == 11
