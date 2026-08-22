"""
ALLOW_SHORTS — the switch that stops the screener proposing short trades.

Off by default on measured evidence: across the 10-fold walk-forward,
shorts paid -1.069% per trade at a 41.6% win rate while longs paid +0.225%
at 52.1%. Shorts were ~5% of trades and still dragged the book down.

What these tests pin down:
  - short candidates are DROPPED before sizing, never sized to zero, so
    they never reach the approval gate;
  - a dropped short frees its slot for the next long rather than shrinking
    the book;
  - the short code path still works when the switch is on, so this stays a
    policy decision and not a one-way deletion.
"""
from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings
from models.regime.trend_chop_classifier import TREND
from models.screener import select_concentrated_trades, select_trades


def _scored_df(rows):
    df = pd.DataFrame(rows)
    df["conviction_score"] = df["direction_agreement"] * df["predicted_return"].abs()
    return df


def _identity_corr(symbols):
    return pd.DataFrame(
        {s: [1.0 if s == t else 0.0 for t in symbols] for s in symbols}, index=symbols
    )


def _select(scored, allow_shorts, top_k=10):
    return select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=_identity_corr(list(scored["symbol"])),
        top_k=top_k,
        allow_shorts=allow_shorts,
    )


# --------------------------------------------------------------------------
# the setting itself
# --------------------------------------------------------------------------


def test_allow_shorts_defaults_to_false():
    """The evidence says off; a default of on would need new evidence."""
    assert Settings(_env_file=None).allow_shorts is False


def test_allow_shorts_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALLOW_SHORTS", "true")
    assert Settings(_env_file=None).allow_shorts is True


# --------------------------------------------------------------------------
# diversified book
# --------------------------------------------------------------------------


def test_shorts_are_dropped_entirely_not_sized_to_zero():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True},
        ]
    )

    candidates = _select(scored, allow_shorts=False)

    # Not "present with target_position_pct == 0" — absent. A zero-size
    # candidate would still be logged as a decision and shown at the
    # approval gate as a trade nobody intends to make.
    assert [c.symbol for c in candidates] == ["AAPL"]
    assert all(c.side == "long" for c in candidates)


def test_shorts_are_kept_when_the_switch_is_on():
    """The code path stays intact so shorts can be re-enabled if they earn it."""
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True}]
    )

    candidates = _select(scored, allow_shorts=True)

    assert len(candidates) == 1
    assert candidates[0].side == "short"
    assert candidates[0].target_position_pct < 0


def test_a_dropped_short_frees_its_slot_for_the_next_long():
    """
    The top_k budget is for trades that will actually be placed. A rejected
    short must not consume a slot and leave the book underfilled.
    """
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.09, "direction_agreement": 1.0, "confident": True},
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
        ]
    )

    candidates = _select(scored, allow_shorts=False, top_k=2)

    assert [c.symbol for c in candidates] == ["AAPL", "MMM"]


def test_an_all_short_shortlist_becomes_an_empty_book_not_a_long_one():
    """Suppressing shorts must never flip a bearish call into a bullish trade."""
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.09, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": -0.04, "direction_agreement": 1.0, "confident": True},
        ]
    )

    assert _select(scored, allow_shorts=False) == []


def test_longs_are_untouched_by_the_switch():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )

    on = _select(scored, allow_shorts=True)
    off = _select(scored, allow_shorts=False)

    assert [c.symbol for c in on] == [c.symbol for c in off] == ["AAPL"]
    assert on[0].target_position_pct == pytest.approx(off[0].target_position_pct)


# --------------------------------------------------------------------------
# concentrated 2-trade split
# --------------------------------------------------------------------------


def test_concentrated_split_skips_shorts_and_falls_through_the_ranking():
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.09, "direction_agreement": 1.0, "confident": True},
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
        ]
    )

    candidates = select_concentrated_trades(
        scored, max_leg_pct=0.70, min_leg_pct=0.30, allow_shorts=False
    )

    assert [c.symbol for c in candidates] == ["AAPL", "MMM"]
    assert all(c.target_position_pct > 0 for c in candidates)
    # Both legs still add up to the full deployment — dropping the short
    # must not leave capital stranded.
    assert sum(c.target_position_pct for c in candidates) == pytest.approx(1.0)


def test_concentrated_split_keeps_shorts_when_allowed():
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.09, "direction_agreement": 1.0, "confident": True},
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
        ]
    )

    candidates = select_concentrated_trades(
        scored, max_leg_pct=0.70, min_leg_pct=0.30, allow_shorts=True
    )

    assert {c.side for c in candidates} == {"short", "long"}


def test_concentrated_split_with_only_shorts_available_returns_nothing():
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.09, "direction_agreement": 1.0, "confident": True}]
    )

    assert select_concentrated_trades(
        scored, max_leg_pct=0.70, min_leg_pct=0.30, allow_shorts=False
    ) == []


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_run_screen_passes_the_configured_setting_to_the_selector(monkeypatch):
    """
    The policy has to actually reach the selector — a setting nothing reads
    is the failure mode this test exists for.
    """
    import models.screener as scr

    captured = {}

    monkeypatch.setattr(scr.settings, "strategy_mode", "diversified")
    monkeypatch.setattr(scr.settings, "allow_shorts", False)
    monkeypatch.setattr(scr.settings, "screener_top_k", 5)
    monkeypatch.setattr(scr.settings, "full_deployment", False)

    dates = pd.bdate_range("2026-01-01", periods=3, tz="UTC")
    train_df = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "ts": dates,
            "close": [1.0, 1.1, 1.2],
            "fwd_return": [0.01, 0.02, 0.03],
            "target": [0.01, 0.02, 0.03],
            "f1": [1, 2, 3],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    class _NoopEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y): ...

    monkeypatch.setattr(scr, "EnsembleForecastModel", _NoopEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["A"], "f1": [3]}))
    monkeypatch.setattr(scr, "score_universe", lambda *a, **k: _scored_df(
        [{"symbol": "A", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    ))
    monkeypatch.setattr(scr, "build_correlation_matrix", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)

    def fake_select_trades(scored, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(scr, "select_trades", fake_select_trades)

    scr.run_screen("v3", ["A"])
    assert captured["allow_shorts"] is False

    monkeypatch.setattr(scr.settings, "allow_shorts", True)
    scr.run_screen("v3", ["A"])
    assert captured["allow_shorts"] is True


def test_suppressing_shorts_leaves_the_scored_frame_intact(monkeypatch):
    """
    execution/hold_rules.py reads predicted_return for HELD positions to
    spot a forecast that has flipped sign. Filtering shorts out of `scored`
    itself would blind it, so the suppression must happen at selection time
    only.
    """
    import models.screener as scr

    monkeypatch.setattr(scr.settings, "strategy_mode", "diversified")
    monkeypatch.setattr(scr.settings, "allow_shorts", False)
    monkeypatch.setattr(scr.settings, "screener_top_k", 5)
    monkeypatch.setattr(scr.settings, "full_deployment", False)

    dates = pd.bdate_range("2026-01-01", periods=3, tz="UTC")
    train_df = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "ts": dates,
            "close": [1.0, 1.1, 1.2],
            "fwd_return": [0.01, 0.02, 0.03],
            "target": [0.01, 0.02, 0.03],
            "f1": [1, 2, 3],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    class _NoopEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y): ...

    monkeypatch.setattr(scr, "EnsembleForecastModel", _NoopEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["HELD"], "f1": [3]}))
    monkeypatch.setattr(scr, "score_universe", lambda *a, **k: _scored_df(
        [{"symbol": "HELD", "predicted_return": -0.07, "direction_agreement": 1.0, "confident": True}]
    ))
    monkeypatch.setattr(scr, "build_correlation_matrix", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)

    result = scr.run_screen_with_scores("v3", ["HELD"])

    assert result.candidates == []  # no short proposed
    # ...but the bearish forecast is still visible to the hold rules.
    assert result.predicted_return_by_symbol()["HELD"] == pytest.approx(-0.07)
