import numpy as np
import pandas as pd
import pytest

from models.regime.trend_chop_classifier import TREND
from models.screener import (
    _attach_reasoning,
    build_correlation_matrix,
    score_universe,
    select_concentrated_trades,
    select_trades,
)


class _FakeEnsemble:
    """Fakes EnsembleForecastModel.predict with a preset mapping of row index -> stats."""

    def __init__(self, mean_prediction, direction_agreement, std_prediction=None, contributions=None):
        self.mean_prediction = mean_prediction
        self.direction_agreement = direction_agreement
        self.std_prediction = std_prediction if std_prediction is not None else [0.0] * len(mean_prediction)
        self._contributions = contributions

    def predict(self, X):
        return pd.DataFrame(
            {
                "mean_prediction": self.mean_prediction,
                "std_prediction": self.std_prediction,
                "direction_agreement": self.direction_agreement,
            },
            index=X.index,
        )

    def predict_contributions(self, X):
        return self._contributions.loc[X.index]


def test_score_universe_computes_conviction_and_confident_flag():
    latest = pd.DataFrame({"symbol": ["AAPL", "TSLA", "MMM"], "f1": [0.1, 0.2, 0.3]})
    ensemble = _FakeEnsemble(mean_prediction=[0.05, -0.03, 0.01], direction_agreement=[1.0, 0.6, 0.9])

    result = score_universe(ensemble, latest, feature_cols=["f1"], min_direction_agreement=0.8, min_abs_return=0.02)

    by_symbol = result.set_index("symbol")
    assert by_symbol.loc["AAPL", "confident"]  # agreement 1.0 >= 0.8, |return| 0.05 >= 0.02
    assert not by_symbol.loc["TSLA", "confident"]  # agreement 0.6 < 0.8
    assert not by_symbol.loc["MMM", "confident"]  # |return| 0.01 < 0.02 even though agreement is fine
    # ranked by conviction_score = agreement * |return|, descending
    assert list(result["symbol"]) == sorted(result["symbol"], key=lambda s: -by_symbol.loc[s, "conviction_score"])


def test_score_universe_empty_input_returns_empty_with_columns():
    ensemble = _FakeEnsemble(mean_prediction=[], direction_agreement=[])
    result = score_universe(ensemble, pd.DataFrame(), feature_cols=["f1"])
    assert result.empty
    assert list(result.columns) == ["symbol", "predicted_return", "direction_agreement", "conviction_score", "confident"]


def _scored_df(rows):
    df = pd.DataFrame(rows)
    df["conviction_score"] = df["direction_agreement"] * df["predicted_return"].abs()
    return df


def test_select_trades_sizes_via_target_position_size_and_respects_top_k():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.04, "direction_agreement": 0.9, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.03, "direction_agreement": 0.85, "confident": True},
        ]
    )
    corr = pd.DataFrame(
        {"AAPL": [1.0, 0.0, 0.0], "TSLA": [0.0, 1.0, 0.0], "MMM": [0.0, 0.0, 1.0]},
        index=["AAPL", "TSLA", "MMM"],
    )

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
        top_k=2,
    )

    assert len(candidates) == 2
    assert [c.symbol for c in candidates] == ["AAPL", "TSLA"]  # highest conviction first
    assert all(c.side == "long" for c in candidates)


def test_select_trades_picks_short_side_for_negative_forecast():
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True}]
    )
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
    )

    assert len(candidates) == 1
    assert candidates[0].side == "short"
    assert candidates[0].target_position_pct < 0
    assert candidates[0].target_position_pct >= -0.15  # short cap, not the (looser) long cap


def test_select_trades_drops_unconfident_rows():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.20, "direction_agreement": 0.55, "confident": False},
        ]
    )
    corr = pd.DataFrame({"AAPL": [1.0, 0.0], "MMM": [0.0, 1.0]}, index=["AAPL", "MMM"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
    )

    assert [c.symbol for c in candidates] == ["AAPL"]


def test_select_trades_skips_unshortable_candidates():
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True}]
    )
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        is_shortable_fn=lambda symbol: False,
    )

    assert candidates == []


def test_select_trades_shortable_check_does_not_affect_longs():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        is_shortable_fn=lambda symbol: False,  # would block a short, must not block this long
    )

    assert len(candidates) == 1


def test_select_concentrated_trades_splits_by_relative_conviction():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    # conviction_score: AAPL=0.04, TSLA=0.02 -> raw split 2:1 -> 0.667/0.333, within [0.30, 0.70] bounds.
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30)

    assert len(candidates) == 2
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].target_position_pct == pytest.approx(2 / 3, rel=1e-6)
    assert by_symbol["TSLA"].target_position_pct == pytest.approx(1 / 3, rel=1e-6)
    # fully deployed
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(1.0)


def test_select_concentrated_trades_clamps_dominant_leg_at_bound():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.50, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.01, "direction_agreement": 1.0, "confident": True},
        ]
    )
    # Raw split would be ~98/2 — clamped to 70/30 so one pick can't swallow the whole deployment.
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30)

    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].target_position_pct == pytest.approx(0.70)
    assert by_symbol["TSLA"].target_position_pct == pytest.approx(0.30)


def test_select_concentrated_trades_signs_match_forecast_direction():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": -0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30)
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].side == "long"
    assert by_symbol["AAPL"].target_position_pct > 0
    assert by_symbol["TSLA"].side == "short"
    assert by_symbol["TSLA"].target_position_pct < 0


def test_select_concentrated_trades_single_confident_candidate_goes_all_in():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True}]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30)

    assert len(candidates) == 1
    assert candidates[0].target_position_pct == pytest.approx(1.0)


def test_select_concentrated_trades_no_confident_candidates_returns_empty():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.20, "direction_agreement": 0.55, "confident": False}]
    )
    assert select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30) == []


def test_select_concentrated_trades_respects_total_deploy_pct():
    """Regime-based damping (see run_screen) scales both legs down together."""
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30, total_deploy_pct=0.35)
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(0.35)


def test_select_concentrated_trades_skips_unshortable_and_falls_through_ranking():
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 1.0, "confident": True},  # unshortable
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MSFT", "predicted_return": 0.03, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(
        scored, max_leg_pct=0.70, min_leg_pct=0.30,
        is_shortable_fn=lambda symbol: symbol != "TSLA",
    )

    symbols = {c.symbol for c in candidates}
    assert symbols == {"AAPL", "MSFT"}  # TSLA skipped, next two ranked candidates picked instead


def test_attach_reasoning_picks_top_features_by_absolute_contribution():
    scored = _scored_df([{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}])
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_pct=0.30)
    latest = pd.DataFrame({"symbol": ["AAPL"], "f1": [1.5], "f2": [-0.5], "f3": [0.1]})
    # Indexed by symbol, matching what _attach_reasoning's X (latest.set_index("symbol")) actually looks like.
    contributions = pd.DataFrame(
        {"f1": [0.03], "f2": [-0.08], "f3": [0.001], "base_value": [0.0]}, index=pd.Index(["AAPL"], name="symbol")
    )
    ensemble = _FakeEnsemble(mean_prediction=[0.05], direction_agreement=[1.0], contributions=contributions)

    _attach_reasoning(
        candidates, ensemble, latest, feature_cols=["f1", "f2", "f3"], scored=scored, regime=TREND,
        max_leg_pct=0.70, min_leg_pct=0.30, min_direction_agreement=0.8, top_n=2,
    )

    phases = {p["phase"]: p for p in candidates[0].reasoning}
    assert set(phases) == {2, 3, 4}
    lines = " ".join(phases[2]["lines"]).lower()
    assert lines.index("f2") < lines.index("f1")  # ranked by |contribution|, f3 excluded (top_n=2)
    assert "f3" not in lines


def test_attach_reasoning_empty_candidates_is_noop():
    ensemble = _FakeEnsemble(mean_prediction=[], direction_agreement=[])
    _attach_reasoning(
        [], ensemble, pd.DataFrame(), feature_cols=[], scored=pd.DataFrame({"confident": []}),
        regime=TREND, max_leg_pct=0.70, min_leg_pct=0.30, min_direction_agreement=0.8,
    )  # should not raise


def test_build_correlation_matrix_from_prices():
    dates = pd.bdate_range("2026-01-01", periods=30, tz="UTC")
    rng = np.random.default_rng(0)
    shared = rng.normal(size=30)
    prices = pd.DataFrame(
        {
            "symbol": ["A"] * 30 + ["B"] * 30,
            "ts": list(dates) * 2,
            "close": list(100 * np.cumprod(1 + 0.001 * shared)) + list(100 * np.cumprod(1 + 0.001 * shared)),
        }
    )
    corr = build_correlation_matrix(prices)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)  # identical return series
