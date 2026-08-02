import numpy as np
import pandas as pd
import pytest

from models.regime.trend_chop_classifier import CHOP, TREND
from models.screener import (
    TradeCandidate,
    build_correlation_matrix,
    log_candidates,
    per_symbol_regimes,
    score_universe,
    select_trades,
)


class _FakeEnsemble:
    """Fakes EnsembleForecastModel.predict with a preset mapping of row index -> stats."""

    def __init__(self, mean_prediction, direction_agreement, std_prediction=None):
        self.mean_prediction = mean_prediction
        self.direction_agreement = direction_agreement
        self.std_prediction = std_prediction if std_prediction is not None else [0.0] * len(mean_prediction)

    def predict(self, X):
        return pd.DataFrame(
            {
                "mean_prediction": self.mean_prediction,
                "std_prediction": self.std_prediction,
                "direction_agreement": self.direction_agreement,
            },
            index=X.index,
        )


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


def test_per_symbol_regimes_splits_on_adx_threshold():
    latest = pd.DataFrame({"symbol": ["AAPL", "TSLA", "MMM"], "adx_14": [30.0, 10.0, np.nan]})

    regimes = per_symbol_regimes(latest, adx_threshold=25.0)

    assert regimes == {"AAPL": TREND, "TSLA": CHOP}  # NaN ADX omitted -> caller falls back to TREND


def test_per_symbol_regimes_without_adx_column_returns_empty():
    assert per_symbol_regimes(pd.DataFrame({"symbol": ["AAPL"], "f1": [0.1]})) == {}


def test_select_trades_damps_chop_symbols_and_tags_regime():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
        ]
    )
    corr = pd.DataFrame(
        {"AAPL": [1.0, 0.0], "MMM": [0.0, 1.0]}, index=["AAPL", "MMM"]
    )

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
        regime_by_symbol={"AAPL": TREND, "MMM": CHOP},
    )

    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].regime == TREND
    assert by_symbol["MMM"].regime == CHOP
    # identical forecasts, but the chop symbol must be sized strictly smaller
    assert abs(by_symbol["MMM"].target_position_pct) < abs(by_symbol["AAPL"].target_position_pct)


def test_select_trades_falls_back_to_global_regime_for_unmapped_symbols():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        regime_by_symbol={"TSLA": CHOP},  # AAPL not in the map
    )

    assert candidates[0].regime == TREND


def test_log_candidates_writes_regime(monkeypatch):
    captured = {}

    def fake_to_sql(self, name, engine, **kwargs):
        captured["table"] = name
        captured["df"] = self

    monkeypatch.setattr("models.screener.get_engine", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    candidate = TradeCandidate(
        symbol="MMM", side="long", predicted_return=0.03, direction_agreement=0.9,
        conviction_score=0.027, target_position_pct=0.05, regime=CHOP,
    )
    n = log_candidates([candidate], feature_set_id="v3")

    assert n == 1
    assert captured["table"] == "decisions"
    assert list(captured["df"]["regime"]) == [CHOP]


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
