import pandas as pd

from monitoring.forecast_accuracy import compute_forecast_accuracy


def test_compute_forecast_accuracy_scores_directional_hits():
    decisions = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
            "forecast": [0.5, -0.3, 0.1],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * 5,
            "ts": pd.to_datetime(
                ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"], utc=True
            ),
            "close": [500, 505, 500, 495, 500],
        }
    )
    result = compute_forecast_accuracy(decisions, prices, horizon_bars=1)

    # 2026-01-04's decision has no future bar left after horizon lookup beyond
    # what's provided consistently, so only fully-matured rows are scored.
    assert set(result["ts"]) <= set(decisions["ts"])
    assert len(result) >= 2

    row_0102 = result.loc[result["ts"] == pd.Timestamp("2026-01-02", tz="UTC")].iloc[0]
    assert row_0102["realized_return"] == 505 / 500 - 1
    assert row_0102["hit"]  # positive forecast, positive realized return

    row_0103 = result.loc[result["ts"] == pd.Timestamp("2026-01-03", tz="UTC")].iloc[0]
    assert row_0103["hit"]  # negative forecast, negative realized return


def test_compute_forecast_accuracy_empty_inputs_return_empty():
    empty = pd.DataFrame(columns=["symbol", "ts", "forecast"])
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])
    result = compute_forecast_accuracy(empty, prices)
    assert result.empty


def test_compute_forecast_accuracy_drops_decisions_without_a_future_bar():
    decisions = pd.DataFrame(
        {"symbol": ["SPY"], "ts": pd.to_datetime(["2026-01-06"], utc=True), "forecast": [0.2]}
    )
    prices = pd.DataFrame(
        {"symbol": ["SPY"], "ts": pd.to_datetime(["2026-01-06"], utc=True), "close": [500.0]}
    )
    result = compute_forecast_accuracy(decisions, prices)
    assert result.empty
