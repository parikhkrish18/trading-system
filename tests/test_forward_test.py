import pandas as pd
import pytest

from monitoring.forward_test import graded_real_decisions, headline_metrics, monthly_success_buckets


def test_graded_real_decisions_excludes_backfill_mode():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "mode": ["paper", "backfill"],
            "forecast": [0.5, 0.5],
        }
    )
    prices = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"] * 2, "ts": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"], utc=True), "close": [100.0, 200.0, 110.0, 190.0]}
    )
    scored = graded_real_decisions(decisions, prices, horizon_bars=1)
    assert list(scored["symbol"]) == ["AAPL"]


def test_graded_real_decisions_empty_input_returns_empty():
    decisions = pd.DataFrame(columns=["symbol", "ts", "mode", "forecast"])
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])
    assert graded_real_decisions(decisions, prices, horizon_bars=1).empty


def test_monthly_success_buckets_groups_by_calendar_month():
    scored = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "TSLA"],
            "ts": pd.to_datetime(["2026-07-02", "2026-07-15", "2026-08-01"], utc=True),
            "forecast": [0.5, -0.3, 0.2],
            "realized_return": [0.05, 0.02, -0.01],
            "hit": [True, False, False],
        }
    )
    buckets = monthly_success_buckets(scored)
    assert list(buckets["month"]) == ["2026-07", "2026-08"]
    assert buckets.iloc[0]["n_graded"] == 2
    assert buckets.iloc[0]["success_rate"] == 0.5
    assert buckets.iloc[1]["n_graded"] == 1


def test_monthly_success_buckets_empty_input_returns_empty():
    scored = pd.DataFrame(columns=["symbol", "ts", "forecast", "realized_return", "hit"])
    buckets = monthly_success_buckets(scored)
    assert buckets.empty
    assert list(buckets.columns) == ["month", "n_graded", "success_rate", "avg_realized_return"]


def test_headline_metrics_pools_across_months_rather_than_averaging_bucket_rates():
    # July: 1/1 hit. August: 0/3 hits. A naive average-of-buckets would say
    # (100% + 0%) / 2 = 50%; pooled across all 4 graded decisions it's 25%.
    scored = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "ts": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-08-02", "2026-08-03"], utc=True),
            "forecast": [0.5, 0.5, 0.5, 0.5],
            "realized_return": [0.05, -0.01, -0.02, -0.03],
            "hit": [True, False, False, False],
        }
    )
    headline = headline_metrics(scored)
    assert headline["n_months"] == 2
    assert headline["n_graded"] == 4
    assert headline["success_rate"] == 0.25
    assert headline["avg_realized_return"] == pytest.approx((0.05 - 0.01 - 0.02 - 0.03) / 4)


def test_headline_metrics_empty_input():
    scored = pd.DataFrame(columns=["symbol", "ts", "forecast", "realized_return", "hit"])
    headline = headline_metrics(scored)
    assert headline == {"n_months": 0, "n_graded": 0, "success_rate": None, "avg_realized_return": None}
