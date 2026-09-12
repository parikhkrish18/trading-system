import datetime as dt

import pandas as pd

from monitoring.model_report_card import (
    compute_report_card,
    current_capital_deployed_pct,
    real_trades_taken,
)

# --------------------------------------------------------------------------
# real_trades_taken
# --------------------------------------------------------------------------


def test_real_trades_taken_excludes_backfill_mode():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "mode": ["paper", "backfill"],
            "target_position": [0.3, 0.5],
        }
    )
    result = real_trades_taken(decisions)
    assert list(result["symbol"]) == ["AAPL"]


def test_real_trades_taken_excludes_holds_and_closes():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "TSLA"],
            "ts": pd.to_datetime(["2026-01-02"] * 3, utc=True),
            "mode": ["paper"] * 3,
            "target_position": [0.3, None, 0.0],  # opened, held, closed
        }
    )
    result = real_trades_taken(decisions)
    assert list(result["symbol"]) == ["AAPL"]


def test_real_trades_taken_empty_input_returns_empty():
    decisions = pd.DataFrame(columns=["symbol", "ts", "mode", "target_position"])
    assert real_trades_taken(decisions).empty


# --------------------------------------------------------------------------
# current_capital_deployed_pct
# --------------------------------------------------------------------------


def test_capital_deployed_sums_the_latest_decision_per_symbol():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-09"], utc=True),
            "mode": ["paper"] * 3,
            # AAPL's older 0.6 must not double-count -- only its latest (0.4) does.
            "target_position": [0.6, 0.4, 0.3],
        }
    )
    assert current_capital_deployed_pct(decisions) == 0.7


def test_capital_deployed_counts_a_short_by_its_absolute_size():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "mode": ["paper"],
            "target_position": [-0.5],
        }
    )
    assert current_capital_deployed_pct(decisions) == 0.5


def test_capital_deployed_is_zero_when_the_latest_decision_is_a_close():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-09"], utc=True),
            "mode": ["paper", "paper"],
            "target_position": [0.6, 0.0],
        }
    )
    assert current_capital_deployed_pct(decisions) == 0.0


def test_capital_deployed_ignores_backfill_mode():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "mode": ["backfill"],
            "target_position": [0.9],
        }
    )
    assert current_capital_deployed_pct(decisions) == 0.0


def test_capital_deployed_empty_input_is_zero():
    decisions = pd.DataFrame(columns=["symbol", "ts", "mode", "target_position"])
    assert current_capital_deployed_pct(decisions) == 0.0


# --------------------------------------------------------------------------
# compute_report_card
# --------------------------------------------------------------------------


def test_compute_report_card_grades_only_real_matured_trades():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "TSLA"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-02"], utc=True),
            "mode": ["paper", "paper", "backfill"],  # TSLA must never count
            "forecast": [0.5, -0.3, 0.9],
            "target_position": [0.4, -0.2, 0.9],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-02", "2026-01-05"], utc=True),
            "close": [100.0, 110.0, 200.0, 190.0],  # AAPL up (hit), MSFT down (hit for the short)
        }
    )

    result = compute_report_card(decisions, prices, horizon_bars=1, as_of_date=dt.date(2026, 1, 10))

    assert result["n_trades_taken"] == 2  # AAPL + MSFT, not the backfill TSLA row
    assert result["n_matured"] == 2
    assert result["n_hits"] == 2
    assert result["hit_rate"] == 1.0
    assert result["as_of_date"] == "2026-01-10"


def test_compute_report_card_handles_no_real_trades_at_all():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "mode": ["backfill"],
            "forecast": [0.5],
            "target_position": [0.4],
        }
    )
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])

    result = compute_report_card(decisions, prices, horizon_bars=1, as_of_date=dt.date(2026, 1, 10))

    assert result["n_trades_taken"] == 0
    assert result["n_matured"] == 0
    assert result["hit_rate"] is None
    assert result["avg_realized_return"] is None
    assert result["capital_deployed_pct"] == 0.0


def test_compute_report_card_unmatured_trades_still_count_as_taken_not_matured():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-09"], utc=True),
            "mode": ["paper"],
            "forecast": [0.5],
            "target_position": [0.4],
        }
    )
    # Only one price bar exists at the decision date itself -- no future bar
    # to grade against yet.
    prices = pd.DataFrame({"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-01-09"], utc=True), "close": [100.0]})

    result = compute_report_card(decisions, prices, horizon_bars=1, as_of_date=dt.date(2026, 1, 10))

    assert result["n_trades_taken"] == 1
    assert result["n_matured"] == 0
    assert result["hit_rate"] is None
    assert result["capital_deployed_pct"] == 0.4
