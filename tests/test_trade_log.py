import pandas as pd
import pytest

from monitoring.trade_log import (
    attach_actual_outcomes,
    attach_feature_vectors,
    attach_nearby_headlines,
    grade_outcomes,
    nearby_headlines,
    nearby_news,
    pivot_feature_vector,
    real_trade_rows,
    trade_direction,
)


def test_real_trade_rows_excludes_backfill_mode():
    decisions = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"], "mode": ["paper", "backfill"], "executed_position": [0.3, 0.5]}
    )
    result = real_trade_rows(decisions)
    assert list(result["symbol"]) == ["AAPL"]


def test_real_trade_rows_includes_closes_unlike_capital_deployed_accounting():
    """A close (executed_position == 0) is still an executed trade here."""
    decisions = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "mode": ["paper", "paper"], "executed_position": [0.0, None]})
    result = real_trade_rows(decisions)
    assert list(result["symbol"]) == ["AAPL"]  # the None (a hold, nothing executed) is dropped


def test_trade_direction():
    assert trade_direction(0.3) == "long"
    assert trade_direction(-0.3) == "short"
    assert trade_direction(0.0) == "close"
    assert trade_direction(None) == "unknown"
    assert trade_direction(float("nan")) == "unknown"


def test_grade_outcomes_attaches_realized_return_and_hit():
    trades = pd.DataFrame({"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-01-02"], utc=True), "forecast": [0.5]})
    prices = pd.DataFrame(
        {"symbol": ["AAPL", "AAPL"], "ts": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True), "close": [100.0, 110.0]}
    )
    result = grade_outcomes(trades, prices, horizon_bars=1)
    assert result.iloc[0]["hit"]
    assert result.iloc[0]["realized_return"] == pytest.approx(0.1)


def test_grade_outcomes_close_with_no_forecast_gets_null_outcome_not_dropped():
    trades = pd.DataFrame({"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-01-02"], utc=True), "forecast": [None]})
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])
    result = grade_outcomes(trades, prices, horizon_bars=1)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["realized_return"])


def test_grade_outcomes_empty_input_returns_empty_with_outcome_columns():
    trades = pd.DataFrame(columns=["symbol", "ts", "forecast"])
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])
    result = grade_outcomes(trades, prices, horizon_bars=1)
    assert result.empty
    assert "realized_return" in result.columns
    assert "hit" in result.columns


def test_attach_actual_outcomes_folds_the_close_into_the_entry_row():
    trades = pd.DataFrame(
        [
            {"id": 1, "symbol": "TROW", "ts": pd.Timestamp("2026-09-14T22:51:39Z"), "executed_position": -267.0},
            {"id": 2, "symbol": "TROW", "ts": pd.Timestamp("2026-09-16T21:08:21Z"), "executed_position": 0.0},
        ]
    )
    episodes = [
        {"symbol": "TROW", "entry_id": 1, "close_id": 2, "entry_ts": pd.Timestamp("2026-09-14T22:51:40Z"), "realized_pnl_pct": 0.0153}
    ]

    result = attach_actual_outcomes(trades, episodes, open_symbols=set())

    assert len(result) == 1  # the close row folded into the entry row, not shown separately
    row = result.iloc[0]
    assert row["id"] == 1
    assert row["actual_outcome"] == "closed"
    assert row["actual_realized_return"] == pytest.approx(0.0153)


def test_attach_actual_outcomes_still_open_position_reads_open_not_pending():
    trades = pd.DataFrame(
        [{"id": 1, "symbol": "CL", "ts": pd.Timestamp("2026-09-14T22:51:38Z"), "executed_position": -408.4389}]
    )

    result = attach_actual_outcomes(trades, episodes=[], open_symbols={"CL"})

    assert result.iloc[0]["actual_outcome"] == "open"
    assert pd.isna(result.iloc[0]["actual_realized_return"])


def test_attach_actual_outcomes_no_broker_confirmation_falls_back_to_pending():
    """No closing decision AND the broker no longer shows this symbol held -- can't confirm either way."""
    trades = pd.DataFrame(
        [{"id": 1, "symbol": "GHOST", "ts": pd.Timestamp("2026-09-14T22:51:38Z"), "executed_position": 10.0}]
    )

    result = attach_actual_outcomes(trades, episodes=[], open_symbols=set())

    assert result.iloc[0]["actual_outcome"] == "pending"


def test_attach_actual_outcomes_resize_row_also_reads_open():
    """A mid-episode resize (still nonzero, not a close) is its own row and should also read 'open', not 'pending'."""
    trades = pd.DataFrame(
        [
            {"id": 1, "symbol": "CL", "ts": pd.Timestamp("2026-09-14T22:51:38Z"), "executed_position": -300.0},
            {"id": 2, "symbol": "CL", "ts": pd.Timestamp("2026-09-16T21:08:22Z"), "executed_position": -408.4389},
        ]
    )

    result = attach_actual_outcomes(trades, episodes=[], open_symbols={"CL"})

    assert len(result) == 2  # nothing dropped -- no close happened
    assert set(result["actual_outcome"]) == {"open"}


def test_attach_actual_outcomes_keeps_unrelated_columns_untouched():
    trades = pd.DataFrame(
        [
            {
                "id": 1, "symbol": "AAPL", "ts": pd.Timestamp("2026-09-14T00:00:00Z"), "executed_position": 10.0,
                "forecast": 0.02, "hit": True, "realized_return": 0.05,
            },
            {"id": 2, "symbol": "AAPL", "ts": pd.Timestamp("2026-09-16T00:00:00Z"), "executed_position": 0.0},
        ]
    )

    result = attach_actual_outcomes(trades, episodes=[], open_symbols=set())

    row = result.iloc[0]
    assert row["forecast"] == pytest.approx(0.02)
    assert row["hit"] is True
    assert row["realized_return"] == pytest.approx(0.05)


def test_attach_actual_outcomes_no_matching_episode_still_marks_closed_with_no_return():
    """The decisions table says it closed, but no Alpaca episode was found (e.g. outside the fetched fill window)."""
    trades = pd.DataFrame(
        [
            {"id": 1, "symbol": "AAPL", "ts": pd.Timestamp("2026-09-14T00:00:00Z"), "executed_position": 10.0},
            {"id": 2, "symbol": "AAPL", "ts": pd.Timestamp("2026-09-16T00:00:00Z"), "executed_position": 0.0},
        ]
    )

    result = attach_actual_outcomes(trades, episodes=[], open_symbols=set())

    row = result.iloc[0]
    assert row["actual_outcome"] == "closed"
    assert pd.isna(row["actual_realized_return"])


def test_attach_actual_outcomes_matches_by_entry_id_not_position_in_a_list():
    """
    Two round trips for the same symbol: episodes must match their own
    entry decision exactly (by entry_id), not by guessing "next episode in
    the list" -- an out-of-order episodes list must still land correctly.
    """
    trades = pd.DataFrame(
        [
            {"id": 1, "symbol": "AAPL", "ts": pd.Timestamp("2026-01-01T00:00:00Z"), "executed_position": 10.0},
            {"id": 2, "symbol": "AAPL", "ts": pd.Timestamp("2026-01-02T00:00:00Z"), "executed_position": 0.0},
            {"id": 3, "symbol": "AAPL", "ts": pd.Timestamp("2026-02-01T00:00:00Z"), "executed_position": -5.0},
            {"id": 4, "symbol": "AAPL", "ts": pd.Timestamp("2026-02-02T00:00:00Z"), "executed_position": 0.0},
        ]
    )
    # Deliberately out of order, and keyed to the SECOND round trip's entry.
    episodes = [
        {"symbol": "AAPL", "entry_id": 3, "close_id": 4, "entry_ts": pd.Timestamp("2026-02-01T00:00:00Z"), "realized_pnl_pct": -0.02},
        {"symbol": "AAPL", "entry_id": 1, "close_id": 2, "entry_ts": pd.Timestamp("2026-01-01T00:00:00Z"), "realized_pnl_pct": 0.10},
    ]

    result = attach_actual_outcomes(trades, episodes, open_symbols=set())

    by_id = result.set_index("id")
    assert by_id.loc[1, "actual_realized_return"] == pytest.approx(0.10)
    assert by_id.loc[3, "actual_realized_return"] == pytest.approx(-0.02)


def test_attach_actual_outcomes_empty_input():
    trades = pd.DataFrame(columns=["id", "symbol", "ts", "executed_position"])
    result = attach_actual_outcomes(trades, episodes=[], open_symbols=set())
    assert result.empty
    assert "actual_outcome" in result.columns
    assert "actual_realized_return" in result.columns


def test_pivot_feature_vector():
    features = pd.DataFrame({"feature_name": ["mom_ret_20d", "vol_atr_14"], "value": [0.08, 1.2]})
    assert pivot_feature_vector(features) == {"mom_ret_20d": 0.08, "vol_atr_14": 1.2}


def test_pivot_feature_vector_empty_input():
    assert pivot_feature_vector(pd.DataFrame(columns=["feature_name", "value"])) == {}


def test_attach_feature_vectors_merges_on_symbol_feature_set_and_ts():
    trades = pd.DataFrame(
        {"symbol": ["AAPL"], "feature_set_id": ["v4"], "ts": pd.to_datetime(["2026-01-02"], utc=True)}
    )
    features_long = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "feature_set_id": ["v4", "v4"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "feature_name": ["mom_ret_20d", "vol_atr_14"],
            "value": [0.08, 1.2],
        }
    )
    result = attach_feature_vectors(trades, features_long)
    assert result.iloc[0]["feat_mom_ret_20d"] == 0.08
    assert result.iloc[0]["feat_vol_atr_14"] == 1.2


def test_attach_feature_vectors_no_matching_features_returns_trades_unchanged():
    trades = pd.DataFrame(
        {"symbol": ["AAPL"], "feature_set_id": ["v4"], "ts": pd.to_datetime(["2026-01-02"], utc=True)}
    )
    result = attach_feature_vectors(trades, pd.DataFrame(columns=["symbol", "feature_set_id", "ts", "feature_name", "value"]))
    assert "feat_mom_ret_20d" not in result.columns


def test_nearby_news_keeps_full_row_data_not_just_the_headline():
    news = pd.DataFrame(
        {"ts": pd.to_datetime(["2026-01-03"], utc=True), "headline": ["in window"], "source": ["wire"], "sentiment": [0.4]}
    )
    result = nearby_news(news, pd.to_datetime("2026-01-02", utc=True), window_days=3)
    assert result.iloc[0]["source"] == "wire"
    assert result.iloc[0]["sentiment"] == 0.4


def test_nearby_news_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["ts", "headline"])
    assert nearby_news(empty, pd.to_datetime("2026-01-02", utc=True)).empty


def test_nearby_headlines_filters_to_the_window_and_sorts_newest_first():
    news = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-12-27", "2026-01-03", "2026-01-10"], utc=True),
            "headline": ["too early", "in window", "too late"],
        }
    )
    result = nearby_headlines(news, pd.to_datetime("2026-01-02", utc=True), window_days=3)
    assert result == ["in window"]


def test_nearby_headlines_empty_input_returns_empty_list():
    assert nearby_headlines(pd.DataFrame(columns=["ts", "headline"]), pd.to_datetime("2026-01-02", utc=True)) == []


def test_attach_nearby_headlines_joins_per_symbol_window():
    trades = pd.DataFrame({"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-01-02"], utc=True)})
    news = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "ts": pd.to_datetime(["2026-01-03", "2026-01-02"], utc=True),
            "headline": ["AAPL news", "MSFT news (different symbol, must not appear)"],
        }
    )
    result = attach_nearby_headlines(trades, news, window_days=3)
    assert result.iloc[0]["nearby_headlines"] == "AAPL news"
