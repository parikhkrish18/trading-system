import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestResult


def _flat_price_frame(dates, price, volume=1e9):
    return pd.DataFrame({"close": price, "volume": volume}, index=dates)


def test_equity_curve_reflects_same_day_transaction_costs_on_a_rebalance_date():
    """
    Regression: portfolio_value used to be snapshotted BEFORE the rebalance
    loop applied that date's trades (and their transaction costs), then
    stored into equity_curve[date] unchanged — so a rebalance date's own
    costs never showed up in its own equity, only as an artificial
    one-day-lagged drop the next day. Using the backtest's LAST date as the
    (only) rebalance date is the sharpest version of this: with the old
    code that day's cost would never appear in the equity curve at all,
    since there is no "next day" for it to leak into.
    """
    dates = pd.bdate_range("2024-01-01", periods=5)  # Mon..Fri; Fri is the only W-FRI date in range
    prices = {"AAA": _flat_price_frame(dates, price=100.0)}

    engine = BacktestEngine(
        initial_capital=100_000.0,
        rebalance_freq="W-FRI",
        commission_per_share=0.0,
        base_spread_bps=100.0,  # 1% flat slippage; impact term zeroed below so this is exact
        impact_coefficient=0.0,
    )

    def decision_fn(date, history):
        return {"AAA": 1.0}  # fully invested

    result = engine.run(prices, decision_fn)

    rebalance_date = dates[-1]
    assert rebalance_date in result.trades["date"].values  # the single trade happened on the LAST date
    trade_cost = result.trades.loc[result.trades["date"] == rebalance_date, "cost"].iloc[0]
    assert trade_cost == pytest.approx(1_000.0)  # 100_000 (fully deployed) * 1% flat slippage

    # No prior trades, so pre-trade equity that day was exactly
    # initial_capital — the fix means the STORED equity is already net of
    # that same day's cost, not the stale pre-trade snapshot.
    assert result.equity_curve.loc[rebalance_date] == pytest.approx(100_000.0 - trade_cost)


def test_equity_curve_cost_shows_up_on_the_rebalance_date_not_the_day_after():
    """Same shape but with trading days AFTER the rebalance date, to pin down that the cost isn't lagged by one day."""
    dates = pd.bdate_range("2024-01-01", periods=10)  # two W-FRI dates: day 5 and day 10
    prices = {"AAA": _flat_price_frame(dates, price=100.0)}

    engine = BacktestEngine(
        initial_capital=100_000.0,
        rebalance_freq="W-FRI",
        commission_per_share=0.0,
        base_spread_bps=100.0,
        impact_coefficient=0.0,
    )

    def decision_fn(date, history):
        return {"AAA": 1.0}

    result = engine.run(prices, decision_fn)

    first_rebalance = dates[4]  # first Friday
    day_after = dates[5]
    trade_cost = result.trades.loc[result.trades["date"] == first_rebalance, "cost"].iloc[0]

    assert result.equity_curve.loc[first_rebalance] == pytest.approx(100_000.0 - trade_cost)
    # Flat prices and no further rebalancing until the next Friday: the day
    # after should show NO additional drop — the cost already landed on the
    # rebalance date itself, not leaked into the next day's mark.
    assert result.equity_curve.loc[day_after] == pytest.approx(result.equity_curve.loc[first_rebalance])


def test_equity_curve_unaffected_on_non_rebalance_dates():
    """Sanity check the fix didn't touch mark-to-market on days nothing traded."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = {"AAA": _flat_price_frame(dates, price=100.0)}

    engine = BacktestEngine(initial_capital=50_000.0, rebalance_freq="W-FRI")

    def decision_fn(date, history):
        return {}  # never trades

    result = engine.run(prices, decision_fn)

    assert (result.equity_curve == 50_000.0).all()


# ---------------------------------------------------------------------
# General engine behavior beyond the transaction-cost regression above.
# ---------------------------------------------------------------------


def test_equity_curve_covers_every_bar_not_just_rebalance_dates():
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = {"AAA": _flat_price_frame(dates, price=100.0)}
    engine = BacktestEngine(initial_capital=10_000.0, rebalance_freq="W-FRI")

    result = engine.run(prices, lambda date, history: {})

    assert len(result.equity_curve) == len(dates)
    assert list(result.equity_curve.index) == list(dates)


def test_a_long_only_full_deployment_tracks_price_moves_after_the_only_trade():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = {"AAA": _flat_price_frame([dates[0], dates[1], dates[2]], price=[100.0, 100.0, 110.0])}
    engine = BacktestEngine(initial_capital=10_000.0, rebalance_freq="D", commission_per_share=0.0, base_spread_bps=0.0, impact_coefficient=0.0)

    def decision_fn(date, history):
        return {"AAA": 1.0} if date == dates[0] else {}

    result = engine.run(prices, decision_fn)

    # 100 shares bought at $100 with zero costs; day 3's close is $110.
    assert result.equity_curve.loc[dates[-1]] == pytest.approx(11_000.0)


def test_a_short_position_loses_money_when_price_rises():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = {"AAA": _flat_price_frame([dates[0], dates[1], dates[2]], price=[100.0, 100.0, 120.0])}
    engine = BacktestEngine(initial_capital=10_000.0, rebalance_freq="D", commission_per_share=0.0, base_spread_bps=0.0, impact_coefficient=0.0)

    def decision_fn(date, history):
        return {"AAA": -1.0} if date == dates[0] else {}

    result = engine.run(prices, decision_fn)

    # Fully short $10,000 notional at $100 (-100 shares); price rises to $120
    # -> the short leg loses $2,000, portfolio drops to $8,000.
    assert result.equity_curve.loc[dates[-1]] == pytest.approx(8_000.0)


def test_multi_symbol_rebalance_splits_capital_per_target_weight():
    dates = pd.bdate_range("2024-01-01", periods=2)
    prices = {
        "AAA": _flat_price_frame(dates, price=100.0),
        "BBB": _flat_price_frame(dates, price=50.0),
    }
    engine = BacktestEngine(initial_capital=10_000.0, rebalance_freq="D", commission_per_share=0.0, base_spread_bps=0.0, impact_coefficient=0.0)

    def decision_fn(date, history):
        return {"AAA": 0.6, "BBB": 0.4}

    result = engine.run(prices, decision_fn)

    day0_trades = result.trades[result.trades["date"] == dates[0]]
    aaa_shares = day0_trades.loc[day0_trades["symbol"] == "AAA", "trade_shares"].iloc[0]
    bbb_shares = day0_trades.loc[day0_trades["symbol"] == "BBB", "trade_shares"].iloc[0]
    assert aaa_shares == pytest.approx(60.0)  # 60% of $10,000 / $100
    assert bbb_shares == pytest.approx(80.0)  # 40% of $10,000 / $50


def test_a_symbol_missing_a_bar_on_the_rebalance_date_is_skipped_not_errored():
    dates = pd.bdate_range("2024-01-01", periods=3)
    aaa = _flat_price_frame(dates, price=100.0)
    bbb = _flat_price_frame(dates, price=50.0).drop(dates[0])  # BBB has no bar on day 0
    prices = {"AAA": aaa, "BBB": bbb}
    engine = BacktestEngine(initial_capital=10_000.0, rebalance_freq="D")

    def decision_fn(date, history):
        return {"AAA": 0.5, "BBB": 0.5}

    result = engine.run(prices, decision_fn)  # must not raise

    day0_trades = result.trades[result.trades["date"] == dates[0]]
    assert set(day0_trades["symbol"]) == {"AAA"}  # BBB silently skipped for that date only


def test_summary_reports_total_return_and_max_drawdown():
    dates = pd.bdate_range("2024-01-01", periods=4)
    # Equity: 10000 -> 12000 -> 9000 -> 10500 (a drawdown then partial recovery)
    equity = pd.Series([10_000.0, 12_000.0, 9_000.0, 10_500.0], index=dates, name="equity")
    trades = pd.DataFrame(columns=["date", "symbol", "trade_shares", "price", "cost"])
    positions = pd.DataFrame(index=dates)

    result = BacktestResult(equity_curve=equity, trades=trades, positions=positions)
    summary = result.summary

    assert summary["total_return"] == pytest.approx(0.05)  # 10500/10000 - 1
    assert summary["max_drawdown"] == pytest.approx((9_000.0 / 12_000.0) - 1)  # -25% from the peak
    assert summary["n_trades"] == 0


def test_summary_sharpe_is_nan_when_returns_have_zero_variance():
    dates = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([10_000.0] * 4, index=dates, name="equity")  # perfectly flat -- std() == 0
    trades = pd.DataFrame(columns=["date", "symbol", "trade_shares", "price", "cost"])
    positions = pd.DataFrame(index=dates)

    result = BacktestResult(equity_curve=equity, trades=trades, positions=positions)

    assert pd.isna(result.summary["sharpe"])
