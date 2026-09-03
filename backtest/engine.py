"""
Event-driven backtester: processes bars sequentially and only ever exposes
a strategy function to data up to and including the current bar — it can't
accidentally look ahead, unlike a fully vectorized backtest where it's easy
to (say) use a rolling feature computed with a centered window by mistake.

Relationship to decay_sim.py: this engine consumes whatever daily close
price series you give it for each symbol. For symbols with enough real
trading history (e.g. TQQQ since 2010), just use the real price series —
it already reflects true daily-reset compounding. decay_sim.py is for
constructing a *synthetic* leveraged-ETF price series (e.g. to backtest
further back than a fund's actual inception, using the underlying index),
or for auditing how much naive modeling would have overstated returns.
Keep that construction step separate from this engine, upstream of it.

Rebalance cadence matches the plan: weekly or biweekly, not daily.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from backtest.cost_model import total_transaction_cost


@dataclasses.dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame

    @property
    def summary(self) -> dict:
        returns = self.equity_curve.pct_change().dropna()
        n_years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else float("nan")
        running_max = self.equity_curve.cummax()
        drawdown = self.equity_curve / running_max - 1
        max_drawdown = drawdown.min()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else float("nan")
        return {
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "n_trades": len(self.trades),
        }


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        rebalance_freq: str = "W-FRI",   # weekly, Fridays; use "2W-FRI" for biweekly
        commission_per_share: float = 0.0,
        base_spread_bps: float = 1.0,
        impact_coefficient: float = 10.0,
    ):
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.commission_per_share = commission_per_share
        self.base_spread_bps = base_spread_bps
        self.impact_coefficient = impact_coefficient

    def run(
        self,
        prices: dict[str, pd.DataFrame],
        decision_fn,
    ) -> BacktestResult:
        """
        prices: {symbol: DataFrame indexed by date with columns [close, volume]}.
        decision_fn(date, price_history) -> {symbol: target_weight in [-1, 1]}
            price_history is a {symbol: DataFrame} dict truncated to rows with
            index <= date — decision_fn physically cannot see the future.

        Returns a BacktestResult with the full equity curve (every bar, not
        just rebalance dates) so drawdown/vol stats reflect real day-to-day
        movement even though trading only happens on rebalance dates.
        """
        all_dates = sorted(set().union(*[df.index for df in prices.values()]))
        all_dates = pd.DatetimeIndex(all_dates)

        # Build rebalance date set directly from calendar, then snap each to
        # the nearest available trading date at or before it.
        cal = pd.date_range(all_dates.min(), all_dates.max(), freq=self.rebalance_freq)
        rebalance_dates = set()
        for rd in cal:
            candidates = all_dates[all_dates <= rd]
            if len(candidates):
                rebalance_dates.add(candidates[-1])

        cash = self.initial_capital
        shares: dict[str, float] = dict.fromkeys(prices, 0.0)
        equity_curve = {}
        trade_log = []
        position_log = []

        for date in all_dates:
            # Mark-to-market with today's close, using only data up to today.
            portfolio_value = cash
            for sym, df in prices.items():
                if date in df.index:
                    portfolio_value += shares[sym] * df.loc[date, "close"]

            if date in rebalance_dates:
                history = {sym: df.loc[:date] for sym, df in prices.items()}
                target_weights = decision_fn(date, history)

                for sym, target_weight in target_weights.items():
                    df = prices[sym]
                    if date not in df.index:
                        continue
                    price = df.loc[date, "close"]
                    volume = df.loc[date, "volume"] if "volume" in df.columns else 1e9
                    target_value = target_weight * portfolio_value
                    target_shares = target_value / price
                    trade_shares = target_shares - shares[sym]

                    if abs(trade_shares) < 1e-6:
                        continue

                    cost = total_transaction_cost(
                        trade_shares, price, volume,
                        self.commission_per_share, self.base_spread_bps, self.impact_coefficient,
                    )
                    cash -= trade_shares * price + cost
                    shares[sym] += trade_shares
                    trade_log.append(
                        {"date": date, "symbol": sym, "trade_shares": trade_shares, "price": price, "cost": cost}
                    )

                # Re-mark AFTER the rebalance: cash/shares above were just
                # mutated by today's trades (and their transaction costs),
                # so the pre-trade portfolio_value snapshotted at the top of
                # this iteration is now stale for THIS date. Storing that
                # stale value would report every rebalance date's equity as
                # if its own costs hadn't happened yet — they'd only show up
                # as an artificial one-day-lagged drop tomorrow, and
                # wouldn't show up at all if this date is the backtest's
                # last one.
                portfolio_value = cash
                for sym, df in prices.items():
                    if date in df.index:
                        portfolio_value += shares[sym] * df.loc[date, "close"]

            position_log.append({"date": date, **{sym: shares[sym] for sym in prices}})
            equity_curve[date] = portfolio_value

        equity_series = pd.Series(equity_curve).sort_index()
        equity_series.name = "equity"
        trades_df = pd.DataFrame(trade_log)
        positions_df = pd.DataFrame(position_log).set_index("date") if position_log else pd.DataFrame()

        return BacktestResult(equity_curve=equity_series, trades=trades_df, positions=positions_df)
