"""
The do-nothing baseline, and the cross-sectional transforms that exist
because of it.

Why this module exists: for its whole life this project reported "the model
made +0.71% per trade after costs at a 20-day horizon" and nobody asked the
obvious follow-up — compared to what? Buying every candidate row in the same
test window and holding it for the same 20 days paid +1.13%. The model was
measuring market drift and calling it skill: 94.6% of its trades were longs
in a rising market. Every return figure this repo prints must now carry its
benchmark, and the headline number is the DIFFERENCE (excess_return), not
the raw return.

Two families of helpers:

  trade_metrics / benchmark_return
      The evaluation side. What a set of trades paid, what doing nothing
      paid, the gap between them, and the long/short split — because an
      aggregate that is 95% longs tells you almost nothing about the 5%.

  cross_sectional_excess / cross_sectional_zscore
      The training side (TARGET_MODE=relative). If the model is graded on
      beating the market, train it to predict beating the market: label a
      stock by how much it outruns the equal-weight universe *on its own
      date*, and rank features against same-day peers rather than against
      absolute levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sentinel for "this fold had no trades of this kind" — never 0.0, which
# would silently average in as a real observation of a break-even trade.
_NO_DATA = float("nan")


def benchmark_return(universe_returns: np.ndarray | pd.Series) -> float:
    """
    The do-nothing baseline: equal-weight buy-and-hold over every candidate
    row in the window. One row = one stock on one date held for the target
    horizon, so this is what a coin-flip-free investor earns by owning the
    whole universe over the same period the model was trading it.

    Deliberately GROSS of transaction costs while the model's number is net.
    Buy-and-hold pays the spread once over the whole period; a trader pays it
    on every entry and exit. Charging the benchmark nothing is the harder
    test, and the harder test is the honest one.
    """
    values = pd.Series(universe_returns).dropna()
    return float(values.mean()) if len(values) else _NO_DATA


def _side_metrics(realized: np.ndarray, sign: float, cost: float) -> tuple[float, float, int]:
    """(net return per trade, win rate, n) for one side. Win rate is AFTER
    costs — a trade that made 1bp on a 2bp round trip lost money."""
    if realized.size == 0:
        return _NO_DATA, _NO_DATA, 0
    net = sign * realized - cost
    return float(net.mean()), float((net > 0).mean()), int(realized.size)


def trade_metrics(
    predictions: np.ndarray | pd.Series,
    realized_returns: np.ndarray | pd.Series,
    universe_returns: np.ndarray | pd.Series,
    cost: float,
) -> dict[str, float]:
    """
    Everything needed to judge a set of trades against doing nothing.

    predictions       model output on the rows actually traded (the confident
                      subset). Only its SIGN is used: positive = go long.
    realized_returns  the ABSOLUTE forward returns of those same rows. Must
                      stay absolute even when TARGET_MODE=relative — the
                      training label can be market-relative, but what a trade
                      paid is measured in real money.
    universe_returns  absolute forward returns of EVERY candidate row in the
                      window, traded or not. This is the benchmark pool.
    cost              round-trip transaction cost fraction (see
                      backtest/cost_model.py).

    The headline is excess_return = model_return_net - benchmark_return.
    A positive model_return_net with a negative excess_return means the
    strategy made money and still would have done better owning the index.
    """
    preds = np.asarray(predictions, dtype=float)
    realized = np.asarray(realized_returns, dtype=float)
    bench = benchmark_return(universe_returns)

    if preds.size == 0:
        return {
            "benchmark_return": bench,
            "model_return_gross": _NO_DATA,
            "model_return_net": _NO_DATA,
            "excess_return": _NO_DATA,
            "n_trades": 0,
            "pct_long": _NO_DATA,
            "long_return_net": _NO_DATA,
            "long_win_rate": _NO_DATA,
            "n_long": 0,
            "short_return_net": _NO_DATA,
            "short_win_rate": _NO_DATA,
            "n_short": 0,
        }

    # sign(0) is 0, which would book a zero-return trade that never happened.
    # A prediction of exactly zero is treated as long, matching the screener's
    # `side = "long" if forecast >= 0 else "short"`.
    sides = np.where(preds >= 0, 1.0, -1.0)
    traded = sides * realized
    model_net = float(np.mean(traded - cost))

    long_net, long_win, n_long = _side_metrics(realized[sides > 0], 1.0, cost)
    short_net, short_win, n_short = _side_metrics(realized[sides < 0], -1.0, cost)

    return {
        "benchmark_return": bench,
        "model_return_gross": float(np.mean(traded)),
        "model_return_net": model_net,
        "excess_return": model_net - bench,
        "n_trades": int(preds.size),
        "pct_long": float(n_long / preds.size),
        "long_return_net": long_net,
        "long_win_rate": long_win,
        "n_long": n_long,
        "short_return_net": short_net,
        "short_win_rate": short_win,
        "n_short": n_short,
    }


def cross_sectional_excess(
    df: pd.DataFrame, return_col: str = "fwd_return", date_col: str = "ts"
) -> pd.Series:
    """
    A stock's forward return minus the equal-weight mean forward return of
    everything else measured on the SAME date — the training label for
    TARGET_MODE=relative.

    Subtracting the same-date mean strips out whatever the whole market did
    that day, which is the term that dominates an absolute forward return and
    is exactly what the model was previously being rewarded for predicting.
    What survives is the part a stock-picker could actually have added.

    The mean includes the stock itself. With ~500 names the self-inclusion
    bias is ~1/500 of the stock's own return and shrinking it would introduce
    a date-dependent scale factor for no real gain.
    """
    market = df.groupby(date_col)[return_col].transform("mean")
    return df[return_col] - market


def cross_sectional_zscore(
    df: pd.DataFrame, feature_cols: list[str], date_col: str = "ts"
) -> pd.DataFrame:
    """
    Per-date z-scores of each feature across the universe, so the model ranks
    stocks against their same-day peers instead of against absolute levels.

    Without this, a feature like 14-day RSI carries a market-wide component:
    in a rally nearly everything reads high, and the model learns "high RSI
    -> good" when it has really only learned "rallies go up". Z-scoring per
    date removes the common level and leaves the relative position, which is
    the only thing a market-relative label can be predicted from.

    Dates where a feature has zero variance (or a single symbol) would divide
    by zero; those become 0.0 — no information, no rank, rather than inf.
    """
    out = df.copy()
    grouped = df.groupby(date_col)[feature_cols]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (df[feature_cols] - mean) / std.where(std > 0)
    out[feature_cols] = z.fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Measuring the book that actually trades
# --------------------------------------------------------------------------
#
# The walk-forward used to score every confident row, long and short alike,
# and report the result as though it described the live system. It did not:
# production runs with ALLOW_SHORTS=false, so roughly two thirds of the
# scored picks were trades that would never have been placed. The headline
# excess described a long-short book nobody runs.
#
# That is the same failure as reporting a return with no benchmark beside
# it — not a wrong number, a number measuring something other than the
# question. So selection here mirrors models/screener.py: the cost floor,
# then the shorts filter, then top-k by conviction, per date.


def production_book_mask(
    predictions: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    cost: float,
    allow_shorts: bool,
    top_k: int,
) -> np.ndarray:
    """
    Which rows the live screener would actually have traded.

    `predictions`  model output for every candidate row in the window.
    `dates`        each row's date. Top-k is applied PER DATE, because the
                   screener picks a fresh book each time it runs — ranking
                   the whole test window at once would let a great pick in
                   March crowd out every pick in April.
    `cost`         round-trip cost floor; a predicted move smaller than what
                   the trade costs is not tradeable (score_universe).
    `allow_shorts` when False, negative predictions are dropped outright,
                   the same way select_trades skips a short candidate.
    `top_k`        how many names the book holds at most on any one date.

    Returns a boolean mask over the input rows.
    """
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        return np.zeros(0, dtype=bool)

    eligible = np.abs(preds) >= cost
    if not allow_shorts:
        # sign(0) is treated as long everywhere else in this module; keep
        # that consistent so a zero prediction isn't silently dropped here
        # and traded there.
        eligible &= preds >= 0

    frame = pd.DataFrame({"pred": preds, "date": np.asarray(dates), "eligible": eligible})
    frame["rank"] = (
        frame["pred"].abs().where(frame["eligible"]).groupby(frame["date"]).rank(ascending=False, method="first")
    )
    return (frame["eligible"] & (frame["rank"] <= top_k)).to_numpy()


def describe_book(*, allow_shorts: bool, strategy_mode: str, top_k: int, cost: float) -> str:
    """
    What configuration a set of results measured. Printed with the verdict,
    because two runs of this harness can disagree simply by measuring
    different systems, and a reader has no way to tell from the numbers.
    """
    sides = "long and short" if allow_shorts else "long only"
    return (
        f"{strategy_mode} book, {sides}, top {top_k} per date, "
        f"minimum move {cost:.2%} (the round-trip cost)"
    )
