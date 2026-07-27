"""
Hard circuit breakers. These are the last line of defense and should be the
most conservatively coded, most heavily tested part of the repo — the plan
explicitly calls out risk/circuit_breakers.py and execution/broker_*.py
as the two files where a bug does the most damage (Security section).

Design intent: every check here returns a plain (triggered: bool, reason:
str) pair. Nothing here places orders directly — execution/ decides what to
do with a triggered breaker (e.g. flatten via get_broker()), keeping "did a
limit get breached" separate from "what do we do about it".
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class BreakerResult:
    triggered: bool
    reason: str = ""


def max_drawdown_breaker(equity_curve, max_drawdown_pct: float) -> BreakerResult:
    """
    equity_curve: array-like of portfolio equity over time (most recent
    last). Triggers if current drawdown from the running peak exceeds
    max_drawdown_pct.
    """
    if len(equity_curve) < 2:
        return BreakerResult(False)
    peak = max(equity_curve)
    current = equity_curve[-1]
    if peak <= 0:
        return BreakerResult(False)
    drawdown = (current / peak) - 1.0
    if drawdown < -abs(max_drawdown_pct):
        return BreakerResult(True, f"Drawdown {drawdown:.2%} exceeds limit -{max_drawdown_pct:.2%}")
    return BreakerResult(False)


def max_single_position_breaker(
    position_value: float, portfolio_value: float, max_single_position_pct: float
) -> BreakerResult:
    if portfolio_value <= 0:
        return BreakerResult(False)
    pct = abs(position_value) / portfolio_value
    if pct > max_single_position_pct:
        return BreakerResult(
            True, f"Position is {pct:.2%} of portfolio, exceeds limit {max_single_position_pct:.2%}"
        )
    return BreakerResult(False)


def max_correlated_exposure_breaker(
    positions_by_symbol: dict[str, float],
    portfolio_value: float,
    correlation_matrix,
    max_correlated_exposure_pct: float,
    correlation_threshold: float = 0.7,
) -> list[BreakerResult]:
    """
    Checks every symbol's cluster of highly-correlated open positions against
    the portfolio-level cap. Returns one result per symbol that breaches —
    empty list if none do.
    """
    if portfolio_value <= 0 or not positions_by_symbol:
        return []

    results = []
    for symbol, value in positions_by_symbol.items():
        cluster_exposure = abs(value)
        for other_symbol, other_value in positions_by_symbol.items():
            if other_symbol == symbol:
                continue
            if symbol in correlation_matrix.index and other_symbol in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol, other_symbol]
                if corr > correlation_threshold:
                    cluster_exposure += abs(other_value)
        pct = cluster_exposure / portfolio_value
        if pct > max_correlated_exposure_pct:
            results.append(
                BreakerResult(
                    True,
                    f"{symbol}'s correlated cluster is {pct:.2%} of portfolio, "
                    f"exceeds limit {max_correlated_exposure_pct:.2%}",
                )
            )
    return results


def run_all_breakers(
    equity_curve,
    positions_by_symbol: dict[str, float],
    portfolio_value: float,
    correlation_matrix,
    max_drawdown_pct: float,
    max_single_position_pct: float,
    max_correlated_exposure_pct: float,
) -> list[BreakerResult]:
    """Runs every breaker, returns only the ones that triggered."""
    triggered = []

    dd = max_drawdown_breaker(equity_curve, max_drawdown_pct)
    if dd.triggered:
        triggered.append(dd)

    for value in positions_by_symbol.values():
        pos = max_single_position_breaker(value, portfolio_value, max_single_position_pct)
        if pos.triggered:
            triggered.append(pos)

    triggered += [
        r
        for r in max_correlated_exposure_breaker(
            positions_by_symbol, portfolio_value, correlation_matrix, max_correlated_exposure_pct
        )
        if r.triggered
    ]

    return triggered
