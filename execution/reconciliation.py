"""
Compares intended target positions (what risk/sizing.py decided) against
what actually got filled at the broker, and flags divergence beyond a
tolerance. This is what Phase 6, point 3 means by "compare paper results to
what the backtest predicted" at the position level, not just P&L.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ReconciliationResult:
    symbol: str
    intended_shares: float
    actual_shares: float
    diff_shares: float
    diff_pct: float
    flagged: bool


def reconcile_positions(
    intended: dict[str, float],
    actual: dict[str, float],
    tolerance_pct: float = 0.02,
) -> list[ReconciliationResult]:
    """
    `intended`: {symbol: target_shares} as decided by sizing logic.
    `actual`: {symbol: filled_shares} as reported by the broker.
    Flags any symbol where the actual position diverges from intended by
    more than `tolerance_pct` (as a fraction of the intended size), or where
    a position exists on one side but not the other.
    """
    symbols = set(intended) | set(actual)
    results = []
    for symbol in sorted(symbols):
        intended_shares = intended.get(symbol, 0.0)
        actual_shares = actual.get(symbol, 0.0)
        diff = actual_shares - intended_shares
        denom = abs(intended_shares) if intended_shares != 0 else max(abs(actual_shares), 1.0)
        diff_pct = diff / denom
        flagged = abs(diff_pct) > tolerance_pct
        results.append(
            ReconciliationResult(
                symbol=symbol,
                intended_shares=intended_shares,
                actual_shares=actual_shares,
                diff_shares=diff,
                diff_pct=diff_pct,
                flagged=flagged,
            )
        )
    return results


def summarize(results: list[ReconciliationResult]) -> str:
    flagged = [r for r in results if r.flagged]
    if not flagged:
        return f"All {len(results)} position(s) reconciled within tolerance."
    lines = [f"{len(flagged)} of {len(results)} position(s) diverged beyond tolerance:"]
    lines.extend(
        f"  {r.symbol}: intended={r.intended_shares}, actual={r.actual_shares}, "
        f"diff={r.diff_shares} ({r.diff_pct:.1%})"
        for r in flagged
    )
    return "\n".join(lines)
