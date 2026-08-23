"""
Compares intended target positions (what risk/sizing.py decided) against
what actually happened at the broker, and flags real divergence. This is
what Phase 6, point 3 means by "compare paper results to what the backtest
predicted" at the position level, not just P&L.

The distinction this module exists to make: **an order that hasn't filled
yet is not an order that went wrong.** Submitting outside market hours —
which every cycle does when it runs after the close — leaves the position
at zero until the next open. Comparing intent against holdings alone reads
that as a total failure, so a healthy run reported "3 of 3 positions
diverged beyond tolerance" and a warning that fires on success is one
nobody reads by the time it matters.

So reconciliation needs the order outcome, not just the position count:

    queued    an order is live and waiting to fill — informational
    filled    it filled, and the position matches — silent
    partial   it filled short — flagged, with the actual gap
    rejected  the broker refused it — flagged
    diverged  the position is wrong and no order explains why — flagged

Only the last three are problems.
"""
from __future__ import annotations

import dataclasses

# Broker order states, grouped by what they mean for us. Named for Alpaca's
# vocabulary (the broker in use); anything unrecognised is treated as
# pending rather than as a failure, because inventing a warning from a
# status we don't understand is the behaviour this module is fixing.
_FILLED_STATUSES = frozenset({"filled"})
_PARTIAL_STATUSES = frozenset({"partially_filled"})
_FAILED_STATUSES = frozenset({"rejected", "canceled", "cancelled", "expired", "suspended", "done_for_day"})

# What a symbol's reconciliation concluded.
QUEUED = "queued"
FILLED = "filled"
PARTIAL = "partial"
REJECTED = "rejected"
DIVERGED = "diverged"

# Outcomes that mean something needs a human's attention.
_PROBLEM_OUTCOMES = frozenset({PARTIAL, REJECTED, DIVERGED})


@dataclasses.dataclass
class ReconciliationResult:
    symbol: str
    intended_shares: float
    actual_shares: float
    diff_shares: float
    diff_pct: float
    outcome: str
    flagged: bool
    order_status: str | None = None

    @property
    def is_problem(self) -> bool:
        return self.flagged


def classify_order_status(status: str | None) -> str:
    """
    Map a broker's order status onto what it means for reconciliation.
    Unknown and missing statuses count as pending — see the module
    docstring on not inventing warnings.
    """
    if status is None:
        return QUEUED
    normalized = str(status).lower().strip()
    if normalized in _FILLED_STATUSES:
        return FILLED
    if normalized in _PARTIAL_STATUSES:
        return PARTIAL
    if normalized in _FAILED_STATUSES:
        return REJECTED
    return QUEUED


def reconcile_positions(
    intended: dict[str, float],
    actual: dict[str, float],
    orders: dict[str, dict] | None = None,
    tolerance_pct: float = 0.02,
) -> list[ReconciliationResult]:
    """
    `intended`: {symbol: target_shares} as decided by sizing logic.
    `actual`:   {symbol: filled_shares} as reported by the broker.
    `orders`:   {symbol: {"status": ...}} as reported by the broker for the
                order submitted this cycle. Omitted or missing entries mean
                "no order was submitted for this symbol", which is only a
                problem if the position is wrong anyway.

    A symbol is flagged only when something actually went wrong. Matching
    within `tolerance_pct` is always fine; beyond it, the order's status
    decides whether this is a queue or a failure.
    """
    orders = orders or {}
    symbols = set(intended) | set(actual)
    results = []
    for symbol in sorted(symbols):
        intended_shares = intended.get(symbol, 0.0)
        actual_shares = actual.get(symbol, 0.0)
        diff = actual_shares - intended_shares
        denom = abs(intended_shares) if intended_shares != 0 else max(abs(actual_shares), 1.0)
        diff_pct = diff / denom

        order = orders.get(symbol)
        raw_status = order.get("status") if isinstance(order, dict) else None

        if abs(diff_pct) <= tolerance_pct:
            # The position is what we asked for. How it got there doesn't
            # matter, and a stale "accepted" status must not manufacture a
            # warning about a position that is already correct.
            outcome = FILLED
        elif order is None:
            # Nothing was submitted for this symbol, yet the position is
            # wrong — that is exactly the case worth shouting about.
            outcome = DIVERGED
        else:
            classified = classify_order_status(raw_status)
            # A "filled" order whose position still doesn't match means it
            # filled at a size we didn't ask for — a real gap, not a queue.
            outcome = PARTIAL if classified == FILLED else classified

        results.append(
            ReconciliationResult(
                symbol=symbol,
                intended_shares=intended_shares,
                actual_shares=actual_shares,
                diff_shares=diff,
                diff_pct=diff_pct,
                outcome=outcome,
                flagged=outcome in _PROBLEM_OUTCOMES,
                order_status=raw_status,
            )
        )
    return results


def _shares(value: float) -> str:
    """Share counts for humans. 32.063831455312794 is not a share count."""
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def summarize(results: list[ReconciliationResult]) -> str:
    """
    One line when everything is fine, detail only for what needs it.
    Queued orders are reported, not warned about — they are the expected
    state for any cycle that runs while the market is shut.
    """
    if not results:
        return "No positions to reconcile."

    flagged = [r for r in results if r.flagged]
    queued = [r for r in results if r.outcome == QUEUED]
    settled = len(results) - len(flagged) - len(queued)

    if not flagged:
        parts = []
        if settled:
            parts.append(f"{settled} position(s) reconciled")
        if queued:
            names = ", ".join(r.symbol for r in queued)
            parts.append(f"{len(queued)} order(s) queued, will fill at the next open ({names})")
        return "; ".join(parts) + "."

    lines = [f"{len(flagged)} of {len(results)} position(s) need attention:"]
    lines.extend(
        f"  {r.symbol}: {r.outcome} — intended {_shares(r.intended_shares)}, "
        f"actual {_shares(r.actual_shares)} (off by {_shares(r.diff_shares)})"
        + (f", order status {r.order_status}" if r.order_status else "")
        for r in flagged
    )
    if queued:
        lines.append(f"  ({len(queued)} other order(s) queued and waiting to fill — not a problem.)")
    return "\n".join(lines)
