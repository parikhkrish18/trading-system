"""
Transaction cost model. Commission is effectively $0 for equities/ETFs at
most brokers now, but the plan says to model it anyway (fee schedules
change, and this keeps the interface ready for a broker where it isn't
free). Slippage is modeled as a function of trade size vs. average volume,
which is the part that actually matters at any real size.
"""
from __future__ import annotations


def commission(trade_value: float, commission_per_share: float = 0.0, shares: float = 0.0) -> float:
    return commission_per_share * shares


def slippage_bps(
    trade_shares: float,
    avg_daily_volume: float,
    base_spread_bps: float = 1.0,
    impact_coefficient: float = 10.0,
) -> float:
    """
    Slippage in basis points as base spread cost plus a market-impact term
    that grows with participation rate (trade size as a fraction of ADV).
    `impact_coefficient` is a tunable knob — calibrate it against real fill
    data once you have paper/live fills to compare against (Phase 6).
    """
    if avg_daily_volume <= 0:
        return base_spread_bps
    participation = abs(trade_shares) / avg_daily_volume
    return base_spread_bps + impact_coefficient * participation


def total_transaction_cost(
    trade_shares: float,
    price: float,
    avg_daily_volume: float,
    commission_per_share: float = 0.0,
    base_spread_bps: float = 1.0,
    impact_coefficient: float = 10.0,
) -> float:
    """Total dollar cost of a trade: commission + (slippage_bps * trade value)."""
    trade_value = abs(trade_shares) * price
    slip_bps = slippage_bps(trade_shares, avg_daily_volume, base_spread_bps, impact_coefficient)
    slip_cost = trade_value * (slip_bps / 10_000)
    comm_cost = commission(trade_value, commission_per_share, abs(trade_shares))
    return slip_cost + comm_cost
