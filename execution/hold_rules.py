"""
When a position deserves to be closed — and, more importantly, when it
doesn't. Weekly shortlist membership controls thesis/rotation exits, while
stop-loss and take-profit levels are hard risk/execution boundaries that
apply to every held position, including a symbol the fresh screen re-picked.

A held position is closed when any real exit condition fires:

  1. Its own stop-loss or take-profit is hit. These are hard exits and take
     precedence over shortlist membership — re-picking a name cannot waive
     the terms the position was opened with.
  2. It has missed the shortlist for HOLD_MAX_MISSED_CYCLES consecutive
     weekly cycles.
  3. The model's fresh prediction points against the held side by at least
     the round-trip cost floor.
  4. The hourly contradiction monitor flags it (that path lives in
     execution/contradiction_monitor.py).

A symbol that made this cycle's shortlist resets its consecutive-miss
counter, but only suppresses shortlist/forecast-rotation exits. It does not
suppress a stop or target that has already been reached.
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
from sqlalchemy import text

from backtest.cost_model import round_trip_cost_fraction
from execution.exit_levels import ExitLevels

DEFAULT_MIN_FLIP_RETURN = round_trip_cost_fraction()


@dataclasses.dataclass(frozen=True)
class StopTargetHit:
    """A position's unrealized P&L has breached its stop-loss or reached its take-profit."""

    kind: str  # "stop_loss" | "take_profit"
    message: str


def check_stop_or_target(
    pnl_pct: float | None,
    levels: ExitLevels | None,
    fallback_stop_loss_pct: float,
    fallback_take_profit_pct: float,
) -> StopTargetHit | None:
    """Return the hard exit hit, if any, using the position's recorded levels first."""
    if pnl_pct is None:
        return None
    stop = levels.stop_loss_pct if levels else fallback_stop_loss_pct
    target = levels.take_profit_pct if levels else fallback_take_profit_pct
    if pnl_pct <= -stop:
        return StopTargetHit("stop_loss", f"stop loss: {pnl_pct:+.1%} unrealized, limit -{stop:.1%}")
    if pnl_pct >= target:
        return StopTargetHit("take_profit", f"profit target reached: {pnl_pct:+.1%} unrealized, target +{target:.1%}")
    return None


@dataclasses.dataclass
class HoldDecision:
    symbol: str
    close: bool
    missed_cycles: int
    reasons: list[str]


def evaluate_holds(
    positions: dict[str, float],
    shortlist: set[str],
    predictions: dict[str, float],
    pnl_pct: dict[str, float | None],
    prior_missed: dict[str, int],
    *,
    max_missed_cycles: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    levels_by_symbol: dict[str, ExitLevels] | None = None,
    min_flip_return: float = DEFAULT_MIN_FLIP_RETURN,
) -> list[HoldDecision]:
    """
    Pure exit policy. Stop/target checks are evaluated before shortlist
    handling because they are hard boundaries. A re-picked position resets
    its miss counter but still closes if its own stop or target fired.
    """
    levels_by_symbol = levels_by_symbol or {}
    decisions: list[HoldDecision] = []
    for symbol, qty in positions.items():
        if qty == 0:
            continue

        hit = check_stop_or_target(
            pnl_pct.get(symbol),
            levels_by_symbol.get(symbol),
            stop_loss_pct,
            take_profit_pct,
        )
        if hit:
            decisions.append(HoldDecision(symbol=symbol, close=True, missed_cycles=0, reasons=[hit.message]))
            continue

        if symbol in shortlist:
            decisions.append(HoldDecision(symbol=symbol, close=False, missed_cycles=0, reasons=[]))
            continue

        side = "long" if qty > 0 else "short"
        sign = 1.0 if qty > 0 else -1.0
        missed = prior_missed.get(symbol, 0) + 1
        reasons: list[str] = []

        if missed >= max_missed_cycles:
            reasons.append(
                f"out of the shortlist {missed} consecutive cycle(s) (limit {max_missed_cycles})"
            )

        prediction = predictions.get(symbol)
        if prediction is not None and sign * prediction < 0 and abs(prediction) >= min_flip_return:
            reasons.append(f"model now predicts {prediction:+.2%} against the {side} position")

        decisions.append(HoldDecision(symbol=symbol, close=bool(reasons), missed_cycles=missed, reasons=reasons))
    return decisions


def load_missed_cycles(engine) -> dict[str, int]:
    """symbol -> consecutive shortlist misses, as of the previous cycle."""
    df = pd.read_sql("SELECT symbol, missed_cycles FROM position_hold_state", engine)
    return dict(zip(df["symbol"], df["missed_cycles"].astype(int), strict=False))


def load_exit_levels(engine) -> dict[str, ExitLevels]:
    """symbol -> the exit levels each open position is currently held to."""
    df = pd.read_sql("SELECT symbol, take_profit_pct, stop_loss_pct FROM position_hold_state", engine)
    levels: dict[str, ExitLevels] = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.take_profit_pct) or pd.isna(row.stop_loss_pct):
            continue
        levels[row.symbol] = ExitLevels(
            take_profit_pct=float(row.take_profit_pct), stop_loss_pct=float(row.stop_loss_pct)
        )
    return levels


def store_missed_cycles(
    engine, counts: dict[str, int], levels_by_symbol: dict[str, ExitLevels] | None = None
) -> None:
    """Rewrite hold state to exactly the currently-open positions."""
    levels_by_symbol = levels_by_symbol or {}
    now = dt.datetime.now(tz=dt.UTC)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM position_hold_state"))
        for symbol, missed in counts.items():
            levels = levels_by_symbol.get(symbol)
            conn.execute(
                text(
                    "INSERT INTO position_hold_state "
                    "(symbol, missed_cycles, updated_at, take_profit_pct, stop_loss_pct) "
                    "VALUES (:symbol, :missed, :now, :take_profit, :stop_loss)"
                ),
                {
                    "symbol": symbol,
                    "missed": int(missed),
                    "now": now,
                    "take_profit": levels.take_profit_pct if levels else None,
                    "stop_loss": levels.stop_loss_pct if levels else None,
                },
            )
