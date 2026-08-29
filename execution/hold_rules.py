"""
When a position deserves to be closed — and, more importantly, when it
doesn't. The old weekly cycle closed anything not in this week's shortlist,
which is the right rule for 5-day forecasts and exactly the wrong one for
multi-week swing holds (TARGET_HORIZON_DAYS): a good 4-week position got
sold because something else scored marginally higher on a Monday.

A held position is now closed only when a REAL exit condition fires:

  1. It has missed the shortlist for HOLD_MAX_MISSED_CYCLES consecutive
     weekly cycles (one bad Monday is noise; several in a row means the
     model has genuinely moved on).
  2. The model's fresh prediction points AGAINST the held side, by at
     least the round-trip cost floor (a flip smaller than the cost of
     trading on it is noise, the same bar the screener applies).
  3. Unrealized P&L breaches the stop loss (HOLD_STOP_LOSS_PCT) or the
     profit target (HOLD_TAKE_PROFIT_PCT).
  4. The hourly contradiction monitor flags it — that path lives in
     execution/contradiction_monitor.py and is deliberately NOT duplicated
     here; the two interact only through the broker's position list (a
     position the monitor already closed simply isn't held anymore by the
     time the weekly cycle looks).

A symbol that made this cycle's shortlist is never an exit candidate: the
model just re-picked it, and the approval gate will resize (or reverse) it.

Every close this module recommends still goes through execution/approval_gate.py
— this file only decides what to PROPOSE. By default that gate executes
proposals immediately (APPROVAL_MODE=auto); it can be switched back to a
pre-trade human gate (APPROVAL_MODE=telegram) if that's ever wanted again.

State (the consecutive-miss counter) lives in the position_hold_state
table, rewritten each cycle to exactly the currently-held set, so closes
from any path fall out automatically.
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
from sqlalchemy import text

from backtest.cost_model import round_trip_cost_fraction
from execution.exit_levels import ExitLevels

# A prediction against the position smaller than the cost of a round trip
# is not a reason to pay for a round trip.
DEFAULT_MIN_FLIP_RETURN = round_trip_cost_fraction()


@dataclasses.dataclass(frozen=True)
class StopTargetHit:
    """A position's unrealized P&L has breached its stop-loss or reached its take-profit."""

    kind: str  # "stop_loss" | "take_profit"
    message: str  # human-readable, e.g. "stop loss: -8.2% unrealized, limit -8.0%"


def check_stop_or_target(
    pnl_pct: float | None,
    levels: ExitLevels | None,
    fallback_stop_loss_pct: float,
    fallback_take_profit_pct: float,
) -> StopTargetHit | None:
    """
    Whether unrealized P&L has breached this position's own stop-loss or
    reached its own take-profit — the levels it was actually approved with
    (`levels`), falling back to the global HOLD_*_PCT settings only for a
    position with none recorded.

    Shared by two callers on two different clocks: evaluate_holds below
    (the weekly cycle, for positions that missed this week's shortlist) and
    execution/contradiction_monitor.py's hourly check (every held position,
    every hour, regardless of shortlist status). The point of running it on
    both clocks is that a swing trade should close when IT resolves — a
    volatile stock's larger target reached in 3-4 days, a calm stock's
    smaller target taking a week or more — rather than sitting past its own
    target or stop until the next weekly checkpoint just because the
    calendar hadn't come around yet.
    """
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
    """One held position's verdict for this cycle."""

    symbol: str
    close: bool
    missed_cycles: int  # consecutive cycles out of the shortlist, AFTER this cycle
    reasons: list[str]  # human-readable exit conditions that fired; empty = hold


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
    Pure function of explicit inputs (no DB, no broker) — the whole exit
    policy in one testable place. `positions` is symbol -> signed shares;
    `predictions` is this cycle's fresh predicted return per symbol (from
    the full scored universe, so held names that didn't make the shortlist
    still have one); `pnl_pct` is unrealized P&L as a fraction (None when
    the broker can't report it — those conditions simply don't fire).

    `levels_by_symbol` is the take-profit/stop-loss pair each position was
    actually approved with. A position is judged against its own levels, not
    against whatever the global settings happen to say now — otherwise
    editing HOLD_STOP_LOSS_PCT on a Tuesday silently re-writes the terms of
    every open position, including ones a human approved on different ones.
    `stop_loss_pct` / `take_profit_pct` remain the fallback for positions
    with no recorded levels (opened before they existed, or opened when
    volatility couldn't be measured).
    """
    levels_by_symbol = levels_by_symbol or {}
    decisions: list[HoldDecision] = []
    for symbol, qty in positions.items():
        if qty == 0:
            continue
        if symbol in shortlist:
            # Re-picked: the fresh screen is the strongest possible "keep".
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

        hit = check_stop_or_target(pnl_pct.get(symbol), levels_by_symbol.get(symbol), stop_loss_pct, take_profit_pct)
        if hit:
            reasons.append(hit.message)

        decisions.append(
            HoldDecision(symbol=symbol, close=bool(reasons), missed_cycles=missed, reasons=reasons)
        )
    return decisions


# --------------------------------------------------------------------------
# State — the consecutive-miss counter
# --------------------------------------------------------------------------


def load_missed_cycles(engine) -> dict[str, int]:
    """symbol -> consecutive shortlist misses, as of the previous cycle."""
    df = pd.read_sql("SELECT symbol, missed_cycles FROM position_hold_state", engine)
    return dict(zip(df["symbol"], df["missed_cycles"].astype(int), strict=False))


def load_exit_levels(engine) -> dict[str, ExitLevels]:
    """
    symbol -> the levels each open position is being enforced against.

    Positions with no recorded levels are simply absent, and the caller
    falls back to the globals for them — a position opened before this
    existed must still be evaluated, not skipped.
    """
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
    """
    Rewrite the table to exactly `counts` (the currently-held set). A full
    rewrite, not an upsert, so a position closed by ANY path — weekly exit,
    contradiction monitor, breaker flatten — disappears without every one
    of those paths having to know this table exists.

    The levels travel with the counter so a position keeps being judged
    against the terms it was approved on, cycle after cycle.
    """
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
