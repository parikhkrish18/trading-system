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
model just re-picked it, and the approval flow will resize (or reverse) it.

Every close this module recommends still goes through the human approval
gate — this file only decides what to PROPOSE.

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

# A prediction against the position smaller than the cost of a round trip
# is not a reason to pay for a round trip.
DEFAULT_MIN_FLIP_RETURN = round_trip_cost_fraction()


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
    min_flip_return: float = DEFAULT_MIN_FLIP_RETURN,
) -> list[HoldDecision]:
    """
    Pure function of explicit inputs (no DB, no broker) — the whole exit
    policy in one testable place. `positions` is symbol -> signed shares;
    `predictions` is this cycle's fresh predicted return per symbol (from
    the full scored universe, so held names that didn't make the shortlist
    still have one); `pnl_pct` is unrealized P&L as a fraction (None when
    the broker can't report it — those conditions simply don't fire).
    """
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

        pnl = pnl_pct.get(symbol)
        if pnl is not None:
            if pnl <= -stop_loss_pct:
                reasons.append(f"stop loss: {pnl:+.1%} unrealized, limit -{stop_loss_pct:.0%}")
            elif pnl >= take_profit_pct:
                reasons.append(f"profit target reached: {pnl:+.1%} unrealized, target +{take_profit_pct:.0%}")

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


def store_missed_cycles(engine, counts: dict[str, int]) -> None:
    """
    Rewrite the table to exactly `counts` (the currently-held set). A full
    rewrite, not an upsert, so a position closed by ANY path — weekly exit,
    contradiction monitor, breaker flatten — disappears without every one
    of those paths having to know this table exists.
    """
    now = dt.datetime.now(tz=dt.UTC)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM position_hold_state"))
        for symbol, missed in counts.items():
            conn.execute(
                text(
                    "INSERT INTO position_hold_state (symbol, missed_cycles, updated_at) "
                    "VALUES (:symbol, :missed, :now)"
                ),
                {"symbol": symbol, "missed": int(missed), "now": now},
            )
