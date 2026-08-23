"""
Per-pick take-profit and stop-loss levels.

Until now every position shared one pair of numbers: close at -8%, close at
+10%, for a utility and a biotech alike. Those are the wrong shape for two
reasons. A calm stock rarely moves 8% at all, so the stop sits somewhere it
will only be reached by something genuinely broken — fine — while a
volatile one wanders through 8% most months, so the same stop closes it on
noise. And a fixed +10% target is unrelated to what the model actually
predicted for that stock: taking profit at 10% on a pick forecast to move
3% means holding long past the thesis, and on one forecast to move 15%
means leaving most of it behind.

So the levels are derived per pick, from the two things that describe it:

    target  what the model expects this stock to do
    stop    how far this stock normally wanders anyway

Both are expressed against the forecast horizon rather than per day, since
that is the period the position is meant to be held for.

    horizon_sigma = daily volatility x sqrt(horizon days)

    take profit = the predicted move, floored so it is worth the round trip
                  and capped so an extreme forecast doesn't set a target the
                  stock has no history of reaching

    stop loss   = a multiple of horizon_sigma, bounded — wide enough that
                  ordinary movement doesn't reach it, tight enough to still
                  be a stop

The bounds matter as much as the formula. Volatility is estimated from a
short window and can be badly wrong for a stock that has just gapped;
without bounds, one quiet month would produce a 1% stop that closes on the
first ordinary day.

When volatility is unknown — a new listing, a gap in prices — this falls
back to the global settings rather than guessing. Guessing a stop is worse
than using a blunt one.
"""
from __future__ import annotations

import dataclasses
import math

from config.settings import settings


@dataclasses.dataclass(frozen=True)
class ExitLevels:
    """
    The levels a single pick was proposed, and approved, with.

    Both are positive fractions of the entry price, regardless of side: a
    long's stop is below entry and a short's is above, and the direction is
    already carried by the position. 0.08 means "8% against me".
    """

    take_profit_pct: float
    stop_loss_pct: float
    # False when volatility was unavailable and the globals were used, so
    # the proposal message can be honest about which it is.
    derived: bool = True


def global_levels() -> ExitLevels:
    """The blunt instrument: one pair of numbers for everything."""
    return ExitLevels(
        take_profit_pct=settings.hold_take_profit_pct,
        stop_loss_pct=settings.hold_stop_loss_pct,
        derived=False,
    )


def exit_levels_for(
    predicted_return: float | None,
    daily_volatility: float | None,
    horizon_days: int | None = None,
) -> ExitLevels:
    """
    `predicted_return`: the model's forecast for this pick, signed. Only its
        size matters here — a -5% forecast on a short is a 5% target.
    `daily_volatility`: standard deviation of this stock's daily returns,
        NOT annualized. None when it couldn't be measured.
    `horizon_days`: trading days the position is meant to be held.
    """
    if daily_volatility is None or not math.isfinite(daily_volatility) or daily_volatility <= 0:
        return global_levels()
    if predicted_return is None or not math.isfinite(predicted_return):
        return global_levels()

    horizon = horizon_days if horizon_days is not None else settings.target_horizon_days
    horizon_sigma = daily_volatility * math.sqrt(max(horizon, 1))

    take_profit = min(
        max(abs(predicted_return), settings.exit_min_take_profit_pct),
        settings.exit_take_profit_max_sigmas * horizon_sigma,
    )
    # The floor is applied last as well: capping at a small sigma must not
    # produce a target below what the round trip costs, or the position
    # would be closed into a guaranteed loss.
    take_profit = max(take_profit, settings.exit_min_take_profit_pct)

    stop_loss = min(
        max(settings.exit_stop_loss_sigmas * horizon_sigma, settings.exit_min_stop_loss_pct),
        settings.exit_max_stop_loss_pct,
    )
    return ExitLevels(take_profit_pct=take_profit, stop_loss_pct=stop_loss, derived=True)


def describe(levels: ExitLevels) -> str:
    """One line for the proposal message, so the human approves known levels."""
    basis = "sized to this stock" if levels.derived else "default levels — volatility unavailable"
    return f"take profit +{levels.take_profit_pct:.1%} / stop -{levels.stop_loss_pct:.1%} ({basis})"
