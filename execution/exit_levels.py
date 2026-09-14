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

So the levels are derived per pick, from the things that describe it:

    target  what the model expects this stock to do, bounded to what its
            own ATR and its own recent trading range actually support
    stop    how far this stock normally wanders anyway, tightened toward
            a real support/resistance level when one sits in the way

Both are expressed against the forecast horizon rather than per day, since
that is the period the position is meant to be held for — a weekly swing
book (see settings.target_horizon_days), not a buy-and-hold one.

    horizon_sigma = daily volatility x sqrt(horizon days)

    take profit = the predicted move, bounded to a MULTIPLE OF THIS STOCK'S
                  OWN ATR (2x-4x by default — see
                  exit_take_profit_min_atr_mult/exit_take_profit_max_atr_mult),
                  rather than a flat percentage, so a target on a stock that
                  genuinely moves is a real, tradeable one instead of a
                  floor sized for a much calmer name. Falls back to the
                  older horizon_sigma-based bounds when ATR isn't available
                  for this stock (see atr_pct below) rather than guessing.

    stop loss   = a multiple of horizon_sigma, bounded — wide enough that
                  ordinary movement doesn't reach it, tight enough to still
                  be a stop.

Both are then SOLIDIFIED against this stock's own Donchian channel (its
rolling settings.donchian_exit_window high/low — support and resistance in
the plainest sense): whichever of the ATR/sigma bound or the distance to
the nearer real level is CLOSER wins, so a target never sits past a
resistance this stock has actually failed to clear recently, and a stop
never sits past a support it has actually held — but neither is ever
pushed FURTHER out than the ATR/sigma bound alone would set; a level with
no room left (price already through it) simply doesn't tighten anything.
See _channel_cap and resistance_distance_pct/support_distance_pct below.

The bounds matter as much as the formula. Volatility is estimated from a
short window and can be badly wrong for a stock that has just gapped;
without bounds, one quiet month would produce a 1% stop that closes on the
first ordinary day.

When volatility, ATR, or a channel level is unknown — a new listing, a gap
in prices — this falls back to the global settings rather than guessing.
Guessing a stop is worse than using a blunt one.
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


def _take_profit_bounds(daily_volatility: float, atr_pct: float | None, horizon: int) -> tuple[float, float]:
    """
    (floor, ceiling) for take-profit, in units of this stock's own movement,
    before any Donchian channel solidification (see _channel_cap).

    ATR-relative when this stock's ATR is available (atr_pct — Average True
    Range as a fraction of price, e.g. vol_atr_14 / close): 2x-4x its
    horizon-scaled ATR by default, matching the weekly/biweekly swing
    target this system is sized for. Falls back to the older
    horizon-sigma-based bounds (a multiple of close-to-close volatility)
    when ATR isn't available for this stock — a new listing, or a gap in
    the high/low history ATR needs — rather than guessing an ATR.
    """
    if atr_pct is not None and math.isfinite(atr_pct) and atr_pct > 0:
        atr_horizon = atr_pct * math.sqrt(max(horizon, 1))
        floor = settings.exit_take_profit_min_atr_mult * atr_horizon
        ceiling = max(settings.exit_take_profit_max_atr_mult * atr_horizon, floor)
        return floor, ceiling

    horizon_sigma = daily_volatility * math.sqrt(max(horizon, 1))
    ceiling = max(settings.exit_take_profit_max_sigmas * horizon_sigma, settings.exit_min_take_profit_pct)
    return settings.exit_min_take_profit_pct, ceiling


def _channel_cap(ceiling: float, distance_pct: float | None, floor: float) -> float:
    """
    Tightens `ceiling` toward a real Donchian support/resistance distance
    when one is available, positive, and closer than the ceiling already
    is — never widens it, and never pulls it below `floor` (a level with no
    room left, or sitting right at the floor, simply doesn't tighten
    anything further).

    `distance_pct`: this stock's distance from its current price to the
    relevant channel boundary, as a positive fraction of price — None (or
    non-finite, non-positive — price already through the level, or the
    channel couldn't be measured) leaves `ceiling` untouched.
    """
    if distance_pct is None or not math.isfinite(distance_pct) or distance_pct <= 0:
        return ceiling
    return max(min(ceiling, distance_pct), floor)


def exit_levels_for(
    predicted_return: float | None,
    daily_volatility: float | None,
    horizon_days: int | None = None,
    atr_pct: float | None = None,
    resistance_distance_pct: float | None = None,
    support_distance_pct: float | None = None,
) -> ExitLevels:
    """
    `predicted_return`: the model's forecast for this pick, signed. Only its
        size matters here — a -5% forecast on a short is a 5% target. Also
        decides which channel distance applies to which leg: a long's
        take-profit is bounded by resistance overhead and its stop-loss by
        support underneath; a short is the mirror image.
    `daily_volatility`: standard deviation of this stock's daily returns,
        NOT annualized. None when it couldn't be measured. Still the sole
        basis for stop-loss sizing, and the take-profit fallback when ATR
        isn't available.
    `horizon_days`: trading days the position is meant to be held.
    `atr_pct`: this stock's 14-day Average True Range as a fraction of
        price (vol_atr_14 / close). When available, take-profit is sized as
        a multiple of it instead of daily_volatility — see
        _take_profit_bounds. None falls back to the older behavior.
    `resistance_distance_pct`/`support_distance_pct`: this stock's distance
        to its rolling settings.donchian_exit_window high/low, as a
        positive fraction of price (None/non-positive if unmeasurable or
        price is already through that level) — see _channel_cap. Tightens
        whichever leg (take-profit or stop-loss) that level applies to,
        per predicted_return's sign; never widens either leg.
    """
    if daily_volatility is None or not math.isfinite(daily_volatility) or daily_volatility <= 0:
        return global_levels()
    if predicted_return is None or not math.isfinite(predicted_return):
        return global_levels()

    is_long = predicted_return >= 0
    tp_channel = resistance_distance_pct if is_long else support_distance_pct
    sl_channel = support_distance_pct if is_long else resistance_distance_pct

    horizon = horizon_days if horizon_days is not None else settings.target_horizon_days
    horizon_sigma = daily_volatility * math.sqrt(max(horizon, 1))

    tp_floor, tp_ceiling = _take_profit_bounds(daily_volatility, atr_pct, horizon)
    tp_ceiling = _channel_cap(tp_ceiling, tp_channel, tp_floor)
    take_profit = min(max(abs(predicted_return), tp_floor), tp_ceiling)
    # The floor is applied last as well: capping at a small ATR/sigma/channel
    # distance must not produce a target below what the round trip costs,
    # or the position would be closed into a guaranteed loss.
    take_profit = max(take_profit, tp_floor, settings.exit_min_take_profit_pct)

    sl_ceiling = min(
        max(settings.exit_stop_loss_sigmas * horizon_sigma, settings.exit_min_stop_loss_pct),
        settings.exit_max_stop_loss_pct,
    )
    stop_loss = _channel_cap(sl_ceiling, sl_channel, settings.exit_min_stop_loss_pct)
    return ExitLevels(take_profit_pct=take_profit, stop_loss_pct=stop_loss, derived=True)


def exit_levels_advised(
    predicted_return: float | None,
    daily_volatility: float | None,
    llm_take_profit_pct: float | None,
    llm_stop_loss_pct: float | None,
    horizon_days: int | None = None,
    atr_pct: float | None = None,
    resistance_distance_pct: float | None = None,
    support_distance_pct: float | None = None,
) -> ExitLevels:
    """
    Same bounds as exit_levels_for, but the take-profit/stop-loss VALUE
    within those bounds comes from an LLM's per-stock suggestion when one
    is given and finite. Advisory, not authoritative: the bounds
    themselves (ATR-, volatility-, and Donchian-channel-derived, or the
    global fallback) are computed exactly as exit_levels_for always has, so
    a stray, missing, or overly aggressive LLM number can only ever be
    clamped into the existing safe range, never escape it.
    """
    baseline = exit_levels_for(
        predicted_return, daily_volatility, horizon_days, atr_pct=atr_pct,
        resistance_distance_pct=resistance_distance_pct, support_distance_pct=support_distance_pct,
    )
    if not baseline.derived:
        return baseline  # global fallback -- no per-stock bounds to advise within

    is_long = predicted_return >= 0
    tp_channel = resistance_distance_pct if is_long else support_distance_pct
    sl_channel = support_distance_pct if is_long else resistance_distance_pct

    horizon = horizon_days if horizon_days is not None else settings.target_horizon_days
    tp_lo, tp_hi = _take_profit_bounds(daily_volatility, atr_pct, horizon)
    tp_hi = _channel_cap(tp_hi, tp_channel, tp_lo)
    sl_lo, sl_hi = settings.exit_min_stop_loss_pct, settings.exit_max_stop_loss_pct
    sl_hi = _channel_cap(sl_hi, sl_channel, sl_lo)

    take_profit = baseline.take_profit_pct
    if llm_take_profit_pct is not None and math.isfinite(llm_take_profit_pct):
        take_profit = min(max(llm_take_profit_pct, tp_lo), tp_hi)

    stop_loss = baseline.stop_loss_pct
    if llm_stop_loss_pct is not None and math.isfinite(llm_stop_loss_pct):
        stop_loss = min(max(llm_stop_loss_pct, sl_lo), sl_hi)

    return ExitLevels(take_profit_pct=take_profit, stop_loss_pct=stop_loss, derived=True)


def describe(levels: ExitLevels) -> str:
    """One line for the proposal message, so the human approves known levels."""
    basis = "sized to this stock" if levels.derived else "default levels — volatility unavailable"
    return f"take profit +{levels.take_profit_pct:.1%} / stop -{levels.stop_loss_pct:.1%} ({basis})"
