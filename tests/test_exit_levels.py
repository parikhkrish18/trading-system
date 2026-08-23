"""
Per-pick exit levels.

The point of these is that one pair of numbers cannot be right for a
utility and a biotech at once, so the tests are mostly about the levels
actually differing with the stock — and about the bounds, which are what
stop a short volatility window producing something absurd.
"""
import math

import pytest

from execution.exit_levels import ExitLevels, describe, exit_levels_for, global_levels


def _levels(predicted=0.06, daily_vol=0.02, horizon=20):
    return exit_levels_for(predicted_return=predicted, daily_volatility=daily_vol, horizon_days=horizon)


# --------------------------------------------------------------------------
# The levels follow the stock
# --------------------------------------------------------------------------


def test_a_volatile_stock_gets_a_wider_stop_than_a_calm_one():
    """
    The whole reason for doing this. A stock that routinely swings 3% a day
    must not be held to the same stop as one that barely moves 0.5%, or the
    volatile one is closed on ordinary movement every time.
    """
    calm = _levels(daily_vol=0.005)
    volatile = _levels(daily_vol=0.03)

    assert volatile.stop_loss_pct > calm.stop_loss_pct


def test_the_target_follows_what_the_model_predicted():
    """
    Taking profit at a fixed +10% means holding a 3% forecast long past its
    thesis and leaving most of a 15% one on the table.
    """
    small = _levels(predicted=0.04, daily_vol=0.03)
    large = _levels(predicted=0.09, daily_vol=0.03)

    assert large.take_profit_pct > small.take_profit_pct


def test_direction_does_not_change_the_levels():
    """
    Both are distances from entry, not prices. A short's stop is above the
    entry and a long's below, and the position already carries the side.
    """
    assert _levels(predicted=0.06) == _levels(predicted=-0.06)


def test_levels_scale_with_the_holding_horizon():
    """A month of wandering is more than a week of it."""
    short_hold = _levels(horizon=5)
    long_hold = _levels(horizon=40)

    assert long_hold.stop_loss_pct > short_hold.stop_loss_pct


# --------------------------------------------------------------------------
# The bounds, which are what keep a bad volatility estimate survivable
# --------------------------------------------------------------------------


def test_a_very_quiet_stock_still_gets_a_usable_stop(monkeypatch):
    """
    Volatility comes from a short window and can be far too low right after
    a quiet stretch. Unbounded, that produces a 1% stop that closes on the
    first ordinary day.
    """
    levels = _levels(daily_vol=0.0001)

    assert levels.stop_loss_pct >= 0.05


def test_a_wildly_volatile_stock_does_not_get_an_unbounded_stop():
    levels = _levels(daily_vol=0.5)

    assert levels.stop_loss_pct <= 0.20


def test_the_target_never_falls_below_what_a_round_trip_costs():
    """
    Closing at a profit smaller than the cost of the trade books a loss and
    calls it a win.
    """
    levels = _levels(predicted=0.0001, daily_vol=0.0005)

    assert levels.take_profit_pct >= 0.03


def test_an_extreme_forecast_does_not_set_an_unreachable_target():
    """
    A 60% forecast on a stock that moves 1% a day is the model being wrong,
    not an opportunity. The target is capped in units of what the stock
    actually does.
    """
    levels = _levels(predicted=0.60, daily_vol=0.01)

    assert levels.take_profit_pct < 0.60


# --------------------------------------------------------------------------
# Falling back rather than guessing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_vol", [None, 0.0, -0.01, float("nan"), math.inf])
def test_unmeasurable_volatility_falls_back_to_the_globals(bad_vol):
    """
    A new listing or a gap in prices. Guessing a stop from a number we
    don't have is worse than using a blunt one everybody knows about.
    """
    levels = exit_levels_for(predicted_return=0.05, daily_volatility=bad_vol)

    assert levels == global_levels()
    assert levels.derived is False


def test_a_missing_forecast_falls_back_too():
    assert exit_levels_for(predicted_return=None, daily_volatility=0.02).derived is False


def test_derived_levels_are_marked_as_derived():
    assert _levels().derived is True


# --------------------------------------------------------------------------
# What the human reads before approving
# --------------------------------------------------------------------------


def test_description_states_both_levels_and_their_direction():
    text = describe(ExitLevels(take_profit_pct=0.08, stop_loss_pct=0.12))

    assert "+8.0%" in text
    assert "-12.0%" in text


def test_description_admits_when_the_levels_are_only_the_defaults():
    """
    A reader should be able to tell "sized to this stock" from "we couldn't
    measure it" without going and looking.
    """
    assert "unavailable" in describe(global_levels())
    assert "this stock" in describe(ExitLevels(0.08, 0.12, derived=True))
