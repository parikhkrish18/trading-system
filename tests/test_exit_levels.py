"""
Per-pick exit levels.

The point of these is that one pair of numbers cannot be right for a
utility and a biotech at once, so the tests are mostly about the levels
actually differing with the stock — and about the bounds, which are what
stop a short volatility window producing something absurd.
"""
import math

import pytest

from execution.exit_levels import ExitLevels, describe, exit_levels_advised, exit_levels_for, global_levels


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

    daily_vol left at _levels' own default (0.02, not a highly volatile
    stock): at 0.03 the stop-loss hits its 20% cap for both cases, and
    exit_min_reward_risk_ratio's floor (0.6x that capped stop) then swamps
    both forecasts up to the same number -- a real, intended effect of
    that floor, just not what THIS test is checking, so it needs inputs
    that don't collide with it.
    """
    small = _levels(predicted=0.04)
    large = _levels(predicted=0.09)

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


# --------------------------------------------------------------------------
# exit_levels_advised: an LLM's suggestion, clamped to the same bounds
# --------------------------------------------------------------------------


def _advised(llm_tp=None, llm_sl=None, predicted=0.06, daily_vol=0.02, horizon=20):
    return exit_levels_advised(
        predicted_return=predicted, daily_volatility=daily_vol,
        llm_take_profit_pct=llm_tp, llm_stop_loss_pct=llm_sl, horizon_days=horizon,
    )


def test_a_suggestion_within_bounds_is_used_as_given():
    baseline = _levels(daily_vol=0.02)
    # Pick a value strictly between the quant floor and ceiling for this vol.
    suggestion = (baseline.take_profit_pct + 0.001, baseline.stop_loss_pct + 0.001)
    levels = _advised(llm_tp=suggestion[0], llm_sl=suggestion[1], daily_vol=0.02)
    assert levels.take_profit_pct == pytest.approx(suggestion[0])
    assert levels.stop_loss_pct == pytest.approx(suggestion[1])


def test_a_suggestion_above_the_ceiling_is_clamped_not_used_outright():
    levels = _advised(llm_tp=5.0, llm_sl=5.0, daily_vol=0.02)
    assert levels.take_profit_pct <= 0.20  # nowhere near the absurd 5.0 (500%) suggested
    assert levels.stop_loss_pct <= 0.20


def test_a_suggestion_below_the_floor_is_clamped_up():
    levels = _advised(llm_tp=0.0001, llm_sl=0.0001, daily_vol=0.02)
    assert levels.take_profit_pct >= 0.03
    assert levels.stop_loss_pct >= 0.05


def test_no_suggestion_falls_back_to_the_plain_quant_formula():
    assert _advised(llm_tp=None, llm_sl=None) == _levels()


def test_a_non_finite_suggestion_is_ignored_like_no_suggestion():
    levels = _advised(llm_tp=float("nan"), llm_sl=float("inf"))
    assert levels == _levels()


def test_unmeasurable_volatility_ignores_the_llm_suggestion_too():
    """No per-stock bounds to advise within -- same global fallback as exit_levels_for."""
    levels = _advised(llm_tp=0.07, llm_sl=0.04, daily_vol=None)
    assert levels == global_levels()
    assert levels.derived is False


# --------------------------------------------------------------------------
# atr_pct: take-profit sized as a multiple of this stock's own ATR
# --------------------------------------------------------------------------


def test_take_profit_is_bounded_by_atr_multiples_when_atr_is_available():
    """
    A stock with a real ATR gets its take-profit sized between
    exit_take_profit_min_atr_mult and exit_take_profit_max_atr_mult times
    its own horizon-scaled ATR, not the flat/sigma-based bounds.
    """
    # atr_pct=0.02, horizon=5 -> atr_horizon = 0.02 * sqrt(5) ~= 0.0447
    # bounds: [2x, 4x] ~= [0.0894, 0.1789]
    levels = exit_levels_for(predicted_return=0.15, daily_volatility=0.01, horizon_days=5, atr_pct=0.02)
    assert 0.089 < levels.take_profit_pct < 0.179


def test_take_profit_floors_at_the_minimum_atr_multiple_even_for_a_tiny_forecast():
    levels = exit_levels_for(predicted_return=0.0001, daily_volatility=0.01, horizon_days=5, atr_pct=0.02)
    assert levels.take_profit_pct == pytest.approx(2.0 * 0.02 * math.sqrt(5))


def test_take_profit_caps_at_the_maximum_atr_multiple_for_an_extreme_forecast():
    levels = exit_levels_for(predicted_return=0.90, daily_volatility=0.01, horizon_days=5, atr_pct=0.02)
    assert levels.take_profit_pct == pytest.approx(4.0 * 0.02 * math.sqrt(5))


# --------------------------------------------------------------------------
# exit_min_reward_risk_ratio: take-profit is never left far below stop-loss
# --------------------------------------------------------------------------


def test_a_very_calm_stock_still_gets_a_take_profit_worth_the_stop_loss_risked():
    """
    Hit live: TECH's ATR was so small that 2x-ATR (the take-profit floor)
    came out to a 2% target, while the stop-loss -- sized off a completely
    different, unrelated formula (volatility-sigma, floored at
    exit_min_stop_loss_pct) -- landed at 5%. A trade risking more than
    double what it targeted. The take-profit must never sit below
    settings.exit_min_reward_risk_ratio (0.6x by default) of the stop-loss
    it's paired with.
    """
    # atr_pct tiny enough that 2x-ATR-horizon is well under 1%; daily_vol
    # large enough (relative to atr_pct) that stop-loss lands at its own
    # 5% floor -- the exact TECH shape.
    levels = exit_levels_for(predicted_return=0.001, daily_volatility=0.005, horizon_days=5, atr_pct=0.002)
    assert levels.stop_loss_pct == pytest.approx(0.05)
    assert levels.take_profit_pct == pytest.approx(0.6 * 0.05)
    assert levels.take_profit_pct >= 0.6 * levels.stop_loss_pct


def test_reward_risk_floor_only_ever_raises_take_profit_never_lowers_stop_loss():
    """The fix for a lopsided ratio is a bigger target, not a tighter stop closer to ordinary noise."""
    floored = exit_levels_for(predicted_return=0.0001, daily_volatility=0.005, horizon_days=5, atr_pct=0.002)
    # take-profit was lifted by the ratio floor, above what 2x-ATR alone
    # would have set -- stop-loss is untouched, still exactly its own
    # 5% floor, not tightened to bring the ratio into line some other way.
    assert floored.take_profit_pct > 2 * 0.002 * math.sqrt(5)
    assert floored.stop_loss_pct == pytest.approx(0.05)


def test_reward_risk_floor_does_not_touch_an_already_healthy_ratio():
    """A take-profit already comfortably above the ratio floor is left exactly as sized."""
    levels = exit_levels_for(predicted_return=0.15, daily_volatility=0.01, horizon_days=5, atr_pct=0.02)
    assert levels.take_profit_pct > 0.6 * levels.stop_loss_pct
    assert levels.take_profit_pct == pytest.approx(0.15)  # the raw forecast, unchanged -- well within the ATR band


def test_advised_reapplies_the_reward_risk_floor_after_an_llm_suggestion():
    """
    An LLM's own take-profit suggestion is clamped into [tp_lo, tp_hi] --
    tp_lo is the SAME unfloored ATR-derived value the ratio floor exists
    to guard against, so a low LLM suggestion on a TECH-shaped stock must
    still come out no lower than the ratio floor, exactly like the
    unadvised path.
    """
    levels = exit_levels_advised(
        predicted_return=0.001, daily_volatility=0.005, horizon_days=5, atr_pct=0.002,
        llm_take_profit_pct=0.0001, llm_stop_loss_pct=None,
    )
    assert levels.stop_loss_pct == pytest.approx(0.05)
    assert levels.take_profit_pct == pytest.approx(0.6 * 0.05)


def test_a_more_volatile_stocks_atr_produces_a_bigger_take_profit_band():
    calm = exit_levels_for(predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.01)
    volatile = exit_levels_for(predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.05)
    assert volatile.take_profit_pct > calm.take_profit_pct


def test_atr_pct_does_not_change_stop_loss_sizing():
    """Stop-loss stays on its existing volatility-sigma basis regardless of atr_pct -- only take-profit moves."""
    without_atr = exit_levels_for(predicted_return=0.06, daily_volatility=0.02, horizon_days=5, atr_pct=None)
    with_atr = exit_levels_for(predicted_return=0.06, daily_volatility=0.02, horizon_days=5, atr_pct=0.09)
    assert without_atr.stop_loss_pct == pytest.approx(with_atr.stop_loss_pct)


@pytest.mark.parametrize("bad_atr", [None, 0.0, -0.01, float("nan"), float("inf")])
def test_an_unusable_atr_pct_falls_back_to_the_sigma_based_bounds(bad_atr):
    """A missing/invalid atr_pct (new listing, gap in high/low history) must not raise or silently zero the target."""
    with_bad_atr = exit_levels_for(predicted_return=0.06, daily_volatility=0.02, horizon_days=5, atr_pct=bad_atr)
    without_atr = exit_levels_for(predicted_return=0.06, daily_volatility=0.02, horizon_days=5, atr_pct=None)
    assert with_bad_atr == without_atr


def test_advised_clamps_the_llm_suggestion_into_the_atr_band_not_the_sigma_one():
    """
    Regression test: exit_levels_advised must advise within the SAME bounds
    exit_levels_for actually used (ATR-relative here), not the older
    sigma-based ones -- otherwise a suggestion inside the sigma band but
    outside the (tighter or wider) ATR band would pass through unclamped.
    """
    # ATR band (2x-4x of 0.02*sqrt(5)~=0.0447): [0.0894, 0.1789]
    levels = exit_levels_advised(
        predicted_return=0.15, daily_volatility=0.01, llm_take_profit_pct=0.50,
        llm_stop_loss_pct=None, horizon_days=5, atr_pct=0.02,
    )
    assert levels.take_profit_pct == pytest.approx(4.0 * 0.02 * math.sqrt(5))


def test_advised_uses_the_llm_suggestion_within_the_atr_band_as_given():
    levels = exit_levels_advised(
        predicted_return=0.15, daily_volatility=0.01, llm_take_profit_pct=0.10,
        llm_stop_loss_pct=None, horizon_days=5, atr_pct=0.02,
    )
    assert levels.take_profit_pct == pytest.approx(0.10)


# --------------------------------------------------------------------------
# Donchian channel solidification: exits tighten toward a real level,
# never widen past the ATR/sigma bound
# --------------------------------------------------------------------------


def test_take_profit_tightens_toward_a_closer_resistance_on_a_long():
    """
    ATR band is [2x, 4x]*0.02*sqrt(5) ~= [0.089, 0.179] -- a resistance
    distance strictly inside that band should replace the ceiling.
    """
    levels = exit_levels_for(
        predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.02,
        resistance_distance_pct=0.12,
    )
    assert levels.take_profit_pct == pytest.approx(0.12)


def test_take_profit_never_widens_past_the_atr_ceiling_even_with_a_far_resistance():
    atr_ceiling = 4.0 * 0.02 * math.sqrt(5)
    levels = exit_levels_for(
        predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.02,
        resistance_distance_pct=10.0,  # miles away -- must not stretch the target out to it
    )
    assert levels.take_profit_pct == pytest.approx(atr_ceiling)


def test_take_profit_never_crushes_below_the_atr_floor_even_with_a_very_close_resistance():
    atr_floor = 2.0 * 0.02 * math.sqrt(5)
    levels = exit_levels_for(
        predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.02,
        resistance_distance_pct=0.001,  # resistance right on top of entry
    )
    assert levels.take_profit_pct == pytest.approx(atr_floor)


def test_a_resistance_the_price_has_already_cleared_does_not_tighten_anything():
    """distance_pct <= 0 means price is already through the level -- no ceiling left to speak of."""
    atr_ceiling = 4.0 * 0.02 * math.sqrt(5)
    levels = exit_levels_for(
        predicted_return=0.5, daily_volatility=0.01, horizon_days=5, atr_pct=0.02,
        resistance_distance_pct=0.0,
    )
    assert levels.take_profit_pct == pytest.approx(atr_ceiling)


def test_stop_loss_tightens_toward_a_closer_support_on_a_long():
    # sigma stop = min(max(1.5*0.02*sqrt(5), 0.05), 0.20) ~= 0.067 -- pick a
    # support distance strictly between the 0.05 floor and that ceiling.
    sigma_stop = min(max(1.5 * 0.02 * math.sqrt(5), 0.05), 0.20)
    levels = exit_levels_for(predicted_return=0.06, daily_volatility=0.02, horizon_days=5, support_distance_pct=0.06)
    assert levels.stop_loss_pct == pytest.approx(0.06)
    assert levels.stop_loss_pct < sigma_stop


def test_stop_loss_never_tightens_below_its_own_minimum():
    levels = exit_levels_for(
        predicted_return=0.06, daily_volatility=0.02, horizon_days=5, support_distance_pct=0.001,
    )
    assert levels.stop_loss_pct == pytest.approx(0.05)  # exit_min_stop_loss_pct default


def test_short_uses_support_for_take_profit_and_resistance_for_stop_loss():
    """
    A short's roles are the mirror of a long's -- profit is downside room
    (support), risk is upside room (resistance). Both distances chosen
    strictly inside their respective leg's [floor, ceiling] band -- TP:
    [0.089, 0.179] (ATR-based), SL: [0.05, 0.067] (sigma-based) -- so each
    actually tightens rather than getting floored back out.
    """
    levels = exit_levels_for(
        predicted_return=-0.5, daily_volatility=0.02, horizon_days=5, atr_pct=0.02,
        support_distance_pct=0.12, resistance_distance_pct=0.06,
    )
    assert levels.take_profit_pct == pytest.approx(0.12)
    assert levels.stop_loss_pct == pytest.approx(0.06)


def test_advised_clamps_the_llm_suggestion_into_the_channel_tightened_band():
    levels = exit_levels_advised(
        predicted_return=0.5, daily_volatility=0.01, llm_take_profit_pct=0.15,
        llm_stop_loss_pct=None, horizon_days=5, atr_pct=0.02, resistance_distance_pct=0.12,
    )
    assert levels.take_profit_pct == pytest.approx(0.12)  # clamped down to resistance, not the raw 0.15
