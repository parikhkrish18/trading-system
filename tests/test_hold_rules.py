"""
The exit policy in isolation (execution/hold_rules.py): pure decisions in
evaluate_holds, and the consecutive-miss counter's persistence round-trip.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from execution.exit_levels import ExitLevels
from execution.hold_rules import (
    HoldDecision,
    check_stop_or_target,
    evaluate_holds,
    load_exit_levels,
    load_missed_cycles,
    store_missed_cycles,
)

_DEFAULTS = {"max_missed_cycles": 2, "stop_loss_pct": 0.08, "take_profit_pct": 0.10}


def _decide(positions, shortlist=(), predictions=None, pnl=None, prior=None, **overrides):
    return evaluate_holds(
        positions=positions,
        shortlist=set(shortlist),
        predictions=predictions or {},
        pnl_pct=pnl or {},
        prior_missed=prior or {},
        **{**_DEFAULTS, **overrides},
    )


def _one(decisions, symbol) -> HoldDecision:
    return next(d for d in decisions if d.symbol == symbol)


class TestCheckStopOrTarget:
    """
    The shared building block behind both evaluate_holds' stop/target check
    (weekly cycle) and contradiction_monitor's hourly one — a swing trade
    should close when ITS OWN target/stop is hit, not wait on whichever
    clock happens to run next.
    """

    def test_no_pnl_is_no_verdict(self):
        assert check_stop_or_target(None, None, fallback_stop_loss_pct=0.08, fallback_take_profit_pct=0.10) is None

    def test_within_band_is_no_verdict(self):
        assert check_stop_or_target(0.02, None, fallback_stop_loss_pct=0.08, fallback_take_profit_pct=0.10) is None

    def test_uses_recorded_levels_over_fallback(self):
        levels = ExitLevels(take_profit_pct=0.05, stop_loss_pct=0.03)
        # -4% breaches the recorded 3% stop but not the 8% fallback -- recorded levels must win.
        hit = check_stop_or_target(-0.04, levels, fallback_stop_loss_pct=0.08, fallback_take_profit_pct=0.10)
        assert hit is not None
        assert hit.kind == "stop_loss"
        assert "3.0%" in hit.message

    def test_falls_back_when_no_levels_recorded(self):
        hit = check_stop_or_target(-0.09, None, fallback_stop_loss_pct=0.08, fallback_take_profit_pct=0.10)
        assert hit.kind == "stop_loss"

    def test_take_profit_hit(self):
        levels = ExitLevels(take_profit_pct=0.05, stop_loss_pct=0.03)
        hit = check_stop_or_target(0.06, levels, fallback_stop_loss_pct=0.08, fallback_take_profit_pct=0.10)
        assert hit.kind == "take_profit"
        assert "5.0%" in hit.message


def test_first_missed_cycle_is_a_hold():
    (d,) = _decide({"AAPL": 10.0})

    assert not d.close
    assert d.missed_cycles == 1
    assert d.reasons == []


def test_hitting_the_miss_limit_fires_the_exit():
    (d,) = _decide({"AAPL": 10.0}, prior={"AAPL": 1})

    assert d.close
    assert d.missed_cycles == 2
    assert any("consecutive" in r for r in d.reasons)


def test_shortlisted_position_resets_its_miss_counter_and_is_never_an_exit():
    (d,) = _decide(
        {"AAPL": 10.0},
        shortlist={"AAPL"},
        prior={"AAPL": 5},
        pnl={"AAPL": -0.5},  # even a huge drawdown: the fresh screen re-picked it
    )

    assert not d.close
    assert d.missed_cycles == 0


def test_confident_prediction_flip_fires_even_on_the_first_miss():
    (d,) = _decide({"AAPL": 10.0}, predictions={"AAPL": -0.03})

    assert d.close
    assert any("against the long" in r for r in d.reasons)


def test_prediction_flip_below_the_cost_floor_is_noise_not_an_exit():
    (d,) = _decide({"AAPL": 10.0}, predictions={"AAPL": -0.0001}, min_flip_return=0.001)

    assert not d.close


def test_prediction_in_the_held_direction_is_not_a_flip():
    (short,) = _decide({"TSLA": -10.0}, predictions={"TSLA": -0.05})  # short, model says down: agree

    assert not short.close


def test_flip_logic_respects_the_short_side():
    (short,) = _decide({"TSLA": -10.0}, predictions={"TSLA": 0.05})  # short, model says up 5%

    assert short.close
    assert any("against the short" in r for r in short.reasons)


def test_stop_loss_and_take_profit_fire_on_pnl():
    decisions = _decide(
        {"DOWN": 10.0, "UP": 10.0, "FLAT": 10.0},
        pnl={"DOWN": -0.09, "UP": 0.12, "FLAT": 0.01},
    )

    assert _one(decisions, "DOWN").close and any("stop loss" in r for r in _one(decisions, "DOWN").reasons)
    assert _one(decisions, "UP").close and any("profit target" in r for r in _one(decisions, "UP").reasons)
    assert not _one(decisions, "FLAT").close


def test_missing_pnl_never_fires_a_pnl_exit():
    (d,) = _decide({"AAPL": 10.0}, pnl={"AAPL": None})

    assert not d.close


def test_zero_quantity_positions_are_ignored():
    assert _decide({"AAPL": 0.0}) == []


# --- state persistence round-trip -------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE position_hold_state ("
                "symbol TEXT PRIMARY KEY, missed_cycles INTEGER NOT NULL, updated_at TIMESTAMP NOT NULL, "
                "take_profit_pct REAL, stop_loss_pct REAL)"
            )
        )
    return eng


def test_store_then_load_round_trips(engine):
    store_missed_cycles(engine, {"AAPL": 2, "TSLA": 0})

    assert load_missed_cycles(engine) == {"AAPL": 2, "TSLA": 0}


def test_store_rewrites_the_whole_table(engine):
    """A position closed by ANY path must fall out on the next cycle's write."""
    store_missed_cycles(engine, {"AAPL": 1, "GONE": 3})
    store_missed_cycles(engine, {"AAPL": 2})

    assert load_missed_cycles(engine) == {"AAPL": 2}


# --------------------------------------------------------------------------
# A position is judged against the levels it was approved with
# --------------------------------------------------------------------------


def _held(pnl, levels=None):
    return evaluate_holds(
        positions={"AAPL": 10.0},
        shortlist=set(),
        predictions={},
        pnl_pct={"AAPL": pnl},
        prior_missed={},
        max_missed_cycles=99,  # keep the miss rule out of the way
        stop_loss_pct=0.08,
        take_profit_pct=0.10,
        levels_by_symbol=levels,
    )[0]


def test_a_positions_own_stop_is_used_instead_of_the_global_one():
    """
    Editing HOLD_STOP_LOSS_PCT must not silently rewrite the terms of a
    position a human already approved on different ones.
    """
    own = {"AAPL": ExitLevels(take_profit_pct=0.20, stop_loss_pct=0.15)}

    # -10% trips the global 8% stop but not this position's 15% one.
    assert _held(-0.10).close is True
    assert _held(-0.10, own).close is False
    assert _held(-0.16, own).close is True


def test_a_positions_own_target_is_used_instead_of_the_global_one():
    own = {"AAPL": ExitLevels(take_profit_pct=0.25, stop_loss_pct=0.10)}

    assert _held(0.12).close is True  # clears the global 10% target
    assert _held(0.12, own).close is False
    assert _held(0.26, own).close is True


def test_the_globals_still_apply_to_a_position_with_no_recorded_levels():
    """
    Positions opened before per-pick levels existed must still be
    evaluated, not skipped for want of a column.
    """
    decision = _held(-0.09, {"SOMETHING_ELSE": ExitLevels(0.2, 0.2)})

    assert decision.close is True
    assert "stop loss" in " ".join(decision.reasons)


def test_the_reason_quotes_the_level_that_actually_fired():
    """A reason naming the global limit when a per-pick one fired would be a lie."""
    own = {"AAPL": ExitLevels(take_profit_pct=0.20, stop_loss_pct=0.15)}

    reasons = " ".join(_held(-0.16, own).reasons)

    assert "15.0%" in reasons
    assert "8" not in reasons.replace("15.0%", "")


def test_levels_round_trip_through_the_hold_state(engine):
    levels = {"AAPL": ExitLevels(take_profit_pct=0.11, stop_loss_pct=0.07)}

    store_missed_cycles(engine, {"AAPL": 1, "TSLA": 0}, levels)
    loaded = load_exit_levels(engine)

    assert loaded["AAPL"].take_profit_pct == pytest.approx(0.11)
    assert loaded["AAPL"].stop_loss_pct == pytest.approx(0.07)
    # A position stored without levels is absent, not zeroed — the caller
    # has to be able to tell "no levels" from "a zero stop".
    assert "TSLA" not in loaded
