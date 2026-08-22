"""
The exit policy in isolation (execution/hold_rules.py): pure decisions in
evaluate_holds, and the consecutive-miss counter's persistence round-trip.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from execution.hold_rules import HoldDecision, evaluate_holds, load_missed_cycles, store_missed_cycles

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
            text("CREATE TABLE position_hold_state (symbol TEXT PRIMARY KEY, missed_cycles INTEGER NOT NULL, updated_at TIMESTAMP NOT NULL)")
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
