from __future__ import annotations

import pytest

from config.settings import settings
from execution.exit_levels import ExitLevels
from execution.hold_rules import evaluate_holds
from risk.sizing import allocate_by_conviction


def test_repicked_position_still_exits_at_its_recorded_profit_target():
    """A fresh screen re-pick cannot waive the take-profit the trade was opened with."""
    (decision,) = evaluate_holds(
        positions={"SNDK": 10.0},
        shortlist={"SNDK"},
        predictions={"SNDK": 0.04},
        pnl_pct={"SNDK": 0.101},
        prior_missed={"SNDK": 0},
        max_missed_cycles=2,
        stop_loss_pct=0.08,
        take_profit_pct=0.10,
        levels_by_symbol={"SNDK": ExitLevels(take_profit_pct=0.095, stop_loss_pct=0.07)},
    )

    assert decision.close is True
    assert decision.missed_cycles == 0
    assert any("profit target reached" in reason for reason in decision.reasons)
    assert any("9.5%" in reason for reason in decision.reasons)


def test_repicked_position_still_exits_at_its_recorded_stop_loss():
    (decision,) = evaluate_holds(
        positions={"AAPL": 10.0},
        shortlist={"AAPL"},
        predictions={"AAPL": 0.03},
        pnl_pct={"AAPL": -0.081},
        prior_missed={"AAPL": 4},
        max_missed_cycles=2,
        stop_loss_pct=0.08,
        take_profit_pct=0.10,
        levels_by_symbol={"AAPL": ExitLevels(take_profit_pct=0.12, stop_loss_pct=0.08)},
    )

    assert decision.close is True
    assert any("stop loss" in reason for reason in decision.reasons)


def test_concentrated_post_approval_allocation_is_not_forced_to_40_40_20(monkeypatch):
    """
    Regression for the observed slot-like allocation. The caller may pass
    MAX_SINGLE_POSITION_PCT=.40 for the general/diversified risk envelope,
    but concentrated mode has its own 70% per-leg cap. Conviction should
    therefore be allowed to produce 60/30/10 instead of mechanically
    clipping the first two legs to 40/40 and dumping the remainder into the
    third.
    """
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)

    result = allocate_by_conviction(
        {"HIGH": 0.60, "MID": 0.30, "LOW": 0.10},
        max_position_pct=0.40,
        max_short_position_pct=0.15,
        target_allocation=1.0,
    )

    assert result.reached_target is True
    assert result.sizes["HIGH"] == pytest.approx(0.60)
    assert result.sizes["MID"] == pytest.approx(0.30)
    assert result.sizes["LOW"] == pytest.approx(0.10)


def test_diversified_post_approval_allocation_still_honors_generic_cap(monkeypatch):
    monkeypatch.setattr(settings, "strategy_mode", "diversified")

    result = allocate_by_conviction(
        {"HIGH": 0.60, "MID": 0.30, "LOW": 0.10},
        max_position_pct=0.40,
        max_short_position_pct=0.15,
        target_allocation=1.0,
    )

    assert result.sizes["HIGH"] == pytest.approx(0.40)
    assert result.sizes["MID"] == pytest.approx(0.40)
    assert result.sizes["LOW"] == pytest.approx(0.20)
