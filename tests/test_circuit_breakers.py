import pandas as pd

from risk.circuit_breakers import (
    max_correlated_exposure_breaker,
    max_drawdown_breaker,
    max_single_position_breaker,
    run_all_breakers,
)


def test_max_drawdown_breaker_triggers_on_breach():
    equity = [100_000, 110_000, 90_000]  # -18% from peak
    result = max_drawdown_breaker(equity, max_drawdown_pct=0.15)
    assert result.triggered
    assert "Drawdown" in result.reason


def test_max_drawdown_breaker_does_not_trigger_within_limit():
    equity = [100_000, 110_000, 100_000]  # ~-9% from peak
    result = max_drawdown_breaker(equity, max_drawdown_pct=0.15)
    assert not result.triggered


def test_max_single_position_breaker():
    assert max_single_position_breaker(30_000, 100_000, max_single_position_pct=0.25).triggered
    assert not max_single_position_breaker(20_000, 100_000, max_single_position_pct=0.25).triggered


def test_max_correlated_exposure_breaker_flags_cluster():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.95], "UPRO": [0.95, 1.0]}, index=["TQQQ", "UPRO"])
    positions = {"TQQQ": 30_000, "UPRO": 30_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert len(results) == 2  # both symbols' clusters breach (60% > 50%)


def test_max_correlated_exposure_breaker_no_flag_when_uncorrelated():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.1], "GLD": [0.1, 1.0]}, index=["TQQQ", "GLD"])
    positions = {"TQQQ": 30_000, "GLD": 30_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert results == []


def test_run_all_breakers_aggregates_only_triggered():
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    triggered = run_all_breakers(
        equity_curve=[100_000, 100_000, 100_000],
        positions_by_symbol={"TQQQ": 10_000},
        portfolio_value=100_000,
        correlation_matrix=corr,
        max_drawdown_pct=0.15,
        max_single_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
    )
    assert triggered == []
