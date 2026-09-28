import pandas as pd
import pytest

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


def test_max_drawdown_breaker_fails_safe_on_too_short_equity_curve():
    """
    Last line of defense: a corrupted/missing equity feed (< 2 points) must
    not be read as "drawdown is fine" — it must halt trading instead.
    """
    for equity in ([], [100_000]):
        result = max_drawdown_breaker(equity, max_drawdown_pct=0.15)
        assert result.triggered
        assert "cannot verify" in result.reason


def test_max_single_position_breaker():
    assert max_single_position_breaker(30_000, 100_000, max_single_position_pct=0.25).triggered
    assert not max_single_position_breaker(20_000, 100_000, max_single_position_pct=0.25).triggered


def test_max_single_position_breaker_computes_a_trim_target_when_symbol_given():
    """
    With a symbol, a breach also carries a target_value to trim TO — 5%
    below the cap (see _TRIM_SAFETY_MARGIN), same sign as the position.
    """
    result = max_single_position_breaker(30_000, 100_000, max_single_position_pct=0.25, symbol="AAPL")
    assert result.triggered
    assert result.breaker_type == "max_single_position"
    assert result.symbol == "AAPL"
    assert result.target_value == pytest.approx(23_750.0)  # 0.25 * 0.95 * 100_000


def test_max_single_position_breaker_trim_target_keeps_a_short_negative():
    result = max_single_position_breaker(-30_000, 100_000, max_single_position_pct=0.25, symbol="TSLA")
    assert result.triggered
    assert result.target_value == pytest.approx(-23_750.0)


def test_max_single_position_breaker_no_symbol_leaves_target_value_unset():
    result = max_single_position_breaker(30_000, 100_000, max_single_position_pct=0.25)
    assert result.triggered
    assert result.symbol is None
    assert result.target_value is None


def test_max_single_position_breaker_fails_safe_on_non_positive_portfolio_value():
    """Last line of defense: a $0 or negative portfolio_value can't verify anything, so it must not pass as 'fine'."""
    for portfolio_value in (0.0, -5_000.0):
        result = max_single_position_breaker(30_000, portfolio_value, max_single_position_pct=0.25)
        assert result.triggered
        assert "cannot verify" in result.reason


def test_max_correlated_exposure_breaker_flags_cluster():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.95], "UPRO": [0.95, 1.0]}, index=["TQQQ", "UPRO"])
    positions = {"TQQQ": 30_000, "UPRO": 30_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert len(results) == 2  # both symbols' clusters breach (60% > 50%)


def test_max_correlated_exposure_breaker_computes_a_trim_target_per_symbol():
    """
    Each flagged symbol gets its OWN target_value: trim just that symbol by
    the cluster's full dollar excess over a margin-padded cap, leaving the
    other correlated member(s) untouched for this trim (see BreakerResult's
    docstring on the self-correcting, one-symbol-at-a-time design).
    """
    corr = pd.DataFrame({"TQQQ": [1.0, 0.95], "UPRO": [0.95, 1.0]}, index=["TQQQ", "UPRO"])
    positions = {"TQQQ": 30_000, "UPRO": 30_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["TQQQ"].breaker_type == "max_correlated_exposure"
    # cluster_exposure 60_000, capped cluster 0.50 * 0.95 * 100_000 = 47_500,
    # excess 12_500 -> trim TQQQ alone from 30_000 down to 17_500.
    assert by_symbol["TQQQ"].target_value == pytest.approx(17_500.0)
    assert by_symbol["UPRO"].target_value == pytest.approx(17_500.0)


def test_max_correlated_exposure_breaker_clamps_target_value_at_zero():
    """
    When one symbol's own excess share would exceed its whole position, the
    trim clamps at fully closing it rather than going negative (flipping a
    long into a short it never asked for).
    """
    corr = pd.DataFrame({"TQQQ": [1.0, 0.95], "UPRO": [0.95, 1.0]}, index=["TQQQ", "UPRO"])
    positions = {"TQQQ": 5_000, "UPRO": 55_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    tqqq = next(r for r in results if r.symbol == "TQQQ")
    assert tqqq.target_value == pytest.approx(0.0)


def test_max_correlated_exposure_breaker_no_flag_when_uncorrelated():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.1], "GLD": [0.1, 1.0]}, index=["TQQQ", "GLD"])
    positions = {"TQQQ": 30_000, "GLD": 30_000}
    results = max_correlated_exposure_breaker(
        positions, portfolio_value=100_000, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert results == []


def test_max_correlated_exposure_breaker_fails_safe_on_non_positive_portfolio_value():
    """Last line of defense: with open positions but an invalid portfolio_value, must trigger, not pass silently."""
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    results = max_correlated_exposure_breaker(
        {"TQQQ": 30_000}, portfolio_value=0.0, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert len(results) == 1
    assert results[0].triggered
    assert "cannot verify" in results[0].reason


def test_max_correlated_exposure_breaker_no_positions_still_returns_empty_even_with_bad_portfolio_value():
    """Nothing to check with no open positions — a non-positive portfolio_value alone shouldn't manufacture a trigger."""
    corr = pd.DataFrame()
    results = max_correlated_exposure_breaker(
        {}, portfolio_value=0.0, correlation_matrix=corr, max_correlated_exposure_pct=0.50
    )
    assert results == []


def test_max_correlated_exposure_breaker_logs_when_a_pair_is_unmeasured(caplog):
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])  # no GLD row/column
    with caplog.at_level("WARNING"):
        results = max_correlated_exposure_breaker(
            {"TQQQ": 10_000, "GLD": 10_000}, portfolio_value=100_000,
            correlation_matrix=corr, max_correlated_exposure_pct=0.50,
        )
    assert results == []  # unmeasured pair treated as uncorrelated, no breach
    assert any("no correlation data" in r.message for r in caplog.records)


def test_run_all_breakers_treats_a_failed_input_check_the_same_as_a_real_breach():
    """
    The caller-facing contract: run_all_breakers (and by extension
    check_and_record_breakers/trading_loop's flatten-and-alert path) only
    ever looks at whether a BreakerResult is triggered — it must treat a
    fail-safe trigger from corrupted input exactly like a real limit breach.
    """
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    triggered = run_all_breakers(
        equity_curve=[100_000],  # too short -> fails safe
        positions_by_symbol={"TQQQ": 10_000},
        portfolio_value=100_000,
        correlation_matrix=corr,
        max_drawdown_pct=0.15,
        max_single_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
    )
    assert len(triggered) == 1
    assert triggered[0].triggered


def test_run_all_breakers_threads_the_symbol_into_max_single_position_triggers():
    """
    run_all_breakers must pass each symbol into max_single_position_breaker
    so the returned trigger carries enough (symbol/target_value) for
    execution/trading_loop.py to trim rather than flatten.
    """
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    triggered = run_all_breakers(
        equity_curve=[100_000, 100_000, 100_000],
        positions_by_symbol={"TQQQ": 30_000},
        portfolio_value=100_000,
        correlation_matrix=corr,
        max_drawdown_pct=0.15,
        max_single_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
    )
    assert len(triggered) == 1
    assert triggered[0].breaker_type == "max_single_position"
    assert triggered[0].symbol == "TQQQ"
    assert triggered[0].target_value == pytest.approx(23_750.0)


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
