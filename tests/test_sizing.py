import pandas as pd
import pytest

from models.regime.trend_chop_classifier import CHOP, TREND
from risk.sizing import (
    confidence_scaled_size,
    correlation_adjusted_size,
    regime_adjusted_size,
    target_position_size,
)


def test_confidence_scaled_size_saturates_at_max():
    size = confidence_scaled_size(forecast=10.0, forecast_scale=1.0, max_position_pct=0.25)
    assert size == pytest.approx(0.25)


def test_confidence_scaled_size_nan_forecast_scale_returns_zero():
    """
    `forecast_scale <= 0` alone doesn't catch NaN (NaN comparisons are
    always False in Python) — a NaN scale (e.g. std() of a too-small
    training frame, see models/screener.py) has to be caught explicitly, or
    it flows straight through np.clip into a NaN position size.
    """
    size = confidence_scaled_size(forecast=0.05, forecast_scale=float("nan"), max_position_pct=0.25)
    assert size == 0.0


def test_correlation_adjusted_size_logs_when_a_pair_is_unmeasured(caplog):
    """A symbol pair missing from the correlation matrix silently contributes
    0 to correlated exposure (conservative-in-math, but the gap must be
    visible rather than silently ignored)."""
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])  # no SQQQ column/row at all
    with caplog.at_level("WARNING"):
        size = correlation_adjusted_size(
            proposed_size=0.30,
            symbol="SQQQ",
            current_positions={"TQQQ": 0.45},
            correlation_matrix=corr,
            max_correlated_exposure_pct=0.50,
        )
    assert size == pytest.approx(0.30)  # unmeasured pair treated as uncorrelated
    assert any("no correlation data" in r.message for r in caplog.records)


def test_confidence_scaled_size_scales_linearly_below_saturation():
    size = confidence_scaled_size(forecast=0.5, forecast_scale=1.0, max_position_pct=0.25)
    assert size == pytest.approx(0.125)


def test_confidence_scaled_size_respects_sign():
    size = confidence_scaled_size(forecast=-0.5, forecast_scale=1.0, max_position_pct=0.25)
    assert size < 0


def test_regime_adjusted_size_dampens_in_chop():
    base = 0.20
    trend_size = regime_adjusted_size(base, TREND)
    chop_size = regime_adjusted_size(base, CHOP)
    assert trend_size == pytest.approx(base)
    assert chop_size < trend_size


def test_regime_adjusted_size_rejects_unknown_regime():
    with pytest.raises(ValueError):
        regime_adjusted_size(0.1, "sideways")


def test_correlation_adjusted_size_shrinks_when_cluster_is_full():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.95], "SQQQ": [0.95, 1.0]}, index=["TQQQ", "SQQQ"])
    current_positions = {"TQQQ": 0.45}
    size = correlation_adjusted_size(
        proposed_size=0.30,
        symbol="SQQQ",
        current_positions=current_positions,
        correlation_matrix=corr,
        max_correlated_exposure_pct=0.50,
    )
    assert size == pytest.approx(0.05)  # only 0.05 of headroom left


def test_correlation_adjusted_size_passes_through_when_uncorrelated():
    corr = pd.DataFrame({"TQQQ": [1.0, 0.1], "GLD": [0.1, 1.0]}, index=["TQQQ", "GLD"])
    size = correlation_adjusted_size(
        proposed_size=0.30,
        symbol="GLD",
        current_positions={"TQQQ": 0.45},
        correlation_matrix=corr,
        max_correlated_exposure_pct=0.50,
    )
    assert size == pytest.approx(0.30)


def test_target_position_size_full_pipeline():
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    size = target_position_size(
        forecast=0.8,
        forecast_scale=1.0,
        regime=CHOP,
        symbol="TQQQ",
        current_positions={},
        correlation_matrix=corr,
        max_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
    )
    # 0.8 * 0.25 = 0.20 confidence-scaled, then * 0.35 chop dampening
    assert size == pytest.approx(0.20 * 0.35)


def test_target_position_size_uses_short_cap_for_negative_forecast():
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])
    size = target_position_size(
        forecast=-10.0,  # deep in saturation for both caps
        forecast_scale=1.0,
        regime=TREND,
        symbol="TSLA",
        current_positions={},
        correlation_matrix=corr,
        max_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
        max_short_position_pct=0.15,
    )
    assert size == pytest.approx(-0.15)  # capped at the short limit, not the long one


def test_target_position_size_long_forecast_unaffected_by_short_cap():
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])
    size = target_position_size(
        forecast=10.0,
        forecast_scale=1.0,
        regime=TREND,
        symbol="TSLA",
        current_positions={},
        correlation_matrix=corr,
        max_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
        max_short_position_pct=0.15,
    )
    assert size == pytest.approx(0.25)  # long cap, not short cap


def test_target_position_size_without_short_cap_falls_back_to_long_cap():
    """Existing (pre-short-selling) callers that don't pass max_short_position_pct keep old behavior."""
    corr = pd.DataFrame({"TQQQ": [1.0]}, index=["TQQQ"])
    size = target_position_size(
        forecast=-10.0,
        forecast_scale=1.0,
        regime=TREND,
        symbol="TQQQ",
        current_positions={},
        correlation_matrix=corr,
        max_position_pct=0.25,
        max_correlated_exposure_pct=0.50,
    )
    assert size == pytest.approx(-0.25)


# --- allocate_by_conviction (post-approval sizing) --------------------------


def test_allocate_by_conviction_splits_by_relative_conviction():
    from risk.sizing import allocate_by_conviction

    result = allocate_by_conviction(
        {"BIG": 0.04, "SMALL": 0.02},
        max_position_pct=1.0,
        target_allocation=1.0,
    )
    assert result.sizes["BIG"] == pytest.approx(2 / 3)
    assert result.sizes["SMALL"] == pytest.approx(1 / 3)
    assert result.reached_target


def test_allocate_by_conviction_respects_per_position_caps_and_reports_shortfall():
    from risk.sizing import allocate_by_conviction

    result = allocate_by_conviction(
        {"AAA": 0.03, "BBB": 0.03},
        max_position_pct=0.25,
        target_allocation=1.0,
    )
    assert result.sizes["AAA"] == pytest.approx(0.25)
    assert result.sizes["BBB"] == pytest.approx(0.25)
    assert result.deployed_pct == pytest.approx(0.50)
    assert not result.reached_target
    assert "cap" in result.reason


def test_allocate_by_conviction_keeps_shorts_short_and_uses_the_short_cap():
    from risk.sizing import allocate_by_conviction

    result = allocate_by_conviction(
        {"LONG": 0.03, "SHRT": -0.03},
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        target_allocation=1.0,
    )
    assert result.sizes["LONG"] == pytest.approx(0.25)
    assert result.sizes["SHRT"] == pytest.approx(-0.15)


def test_allocate_by_conviction_zero_convictions_fall_back_to_equal_split():
    from risk.sizing import allocate_by_conviction

    result = allocate_by_conviction(
        {"AAA": 0.0, "BBB": -0.0},
        max_position_pct=1.0,
        target_allocation=0.5,
    )
    assert result.sizes["AAA"] == pytest.approx(0.25)
    assert result.sizes["BBB"] == pytest.approx(-0.25)  # -0.0 keeps the short side


def test_allocate_by_conviction_clamps_correlated_exposure():
    from risk.sizing import allocate_by_conviction

    corr = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    result = allocate_by_conviction(
        {"AAA": 0.04, "BBB": 0.02},
        max_position_pct=1.0,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
        target_allocation=1.0,
    )
    # AAA (highest conviction) allocates first (2/3); BBB is clamped so the
    # correlated pair stays inside the 50% cap: headroom is 0.5 - 2/3 < 0 -> 0.
    assert result.sizes["AAA"] == pytest.approx(2 / 3)
    assert result.sizes["BBB"] == pytest.approx(0.0)
    assert not result.reached_target
    assert "Correlated-exposure cap" in result.reason


def test_allocate_by_conviction_empty_input_deploys_nothing():
    from risk.sizing import allocate_by_conviction

    result = allocate_by_conviction({}, max_position_pct=0.25)
    assert result.sizes == {}
    assert result.deployed_pct == 0.0
    assert not result.reached_target
