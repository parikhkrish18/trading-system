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
