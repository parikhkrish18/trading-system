"""
Position sizing: forecast confidence + regime signal + current portfolio
correlation -> target position size (as a fraction of portfolio, signed).

Kept as pure functions of explicit inputs (no hidden state, no DB access)
so it's straightforward to unit test and to reason about in isolation from
execution/broker code.
"""
from __future__ import annotations

import numpy as np

from models.regime.trend_chop_classifier import CHOP, TREND


def confidence_scaled_size(
    forecast: float,
    forecast_scale: float,
    max_position_pct: float,
) -> float:
    """
    Maps a raw forecast (e.g. predicted forward return) to a position size
    in [-max_position_pct, max_position_pct], scaled by how large the
    forecast is relative to `forecast_scale` (a typical/expected forecast
    magnitude — e.g. the trailing std of forecasts). Saturates at the max.
    """
    if forecast_scale <= 0:
        return 0.0
    raw = forecast / forecast_scale
    return float(np.clip(raw, -1.0, 1.0) * max_position_pct)


def regime_adjusted_size(base_size: float, regime: str, chop_dampening: float = 0.35) -> float:
    """
    Leveraged ETFs held through chop bleed value via daily-reset decay
    (backtest/decay_sim.py) even when the underlying is flat — so exposure
    should be structurally smaller in a chop regime, independent of how
    confident the forecast model is.
    """
    if regime == CHOP:
        return base_size * chop_dampening
    if regime == TREND:
        return base_size
    raise ValueError(f"Unknown regime: {regime!r}")


def correlation_adjusted_size(
    proposed_size: float,
    symbol: str,
    current_positions: dict[str, float],
    correlation_matrix,   # pandas DataFrame, symbols x symbols
    max_correlated_exposure_pct: float,
) -> float:
    """
    Shrinks `proposed_size` if adding it would push aggregate exposure to
    highly correlated names (corr > 0.7, a conventional threshold) above the
    portfolio-level cap. This is a soft pre-check; circuit_breakers.py is
    the hard backstop that can flatten positions outright.
    """
    if not current_positions:
        return proposed_size

    correlated_exposure = 0.0
    for other_symbol, other_size in current_positions.items():
        if other_symbol == symbol:
            continue
        if symbol in correlation_matrix.index and other_symbol in correlation_matrix.columns:
            corr = correlation_matrix.loc[symbol, other_symbol]
            if corr > 0.7:
                correlated_exposure += abs(other_size)

    headroom = max(max_correlated_exposure_pct - correlated_exposure, 0.0)
    if abs(proposed_size) <= headroom:
        return proposed_size
    return float(np.sign(proposed_size) * headroom)


def target_position_size(
    forecast: float,
    forecast_scale: float,
    regime: str,
    symbol: str,
    current_positions: dict[str, float],
    correlation_matrix,
    max_position_pct: float,
    max_correlated_exposure_pct: float,
    chop_dampening: float = 0.35,
    max_short_position_pct: float | None = None,
) -> float:
    """
    Full sizing pipeline: confidence -> regime adjustment -> correlation adjustment.

    `max_short_position_pct`, when given, caps a negative (short) forecast
    more conservatively than `max_position_pct` caps a long one — a short's
    loss is structurally uncapped, a long's isn't, so the same headroom
    isn't the right default for both. Falls back to `max_position_pct` for
    both directions if not given, to keep the long-only callers unaffected.
    """
    effective_max_pct = max_position_pct
    if forecast < 0 and max_short_position_pct is not None:
        effective_max_pct = max_short_position_pct

    size = confidence_scaled_size(forecast, forecast_scale, effective_max_pct)
    size = regime_adjusted_size(size, regime, chop_dampening)
    size = correlation_adjusted_size(
        size, symbol, current_positions, correlation_matrix, max_correlated_exposure_pct
    )
    return size
