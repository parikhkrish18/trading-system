"""
Position sizing: forecast confidence + regime signal + current portfolio
correlation -> target position size (as a fraction of portfolio, signed).

Kept mostly as pure functions of explicit inputs. The one policy-aware seam
is allocate_by_conviction: post-approval sizing has to honor the active
strategy's risk envelope, otherwise a concentrated book gets accidentally
clamped by the diversified-book cap and collapses into mechanical slot-like
weights such as 40/40/20.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping

import numpy as np

from models.regime.trend_chop_classifier import CHOP, TREND

logger = logging.getLogger(__name__)

_TOLERANCE = 1e-9


def confidence_scaled_size(
    forecast: float,
    forecast_scale: float,
    max_position_pct: float,
) -> float:
    if forecast_scale <= 0 or math.isnan(forecast_scale):
        return 0.0
    raw = forecast / forecast_scale
    return float(np.clip(raw, -1.0, 1.0) * max_position_pct)


def regime_adjusted_size(base_size: float, regime: str, chop_dampening: float = 0.35) -> float:
    if regime == CHOP:
        return base_size * chop_dampening
    if regime == TREND:
        return base_size
    raise ValueError(f"Unknown regime: {regime!r}")


def correlation_adjusted_size(
    proposed_size: float,
    symbol: str,
    current_positions: dict[str, float],
    correlation_matrix,
    max_correlated_exposure_pct: float,
) -> float:
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
        else:
            logger.warning(
                "correlation_adjusted_size: no correlation data for (%s, %s) — treating as "
                "uncorrelated for this check.",
                symbol, other_symbol,
            )

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
    effective_max_pct = max_position_pct
    if forecast < 0 and max_short_position_pct is not None:
        effective_max_pct = max_short_position_pct

    size = confidence_scaled_size(forecast, forecast_scale, effective_max_pct)
    size = regime_adjusted_size(size, regime, chop_dampening)
    size = correlation_adjusted_size(
        size, symbol, current_positions, correlation_matrix, max_correlated_exposure_pct
    )
    return size


@dataclasses.dataclass
class FullDeploymentResult:
    sizes: dict[str, float]
    deployed_pct: float
    target_pct: float
    capped_symbols: list[str]
    reached_target: bool
    reason: str = ""

    @property
    def scale_applied(self) -> float:
        return self.deployed_pct / self.target_pct if self.target_pct else 0.0


def _post_approval_caps(
    max_position_pct: float,
    max_short_position_pct: float | None,
) -> tuple[float, float | None]:
    """
    Return the active strategy's post-approval per-leg caps.

    trading_loop/contradiction_monitor historically passed the generic
    MAX_SINGLE_POSITION_PCT here even when the screener was running the
    concentrated strategy. In a deployment with MAX_SINGLE_POSITION_PCT=.40
    that mechanically produced 40/40/20 once three picks were approved,
    regardless of the concentrated strategy's own 70% cap and relative
    conviction. Use the concentrated strategy's explicit cap in that mode.

    Imported lazily so the lower-level sizing functions remain configuration
    independent and existing offline callers/tests keep their explicit caps.
    """
    try:
        from config.settings import settings

        if settings.strategy_mode == "concentrated":
            cap = float(settings.max_concentrated_position_pct)
            # max_short_position_pct is documented as a diversified-only
            # policy. Concentrated mode's selector uses the same leg cap for
            # either direction, so post-approval sizing must match it.
            return cap, cap
    except Exception:
        logger.debug("Could not resolve strategy-specific allocation caps; using explicit caps.", exc_info=True)
    return max_position_pct, max_short_position_pct


def allocate_by_conviction(
    convictions: Mapping[str, float],
    max_position_pct: float,
    max_short_position_pct: float | None = None,
    max_correlated_exposure_pct: float | None = None,
    correlation_matrix=None,
    target_allocation: float = 1.0,
) -> FullDeploymentResult:
    """
    Distribute target capital across approved picks by signed conviction.

    The allocation is not slot-based: relative conviction determines the
    unconstrained weights, then the active strategy's real risk caps and the
    correlated-exposure cap constrain them. In concentrated mode this uses
    MAX_CONCENTRATED_POSITION_PCT rather than accidentally reusing the
    diversified MAX_SINGLE_POSITION_PCT.
    """
    if not convictions:
        return FullDeploymentResult(
            sizes={},
            deployed_pct=0.0,
            target_pct=target_allocation,
            capped_symbols=[],
            reached_target=False,
            reason="No approved picks to allocate capital to.",
        )

    weights = {symbol: float(c) for symbol, c in convictions.items()}
    if all(abs(w) <= _TOLERANCE for w in weights.values()):
        weights = {symbol: math.copysign(1.0, w) for symbol, w in weights.items()}

    effective_long_cap, effective_short_cap = _post_approval_caps(
        max_position_pct, max_short_position_pct
    )
    result = scale_to_full_deployment(
        weights,
        max_position_pct=effective_long_cap,
        max_short_position_pct=effective_short_cap,
        target_allocation=target_allocation,
    )

    if max_correlated_exposure_pct is None or correlation_matrix is None or getattr(correlation_matrix, "empty", True):
        return result

    accepted: dict[str, float] = {}
    clamped_symbols: list[str] = []
    for symbol in sorted(result.sizes, key=lambda s: abs(convictions.get(s, 0.0)), reverse=True):
        size = result.sizes[symbol]
        adjusted = correlation_adjusted_size(
            size, symbol, accepted, correlation_matrix, max_correlated_exposure_pct
        )
        if abs(adjusted) < abs(size) - _TOLERANCE:
            clamped_symbols.append(symbol)
        accepted[symbol] = adjusted

    if not clamped_symbols:
        return result

    deployed = sum(abs(size) for size in accepted.values())
    reason = (
        f"Correlated-exposure cap bound: {', '.join(sorted(clamped_symbols))} shrunk because "
        f"aggregate exposure to highly correlated names (corr > 0.7) is capped at "
        f"{max_correlated_exposure_pct:.0%} of the portfolio. Deployed {deployed:.1%} of the "
        f"{target_allocation:.1%} target; the shortfall stays in cash rather than being "
        f"pushed back into correlated names."
    )
    if result.reason:
        reason = f"{result.reason} {reason}"
    return FullDeploymentResult(
        sizes={symbol: accepted[symbol] for symbol in result.sizes},
        deployed_pct=deployed,
        target_pct=target_allocation,
        capped_symbols=sorted(set(result.capped_symbols) | set(clamped_symbols)),
        reached_target=deployed >= target_allocation - _TOLERANCE,
        reason=reason,
    )


def scale_to_full_deployment(
    sizes: Mapping[str, float],
    max_position_pct: float,
    max_short_position_pct: float | None = None,
    target_allocation: float = 1.0,
) -> FullDeploymentResult:
    scaled = {symbol: float(size) for symbol, size in sizes.items()}
    short_cap = max_short_position_pct if max_short_position_pct is not None else max_position_pct
    caps = {
        symbol: (max_position_pct if size >= 0 else short_cap) for symbol, size in scaled.items()
    }

    gross = sum(abs(size) for size in scaled.values())
    if not scaled or gross <= _TOLERANCE:
        return FullDeploymentResult(
            sizes=scaled,
            deployed_pct=0.0,
            target_pct=target_allocation,
            capped_symbols=[],
            reached_target=False,
            reason="No sized candidates to deploy — nothing cleared the confidence bar.",
        )
    if target_allocation <= 0:
        return FullDeploymentResult(
            sizes=dict.fromkeys(scaled, 0.0),
            deployed_pct=0.0,
            target_pct=target_allocation,
            capped_symbols=[],
            reached_target=True,
            reason="Target allocation is zero — nothing deployed.",
        )

    free = {symbol for symbol, size in scaled.items() if abs(size) > _TOLERANCE}
    capped: dict[str, float] = {}

    while free:
        free_gross = sum(abs(scaled[symbol]) for symbol in free)
        if free_gross <= _TOLERANCE:
            break
        headroom = target_allocation - sum(capped.values())
        factor = headroom / free_gross

        newly_capped = [s for s in free if abs(scaled[s]) * factor > caps[s] + _TOLERANCE]
        if not newly_capped:
            for symbol in free:
                scaled[symbol] = scaled[symbol] * factor
            break

        for symbol in newly_capped:
            scaled[symbol] = math.copysign(caps[symbol], scaled[symbol])
            capped[symbol] = caps[symbol]
            free.discard(symbol)

    deployed = sum(abs(size) for size in scaled.values())
    capped_symbols = sorted(capped)
    reached = deployed >= target_allocation - _TOLERANCE

    reason = ""
    if not reached:
        reason = (
            f"Full deployment not reached: {deployed:.1%} of the portfolio allocated, "
            f"target {target_allocation:.1%}. Every position that could be scaled is now "
            f"at its per-position cap ({', '.join(capped_symbols)}) — {len(capped_symbols)} "
            f"pick(s) cannot cover the portfolio on their own. Deploying the rest would "
            f"need more confident picks, not larger ones; the caps were not raised and the "
            f"book was not padded with names the model wasn't confident about."
        )

    return FullDeploymentResult(
        sizes=scaled,
        deployed_pct=deployed,
        target_pct=target_allocation,
        capped_symbols=capped_symbols,
        reached_target=reached,
        reason=reason,
    )
