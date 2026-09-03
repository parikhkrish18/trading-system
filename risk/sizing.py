"""
Position sizing: forecast confidence + regime signal + current portfolio
correlation -> target position size (as a fraction of portfolio, signed).

Kept as pure functions of explicit inputs (no hidden state, no DB access)
so it's straightforward to unit test and to reason about in isolation from
execution/broker code.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping

import numpy as np

from models.regime.trend_chop_classifier import CHOP, TREND

logger = logging.getLogger(__name__)

# Everything is a fraction of portfolio value, so sizes that differ by less
# than this are the same size in any currency anyone will actually trade.
_TOLERANCE = 1e-9


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
    # NaN comparisons are always False, so `forecast_scale <= 0` alone lets
    # a NaN scale (e.g. std() of a too-small training frame) straight
    # through — math.isnan catches what the comparison can't.
    if forecast_scale <= 0 or math.isnan(forecast_scale):
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
        else:
            # An unmeasured pair silently contributes 0 to correlated
            # exposure rather than being assumed correlated — logged so a
            # gap in the correlation matrix (a symbol too new for the
            # lookback window, e.g.) is visible instead of quietly
            # understating exposure.
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


# --------------------------------------------------------------------------
# Full deployment
# --------------------------------------------------------------------------


@dataclasses.dataclass
class FullDeploymentResult:
    """
    What scaling to full deployment actually achieved, and — when it fell
    short — why. The shortfall is the interesting part: it means the caps
    bound before the cash ran out, which is a fact about the shortlist that
    the operator should see rather than a number to quietly round up.
    """

    sizes: dict[str, float]  # signed fractions of portfolio, same keys as the input
    deployed_pct: float  # total absolute allocation after scaling
    target_pct: float
    capped_symbols: list[str]  # positions sitting exactly on their per-position cap
    reached_target: bool
    reason: str = ""  # empty when the target was reached

    @property
    def scale_applied(self) -> float:
        """Informational: what the uncapped positions were multiplied by overall."""
        return self.deployed_pct / self.target_pct if self.target_pct else 0.0


def allocate_by_conviction(
    convictions: Mapping[str, float],
    max_position_pct: float,
    max_short_position_pct: float | None = None,
    max_correlated_exposure_pct: float | None = None,
    correlation_matrix=None,
    target_allocation: float = 1.0,
) -> FullDeploymentResult:
    """
    Post-approval sizing: distribute `target_allocation` of the portfolio
    across an already-approved set of picks, weighted by each pick's signed
    conviction (positive = long, negative = short). This exists because the
    approval gate now decides WHICH trades happen before anything decides
    HOW BIG they are — sizing a book and then letting a human veto 7 of 10
    picks leaves the vetoed picks' capital in cash (which is exactly what
    happened on the first live paper run).

    The proportional scaling and the per-position caps are
    scale_to_full_deployment, reused unchanged. On top of that, the
    correlated-exposure cap is enforced with correlation_adjusted_size:
    walking the book from highest to lowest conviction, each position is
    shrunk if the positions already accepted ahead of it (corr > 0.7) have
    used up the correlated headroom. Any shortfall that clamp introduces is
    reported, not silently redistributed — redistributing it back into the
    very names that are correlated would defeat the cap.

    Caps are hard, same contract as scale_to_full_deployment: if they bind
    before the target is reached, the result says so in `reason` and the
    book is NOT padded or scaled past a cap.

    All-zero convictions (possible when every approved pick has conviction
    exactly 0) fall back to an equal split so approved picks still deploy.
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
        # copysign preserves direction even at zero conviction: callers pass
        # -0.0 for a short, and IEEE -0.0 carries its sign into copysign.
        weights = {symbol: math.copysign(1.0, w) for symbol, w in weights.items()}

    result = scale_to_full_deployment(
        weights,
        max_position_pct=max_position_pct,
        max_short_position_pct=max_short_position_pct,
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
    """
    Scale a book of signed position sizes proportionally until total absolute
    allocation reaches `target_allocation` (1.0 = 100% of the portfolio),
    without any single position exceeding its per-position cap.

    Proportional, so the screener's relative conviction ordering survives: a
    pick the model liked twice as much stays twice as large, it just gets a
    bigger share of a fully-deployed book. Signs are preserved, and a short
    is capped by `max_short_position_pct` rather than `max_position_pct` —
    the same asymmetry target_position_size applies, for the same reason (a
    short's loss is structurally uncapped).

    Positions that hit their cap are frozen there and the leftover allocation
    is redistributed across the ones with headroom, repeatedly, until either
    the target is met or everything is capped. That redistribution is the
    whole reason this isn't a single multiply: capping after scaling would
    silently under-deploy, and scaling after capping would silently breach
    the caps.

    Caps are hard. If they bind before the target is reached — two picks
    under a 25% cap can only ever be 50% of a portfolio — this returns the
    cappable maximum and says why. It will not pad the book with picks the
    model wasn't confident about, and it will not exceed a cap to hit a
    round number: the caps exist to bound the damage from one bad forecast,
    and "we wanted to be fully invested" is not a risk argument.

    Scaling runs in both directions: a book already above the target is
    scaled *down* to it, since deploying more than 100% would mean leverage
    that nothing upstream has sized for.
    """
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

    # Water-filling: scale everything that still has headroom by whatever
    # factor would use up the remaining allocation, freeze whatever that
    # would push past its cap, and go again with what's left.
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
            # copysign, not abs * sign: keeps a short short at exactly its cap.
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
