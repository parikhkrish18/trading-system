"""
Confidence-ranked equity screener. Scores every symbol in a universe with
the ensemble forecast model, then concentrates all deployable capital into
the top 2 highest-conviction picks — long only by default; see
settings.allow_shorts, off because shorts lost -1.069% per trade at a
41.6% win rate in the walk-forward — rather than spreading thinly. The
point is to make a small number of high-confidence bets instead of
tracking the market with a diversified book. If nothing in the mega-caps
clears the confidence bar but some other S&P 500 name does, that's what
gets picked; nothing here special-cases which symbols are "important."

This module does NOT place orders — it produces candidates and logs them
to the `decisions` table (mode="paper", executed_position left null).
Wiring the output to execution/broker.py is a separate step.

Usage:
    python -m models.screener --feature-set-id v3 --universe
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from backtest.cost_model import round_trip_cost_fraction
from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from data.ingest.universe import resolve_symbols
from execution.exit_levels import ExitLevels, exit_levels_for
from features.build_features import fundamentals_prior_context
from features.quant.volatility import realized_vol
from models.evaluation import cross_sectional_zscore
from models.forecast.ensemble import EnsembleForecastModel
from models.regime.trend_chop_classifier import TREND
from models.train import cross_sectional_feature_columns, feature_columns, load_feature_frame, load_training_frame
from monitoring import reasoning
from risk.sizing import scale_to_full_deployment, target_position_size

logger = logging.getLogger(__name__)

_MODEL_VERSION = "ensemble_v1"

# A trade whose predicted move doesn't even cover getting in and out is a
# guaranteed loser even when the prediction is right. min_abs_return
# therefore floors at the estimated round-trip transaction cost (see
# backtest/cost_model.py — the spread-only minimum without ADV data)
# instead of the old 0.0, which passed literally any nonzero forecast.
DEFAULT_MIN_ABS_RETURN = round_trip_cost_fraction()


@dataclasses.dataclass
class TradeCandidate:
    symbol: str
    side: str  # "long" | "short"
    predicted_return: float
    direction_agreement: float
    conviction_score: float
    target_position_pct: float
    # The take-profit/stop-loss pair this pick is proposed with, sized to
    # this stock rather than shared with every other position. None only
    # for candidates built outside run_screen; attach_exit_levels fills it.
    exit_levels: ExitLevels | None = None
    # Plain-English phase 2-4 reasoning (see monitoring/reasoning.py), populated
    # by run_screen for both strategies (phase-4 wording follows the strategy).
    # trading_loop.py merges in phases 1/5/6/7 once execution facts exist.
    # None only for candidates built outside run_screen.
    reasoning: list[dict] | None = None


def load_latest_features(
    feature_set_id: str, symbols: list[str], target_mode: str | None = None
) -> pd.DataFrame:
    """
    The most recent feature row per symbol — what gets scored "as of today".

    In relative mode the features are cross-sectionally z-scored across the
    snapshot, exactly as load_training_frame z-scores each training date
    (same NON_CROSS_SECTIONAL_FEATURES exception — see its docstring).
    That has to match: a model trained on per-date z-scores and then scored
    on raw feature levels would be reading a completely different scale from
    the one it learned on, and would produce confident nonsense.

    The z-score is taken across the whole snapshot rather than per ts. The
    rows are the newest per symbol and so are nearly all the same date;
    grouping by ts would give a stale symbol a group of one, whose z-score
    is 0.0 by definition — no information at all, rather than its position
    against today's peers.
    """
    target_mode = settings.target_mode if target_mode is None else target_mode
    df = load_feature_frame(feature_set_id, symbols)
    latest = df.sort_values("ts").groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
    if target_mode == "absolute":
        return latest

    cols = cross_sectional_feature_columns(latest)
    snapshot = latest.assign(_as_of="snapshot")
    return cross_sectional_zscore(snapshot, cols, date_col="_as_of").drop(columns="_as_of")


def build_correlation_matrix(prices: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """
    Trailing daily-return correlation matrix (symbols x symbols) from raw
    OHLCV prices, for risk.sizing.correlation_adjusted_size. `prices`:
    columns [symbol, ts, close].
    """
    pivot = prices.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    returns = pivot.pct_change().tail(lookback_days)
    return returns.corr()


def score_universe(
    ensemble: EnsembleForecastModel,
    latest_features: pd.DataFrame,
    feature_cols: list[str],
    min_abs_return: float = DEFAULT_MIN_ABS_RETURN,
    raw_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    latest_features: one row per symbol (see load_latest_features), with a
    'symbol' column plus all of `feature_cols` (missing ones are fine —
    LightGBM handles NaN features natively).
    Returns: symbol, predicted_return, direction_agreement, conviction_score,
    donchian_breakout_20, trend_pullback_score, confident.

    Two bars decide "confident": the predicted move must be bigger than
    what the round trip costs (a prediction smaller than the cost of acting
    on it is a guaranteed loser even when its direction is right), AND —
    when donchian_breakout_20 is present in the data — price must actually
    be breaking a real 20-day high/low in the SAME direction as the
    prediction: a bullish call needs a fresh 20-day high, a bearish one a
    fresh 20-day low. A feature set built before this feature existed (no
    donchian_breakout_20 column at all) falls back to the cost hurdle alone,
    same as before this existed — this is a stricter bar layered on top,
    not a replacement that could silently break an older feature set.

    `raw_features`: donchian_breakout_20 is read from here when given,
    falling back to latest_features otherwise. In TARGET_MODE=relative,
    latest_features is cross-sectionally z-scored (see load_latest_features)
    — a z-scored breakout flag no longer means "this stock broke out today",
    only "more or less than the day's universe average", which could read a
    stock with NO breakout at all as positive on a day most of the universe
    happens to be breaking out. raw_features (run_screen_with_scores passes
    its unscaled `raw_latest`) is the actual, un-relativized signal. Matched
    to `latest_features` by symbol, not row position, since the two may come
    from separate load_latest_features calls.

    There used to be a second bar here — at least 80% of the ensemble
    agreeing on direction — and it was measured to carry no information.
    The members are near-clones, so ~96% of predictions passed it, accuracy
    on the rows it called "confident" matched accuracy overall, and making
    the members structurally diverse did not change that. A filter that
    admits almost everything and predicts nothing is not a safeguard; it is
    a number that makes a system look more careful than it is. The
    Donchian bar above replaces it with an actual price-action condition
    instead of a model-internal number that didn't predict being right.

    direction_agreement is still computed and recorded, because it is
    evidence about the ensemble worth keeping. It plays no part in
    `confident` above.
    """
    empty = pd.DataFrame(
        columns=[
            "symbol", "predicted_return", "direction_agreement", "conviction_score",
            "donchian_breakout_20", "trend_pullback_score", "confident",
        ]
    )
    if latest_features.empty:
        return empty

    X = latest_features.reindex(columns=feature_cols)
    preds = ensemble.predict(X)

    result = pd.DataFrame(
        {
            "symbol": latest_features["symbol"].to_numpy(),
            "predicted_return": preds["mean_prediction"].to_numpy(),
            "direction_agreement": preds["direction_agreement"].to_numpy(),
        }
    )

    signal_source = raw_features if raw_features is not None else latest_features
    has_breakout_signal = (
        signal_source is not None and "symbol" in signal_source.columns and "donchian_breakout_20" in signal_source.columns
    )
    if has_breakout_signal:
        breakout_by_symbol = signal_source.set_index("symbol")["donchian_breakout_20"]
        result["donchian_breakout_20"] = result["symbol"].map(breakout_by_symbol).fillna(0.0)
    else:
        result["donchian_breakout_20"] = np.nan

    if signal_source is not None and "symbol" in signal_source.columns and "mom_pullback_20_5" in signal_source.columns:
        pullback_by_symbol = signal_source.set_index("symbol")["mom_pullback_20_5"]
        result["trend_pullback_score"] = result["symbol"].map(pullback_by_symbol).fillna(0.0)
    else:
        result["trend_pullback_score"] = 0.0

    # Rank on the size of the predicted move alone. Multiplying by
    # agreement mixed a noise term into the ordering, so which pick got the
    # most capital was partly decided by a number that means nothing.
    result["conviction_score"] = result["predicted_return"].abs()

    breakout_aligned = (
        np.sign(result["predicted_return"]) == result["donchian_breakout_20"] if has_breakout_signal else True
    )
    result["confident"] = (result["predicted_return"].abs() >= min_abs_return) & breakout_aligned
    return result.sort_values("conviction_score", ascending=False).reset_index(drop=True)


def apply_macro_sector_block(
    scored: pd.DataFrame,
    macro_mkt_sentiment: float | None,
    macro_sector_sentiment_by_symbol: dict[str, float] | None,
) -> pd.DataFrame:
    """
    Hard block, not a ranking nudge: when the market-wide sentiment AND a
    stock's own sector sentiment are BOTH negative, a long candidate in
    that stock is never confident, whatever the model itself predicted.
    Symmetric for the mirror case (both positive -> no shorts). Sets
    `confident` to False on the blocked rows rather than dropping them, so
    they still show up in `scored` (counts, reasoning, debugging) exactly
    like any other row that failed to clear the bar — select_trades/
    select_concentrated_trades already only ever pick from
    scored.loc[scored["confident"]].

    Either sentiment being missing/NaN, or the two disagreeing in sign,
    leaves `confident` untouched — this only fires when the backdrop is
    genuinely aligned against the trade's own direction, never guessed
    from incomplete data.

    This is a deliberate exception to how every other macro/sector signal
    in this codebase works (see features/qualitative/macro_sentiment.py's
    own module docstring: those are plain model features the ensemble
    weighs on its own, by design, not hand-coded rules). This one is an
    explicit override: being long against a market AND sector that are
    both genuinely rolling over — or short against both genuinely
    rallying — is a setup this system should never take regardless of
    what one stock's own signals say.

    macro_mkt_sentiment/macro_sector_sentiment_by_symbol must be the RAW
    (un-z-scored) values — see score_universe's own raw_features docstring
    for why: in TARGET_MODE=relative, macro_mkt_sentiment is broadcast
    identically to every symbol on a given date, so z-scoring it across
    the snapshot divides by a zero cross-sectional std and erases it
    entirely. Callers should read these from run_screen_with_scores'
    unscaled `raw_latest`, never from the (possibly z-scored) `latest`
    passed to score_universe.
    """
    if (
        scored.empty
        or macro_mkt_sentiment is None
        or pd.isna(macro_mkt_sentiment)
        or not macro_sector_sentiment_by_symbol
    ):
        return scored

    out = scored.copy()
    sector_sentiment = out["symbol"].map(macro_sector_sentiment_by_symbol)
    aligned_bearish = (macro_mkt_sentiment < 0) & (sector_sentiment < 0)
    aligned_bullish = (macro_mkt_sentiment > 0) & (sector_sentiment > 0)
    blocked = (aligned_bearish & (out["predicted_return"] > 0)) | (aligned_bullish & (out["predicted_return"] < 0))
    out.loc[blocked.fillna(False), "confident"] = False
    return out


def apply_trend_pullback_boost(
    scored: pd.DataFrame,
    boost_pct: float = 0.0,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """
    A ranking tie-breaker (never a sizing change — sizing still runs off the
    model's own conviction_score untouched) for a candidate whose
    trend_pullback_score (see features/quant/momentum.py, and score_universe
    above for where this column comes from) agrees with the direction the
    model predicted: an uptrend pulling back that the model ALSO calls a
    buy, or a downtrend bouncing that it ALSO calls a sell. This is a boost
    on top of the ML forecast, not a substitute for it — a candidate the
    model doesn't like at all gets no boost no matter how textbook the
    pullback looks, and nothing here can select a candidate score_universe
    didn't already mark confident.

    Call this AFTER apply_short_preference (or standalone, if the short
    preference isn't in use) — it boosts whatever `rank_score` already holds,
    defaulting to conviction_score when apply_short_preference hasn't run.

    boost_pct=0.0 (the default) makes this a no-op, matching every existing
    caller that predates this.
    """
    if scored.empty or boost_pct <= 0.0 or "trend_pullback_score" not in scored.columns:
        return scored

    out = scored.copy()
    if "rank_score" not in out.columns:
        out["rank_score"] = out["conviction_score"]

    aligned = (np.sign(out["predicted_return"]) == np.sign(out["trend_pullback_score"])) & (
        out["trend_pullback_score"].abs() >= min_score
    )
    out.loc[aligned, "rank_score"] = out.loc[aligned, "rank_score"] * (1.0 + boost_pct)
    return out


def apply_short_preference(
    scored: pd.DataFrame,
    vol_by_symbol: dict[str, float],
    horizon_days: int,
    penalty: float = 0.0,
    low_risk_stop_loss_pct: float | None = None,
) -> pd.DataFrame:
    """
    Adds a `rank_score` column used to decide WHICH candidates make the
    shortlist (see select_trades/select_concentrated_trades's `rank_score_col`)
    — never how big a selected position is sized, which still runs off the
    real predicted_return/conviction_score untouched.

    Every long keeps rank_score == conviction_score. A short is handicapped
    by `penalty` (rank_score = conviction_score * (1 - penalty)) UNLESS its
    own derived stop-loss — see execution/exit_levels.py, sized to that
    specific stock's volatility, not a blanket number — is at or below
    `low_risk_stop_loss_pct`, in which case it competes on raw conviction
    like a long would. That carve-out is what turns "slight preference for
    longs" into "unless it's a confident short with contained downside":
    the penalty only ever protects longs from a marginal short outbidding
    them on a coin-flip-sized edge, it never blocks a short that is both
    genuinely confident AND backed by a stock that doesn't move much.

    `penalty=0.0` (or `low_risk_stop_loss_pct=None`) makes this a no-op —
    rank_score equals conviction_score for every row, so callers that don't
    care about the preference (e.g. every existing caller before this was
    added) see identical ranking to before.
    """
    if scored.empty:
        return scored.assign(rank_score=pd.Series(dtype=float))

    out = scored.copy()
    if penalty <= 0.0 or low_risk_stop_loss_pct is None:
        out["rank_score"] = out["conviction_score"]
        return out

    def _rank_score(row: pd.Series) -> float:
        if row["predicted_return"] >= 0:
            return float(row["conviction_score"])
        stop_loss = exit_levels_for(
            predicted_return=row["predicted_return"],
            daily_volatility=vol_by_symbol.get(row["symbol"]),
            horizon_days=horizon_days,
        ).stop_loss_pct
        if stop_loss <= low_risk_stop_loss_pct:
            return float(row["conviction_score"])  # confident + contained downside: no handicap
        return float(row["conviction_score"]) * (1.0 - penalty)

    out["rank_score"] = out.apply(_rank_score, axis=1)
    return out


def select_trades(
    scored: pd.DataFrame,
    regime: str,
    forecast_scale: float,
    max_position_pct: float,
    max_short_position_pct: float,
    max_correlated_exposure_pct: float,
    correlation_matrix: pd.DataFrame,
    top_k: int = 10,
    current_positions: dict[str, float] | None = None,
    is_shortable_fn: Callable[[str], bool] | None = None,
    allow_shorts: bool = True,
    rank_score_col: str | None = None,
) -> list[TradeCandidate]:
    """
    The diversified-book path — the default strategy (STRATEGY_MODE=
    diversified). See select_concentrated_trades below for the 2-trade
    alternative; run_screen dispatches between them on settings.strategy_mode.

    Filters `scored` down to the top_k confident candidates and sizes each
    via risk.sizing.target_position_size — reused unchanged; it already
    handles signed forecasts, regime damping, and correlation caps
    generically, nothing here is symbol- or ETF-specific.

    `current_positions` seeds the correlated-exposure check with whatever is
    already held, but the correlation cap has to bind across THIS call's own
    picks too, not just against that starting book: as each candidate is
    sized, its size is folded into a running copy of `current_positions` so
    the next candidate in the same loop is checked against "already held
    plus everything sized so far this call". Without that, four candidates
    that are all pairwise correlated but individually within the cap versus
    an (unchanged) external book can each get sized independently and land
    the combined book well past `max_correlated_exposure_pct`. The caller's
    own dict is never mutated — sizing works on a local copy.

    Short candidates that fail `is_shortable_fn` (when given — pass e.g.
    execution.broker_alpaca.AlpacaBroker.is_shortable) are dropped rather
    than resized to zero, so the caller can see which symbols got skipped.
    Pass None to skip the check entirely (e.g. offline/backtest scoring
    with no live broker connection).

    allow_shorts=False drops every short candidate the same way, before
    sizing, and lets the next long candidate take the freed top_k slot.
    run_screen passes settings.allow_shorts (default False — shorts lost
    -1.069% per trade in the walk-forward). The parameter defaults to True
    here because this function is the mechanism; the policy lives in config.

    `rank_score_col`: which column decides WHO makes the top_k cut and in
    what order — defaults to "conviction_score" (the original behavior,
    unchanged) when None or when the named column isn't present. run_screen
    passes "rank_score" (see apply_short_preference) so a long/short
    ranking preference can apply without touching how a selected
    candidate is actually sized below.
    """
    # Copied, not aliased: this dict is mutated below as each candidate is
    # sized, and the caller's own current_positions must never see that.
    current_positions = dict(current_positions) if current_positions else {}
    sort_col = rank_score_col if rank_score_col and rank_score_col in scored.columns else "conviction_score"
    confident = scored.loc[scored["confident"]].sort_values(sort_col, ascending=False)

    candidates: list[TradeCandidate] = []
    for _, row in confident.iterrows():
        if len(candidates) >= top_k:
            break

        symbol = row["symbol"]
        forecast = row["predicted_return"]
        side = "long" if forecast >= 0 else "short"

        if side == "short" and not allow_shorts:
            continue
        if side == "short" and is_shortable_fn is not None and not is_shortable_fn(symbol):
            continue

        size = target_position_size(
            forecast=forecast,
            forecast_scale=forecast_scale,
            regime=regime,
            symbol=symbol,
            current_positions=current_positions,
            correlation_matrix=correlation_matrix,
            max_position_pct=max_position_pct,
            max_correlated_exposure_pct=max_correlated_exposure_pct,
            max_short_position_pct=max_short_position_pct,
        )
        if abs(size) < 1e-9 or math.isnan(size):
            continue

        candidates.append(
            TradeCandidate(
                symbol=symbol,
                side=side,
                predicted_return=float(forecast),
                direction_agreement=float(row["direction_agreement"]),
                conviction_score=float(row["conviction_score"]),
                target_position_pct=float(size),
            )
        )
        # Fold this pick into the running exposure so the NEXT candidate
        # sized in this same call is checked against it too (see docstring).
        current_positions[symbol] = size

    return candidates


def _make_candidate(row: pd.Series, weight: float, total_deploy_pct: float) -> TradeCandidate:
    forecast = row["predicted_return"]
    side = "long" if forecast >= 0 else "short"
    sign = 1.0 if forecast >= 0 else -1.0
    return TradeCandidate(
        symbol=row["symbol"],
        side=side,
        predicted_return=float(forecast),
        direction_agreement=float(row["direction_agreement"]),
        conviction_score=float(row["conviction_score"]),
        target_position_pct=float(sign * weight * total_deploy_pct),
    )


def _bounded_conviction_weights(
    scores: list[float],
    max_leg_pct: float,
    min_leg_floor_fraction: float,
) -> list[float]:
    """
    Splits 1.0 across len(scores) legs proportional to conviction (each
    score's share of the total), then bounds every leg two ways:

      - never above `max_leg_pct` — one pick can't swallow the whole
        allocation regardless of how many legs there are or how lopsided
        the confidence gap is.
      - never below `min_leg_floor_fraction` of what an EQUAL split would
        have given that leg (1/n) — e.g. with 3 legs and
        min_leg_floor_fraction=0.6, no leg is squeezed below
        0.6 * 1/3 = 20%, so an also-ran third pick still gets a real,
        tradeable slice instead of a token sliver. Expressed relative to
        the equal share (not an absolute percentage) so the floor sums to
        at most `min_leg_floor_fraction` (<= 1 by construction) and stays
        feasible for any leg count, rather than being tuned for one
        specific n.

    Two-phase water-filling, run against a shrinking `target` (starts at
    1.0) rather than fixing every violating leg to its bound in one shot —
    fixing them all at once was the bug: a leg pinned to `max_leg_pct` and
    another pinned to `min_leg_pct` in the SAME pass can together claim more
    than what's actually left, since neither pin checks what the other one
    just took.

    Each outer pass:

      1. Cap-only water-fill `target` across the still-free legs (identical
         in spirit to risk.sizing.scale_to_full_deployment's own
         water-filling): split proportional to conviction, pin any leg
         whose share would exceed `max_leg_pct` there, and re-split the
         leftover among the rest — repeating until nobody still free
         exceeds the cap. This alone can never push a leg over its cap or
         the running total over `target` (a pinned leg's cap is always less
         than the share it would otherwise have gotten, so pinning it only
         ever gives back budget, never take more).
      2. Check that result against the floor. Legs under `min_leg_pct` are
         pinned there instead — which shrinks `target` for whoever is still
         free — and the pass repeats. Re-deriving the cap-only split against
         the smaller `target` (rather than reusing the stale pre-floor
         shares from step 1) is what keeps the combined total from ever
         exceeding 1.0.

    Terminates in at most n passes (each non-final pass pins at least one
    more leg permanently). If a pass produces no floor violations, every
    still-free leg's step-1 share is final and the loop stops.

    A leg can still end up under its floor if honoring every floor would
    require exceeding `max_leg_pct` or the 1.0 total — both of those are the
    hard bounds; the floor is not, once it stops being possible to hit it
    without violating them, the shortfall is logged (see FullDeploymentResult
    /scale_to_full_deployment for the same "reached_target=False, log why"
    convention this mirrors) rather than silently under- or over-deploying.
    """
    n = len(scores)
    if n == 1:
        return [1.0]

    equal_share = 1.0 / n
    min_leg_pct = min_leg_floor_fraction * equal_share
    clipped_scores = [max(s, 0.0) for s in scores]

    weights = [0.0] * n
    free_idx = list(range(n))
    target = 1.0

    for _ in range(n):
        if not free_idx:
            break

        # Phase 1: cap-only water-fill `target` across the free legs.
        capped: dict[int, float] = {}
        active = set(free_idx)
        remaining = target
        while active:
            active_total = sum(clipped_scores[i] for i in active)
            proposal = (
                {i: remaining / len(active) for i in active}
                if active_total <= 0
                else {i: remaining * clipped_scores[i] / active_total for i in active}
            )
            newly_capped = [i for i in active if proposal[i] > max_leg_pct + 1e-9]
            if not newly_capped:
                capped.update(proposal)
                break
            for i in newly_capped:
                capped[i] = max_leg_pct
                remaining -= max_leg_pct
                active.discard(i)

        # Phase 2: pin anyone that leaves under the floor and shrink target
        # for the next pass; otherwise this allocation is final.
        below_floor = [i for i in free_idx if capped.get(i, 0.0) < min_leg_pct - 1e-9]
        if not below_floor:
            for i in free_idx:
                weights[i] = capped[i]
            free_idx = []
            break

        for i in below_floor:
            # min() against max_leg_pct: if min_leg_pct itself exceeds
            # max_leg_pct (a floor/cap combination that leaves no room
            # between them -- not reachable with today's settings, but
            # nothing validates that it can't be configured that way),
            # pinning to the unclamped floor would push this leg's weight
            # above the hard cap the docstring above promises "never above
            # max_leg_pct ... regardless." Clamping here keeps that
            # guarantee true unconditionally; the shortfall this creates
            # still surfaces via the total<1.0 warning below.
            pinned = min(min_leg_pct, max_leg_pct)
            weights[i] = pinned
            target -= pinned
        free_idx = [i for i in free_idx if i not in below_floor]
    else:
        return [equal_share] * n

    total = sum(weights)
    if total < 1.0 - 1e-9:
        logger.warning(
            "Concentrated split under-deployed: max_leg_pct=%.0f%% and min_leg_floor_fraction="
            "%.0f%% can't both be honored for this conviction spread, so only %.1f%% of the "
            "intended 100%% was allocated across %d leg(s) rather than breaching either bound.",
            max_leg_pct * 100, min_leg_floor_fraction * 100, total * 100, n,
        )
    return weights


def select_concentrated_trades(
    scored: pd.DataFrame,
    max_leg_pct: float,
    min_leg_floor_fraction: float,
    total_deploy_pct: float = 1.0,
    is_shortable_fn: Callable[[str], bool] | None = None,
    allow_shorts: bool = True,
    rank_score_col: str | None = None,
    max_positions: int = 2,
) -> list[TradeCandidate]:
    """
    The concentrated strategy (STRATEGY_MODE=concentrated): put all
    deployable capital into a small, high-conviction book of at most
    `max_positions` names instead of spreading across many — the point is
    to make a small number of high-confidence bets, not to track the
    market with a diversified book.

    Takes the top `max_positions` confident candidates by conviction (or
    fewer if fewer clear the confidence bar — see DEFAULT_MIN_ABS_RETURN;
    this never forces a weaker candidate in just to hit a target count).
    The split across however many legs that ends up being is weighted by
    each pick's relative conviction_score, bounded by
    _bounded_conviction_weights so no leg swallows the book and none gets
    squeezed to a token sliver. If only one candidate clears the bar, it
    gets the full total_deploy_pct alone. If none do, returns [].

    allow_shorts=False skips short candidates entirely when filling the
    legs, so a rejected short frees its slot for the next long rather than
    leaving the book under-deployed. See settings.allow_shorts for why it
    defaults off in production.

    `rank_score_col`: which column picks the legs and their order —
    defaults to "conviction_score" (unchanged original behavior) when None
    or when the named column isn't present; run_screen passes "rank_score"
    (see apply_short_preference). The eventual split WEIGHT below still
    uses each pick's real conviction_score, never the ranking column — the
    preference only affects who gets picked, not how big their leg is once
    picked.

    `total_deploy_pct` is the fraction of portfolio value to put to work in
    total, across every leg combined — pass less than 1.0 for e.g.
    execution/contradiction_monitor.py's mid-week reactivation, which only
    has the freed slice of the book to redeploy.
    """
    sort_col = rank_score_col if rank_score_col and rank_score_col in scored.columns else "conviction_score"
    confident = scored.loc[scored["confident"]].sort_values(sort_col, ascending=False)

    picks: list[pd.Series] = []
    for _, row in confident.iterrows():
        if len(picks) >= max_positions:
            break
        is_short = row["predicted_return"] < 0
        if is_short and not allow_shorts:
            continue
        if is_short and is_shortable_fn is not None and not is_shortable_fn(row["symbol"]):
            continue
        picks.append(row)

    if not picks:
        return []

    if len(picks) == 1:
        return [_make_candidate(picks[0], weight=1.0, total_deploy_pct=total_deploy_pct)]

    scores = [max(row["conviction_score"], 0.0) for row in picks]
    weights = _bounded_conviction_weights(scores, max_leg_pct=max_leg_pct, min_leg_floor_fraction=min_leg_floor_fraction)

    return [_make_candidate(row, weight, total_deploy_pct) for row, weight in zip(picks, weights, strict=False)]


def _load_fundamentals_context(symbols: list[str]) -> dict[tuple[str, str], dict]:
    """
    Loads just enough of the raw `fundamentals` history (two filings per
    metric, for these symbols only) for fundamentals_prior_context to give
    _attach_reasoning's narrative something to compare "latest reported X"
    against. Kept as its own function, separate from the pure
    fundamentals_prior_context, purely so tests can mock the I/O here
    without needing a real database — same pattern as load_latest_features.
    """
    if not symbols:
        return {}
    fundamentals = pd.read_sql(
        f"SELECT symbol, ts, metric, value FROM fundamentals WHERE symbol IN ({symbol_in_clause(symbols)})",  # noqa: S608 — symbols validated via symbol_in_clause
        get_engine(),
    )
    return fundamentals_prior_context(fundamentals)


def _explain_features(
    symbols: list[str],
    ensemble: EnsembleForecastModel,
    latest_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    feature_cols: list[str],
    fundamentals_context: dict[tuple[str, str], dict] | None,
    top_n: int = 5,
) -> dict[str, list[dict]]:
    """
    The SHAP core of a symbol's Phase 2 story: its top `top_n` feature
    contributions by |contribution|, each carrying the raw (unscaled)
    display value and any fundamentals trend context. Shared by
    _attach_reasoning (for this cycle's picks) and explain_held_symbols
    (for a position that's being held without being re-picked) so both
    read the exact same attribution logic off the exact same ensemble —
    no second model fit, just a second look at symbols already scored.
    """
    rows = latest_features.set_index("symbol").loc[symbols]
    raw_rows = raw_features.set_index("symbol").loc[symbols]
    X = rows.reindex(columns=feature_cols)
    contributions = ensemble.predict_contributions(X)
    fundamentals_context = fundamentals_context or {}

    out: dict[str, list[dict]] = {}
    for symbol in symbols:
        contrib_row = contributions.loc[symbol].drop("base_value")
        top_feature_names = contrib_row.abs().sort_values(ascending=False).head(top_n).index
        out[symbol] = [
            {
                "feature_name": feat,
                "value": None if pd.isna(raw_rows.loc[symbol, feat]) else float(raw_rows.loc[symbol, feat]),
                "contribution": float(contrib_row[feat]),
                "context": fundamentals_context.get((symbol, feat)),
            }
            for feat in top_feature_names
        ]
    return out


def explain_held_symbols(
    symbols: list[str],
    ensemble: EnsembleForecastModel,
    latest_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    feature_cols: list[str],
    scored: pd.DataFrame,
    regime: str,
) -> dict[str, list[dict]]:
    """
    Phases 2+3 for a position that's held through this cycle's screen
    without being one of the fresh picks -- the same SHAP attribution
    _attach_reasoning gives a fresh candidate, built off the exact same
    ensemble/scored universe this cycle already computed, just for a
    symbol that didn't make the shortlist this time.

    Why this exists: _log_decisions only ever wrote a row for symbols
    that were candidates, closes, or rejections this cycle -- a position
    held with no exit condition fired got no new row at all, so its
    "why this trade" reasoning stayed frozen at whatever it was the last
    time it WAS a fresh pick, however many cycles ago that was. A data
    fix (e.g. the unadjusted-stock-split bug behind an impossible-looking
    "+416% in 20 days" reading) could land in `features` and still never
    reach a held position's card, because nothing ever asked the model
    to look at it again. This closes that gap: trading_loop.py calls it
    for every held-and-not-reshortlisted symbol, every cycle.

    Symbols with no feature row this cycle (e.g. delisted, or a gap in
    the day's ingest) are silently omitted rather than raising -- a
    missing fresh read isn't a reason to fail the whole cycle; the
    position's existing decisions row (however old) still stands as the
    record of what's currently displayed.
    """
    known = set(latest_features["symbol"]) if not latest_features.empty else set()
    usable = [s for s in symbols if s in known]
    if not usable:
        return {}

    fundamentals_context = _load_fundamentals_context(usable)
    per_symbol_top_features = _explain_features(usable, ensemble, latest_features, raw_features, feature_cols, fundamentals_context)
    predicted_by_symbol = dict(zip(scored["symbol"], scored["predicted_return"], strict=False))
    conviction_by_symbol = dict(zip(scored["symbol"], scored["conviction_score"], strict=False))

    return {
        symbol: [
            reasoning.phase_signals(regime, per_symbol_top_features[symbol]),
            reasoning.phase_forecast(predicted_by_symbol.get(symbol, 0.0), conviction_by_symbol.get(symbol, 0.0)),
        ]
        for symbol in usable
    }


def _attach_reasoning(
    candidates: list[TradeCandidate],
    ensemble: EnsembleForecastModel,
    latest_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    feature_cols: list[str],
    scored: pd.DataFrame,
    regime: str,
    max_leg_pct: float,
    min_leg_floor_fraction: float,
    top_n: int = 5,
    strategy: str = "concentrated",
    top_k: int | None = None,
    fundamentals_context: dict[tuple[str, str], dict] | None = None,
) -> None:
    """
    Mutates each candidate in place, attaching plain-English reasoning for
    phases 2-4 of the decision (see monitoring/reasoning.py for the full
    7-phase model; phases 1/5/6/7 get merged in later by trading_loop.py,
    once execution/reconciliation facts exist). Phase 2/3 are built from
    the top `top_n` SHAP feature contributions (LightGBM pred_contrib) —
    genuine per-prediction attribution, not just global feature importance.
    Phase 4 wording follows the strategy: the concentrated split story for
    the 2-trade mode, the top-k book story for the diversified default.

    `latest_features` vs `raw_features`: in TARGET_MODE=relative,
    `latest_features` is cross-sectionally z-scored (see
    load_latest_features) — that's what the model actually scored on, so
    it's what predict_contributions needs to match training. But showing
    a z-score to a human as if it were the real number is exactly how a
    mom_ret_20d z-score of -0.575 (mildly below the day's average) got
    narrated as "sold off sharply (-57.5%)": the formatter in
    monitoring/reasoning.py assumes a literal percent/ATR-in-points/ratio,
    not a standard-deviation count. `raw_features` (always unscaled,
    straight from the features table) supplies the value actually shown
    per top feature; only the model-facing X below uses `latest_features`.

    `fundamentals_context`: {(symbol, feature_name): {"prior_value",
    "pct_change"}} for the fund_*_latest features (see
    _load_fundamentals_context / fundamentals_prior_context) — lets the
    narrative say a reported number moved, instead of stating an
    incomparable level in isolation.
    """
    if not candidates:
        return

    symbols = [c.symbol for c in candidates]
    per_symbol_top_features = _explain_features(
        symbols, ensemble, latest_features, raw_features, feature_cols, fundamentals_context, top_n=top_n
    )
    n_confident = int(scored["confident"].sum())

    for candidate in candidates:
        top_features = per_symbol_top_features[candidate.symbol]
        if strategy == "diversified":
            phase4 = reasoning.phase_selection_diversified(
                candidate.symbol,
                candidate.side,
                candidate.target_position_pct,
                n_confident,
                len(candidates),
                top_k or len(candidates),
            )
        else:
            # reasoning.phase_selection still speaks in absolute per-leg
            # percentages (its own signature/tests are unchanged) — convert
            # the floor-fraction setting to the actual bound for THIS set's
            # leg count right here, rather than pushing the floor-fraction
            # abstraction into the reasoning layer.
            n_selected = len(candidates)
            effective_min_leg_pct = min_leg_floor_fraction * (1.0 / n_selected) if n_selected > 1 else min_leg_floor_fraction
            phase4 = reasoning.phase_selection(
                candidate.symbol,
                candidate.side,
                candidate.target_position_pct,
                n_confident,
                n_selected,
                max_leg_pct,
                effective_min_leg_pct,
            )
        candidate.reasoning = [
            reasoning.phase_signals(regime, top_features),
            reasoning.phase_forecast(candidate.predicted_return, candidate.conviction_score),
            phase4,
        ]


def daily_volatility(prices: pd.DataFrame, window: int = 20) -> dict[str, float]:
    """
    symbol -> standard deviation of that stock's daily returns.

    Not annualized: exit levels are expressed against the forecast horizon
    in trading days, and converting to years and back would only add a
    constant and a chance to get it wrong. Symbols without enough history
    are absent rather than zero — a missing measurement has to stay
    distinguishable from a genuinely motionless stock, because the two
    call for opposite fallbacks.
    """
    if prices.empty:
        return {}
    out: dict[str, float] = {}
    for symbol, group in prices.sort_values("ts").groupby("symbol"):
        if len(group) <= window:
            continue
        vol = realized_vol(group["close"], window=window, annualize=False).iloc[-1]
        if pd.notna(vol) and vol > 0:
            out[str(symbol)] = float(vol)
    return out


def attach_exit_levels(candidates: list[TradeCandidate], vol_by_symbol: dict[str, float]) -> None:
    """
    Give every candidate the levels it will be proposed, approved and later
    judged against. Mutates in place, like _attach_reasoning.
    """
    for candidate in candidates:
        candidate.exit_levels = exit_levels_for(
            predicted_return=candidate.predicted_return,
            daily_volatility=vol_by_symbol.get(candidate.symbol),
        )


@dataclasses.dataclass
class ScreenResult:
    """
    A screen's full output: the shortlist AND the whole scored universe.
    The weekly cycle's hold rules (execution/hold_rules.py) need a fresh
    prediction for every HELD symbol, including ones that didn't make the
    shortlist — "has this position's predicted return flipped sign?" can't
    be answered from the top-k candidates alone.
    """

    candidates: list[TradeCandidate]
    scored: pd.DataFrame  # see score_universe: symbol, predicted_return, direction_agreement, …
    # Everything explain_held needs to re-run this cycle's already-fitted
    # ensemble against a symbol that wasn't one of the picks, without
    # fitting anything a second time. Not meant for other callers — reach
    # for `candidates`/`scored` instead unless you specifically need to
    # explain a symbol outside the shortlist.
    _ensemble: EnsembleForecastModel | None = dataclasses.field(default=None, repr=False)
    _latest_features: pd.DataFrame | None = dataclasses.field(default=None, repr=False)
    _raw_features: pd.DataFrame | None = dataclasses.field(default=None, repr=False)
    _feature_cols: list[str] | None = dataclasses.field(default=None, repr=False)

    def predicted_return_by_symbol(self) -> dict[str, float]:
        if self.scored.empty:
            return {}
        return dict(zip(self.scored["symbol"], self.scored["predicted_return"], strict=False))

    def explain_held(self, symbols: list[str], regime: str) -> dict[str, list[dict]]:
        """See models.screener.explain_held_symbols."""
        if not symbols or self._ensemble is None:
            return {}
        return explain_held_symbols(
            symbols, self._ensemble, self._latest_features, self._raw_features, self._feature_cols, self.scored, regime,
        )


def run_screen(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int | None = None,
    n_ensemble_models: int = 5,
    min_abs_return: float = DEFAULT_MIN_ABS_RETURN,
    regime: str = TREND,
    is_shortable_fn: Callable[[str], bool] | None = None,
    total_deploy_pct: float = 1.0,
    max_positions_override: int | None = None,
    current_positions: dict[str, float] | None = None,
) -> list[TradeCandidate]:
    """The shortlist-only view of run_screen_with_scores — see ScreenResult for who needs more."""
    return run_screen_with_scores(
        feature_set_id,
        symbols,
        target_horizon_days=target_horizon_days,
        n_ensemble_models=n_ensemble_models,
        min_abs_return=min_abs_return,
        regime=regime,
        is_shortable_fn=is_shortable_fn,
        total_deploy_pct=total_deploy_pct,
        max_positions_override=max_positions_override,
        current_positions=current_positions,
    ).candidates


def run_screen_with_scores(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int | None = None,
    n_ensemble_models: int = 5,
    min_abs_return: float = DEFAULT_MIN_ABS_RETURN,
    regime: str = TREND,
    is_shortable_fn: Callable[[str], bool] | None = None,
    total_deploy_pct: float = 1.0,
    max_positions_override: int | None = None,
    current_positions: dict[str, float] | None = None,
) -> ScreenResult:
    """
    Trains a fresh ensemble on all available history, scores today's
    snapshot, and sizes `total_deploy_pct` of capital by whichever strategy
    STRATEGY_MODE selects: the diversified top-k book (default, sized under
    the conservative caps in risk/sizing.py) or the concentrated small book
    (select_concentrated_trades, at most settings.max_concentrated_positions
    names).

    `current_positions`: what's already held, as {symbol: signed fraction of
    portfolio value} — the same units target_position_size/
    correlation_adjusted_size expect. Only consulted in diversified mode, to
    seed the correlated-exposure cap with the real starting book instead of
    an empty one (see select_trades). Defaults to None (treated as {}) for
    callers with no broker/portfolio context to draw it from — offline
    scoring, backtests, and any caller genuinely starting from cash.

    `total_deploy_pct` defaults to 1.0 (the normal weekly-cycle behavior --
    100% of the book). execution/contradiction_monitor.py's mid-week
    reactivation passes a smaller value: the fraction of capital a
    contradiction-close just freed up, so it can redeploy only that slice
    via this exact same selection logic rather than reshuffling the whole
    book. Regime is still used to gate which candidates clear the confidence
    bar upstream, but no longer dampens total deployment on top of that (see
    git history -- the old chop-dampening was leveraged-ETF-specific decay
    protection that doesn't apply to plain equity/short positions).

    `max_positions_override`: only consulted in concentrated mode, in place
    of settings.max_concentrated_positions. contradiction_monitor.py's
    reactivation passes the actual number of open slots left (max positions
    minus what's currently held) so a mid-week refill tops the book back up
    to the target count instead of adding a fresh 2-3 names on top of
    whatever's already held.
    """
    # None -> the configured horizon (TARGET_HORIZON_DAYS), so the screener
    # always trains on the same forward-return definition the evaluation
    # harness graded, rather than a hardcoded 5.
    if target_horizon_days is None:
        target_horizon_days = settings.target_horizon_days
    train_df = load_training_frame(feature_set_id, symbols, target_horizon_days)
    feature_cols = feature_columns(train_df)

    # Fitted on `target` — the absolute forward return, or the
    # cross-sectional excess over the same-day universe, per TARGET_MODE.
    # In relative mode a prediction reads "expected to beat the equal-weight
    # universe by this much over the horizon", not "expected to rise this
    # much", and the confidence thresholds downstream are interpreted in
    # those units.
    ensemble = EnsembleForecastModel(n_models=n_ensemble_models)
    ensemble.fit(train_df[feature_cols], train_df["target"])

    latest = load_latest_features(feature_set_id, symbols)
    # _attach_reasoning needs the unscaled values for its human-readable
    # narrative — reuse `latest` when it's already unscaled (target_mode
    # "absolute"), otherwise fetch it separately rather than reuse the
    # cross-sectionally z-scored `latest` for display (see its docstring).
    raw_latest = latest if settings.target_mode == "absolute" else load_latest_features(
        feature_set_id, symbols, target_mode="absolute"
    )
    scored = score_universe(ensemble, latest, feature_cols, min_abs_return, raw_features=raw_latest)

    if settings.enable_macro_sector_block:
        macro_mkt_sentiment = None
        macro_sector_sentiment_by_symbol: dict[str, float] = {}
        if "macro_mkt_sentiment" in raw_latest.columns:
            mkt_values = raw_latest["macro_mkt_sentiment"].dropna()
            macro_mkt_sentiment = float(mkt_values.iloc[0]) if not mkt_values.empty else None
        if "macro_sector_sentiment" in raw_latest.columns:
            macro_sector_sentiment_by_symbol = raw_latest.set_index("symbol")["macro_sector_sentiment"].to_dict()
        scored = apply_macro_sector_block(scored, macro_mkt_sentiment, macro_sector_sentiment_by_symbol)

    # Computed once, up front, so it can inform BOTH which candidates get
    # picked (the long/short ranking preference below, when configured) and
    # the exit levels every candidate is proposed with — the same volatility
    # read used for both, rather than recomputed twice and risking drift.
    vol_by_symbol = daily_volatility(train_df[["symbol", "ts", "close"]])
    scored = apply_short_preference(
        scored, vol_by_symbol, target_horizon_days,
        penalty=settings.short_ranking_penalty,
        low_risk_stop_loss_pct=settings.short_low_risk_stop_loss_pct,
    )
    scored = apply_trend_pullback_boost(
        scored,
        boost_pct=settings.trend_pullback_boost_pct,
        min_score=settings.trend_pullback_min_score,
    )

    if not settings.allow_shorts and not scored.empty:
        # Logged rather than silent: the shortlist can look thin for a
        # perfectly good reason, and "the model wanted to short 6 names and
        # wasn't allowed to" is the reason worth knowing. `scored` itself is
        # left untouched — the hold rules read predicted_return for HELD
        # positions and need the real sign to spot a forecast that flipped.
        suppressed = int((scored.loc[scored["confident"], "predicted_return"] < 0).sum())
        if suppressed:
            logger.info(
                "ALLOW_SHORTS is off: %d confident short candidate(s) dropped before sizing.",
                suppressed,
            )

    if settings.strategy_mode == "concentrated":
        max_positions = (
            max_positions_override if max_positions_override is not None else settings.max_concentrated_positions
        )
        candidates = select_concentrated_trades(
            scored,
            max_leg_pct=settings.max_concentrated_position_pct,
            min_leg_floor_fraction=settings.min_concentrated_leg_floor_fraction,
            max_positions=max_positions,
            total_deploy_pct=total_deploy_pct,
            is_shortable_fn=is_shortable_fn,
            allow_shorts=settings.allow_shorts,
            rank_score_col="rank_score",
        )
        if max_positions_override is None and len(candidates) < settings.min_concentrated_positions:
            # Informational only — see settings.min_concentrated_positions:
            # this is a target the screen tries to reach, never a reason to
            # force a candidate with no real edge into the book.
            logger.info(
                "Concentrated screen: only %d of the target minimum %d position(s) cleared the confidence bar this cycle.",
                len(candidates), settings.min_concentrated_positions,
            )
    else:
        # Diversified default: top-k book sized through the full risk
        # pipeline (confidence scaling, regime damping, correlation caps).
        # Scaled by the spread of what the model predicts, not of raw
        # returns: in relative mode the two differ by roughly the market's
        # own volatility, and using the wrong one would systematically
        # mis-size every position.
        forecast_scale = float(train_df["target"].std())
        if math.isnan(forecast_scale):
            # A training frame with <=1 usable row for `target` (e.g. a
            # brand-new feature set, or a universe with almost nothing
            # history-eligible this cycle) makes std() NaN. Every downstream
            # size derived from it would be NaN too — confidence_scaled_size
            # guards against that, but sizing against a scale that means
            # nothing isn't a cycle worth running at all, so skip screening
            # outright rather than shortlist against garbage.
            logger.warning(
                "Diversified screen: forecast_scale (std of the training frame's target) is "
                "NaN — the training frame has too few usable rows this cycle. Skipping "
                "screening rather than sizing candidates against a meaningless scale."
            )
            candidates = []
        else:
            correlation_matrix = build_correlation_matrix(train_df[["symbol", "ts", "close"]])
            candidates = select_trades(
                scored,
                regime=regime,
                forecast_scale=forecast_scale,
                max_position_pct=settings.max_single_position_pct,
                max_short_position_pct=settings.max_short_position_pct,
                max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
                correlation_matrix=correlation_matrix,
                top_k=settings.screener_top_k,
                current_positions=current_positions,
                is_shortable_fn=is_shortable_fn,
                allow_shorts=settings.allow_shorts,
                rank_score_col="rank_score",
            )
            # Reactivation semantics: when only a freed slice of the
            # portfolio is on the table, every size shrinks proportionally.
            if total_deploy_pct < 1.0:
                for candidate in candidates:
                    candidate.target_position_pct *= total_deploy_pct
            if settings.full_deployment:
                candidates = apply_full_deployment(
                    candidates,
                    max_position_pct=settings.max_single_position_pct,
                    max_short_position_pct=settings.max_short_position_pct,
                    target_allocation=total_deploy_pct,
                )

    _attach_reasoning(
        candidates,
        ensemble,
        latest,
        raw_latest,
        feature_cols,
        scored,
        regime,
        settings.max_concentrated_position_pct,
        settings.min_concentrated_leg_floor_fraction,
        strategy=settings.strategy_mode,
        top_k=settings.screener_top_k,
        fundamentals_context=_load_fundamentals_context([c.symbol for c in candidates]),
    )
    # Exit levels come from the same price history the model trained on, so
    # a pick is proposed with levels sized to that stock rather than to the
    # average of every stock. Reuses vol_by_symbol computed above (same
    # numbers apply_short_preference already ranked this candidate against)
    # rather than recomputing it a second time.
    attach_exit_levels(candidates, vol_by_symbol)
    return ScreenResult(
        candidates=candidates,
        scored=scored,
        _ensemble=ensemble,
        _latest_features=latest,
        _raw_features=raw_latest,
        _feature_cols=feature_cols,
    )


def apply_full_deployment(
    candidates: list[TradeCandidate],
    max_position_pct: float,
    max_short_position_pct: float,
    target_allocation: float = 1.0,
) -> list[TradeCandidate]:
    """
    Scale the shortlist up to a fully-allocated book (see
    risk.sizing.scale_to_full_deployment) and report what happened.

    Runs after select_trades, so every per-symbol adjustment — regime
    damping, correlation shrinking, the short cap — has already been applied
    and is preserved proportionally. It changes how much of the portfolio the
    picks take up, never which symbols were picked or which way they lean.
    """
    if not candidates:
        return candidates

    result = scale_to_full_deployment(
        {c.symbol: c.target_position_pct for c in candidates},
        max_position_pct=max_position_pct,
        max_short_position_pct=max_short_position_pct,
        target_allocation=target_allocation,
    )

    for candidate in candidates:
        candidate.target_position_pct = result.sizes[candidate.symbol]

    if result.reached_target:
        logger.info(
            "Full deployment: %.1f%% of the portfolio allocated across %d pick(s).",
            result.deployed_pct * 100, len(candidates),
        )
    else:
        logger.warning("%s", result.reason)
    return candidates


def log_candidates(candidates: list[TradeCandidate], feature_set_id: str, mode: str = "paper", regime: str | None = None) -> int:
    """Writes the shortlist to the decisions table — proposed, not executed (executed_position left null)."""
    if not candidates:
        return 0
    now = dt.datetime.now(tz=dt.UTC)
    rows = [
        {
            "ts": now,
            "symbol": c.symbol,
            "feature_set_id": feature_set_id,
            "model_version": _MODEL_VERSION,
            "forecast": c.predicted_return,
            "regime": regime,
            "target_position": c.target_position_pct,
            "executed_position": None,
            "mode": mode,
            "reasoning": json.dumps(c.reasoning) if c.reasoning is not None else None,
        }
        for c in candidates
    ]
    df = pd.DataFrame(rows)
    engine = get_engine()
    df.to_sql("decisions", engine, if_exists="append", index=False, dtype={"reasoning": JSONB})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a universe and shortlist confident trades.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument(
        "--target-horizon-days", type=int, default=None,
        help=f"Forward-return horizon in trading days (default: TARGET_HORIZON_DAYS = {settings.target_horizon_days}).",
    )
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument(
        "--min-abs-return", type=float, default=DEFAULT_MIN_ABS_RETURN,
        help="Minimum |predicted return| to shortlist. Defaults to the estimated round-trip "
             "transaction cost — a prediction below the cost of trading it is a guaranteed loser.",
    )
    parser.add_argument("--log", action="store_true", help="Write the shortlist to the decisions table.")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    candidates = run_screen(
        args.feature_set_id,
        symbols,
        target_horizon_days=args.target_horizon_days,
        n_ensemble_models=args.n_ensemble_models,
        min_abs_return=args.min_abs_return,
    )

    if not candidates:
        print("No candidates cleared the confidence bar this run.")
        return

    print(f"{'symbol':<8}{'side':<7}{'pred_return':>13}{'agreement':>12}{'target_pct':>12}")
    for c in candidates:
        print(f"{c.symbol:<8}{c.side:<7}{c.predicted_return:>13.4f}{c.direction_agreement:>12.2f}{c.target_position_pct:>12.4f}")

    if args.log:
        n = log_candidates(candidates, args.feature_set_id)
        print(f"\nLogged {n} candidate(s) to decisions (mode=paper, not executed).")


if __name__ == "__main__":
    main()
