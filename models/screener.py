"""
Confidence-ranked equity screener. Scores every symbol in a universe with
the ensemble forecast model, then concentrates all deployable capital into
the top 2 highest-conviction picks — long or short, whichever the data
actually supports — rather than spreading thinly across many names. The
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
from collections.abc import Callable

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from config.settings import settings
from data.ingest.db import get_engine
from data.ingest.universe import resolve_symbols
from models.forecast.ensemble import EnsembleForecastModel
from models.regime.trend_chop_classifier import TREND
from models.train import load_feature_frame, load_training_frame
from risk.sizing import target_position_size

_MODEL_VERSION = "ensemble_v1"


@dataclasses.dataclass
class TradeCandidate:
    symbol: str
    side: str  # "long" | "short"
    predicted_return: float
    direction_agreement: float
    conviction_score: float
    target_position_pct: float
    # Top contributing features (LightGBM pred_contrib — genuine per-prediction
    # Tree SHAP, not just global feature importance), populated by run_screen.
    # None for candidates built via the legacy select_trades() path, which
    # doesn't have an ensemble/feature frame in scope to compute this from.
    reasoning: list[dict] | None = None


def load_latest_features(feature_set_id: str, symbols: list[str]) -> pd.DataFrame:
    """The most recent feature row per symbol — what gets scored "as of today"."""
    df = load_feature_frame(feature_set_id, symbols)
    return df.sort_values("ts").groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


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
    min_direction_agreement: float = 0.8,
    min_abs_return: float = 0.0,
) -> pd.DataFrame:
    """
    latest_features: one row per symbol (see load_latest_features), with a
    'symbol' column plus all of `feature_cols` (missing ones are fine —
    LightGBM handles NaN features natively).
    Returns: symbol, predicted_return, direction_agreement, conviction_score, confident.
    """
    empty = pd.DataFrame(
        columns=["symbol", "predicted_return", "direction_agreement", "conviction_score", "confident"]
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
    result["conviction_score"] = result["direction_agreement"] * result["predicted_return"].abs()
    result["confident"] = (result["direction_agreement"] >= min_direction_agreement) & (
        result["predicted_return"].abs() >= min_abs_return
    )
    return result.sort_values("conviction_score", ascending=False).reset_index(drop=True)


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
) -> list[TradeCandidate]:
    """
    LEGACY diversified-book path — not used by run_screen() anymore (see
    select_concentrated_trades below, the active strategy). Kept as tested,
    working infra in case a broader multi-name book is wanted again later.

    Filters `scored` down to the top_k confident candidates and sizes each
    via risk.sizing.target_position_size — reused unchanged; it already
    handles signed forecasts, regime damping, and correlation caps
    generically, nothing here is symbol- or ETF-specific.

    Short candidates that fail `is_shortable_fn` (when given — pass e.g.
    execution.broker_alpaca.AlpacaBroker.is_shortable) are dropped rather
    than resized to zero, so the caller can see which symbols got skipped.
    Pass None to skip the check entirely (e.g. offline/backtest scoring
    with no live broker connection).
    """
    current_positions = current_positions or {}
    confident = scored.loc[scored["confident"]].sort_values("conviction_score", ascending=False)

    candidates: list[TradeCandidate] = []
    for _, row in confident.iterrows():
        if len(candidates) >= top_k:
            break

        symbol = row["symbol"]
        forecast = row["predicted_return"]
        side = "long" if forecast >= 0 else "short"

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
        if abs(size) < 1e-9:
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


def select_concentrated_trades(
    scored: pd.DataFrame,
    max_leg_pct: float,
    min_leg_pct: float,
    total_deploy_pct: float = 1.0,
    is_shortable_fn: Callable[[str], bool] | None = None,
) -> list[TradeCandidate]:
    """
    The active strategy: concentrate all deployable capital into the top 2
    highest-conviction candidates instead of spreading across many names —
    the point is to make a small number of high-confidence bets, not to
    track the market with a diversified book.

    The split between the two legs is weighted by each pick's relative
    conviction_score (direction_agreement * |predicted_return|), bounded to
    [min_leg_pct, max_leg_pct] of the *dominant* leg so one pick can't
    swallow the whole deployment even at an extreme confidence ratio.
    min_leg_pct should be 1 - max_leg_pct so both bounds are satisfiable
    simultaneously with exactly two candidates.

    If only one candidate clears the confidence bar, it gets the full
    total_deploy_pct alone rather than forcing a weaker second trade just
    to fill the split. If none do, returns [].

    `total_deploy_pct` is the fraction of portfolio value to put to work in
    total, both legs combined — pass less than 1.0 for e.g. regime-based
    damping (see run_screen).
    """
    confident = scored.loc[scored["confident"]].sort_values("conviction_score", ascending=False)

    picks: list[pd.Series] = []
    for _, row in confident.iterrows():
        if len(picks) >= 2:
            break
        is_short = row["predicted_return"] < 0
        if is_short and is_shortable_fn is not None and not is_shortable_fn(row["symbol"]):
            continue
        picks.append(row)

    if not picks:
        return []

    if len(picks) == 1:
        return [_make_candidate(picks[0], weight=1.0, total_deploy_pct=total_deploy_pct)]

    scores = [max(row["conviction_score"], 0.0) for row in picks]
    total_score = sum(scores)
    weights = [0.5, 0.5] if total_score <= 0 else [s / total_score for s in scores]

    if weights[0] > max_leg_pct:
        weights = [max_leg_pct, 1.0 - max_leg_pct]
    elif weights[0] < min_leg_pct:
        weights = [min_leg_pct, 1.0 - min_leg_pct]

    return [_make_candidate(row, weight, total_deploy_pct) for row, weight in zip(picks, weights, strict=False)]


def _attach_reasoning(
    candidates: list[TradeCandidate],
    ensemble: EnsembleForecastModel,
    latest_features: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 5,
) -> None:
    """
    Mutates each candidate in place, attaching the top `top_n` features by
    |contribution| (LightGBM pred_contrib) — this is what actually answers
    "why did the model pick this," not just the final forecast number.
    """
    if not candidates:
        return

    symbols = [c.symbol for c in candidates]
    rows = latest_features.set_index("symbol").loc[symbols]
    X = rows.reindex(columns=feature_cols)
    contributions = ensemble.predict_contributions(X)

    for candidate in candidates:
        contrib_row = contributions.loc[candidate.symbol].drop("base_value")
        top_features = contrib_row.abs().sort_values(ascending=False).head(top_n).index
        candidate.reasoning = [
            {
                "feature_name": feat,
                "value": None if pd.isna(rows.loc[candidate.symbol, feat]) else float(rows.loc[candidate.symbol, feat]),
                "contribution": float(contrib_row[feat]),
            }
            for feat in top_features
        ]


def run_screen(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int = 5,
    n_ensemble_models: int = 5,
    min_direction_agreement: float = 0.8,
    min_abs_return: float = 0.0,
    regime: str = TREND,
    is_shortable_fn: Callable[[str], bool] | None = None,
) -> list[TradeCandidate]:
    """
    Trains a fresh ensemble on all available history, scores today's
    snapshot, and concentrates the full deployable capital into the top 2
    highest-conviction picks (select_concentrated_trades) rather than
    spreading across many names.

    Total capital deployed is always 100% across the two legs — regime is
    still used to gate which candidates clear the confidence bar upstream,
    but no longer dampens total deployment on top of that. The old
    chop-dampening of total size (via risk.sizing.regime_adjusted_size) was
    inherited from the leveraged-ETF strategy, where chop causes real daily-
    reset decay independent of confidence; that rationale doesn't carry over
    to plain equity/short positions, and it was silently sitting most of the
    portfolio in cash (35% deployed) whenever the market read as choppy.
    """
    train_df = load_training_frame(feature_set_id, symbols, target_horizon_days)
    feature_cols = [c for c in train_df.columns if c not in ("symbol", "ts", "close", "fwd_return")]

    ensemble = EnsembleForecastModel(n_models=n_ensemble_models)
    ensemble.fit(train_df[feature_cols], train_df["fwd_return"])

    latest = load_latest_features(feature_set_id, symbols)
    scored = score_universe(ensemble, latest, feature_cols, min_direction_agreement, min_abs_return)

    total_deploy_pct = 1.0

    candidates = select_concentrated_trades(
        scored,
        max_leg_pct=settings.max_concentrated_position_pct,
        min_leg_pct=settings.min_concentrated_position_pct,
        total_deploy_pct=total_deploy_pct,
        is_shortable_fn=is_shortable_fn,
    )
    _attach_reasoning(candidates, ensemble, latest, feature_cols)
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
    parser.add_argument("--target-horizon-days", type=int, default=5)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--min-direction-agreement", type=float, default=0.8)
    parser.add_argument("--min-abs-return", type=float, default=0.0)
    parser.add_argument("--log", action="store_true", help="Write the shortlist to the decisions table.")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    candidates = run_screen(
        args.feature_set_id,
        symbols,
        target_horizon_days=args.target_horizon_days,
        n_ensemble_models=args.n_ensemble_models,
        min_direction_agreement=args.min_direction_agreement,
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
