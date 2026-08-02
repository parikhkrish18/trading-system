"""
Confidence-ranked equity screener. Scores every symbol in a universe with
the ensemble forecast model and produces a small, sized shortlist of
trades — long or short, whichever the data actually supports for that
symbol — rather than only ever looking at a fixed handful of names. If
nothing in the mega-caps clears the confidence bar but some other S&P 500
name does, that's what gets picked; nothing here special-cases which
symbols are "important."

This module does NOT place orders — it produces candidates and logs them
to the `decisions` table (mode="paper", executed_position left null).
Wiring the output to execution/broker.py is a separate step.

Usage:
    python -m models.screener --feature-set-id v3 --universe --top-k 10
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from collections.abc import Callable

import pandas as pd

from config.settings import settings
from data.ingest.db import get_engine
from data.ingest.universe import resolve_symbols
from models.forecast.ensemble import EnsembleForecastModel
from models.regime.trend_chop_classifier import TREND, RuleBasedRegime
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
    regime: str = TREND  # "trend" | "chop", per-symbol (see per_symbol_regimes)


def per_symbol_regimes(
    latest_features: pd.DataFrame,
    adx_threshold: float = 25.0,
    adx_col: str = "adx_14",
) -> dict[str, str]:
    """
    Tag each symbol trend/chop from its latest ADX via RuleBasedRegime, so
    risk.sizing's chop dampening actually engages per symbol instead of the
    whole book being sized as if everything were trending.

    Symbols with a missing/NaN ADX (not enough price history yet) fall back
    to TREND — full size, matching the behavior before regimes were wired in —
    rather than silently getting damped by a NaN comparison.
    """
    if adx_col not in latest_features.columns:
        return {}
    known = latest_features.loc[latest_features[adx_col].notna(), ["symbol", adx_col]]
    if known.empty:
        return {}
    labels = RuleBasedRegime(adx_threshold=adx_threshold).predict(known[adx_col])
    return dict(zip(known["symbol"], labels))


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
    regime_by_symbol: dict[str, str] | None = None,
) -> list[TradeCandidate]:
    """
    Filters `scored` down to the top_k confident candidates and sizes each
    via risk.sizing.target_position_size — reused unchanged; it already
    handles signed forecasts, regime damping, and correlation caps
    generically, nothing here is symbol- or ETF-specific.

    `regime_by_symbol` (see per_symbol_regimes) overrides `regime` for the
    symbols it contains; `regime` is the fallback for everything else.

    Short candidates that fail `is_shortable_fn` (when given — pass e.g.
    execution.broker_alpaca.AlpacaBroker.is_shortable) are dropped rather
    than resized to zero, so the caller can see which symbols got skipped.
    Pass None to skip the check entirely (e.g. offline/backtest scoring
    with no live broker connection).
    """
    current_positions = current_positions or {}
    regime_by_symbol = regime_by_symbol or {}
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

        symbol_regime = regime_by_symbol.get(symbol, regime)
        size = target_position_size(
            forecast=forecast,
            forecast_scale=forecast_scale,
            regime=symbol_regime,
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
                regime=symbol_regime,
            )
        )

    return candidates


def run_screen(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int = 5,
    n_ensemble_models: int = 5,
    min_direction_agreement: float = 0.8,
    min_abs_return: float = 0.0,
    top_k: int = 10,
    regime: str = TREND,
    is_shortable_fn: Callable[[str], bool] | None = None,
) -> list[TradeCandidate]:
    """Trains a fresh ensemble on all available history, scores today's snapshot, returns sized candidates."""
    train_df = load_training_frame(feature_set_id, symbols, target_horizon_days)
    feature_cols = [c for c in train_df.columns if c not in ("symbol", "ts", "close", "fwd_return")]
    forecast_scale = float(train_df["fwd_return"].std())

    ensemble = EnsembleForecastModel(n_models=n_ensemble_models)
    ensemble.fit(train_df[feature_cols], train_df["fwd_return"])

    latest = load_latest_features(feature_set_id, symbols)
    scored = score_universe(ensemble, latest, feature_cols, min_direction_agreement, min_abs_return)

    correlation_matrix = build_correlation_matrix(train_df[["symbol", "ts", "close"]])
    regime_by_symbol = per_symbol_regimes(latest)

    return select_trades(
        scored,
        regime=regime,
        regime_by_symbol=regime_by_symbol,
        forecast_scale=forecast_scale,
        max_position_pct=settings.max_single_position_pct,
        max_short_position_pct=settings.max_short_position_pct,
        max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
        correlation_matrix=correlation_matrix,
        top_k=top_k,
        is_shortable_fn=is_shortable_fn,
    )


def log_candidates(candidates: list[TradeCandidate], feature_set_id: str, mode: str = "paper") -> int:
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
            "regime": c.regime,
            "target_position": c.target_position_pct,
            "executed_position": None,
            "mode": mode,
        }
        for c in candidates
    ]
    df = pd.DataFrame(rows)
    engine = get_engine()
    df.to_sql("decisions", engine, if_exists="append", index=False)
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
    parser.add_argument("--top-k", type=int, default=10)
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
        top_k=args.top_k,
    )

    if not candidates:
        print("No candidates cleared the confidence bar this run.")
        return

    print(f"{'symbol':<8}{'side':<7}{'regime':<7}{'pred_return':>13}{'agreement':>12}{'target_pct':>12}")
    for c in candidates:
        print(
            f"{c.symbol:<8}{c.side:<7}{c.regime:<7}"
            f"{c.predicted_return:>13.4f}{c.direction_agreement:>12.2f}{c.target_position_pct:>12.4f}"
        )

    if args.log:
        n = log_candidates(candidates, args.feature_set_id)
        print(f"\nLogged {n} candidate(s) to decisions (mode=paper, not executed).")


if __name__ == "__main__":
    main()
