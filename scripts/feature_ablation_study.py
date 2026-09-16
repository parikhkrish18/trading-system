"""
Offline research script -- NOT part of the live pipeline, never scheduled,
never touches `features`/`trade_log`/any order path. Answers two separate
questions about a set of features (default: donchian_breakout_20,
donchian_pct_20, mom_pullback_20_5):

  1. Information coefficient (IC) -- does the raw feature value, on its
     own, correlate with the forward return? Both a pooled Spearman
     correlation (mixes cross-sectional and time-series variation, easy to
     overstate) and the cross-sectional IC (Spearman WITHIN each date
     across symbols, averaged across dates -- the standard quant-research
     definition, immune to "the whole market went up" inflating every
     feature's pooled number) plus its IR (mean daily IC / std daily IC).

  2. Ablation -- does the walk-forward ENSEMBLE actually rely on these
     features? Reuses models/train.py's run_walk_forward completely
     unmodified (same purge gap, transaction costs, production_book_mask)
     for a baseline (all features) and an ablated run (these features
     dropped), then compares fold-by-fold. This is the more important
     number of the two: a feature can have weak standalone IC but still
     help the model through interactions with other features, or have real
     IC but be redundant with something else already in the feature set --
     only the ablation answers "does removing it change what the model
     does."

Both walk-forward runs write their folds to the existing walk_forward_folds
table (same table run_walk_forward always writes to), tagged under
research-only model_name values ("ablation_baseline_all_features" /
"ablation_excluding_<...>") -- never "forecast_lgbm", so this can never be
confused with, or overwrite, the production model's own recorded results.

Usage:
    python -m scripts.feature_ablation_study --feature-set-id v4 --universe
    python -m scripts.feature_ablation_study --feature-set-id v4 --universe \\
        --exclude-features donchian_breakout_20,donchian_pct_20,mom_pullback_20_5 \\
        --n-folds 10
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats

import models.train as train_mod
from config.settings import settings
from data.ingest.universe import resolve_symbols
from models.train import feature_columns, headline_verdict, load_training_frame, run_walk_forward, spread_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_FEATURES = ["donchian_breakout_20", "donchian_pct_20", "mom_pullback_20_5"]

PAIRED_METRICS = (
    "directional_accuracy",
    "directional_accuracy_when_confident",
    "excess_return",
    "model_return_net",
)


def _spearman_ic(feature: pd.Series, target: pd.Series) -> float:
    valid = feature.notna() & target.notna()
    if valid.sum() < 5 or feature[valid].nunique() < 2:
        return np.nan
    rho, _ = stats.spearmanr(feature[valid], target[valid])
    return float(rho) if np.isfinite(rho) else np.nan


def information_coefficients(df: pd.DataFrame, feature_cols: list[str], target_col: str = "fwd_return") -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        pooled = _spearman_ic(df[col], df[target_col])
        daily = df.groupby("ts", group_keys=False).apply(
            lambda g, col=col: _spearman_ic(g[col], g[target_col]) if len(g) >= 5 else np.nan
        ).dropna()
        mean_ic = float(daily.mean()) if len(daily) else np.nan
        ic_ir = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else np.nan
        rows.append(
            {"feature": col, "pooled_ic": pooled, "mean_daily_ic": mean_ic, "ic_ir": ic_ir, "n_dates": len(daily)}
        )
    out = pd.DataFrame(rows)
    out["_abs_ic"] = out["mean_daily_ic"].abs()
    return out.sort_values("_abs_ic", ascending=False, na_position="last").drop(columns="_abs_ic")


def run_ablation(
    feature_set_id: str, symbols: list[str], exclude_features: list[str], **wf_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs run_walk_forward twice, unmodified, under two different feature
    sets. The exclusion is done by monkeypatching models.train.feature_columns
    for the duration of the ablated call only -- run_walk_forward looks it
    up as a bare name at call time, so this is enough to change what
    feature_cols it fits on without touching the production harness code at
    all. Restored in a finally so a mid-run exception can never leave the
    module patched for any later import of models.train elsewhere.
    """
    logger.info("Baseline walk-forward (all features)...")
    baseline = run_walk_forward(feature_set_id, symbols, model_name="ablation_baseline_all_features", **wf_kwargs)

    original_feature_columns = train_mod.feature_columns
    excluded = set(exclude_features)

    def _filtered_feature_columns(df):
        return [c for c in original_feature_columns(df) if c not in excluded]

    model_name = "ablation_excluding_" + "_".join(sorted(excluded))
    train_mod.feature_columns = _filtered_feature_columns
    try:
        logger.info("Ablated walk-forward (excluding %s)...", sorted(excluded))
        ablated = run_walk_forward(feature_set_id, symbols, model_name=model_name[:60], **wf_kwargs)
    finally:
        train_mod.feature_columns = original_feature_columns

    return baseline, ablated


def paired_test(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict:
    """Same paired-by-fold approach as scripts/compare_horizons.py: a t-test on per-fold (candidate - baseline) diffs."""
    merged = baseline[["fold_id", metric]].merge(
        candidate[["fold_id", metric]], on="fold_id", suffixes=("_base", "_cand")
    ).dropna()
    out = {"metric": metric, "n_pairs": len(merged), "mean_diff": float("nan"), "t_pvalue": float("nan")}
    if len(merged) < 2:
        return out
    diffs = merged[f"{metric}_cand"] - merged[f"{metric}_base"]
    out["mean_diff"] = float(diffs.mean())
    if (diffs != 0).any():
        out["t_pvalue"] = float(stats.ttest_rel(merged[f"{metric}_cand"], merged[f"{metric}_base"]).pvalue)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline feature IC + ablation study -- research only, never deployed.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--exclude-features", default=",".join(DEFAULT_EXCLUDE_FEATURES))
    parser.add_argument("--target-horizon-days", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    exclude_features = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
    horizon = args.target_horizon_days or settings.target_horizon_days

    logger.info("Loading training frame (feature_set_id=%s, horizon=%dd) for IC...", args.feature_set_id, horizon)
    # target_mode="absolute" here specifically so the IC is read against the
    # plain forward return with un-transformed feature values, independent
    # of whatever TARGET_MODE production is currently configured with.
    df = load_training_frame(args.feature_set_id, symbols, horizon, target_mode="absolute")
    all_features = feature_columns(df)
    ic_table = information_coefficients(df, all_features)

    print("\n=== Information coefficients (Spearman, ranked by |mean daily cross-sectional IC|) ===")
    print(ic_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== Ablation target features: {exclude_features} ===")
    ranked = ic_table.reset_index(drop=True)
    for f in exclude_features:
        matches = ranked.index[ranked["feature"] == f]
        if len(matches) == 0:
            print(f"{f}: NOT FOUND in feature_set_id={args.feature_set_id!r} — check spelling / feature_set_id.")
            continue
        rank = int(matches[0]) + 1
        print(f"{f}: rank {rank}/{len(ranked)} by |mean daily IC| -- {ranked.iloc[rank - 1].to_dict()}")

    baseline, ablated = run_ablation(
        args.feature_set_id,
        symbols,
        exclude_features,
        target_horizon_days=args.target_horizon_days,
        n_folds=args.n_folds,
        n_ensemble_models=args.n_ensemble_models,
    )

    print("\n=== Baseline (all features) ===")
    print(spread_summary(baseline))
    print(headline_verdict(baseline))

    print(f"\n=== Ablated (excluding {exclude_features}) ===")
    print(spread_summary(ablated))
    print(headline_verdict(ablated))

    print("\n=== Paired comparison, ablated minus baseline, by fold ===")
    for metric in PAIRED_METRICS:
        result = paired_test(baseline, ablated, metric)
        print(
            f"{metric}: mean_diff={result['mean_diff']:+.4f} (ablated - baseline), "
            f"t_pvalue={result['t_pvalue']:.4f}, n_pairs={result['n_pairs']}"
        )

    print(
        "\nHow to read this: IC measures each feature's own raw correlation with the "
        "forward return, independent of the model. The ablation measures whether the "
        "ENSEMBLE relies on it once every other feature is already available -- the "
        "more decision-relevant number of the two. A near-zero mean_diff with a high "
        "p-value on the ablated run means removing these features didn't change what "
        "the model actually does, consistent with them being redundant or noise (not "
        "proof either way on a handful of folds). A feature can have weak IC alone but "
        "still help the model through interactions with other features, or have real "
        "IC but be redundant with something else already in the feature set -- IC and "
        "the ablation can legitimately disagree."
    )


if __name__ == "__main__":
    main()
