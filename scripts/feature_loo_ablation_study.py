"""
Offline research script -- NOT part of the live pipeline, never scheduled.
Leave-one-out ablation across EVERY feature in the current feature set,
ranked by how much each one actually helps the walk-forward ENSEMBLE (not
just its standalone IC -- see scripts/feature_ablation_study.py, whose
information_coefficients()/paired_test() this module reuses directly).

Runs models/train.py's run_walk_forward ONCE as a baseline (all features),
then once more per feature with just that one feature excluded (feature_columns
is monkeypatched for the duration of each excluded run only, same technique
feature_ablation_study.py uses) -- so the baseline is computed once and
reused for every comparison, not re-run redundantly per feature.

Default settings (5 folds, 3 ensemble models per run) are deliberately
lighter than models/train.py's own defaults (10/5) to keep a 30-condition
sweep (1 baseline + one run per feature) tractable -- a 2-condition study at
10/5 on the full universe took ~13 hours on this project's Railway compute,
and this is ~15x that. Every condition still runs against the FULL
configured universe, since universe breadth is what a feature-importance
ranking is meant to generalize over -- folds/ensemble-size are what got cut
for speed, not that.

Writes each feature's walk-forward folds to the existing walk_forward_folds
table under research-only model_name values
("loo_ablation_baseline"/"loo_ablation_excluding_<feature>"), same
isolation as feature_ablation_study.py -- never "forecast_lgbm".

Usage:
    python -m scripts.feature_loo_ablation_study --feature-set-id v4 --universe
    python -m scripts.feature_loo_ablation_study --feature-set-id v4 --universe --n-folds 10 --n-ensemble-models 5
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from scripts.feature_ablation_study import information_coefficients, paired_test

import models.train as train_mod
from config.settings import settings
from data.ingest.universe import resolve_symbols
from models.train import feature_columns, load_training_frame, run_walk_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_METRIC = "excess_return"


def _run_single_exclusion(feature_set_id: str, symbols: list[str], feature: str, **wf_kwargs) -> pd.DataFrame:
    original_feature_columns = train_mod.feature_columns

    def _filtered_feature_columns(df):
        return [c for c in original_feature_columns(df) if c != feature]

    train_mod.feature_columns = _filtered_feature_columns
    try:
        model_name = f"loo_ablation_excluding_{feature}"[:60]
        return run_walk_forward(feature_set_id, symbols, model_name=model_name, **wf_kwargs)
    finally:
        train_mod.feature_columns = original_feature_columns


def run_leave_one_out(
    feature_set_id: str, symbols: list[str], features: list[str], **wf_kwargs
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    logger.info("Baseline walk-forward (all %d features)...", len(features))
    baseline = run_walk_forward(feature_set_id, symbols, model_name="loo_ablation_baseline", **wf_kwargs)

    per_feature: dict[str, pd.DataFrame] = {}
    for i, feature in enumerate(features, 1):
        logger.info("[%d/%d] Ablated walk-forward excluding %s...", i, len(features), feature)
        per_feature[feature] = _run_single_exclusion(feature_set_id, symbols, feature, **wf_kwargs)

    return baseline, per_feature


def rank_features(
    baseline: pd.DataFrame, per_feature: dict[str, pd.DataFrame], metric: str = DEFAULT_METRIC
) -> pd.DataFrame:
    """
    Importance = baseline[metric] - ablated[metric], paired by fold: how
    much performance DROPS when a feature is removed. Positive means the
    feature helps (removing it hurts the model); negative means the model
    does BETTER without it -- an active-noise candidate to drop outright,
    not merely a redundant one. paired_test computes (ablated - baseline),
    so importance is its negation.
    """
    rows = []
    for feature, ablated in per_feature.items():
        test = paired_test(baseline, ablated, metric)
        mean_diff = test["mean_diff"]
        rows.append(
            {
                "feature": feature,
                "importance": -mean_diff if pd.notna(mean_diff) else float("nan"),
                "p_value": test["t_pvalue"],
                "n_pairs": test["n_pairs"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("importance", ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out ablation across every feature, ranked most to least useful -- research only, never deployed."
    )
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--target-horizon-days", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-ensemble-models", type=int, default=3)
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Walk-forward metric to rank importance on.")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    horizon = args.target_horizon_days or settings.target_horizon_days

    logger.info("Loading training frame (feature_set_id=%s, horizon=%dd) for IC...", args.feature_set_id, horizon)
    # target_mode="absolute" purely for the IC read -- see
    # scripts/feature_ablation_study.py's identical reasoning. The feature
    # NAME list this produces is what drives the leave-one-out loop below;
    # it is identical under "relative" mode too (z-scoring transforms values
    # in place, it never adds/removes/renames columns).
    df = load_training_frame(args.feature_set_id, symbols, horizon, target_mode="absolute")
    all_features = feature_columns(df)
    ic_table = information_coefficients(df, all_features)
    print(f"\n=== Information coefficients, {len(all_features)} features (context only -- see ranking below) ===")
    print(ic_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    baseline, per_feature = run_leave_one_out(
        args.feature_set_id,
        symbols,
        all_features,
        target_horizon_days=args.target_horizon_days,
        n_folds=args.n_folds,
        n_ensemble_models=args.n_ensemble_models,
    )

    ranking = rank_features(baseline, per_feature, metric=args.metric)
    ic_lookup = ic_table.set_index("feature")["mean_daily_ic"].to_dict()
    ranking["mean_daily_ic"] = ranking["feature"].map(ic_lookup)

    print(f"\n=== Feature importance ranking, by {args.metric} drop when removed (most to least useful) ===")
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    n_helpful = int((ranking["importance"] > 0).sum())
    n_harmful = int((ranking["importance"] < 0).sum())
    print(
        f"\n{n_helpful}/{len(ranking)} features help when present (removing them hurts {args.metric}); "
        f"{n_harmful}/{len(ranking)} the model does BETTER without (candidates to drop outright, not just "
        f"deprioritize). Every comparison here is paired over only {args.n_folds} folds, so read the p_value "
        "column: anything above ~0.05 is not distinguishable from noise, even though it still gets a rank in "
        "the table. Treat the ordering among not-significant features as a rough tendency, not a real "
        "difference -- and rerun at the harness's normal 10 folds / 5 ensemble models on any specific feature "
        "you're about to act on, since this sweep deliberately traded precision for tractability."
    )


if __name__ == "__main__":
    main()
