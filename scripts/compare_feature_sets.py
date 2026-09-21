"""
Measure -- don't guess -- whether a new feature_set_id actually helps.
Runs the existing walk-forward harness (models/train.py, purge gap +
transaction costs included) once per candidate feature set, on the SAME
fold boundaries and horizon, then compares each against a baseline with a
paired significance test. Modeled directly on scripts/compare_horizons.py,
just varying feature_set_id instead of the horizon.

Why the same fold boundaries matter: two feature sets can have slightly
different usable date ranges (a longer rolling window needs more warmup
history before it produces a value at all), so naively each would get
its own fold windows and "fold 3 got better" could just mean "fold 3
moved". Fold dates are built once, from the CANDIDATE feature set's own
frame (the one with the longer warmup requirement, since it has fewer
usable dates than the baseline), and shared.

Every feature set is reported against the do-nothing baseline -- the mean
forward return of every candidate row in the same test windows -- and
against the OTHER feature set, paired by fold. excess_return (model minus
buy-and-hold) is the headline metric, same reasoning as compare_horizons.py:
a feature set that merely raises both numbers together isn't skill.

Honest-read caveats, printed with the results (same as compare_horizons.py):
  - n_folds paired points is a small sample; a p-value above ~0.05 here
    means "not distinguishable from noise", not "worse".
  - The universe is today's S&P 500 membership: survivorship-biased.
  - A feature set that looks better here still needs its own live
    monitoring once deployed -- a walk-forward result is evidence, not a
    guarantee, exactly like every other model change in this project.

Usage:
    python -m scripts.compare_feature_sets --baseline v4 --candidate v5 --universe
    python -m scripts.compare_feature_sets --baseline v4 --candidate v5 --universe --n-folds 10 --out-csv results.csv
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from scipy import stats

from data.ingest.universe import resolve_symbols
from models.train import headline_verdict, load_training_frame, run_walk_forward, spread_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same metric set as compare_horizons.py -- kept identical so results from
# either script read the same way.
SUMMARY_METRICS = (
    "directional_accuracy",
    "directional_accuracy_when_confident",
    "pct_rows_confident",
    "benchmark_return",
    "model_return_net",
    "excess_return",
    "pct_long",
    "long_return_net",
    "long_win_rate",
    "short_return_net",
    "short_win_rate",
)

HEADLINE_METRIC = "excess_return"


def summarize_feature_set(results: pd.DataFrame) -> dict:
    """One row of the comparison table from one feature set's per-fold results."""
    summary: dict = {"n_folds": len(results)}
    for metric in SUMMARY_METRICS:
        if metric not in results.columns:
            continue
        vals = results[metric].dropna()
        summary[metric] = float(vals.mean()) if len(vals) else float("nan")
        summary[f"{metric}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
    excess = results[HEADLINE_METRIC].dropna()
    summary["folds_positive_excess"] = int((excess > 0).sum())
    return summary


def excess_vs_zero(results: pd.DataFrame, metric: str = HEADLINE_METRIC) -> dict:
    """Is this feature set's excess return distinguishable from zero at all? Paired at the FOLD level -- see module docstring."""
    vals = results[metric].dropna() if metric in results.columns else pd.Series(dtype=float)
    out = {"n_folds": len(vals), "mean": float("nan"), "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    if len(vals) < 2:
        return out
    out["mean"] = float(vals.mean())
    out["t_pvalue"] = float(stats.ttest_1samp(vals, 0.0).pvalue)
    if (vals != 0).any():
        out["wilcoxon_pvalue"] = float(stats.wilcoxon(vals).pvalue)
    return out


def paired_test(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict:
    """Paired-by-fold comparison of one metric -- t-test plus Wilcoxon, paired on fold_id. See compare_horizons.py's identical helper."""
    merged = baseline[["fold_id", metric]].merge(
        candidate[["fold_id", metric]], on="fold_id", suffixes=("_base", "_cand")
    ).dropna()
    out = {"metric": metric, "n_pairs": len(merged), "mean_diff": float("nan"), "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    if len(merged) < 2:
        return out
    diffs = merged[f"{metric}_cand"] - merged[f"{metric}_base"]
    out["mean_diff"] = float(diffs.mean())
    out["t_pvalue"] = float(stats.ttest_rel(merged[f"{metric}_cand"], merged[f"{metric}_base"]).pvalue)
    if (diffs != 0).any():
        out["wilcoxon_pvalue"] = float(stats.wilcoxon(diffs).pvalue)
    return out


def comparison_table(results_by_feature_set: dict[str, pd.DataFrame], baseline_id: str) -> pd.DataFrame:
    """The headline table: one row per feature set with its benchmark beside its model return, paired p-values vs the baseline."""
    rows = []
    baseline = results_by_feature_set.get(baseline_id)
    for feature_set_id, results in results_by_feature_set.items():
        row = {"feature_set_id": feature_set_id, **summarize_feature_set(results)}
        vs_zero = excess_vs_zero(results)
        row["excess_p_vs_zero"] = vs_zero["t_pvalue"]
        row["excess_wilcoxon_p_vs_zero"] = vs_zero["wilcoxon_pvalue"]
        if baseline is not None and feature_set_id != baseline_id:
            for metric in ("directional_accuracy_when_confident", HEADLINE_METRIC):
                test = paired_test(baseline, results, metric)
                short = "acc" if metric.startswith("directional") else "excess"
                row[f"{short}_diff_vs_{baseline_id}"] = test["mean_diff"]
                row[f"{short}_pvalue"] = test["t_pvalue"]
                row[f"{short}_wilcoxon_p"] = test["wilcoxon_pvalue"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward two feature sets and compare, paired by fold.")
    parser.add_argument("--baseline", required=True, help="feature_set_id currently live, e.g. v4.")
    parser.add_argument("--candidate", required=True, help="feature_set_id to evaluate against the baseline, e.g. v5.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--target-horizon-days", type=int, default=None, help="default: TARGET_HORIZON_DAYS")
    parser.add_argument(
        "--target-mode", choices=("absolute", "relative"), default=None,
        help="What the model predicts (default: TARGET_MODE). See models/train.py::load_training_frame.",
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--out-csv", default=None, help="Optional path for the per-fold results (long form).")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    feature_set_ids = [args.baseline, args.candidate]

    # Shared fold boundaries, built from the CANDIDATE's own usable dates:
    # a wider rolling window (e.g. donchian_pct_200) needs more warmup
    # history before it produces a value at all, so the candidate's usable
    # date range starts later than the baseline's -- building fold dates
    # from the candidate keeps every fold inside both feature sets' usable
    # history (same reasoning as compare_horizons.py building from the
    # longest horizon).
    logger.info("Building shared fold dates from %s's usable history…", args.candidate)
    fold_dates = pd.DatetimeIndex(
        sorted(
            pd.DatetimeIndex(
                load_training_frame(
                    args.candidate, symbols, args.target_horizon_days or 5, "absolute"
                )["ts"]
            ).unique()
        )
    )

    results_by_feature_set: dict[str, pd.DataFrame] = {}
    for feature_set_id in feature_set_ids:
        logger.info("Walk-forward on feature_set_id=%s (%d folds)…", feature_set_id, args.n_folds)
        results = run_walk_forward(
            feature_set_id,
            symbols,
            target_horizon_days=args.target_horizon_days,
            n_folds=args.n_folds,
            n_ensemble_models=args.n_ensemble_models,
            fold_dates=fold_dates,
            target_mode=args.target_mode,
            model_name=f"compare_feature_sets_{feature_set_id}",
        )
        results_by_feature_set[feature_set_id] = results
        print(f"\n=== {feature_set_id}: per-fold spread ===")
        print(spread_summary(results))

    table = comparison_table(results_by_feature_set, args.baseline)
    print(f"\n=== Feature set comparison (means across folds; p-values paired by fold vs {args.baseline} baseline) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Excess over buy-and-hold, per feature set ===")
    for feature_set_id, results in results_by_feature_set.items():
        print(f"\n--- {feature_set_id} ---")
        print(headline_verdict(results))

    print(
        "\nHow to read this: model_return_net is per trade after the round-trip cost "
        "floor; benchmark_return is what equal-weight buy-and-hold of every candidate "
        "row paid over the same windows, gross. excess_return is the difference and "
        "the only column that measures skill.\n"
        f"p-values are over {args.n_folds} paired FOLDS, not rows, which is "
        "deliberate: forward-return windows overlap heavily across consecutive days, "
        "so row-level observations are nowhere near independent and row-level "
        "p-values would be far too optimistic. Even these fold-level p-values are "
        "somewhat optimistic, since folds are adjacent in time and share market "
        "regimes. Anything above ~0.05 is not distinguishable from noise.\n"
        "The universe is TODAY's S&P 500 membership, so every result is "
        "survivorship-biased: companies that fell out of the index (usually after "
        "doing badly) are missing from the history entirely. That inflates the "
        "benchmark and the model's longs alike.\n"
        f"A positive excess_diff_vs_{args.baseline} with a low p-value is the bar "
        f"for actually switching the live FEATURE_SET_ID default to {args.candidate} -- "
        "a candidate that merely doesn't lose is not, on its own, a reason to ship it."
    )

    if args.out_csv:
        long_form = pd.concat(
            [r.assign(feature_set_id=fsid) for fsid, r in results_by_feature_set.items()], ignore_index=True
        )
        long_form.to_csv(args.out_csv, index=False)
        print(f"\nPer-fold results written to {args.out_csv}")


if __name__ == "__main__":
    main()
