"""
Measure — don't guess — what changing the forecast horizon does. Runs the
existing walk-forward harness (models/train.py, purge gap + transaction
costs included, everything logged to MLflow as usual) once per candidate
horizon on the SAME feature set and the SAME fold boundaries, then compares
each horizon against the baseline with a paired significance test.

Why the same fold boundaries matter: a longer horizon loses its last
`horizon` days to the label shift, so naively each horizon would get
slightly different fold windows and "fold 3 got better" could just mean
"fold 3 moved". Fold dates are built once, from the longest horizon's
frame, and shared (run_walk_forward's fold_dates parameter).

Why longer horizons might look better at all: the round-trip cost floor
(backtest/cost_model.py, ~0.02%) is fixed per trade, so it eats
proportionally less of a 5% expected 20-day move than of a 1% expected
5-day move. Whether that survives contact with the data is what this
script exists to answer.

Honest-read caveats, printed with the results:
  - Longer-horizon returns overlap heavily inside a test window, so the
    effective number of independent observations is much smaller than the
    row count — treat p-values as optimistic for 20/40 days.
  - n_folds paired points is a small sample; a p-value above ~0.05 here
    means "not distinguishable from noise", not "worse".

Usage:
    python -m scripts.compare_horizons --feature-set-id v4 --universe
    python -m scripts.compare_horizons --feature-set-id v4 --universe --horizons 5,10,20,40
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from scipy import stats

from data.ingest.universe import resolve_symbols
from models.train import load_training_frame, run_walk_forward, spread_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The columns a horizon gets judged on. mean_return_confident_net is the
# headline: what trading the confident calls would have paid per trade,
# after costs — accuracy can look fine while this is negative.
SUMMARY_METRICS = (
    "directional_accuracy",
    "directional_accuracy_when_confident",
    "pct_rows_confident",
    "mean_return_confident_net",
)


def summarize_horizon(results: pd.DataFrame) -> dict:
    """One row of the comparison table from one horizon's per-fold results."""
    summary: dict = {"n_folds": len(results)}
    for metric in SUMMARY_METRICS:
        vals = results[metric].dropna()
        summary[metric] = float(vals.mean()) if len(vals) else float("nan")
        summary[f"{metric}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
    net = results["mean_return_confident_net"].dropna()
    summary["folds_profitable"] = int((net > 0).sum())
    return summary


def paired_test(
    baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str
) -> dict:
    """
    Paired-by-fold comparison of one metric: t-test on the per-fold
    differences plus Wilcoxon signed-rank as a distribution-free check.
    Pairs on fold_id (inner join) so a fold missing from either side —
    e.g. no confident rows — drops out of both.
    """
    merged = baseline[["fold_id", metric]].merge(
        candidate[["fold_id", metric]], on="fold_id", suffixes=("_base", "_cand")
    ).dropna()
    out = {"metric": metric, "n_pairs": len(merged), "mean_diff": float("nan"), "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    if len(merged) < 2:
        return out
    diffs = merged[f"{metric}_cand"] - merged[f"{metric}_base"]
    out["mean_diff"] = float(diffs.mean())
    out["t_pvalue"] = float(stats.ttest_rel(merged[f"{metric}_cand"], merged[f"{metric}_base"]).pvalue)
    # Wilcoxon needs at least one nonzero difference.
    if (diffs != 0).any():
        out["wilcoxon_pvalue"] = float(stats.wilcoxon(diffs).pvalue)
    return out


def comparison_table(results_by_horizon: dict[int, pd.DataFrame], baseline_horizon: int) -> pd.DataFrame:
    """The headline table: one row per horizon, paired p-values vs the baseline."""
    rows = []
    baseline = results_by_horizon.get(baseline_horizon)
    for horizon, results in sorted(results_by_horizon.items()):
        row = {"horizon_days": horizon, **summarize_horizon(results)}
        if baseline is not None and horizon != baseline_horizon:
            for metric in ("directional_accuracy_when_confident", "mean_return_confident_net"):
                test = paired_test(baseline, results, metric)
                short = "acc" if metric.startswith("directional") else "net"
                row[f"{short}_diff_vs_{baseline_horizon}d"] = test["mean_diff"]
                row[f"{short}_pvalue"] = test["t_pvalue"]
                row[f"{short}_wilcoxon_p"] = test["wilcoxon_pvalue"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward the forecast model across horizons and compare.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--horizons", default="5,10,20,40", help="Comma-separated trading-day horizons to compare.")
    parser.add_argument("--baseline-horizon", type=int, default=5, help="The horizon the paired tests compare against.")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--out-csv", default=None, help="Optional path for the per-fold results (long form).")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    horizons = sorted({int(h.strip()) for h in args.horizons.split(",") if h.strip()})
    if args.baseline_horizon not in horizons:
        horizons = sorted({*horizons, args.baseline_horizon})

    # Shared fold boundaries, built from the LONGEST horizon's usable dates
    # so every horizon has labels inside every fold (see module docstring).
    longest = max(horizons)
    logger.info("Building shared fold dates from the %d-day frame…", longest)
    fold_dates = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(load_training_frame(args.feature_set_id, symbols, longest)["ts"]).unique())
    )

    results_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        logger.info("Walk-forward at horizon=%d trading days (%d folds)…", horizon, args.n_folds)
        results = run_walk_forward(
            args.feature_set_id,
            symbols,
            target_horizon_days=horizon,
            n_folds=args.n_folds,
            n_ensemble_models=args.n_ensemble_models,
            fold_dates=fold_dates,
        )
        results_by_horizon[horizon] = results
        print(f"\n=== horizon {horizon}d: per-fold spread ===")
        print(spread_summary(results))

    table = comparison_table(results_by_horizon, args.baseline_horizon)
    print("\n=== Horizon comparison (means across folds; p-values paired by fold vs "
          f"{args.baseline_horizon}d baseline) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(
        "\nHow to read this: mean_return_confident_net is per trade, after the "
        "round-trip cost floor — a longer horizon holds ~horizon/5 times longer per "
        "trade, so compare per-trade numbers with that in mind. p-values are over "
        f"{args.n_folds} paired folds; anything above ~0.05 is not distinguishable "
        "from noise. Longer-horizon returns overlap within a test window, so their "
        "p-values run optimistic. Do not pick the horizon with the prettiest mean — "
        "pick one only if its improvement is statistically real."
    )

    if args.out_csv:
        long_form = pd.concat(
            [r.assign(horizon_days=h) for h, r in results_by_horizon.items()], ignore_index=True
        )
        long_form.to_csv(args.out_csv, index=False)
        print(f"\nPer-fold results written to {args.out_csv}")


if __name__ == "__main__":
    main()
