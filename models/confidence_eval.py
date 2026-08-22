"""
Does the ensemble's "confident" label actually mean anything?

The v4 walk-forward exposed that it didn't: ~96% of test rows cleared the
direction_agreement >= 0.8 bar and accuracy-when-confident matched overall
accuracy to the third decimal — the five seed-only members almost always
agree, so agreement carries no information. This harness answers, with
per-fold numbers and paired significance tests, whether any cheaper or
deeper fix produces a filter worth keeping:

  (a) stricter agreement plus a minimum predicted magnitude relative to
      the round-trip transaction cost,
  (b) the already-computed std_prediction used as a continuous
      signal-to-noise measure (|mean| / std across members),
  (c) a structurally diverse ensemble (different tree depths, feature
      fractions, training windows per member — see
      models/forecast/ensemble.py) scored with the same filters.

To keep this affordable it trains each ensemble once per fold, persists
every member's raw prediction on every test row (out_dir/*.csv.gz), then
evaluates all candidate confidence definitions offline on identical
predictions. Re-runs reuse the persisted predictions unless
--force-retrain is passed.

The folds, purge gap, and cost floor are exactly models/train.py's — a
filter evaluated under friendlier assumptions than the production harness
would prove nothing.

Usage:
    python -m models.confidence_eval --feature-set-id v4 --universe
"""
from __future__ import annotations

import argparse
import pathlib
from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from backtest.cost_model import round_trip_cost_fraction
from config.settings import settings
from data.ingest.universe import resolve_symbols
from models.evaluation import trade_metrics
from models.forecast.ensemble import EnsembleForecastModel, summarize_members
from models.train import load_training_frame, make_expanding_folds, purged_train_cutoff

# The production filter as of the v4 run — every variant is tested against
# this, on the same per-fold predictions, with a paired test.
BASELINE_VARIANT = "agree80"
BASELINE_DIVERSITY = "seed"


def collect_fold_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    purge_days: int,
    n_ensemble_models: int = 5,
    diversity: str = "seed",
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Walk the same expanding folds as models.train.run_walk_forward, but
    keep every member's raw prediction on every test row instead of only
    aggregate metrics. Returns one row per (fold, symbol, date) with
    columns: fold_id, symbol, ts, fwd_return, member_0..member_K-1.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(df["ts"]).unique()))
    folds = make_expanding_folds(dates, n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough history to build any walk-forward folds.")

    frames = []
    for fold in folds:
        cutoff = purged_train_cutoff(dates, fold.train_end, purge_days)
        train_df = df[(df["ts"] >= fold.train_start) & (df["ts"] < cutoff)]
        test_df = df[(df["ts"] >= fold.test_start) & (df["ts"] < fold.test_end)]
        if train_df.empty or test_df.empty:
            continue

        ensemble = EnsembleForecastModel(
            n_models=n_ensemble_models, base_seed=base_seed, diversity=diversity
        )
        ensemble.fit(train_df[feature_cols], train_df["fwd_return"], ts=train_df["ts"])
        members = ensemble.predict_members(test_df[feature_cols])

        out = test_df[["symbol", "ts", "fwd_return"]].reset_index(drop=True)
        out = pd.concat([out, members.reset_index(drop=True)], axis=1)
        out.insert(0, "fold_id", fold.fold_id)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def add_summary_columns(preds: pd.DataFrame) -> pd.DataFrame:
    """
    mean/std/agreement recomputed from the persisted member columns, plus
    signal_to_noise = |mean| / std — the continuous confidence measure of
    approach (b). Rows where every member says the identical number (std 0)
    get +inf: unanimous-and-identical is maximal confidence by this measure.
    """
    member_cols = [c for c in preds.columns if c.startswith("member_")]
    summary = summarize_members(preds[member_cols])
    out = preds.copy()
    for col in summary.columns:
        out[col] = summary[col].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(out["mean_prediction"].to_numpy()) / out["std_prediction"].to_numpy()
    out["signal_to_noise"] = np.where(np.isnan(snr), np.inf, snr)
    return out


def _per_fold_top_quantile(values: pd.Series, fold_id: pd.Series, q: float) -> pd.Series:
    """True where `values` is at or above its own fold's q-quantile — i.e.
    the top (1-q) fraction of each fold. inf-safe: quantile ranks, not means."""
    finite_or_inf = values.replace(np.inf, np.finfo(np.float64).max)
    threshold = finite_or_inf.groupby(fold_id.to_numpy()).transform(lambda s: s.quantile(q))
    return finite_or_inf >= threshold


def build_variants(cost: float) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """
    Candidate definitions of "confident", each a mask over the prediction
    frame. Selectivity targets ~top 10-20%; the cost-multiple ladder exists
    because the raw cost floor (2 bps) is far below typical 5-day predicted
    moves and filters essentially nothing on its own.
    """
    return {
        # Current production definition (the one that passes ~96% of rows).
        "agree80": lambda d: d["direction_agreement"] >= 0.8,
        # (a) unanimity, then unanimity + magnitude ladders vs the cost floor.
        "agree100": lambda d: d["direction_agreement"] >= 0.999,
        "agree100_cost10x": lambda d: (d["direction_agreement"] >= 0.999)
        & (d["mean_prediction"].abs() >= 10 * cost),
        "agree100_cost25x": lambda d: (d["direction_agreement"] >= 0.999)
        & (d["mean_prediction"].abs() >= 25 * cost),
        # (b) continuous confidence: top of each fold by |mean|/std.
        "snr_top20": lambda d: _per_fold_top_quantile(d["signal_to_noise"], d["fold_id"], 0.80),
        "snr_top10": lambda d: _per_fold_top_quantile(d["signal_to_noise"], d["fold_id"], 0.90),
        # Controls that separate "members agree" from "the move is big":
        # magnitude alone (top |mean|) and pure low disagreement (bottom std).
        "magnitude_top15": lambda d: _per_fold_top_quantile(d["mean_prediction"].abs(), d["fold_id"], 0.85),
        "low_std_top15": lambda d: _per_fold_top_quantile(-d["std_prediction"], d["fold_id"], 0.85),
    }


def summarize_variant(preds: pd.DataFrame, mask: pd.Series, cost: float) -> pd.DataFrame:
    """
    Per-fold report card for one confidence definition: selectivity
    (pct_pass), directional accuracy on all rows vs on the rows the filter
    keeps, and what trading only the kept rows would have paid — beside what
    equal-weight buy-and-hold of every row in that fold paid over the same
    window. NaN metrics where a fold keeps zero rows.

    excess_conf (net minus benchmark) is the column that matters. A filter
    that lifts net_conf by keeping only the longs in a rising market has
    improved nothing; excess_conf is what says so.
    """
    rows = []
    for fold_id, g in preds.groupby("fold_id"):
        kept = mask.loc[g.index].to_numpy()
        correct = np.sign(g["mean_prediction"]) == np.sign(g["fwd_return"])
        metrics = trade_metrics(
            g["mean_prediction"].to_numpy()[kept],
            g["fwd_return"].to_numpy()[kept],
            g["fwd_return"].to_numpy(),
            cost,
        )
        rows.append(
            {
                "fold_id": fold_id,
                "n_test": len(g),
                "pct_pass": float(kept.mean()),
                "acc_all": float(correct.mean()),
                "acc_conf": float(correct[kept].mean()) if kept.any() else float("nan"),
                "benchmark": metrics["benchmark_return"],
                "gross_conf": metrics["model_return_gross"],
                "net_conf": metrics["model_return_net"],
                "excess_conf": metrics["excess_return"],
                "pct_long": metrics["pct_long"],
                "long_net": metrics["long_return_net"],
                "long_win": metrics["long_win_rate"],
                "short_net": metrics["short_return_net"],
                "short_win": metrics["short_win_rate"],
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(variant_folds: pd.DataFrame, baseline_folds: pd.DataFrame, col: str = "excess_conf") -> dict:
    """
    Paired difference (variant - baseline) on per-fold `col`, folds matched
    by fold_id, NaN folds dropped pairwise. Reports both the paired t-test
    and the Wilcoxon signed-rank test — with only ~10 folds the t-test
    alone is easy to fool with one outlier fold.

    Defaults to excess_conf rather than net_conf: two filters can differ in
    net return purely because one happens to hold more market exposure, and
    that difference is not a difference in filter quality.

    Fold-level, not row-level, on purpose — consecutive rows share almost
    all of their forward-return window, so row-level p-values would treat
    heavily overlapping observations as independent.
    """
    merged = variant_folds[["fold_id", col]].merge(
        baseline_folds[["fold_id", col]], on="fold_id", suffixes=("_v", "_b")
    ).dropna()
    if len(merged) < 2:
        return {"n_folds": len(merged), "mean_diff": float("nan"), "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    diff = merged[f"{col}_v"] - merged[f"{col}_b"]
    t_res = stats.ttest_rel(merged[f"{col}_v"], merged[f"{col}_b"])
    if np.allclose(diff, 0):
        wilcoxon_p = 1.0  # identical folds — no evidence of any difference
    else:
        wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
    return {
        "n_folds": len(merged),
        "mean_diff": float(diff.mean()),
        "t_pvalue": float(t_res.pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
    }


def evaluate(
    preds_by_diversity: dict[str, pd.DataFrame], cost: float
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """
    Score every (diversity, variant) combination. Returns the aggregate
    table (one row each) and the per-fold tables keyed by (diversity,
    variant). Aggregates are means over folds — matching how
    models/train.py has always reported — with the paired tests run
    against (BASELINE_DIVERSITY, BASELINE_VARIANT).
    """
    variants = build_variants(cost)
    fold_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for diversity, preds in preds_by_diversity.items():
        for name, mask_fn in variants.items():
            mask = mask_fn(preds)
            fold_tables[(diversity, name)] = summarize_variant(preds, mask, cost)

    baseline = fold_tables[(BASELINE_DIVERSITY, BASELINE_VARIANT)]
    rows = []
    for (diversity, name), table in fold_tables.items():
        comparison = paired_comparison(table, baseline)
        rows.append(
            {
                "diversity": diversity,
                "variant": name,
                "pct_pass": table["pct_pass"].mean(),
                "acc_all": table["acc_all"].mean(),
                "acc_conf": table["acc_conf"].mean(),
                "acc_edge": table["acc_conf"].mean() - table["acc_all"].mean(),
                "benchmark": table["benchmark"].mean(),
                "net_conf": table["net_conf"].mean(),
                "excess_conf": table["excess_conf"].mean(),
                "excess_conf_std": table["excess_conf"].std(),
                "excess_conf_min": table["excess_conf"].min(),
                "excess_conf_max": table["excess_conf"].max(),
                "folds_positive_excess": int((table["excess_conf"] > 0).sum()),
                "pct_long": table["pct_long"].mean(),
                "long_net": table["long_net"].mean(),
                "long_win": table["long_win"].mean(),
                "short_net": table["short_net"].mean(),
                "short_win": table["short_win"].mean(),
                "n_folds_with_rows": int(table["net_conf"].notna().sum()),
                **{f"vs_baseline_{k}": v for k, v in comparison.items() if k != "n_folds"},
            }
        )
    return pd.DataFrame(rows), fold_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate ensemble confidence filters walk-forward.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--target-horizon-days", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--out-dir", default="artifacts/confidence_eval")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore persisted predictions and retrain.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip logging aggregate results to MLflow.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = round_trip_cost_fraction()

    symbols = resolve_symbols(args.symbols, args.universe)
    df = None
    feature_cols: list[str] = []

    preds_by_diversity: dict[str, pd.DataFrame] = {}
    for diversity in ("seed", "structural"):
        cache = out_dir / f"predictions_{args.feature_set_id}_{diversity}.csv.gz"
        if cache.exists() and not args.force_retrain:
            print(f"[{diversity}] reusing {cache}")
            preds = pd.read_csv(cache, parse_dates=["ts"])
        else:
            if df is None:
                df = load_training_frame(args.feature_set_id, symbols, args.target_horizon_days)
                feature_cols = [c for c in df.columns if c not in ("symbol", "ts", "close", "fwd_return")]
            print(f"[{diversity}] training {args.n_folds} folds x {args.n_ensemble_models} members...")
            preds = collect_fold_predictions(
                df,
                feature_cols,
                n_folds=args.n_folds,
                purge_days=args.target_horizon_days,
                n_ensemble_models=args.n_ensemble_models,
                diversity=diversity,
            )
            preds.to_csv(cache, index=False)
            print(f"[{diversity}] saved {len(preds)} rows -> {cache}")
        preds_by_diversity[diversity] = add_summary_columns(preds)

    aggregate, fold_tables = evaluate(preds_by_diversity, cost)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print("\n=== Aggregate (mean over folds; paired tests vs seed/agree80 on excess_conf) ===")
    print(aggregate.to_string(index=False))

    aggregate.to_csv(out_dir / f"aggregate_{args.feature_set_id}.csv", index=False)
    for (diversity, name), table in fold_tables.items():
        table.to_csv(out_dir / f"folds_{args.feature_set_id}_{diversity}_{name}.csv", index=False)
    print(f"\nPer-fold tables written to {out_dir}")

    if not args.no_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("confidence_eval")
        for row in aggregate.itertuples(index=False):
            with mlflow.start_run(run_name=f"{args.feature_set_id}_{row.diversity}_{row.variant}"):
                mlflow.log_params(
                    {
                        "feature_set_id": args.feature_set_id,
                        "diversity": row.diversity,
                        "variant": row.variant,
                        "n_folds": args.n_folds,
                        "n_ensemble_models": args.n_ensemble_models,
                        "target_horizon_days": args.target_horizon_days,
                        "round_trip_cost_fraction": cost,
                    }
                )
                mlflow.log_metrics(
                    {
                        k: getattr(row, k)
                        for k in (
                            "pct_pass", "acc_all", "acc_conf", "acc_edge",
                            "benchmark", "net_conf", "excess_conf", "excess_conf_std",
                            "pct_long", "long_net", "long_win", "short_net", "short_win",
                            "vs_baseline_mean_diff", "vs_baseline_t_pvalue",
                            "vs_baseline_wilcoxon_pvalue",
                        )
                        if pd.notna(getattr(row, k))
                    }
                )
    print(
        "\nRead acc_edge (accuracy-when-confident minus accuracy-overall) together "
        "with pct_pass: a filter only earns its keep if it is selective AND the "
        "kept rows are more often right, with a p-value small enough to take "
        "seriously across this few folds.\n"
        "Judge profitability on excess_conf (net_conf minus benchmark), never on "
        "net_conf alone. benchmark is what equal-weight buy-and-hold of every row "
        "in the fold paid over the same window, gross of costs. A filter can raise "
        "net_conf simply by keeping more longs while the market rises — pct_long, "
        "long_net and short_net are printed so that is visible rather than hidden "
        "inside an average.\n"
        "p-values are paired across ~10 folds, NOT across rows: forward-return "
        "windows overlap heavily between consecutive days, so row-level tests would "
        "count the same information many times over and read far too significant. "
        "The fold-level values here are still mildly optimistic, since adjacent "
        "folds share market regimes.\n"
        "The universe is today's S&P 500 membership — survivorship-biased, which "
        "flatters both the benchmark and any long-heavy strategy."
    )


if __name__ == "__main__":
    main()
