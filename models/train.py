"""
Phase 3, point 1 of the plan: "Build the walk-forward training harness
first, before touching model architecture." This module is deliberately
model-agnostic — it takes any object with .fit(X, y) / .predict(X) and
walks it forward across time, so swapping the forecast model architecture
later doesn't require touching this file.

Walk-forward scheme (expanding window):

    [-------- train --------][test]
    [----------- train -----------][test]
    [-------------- train -------------][test]

Each fold trains on everything up to a cutoff, tests on the next window,
then the cutoff rolls forward. This is the only way to get an honest
out-of-sample estimate on time-series data — a random train/test split
leaks future information into training.

PURGE GAP: the forward-return label of a row at time t is built from the
price `target_horizon_days` later (load_training_frame). A training row
sitting just before the train/test boundary therefore has a label computed
from prices *inside the test window* — that's label leakage, and it
inflates every metric. Each fold purges the last `target_horizon_days`
trading days of its training window so no training label overlaps the test
period.

PREDICTION TARGET: the model is fitted on the `target` column, which is
either the absolute forward return or the cross-sectional excess over the
same-day universe mean (TARGET_MODE / --target-mode, see
load_training_frame). Money is ALWAYS measured on absolute `fwd_return`,
in both modes, against the equal-weight buy-and-hold benchmark — see
models/evaluation.py for why every return figure here carries a baseline.

Usage:
    python -m models.train --feature-set-id v1 --symbols SPY,QQQ,TQQQ,SQQQ \\
        --target-horizon-days 5 --n-folds 10
    python -m models.train --feature-set-id v4 --universe --target-mode absolute
"""
from __future__ import annotations

import argparse
import dataclasses

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backtest.cost_model import round_trip_cost_fraction
from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from data.ingest.universe import resolve_symbols
from models.evaluation import cross_sectional_excess, cross_sectional_zscore, trade_metrics
from models.forecast.ensemble import EnsembleForecastModel


@dataclasses.dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_expanding_folds(dates: pd.DatetimeIndex, n_folds: int, min_train_frac: float = 0.4) -> list[Fold]:
    """
    Splits the sorted unique dates into n_folds expanding-window folds.
    The first `min_train_frac` of history is reserved as the minimum initial
    training window so early folds aren't trained on too little data.
    """
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(dates)
    min_train_end_idx = int(n * min_train_frac)
    remaining = n - min_train_end_idx
    fold_size = max(remaining // n_folds, 1)

    folds = []
    for i in range(n_folds):
        train_end_idx = min_train_end_idx + i * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + fold_size, n - 1)
        if test_start_idx >= n - 1:
            break
        folds.append(
            Fold(
                fold_id=i,
                train_start=dates[0],
                train_end=dates[train_end_idx],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx],
            )
        )
    return folds


def purged_train_cutoff(dates: pd.DatetimeIndex, train_end: pd.Timestamp, horizon_days: int) -> pd.Timestamp:
    """
    The timestamp `horizon_days` trading days before `train_end` — training
    rows at or after this cutoff have forward-return labels that reach into
    the test window and must be dropped. `dates` is the sorted unique
    trading-day index the folds were built from; horizon_days=0 disables
    the purge (used to measure how much leakage was inflating metrics).
    """
    if horizon_days <= 0:
        return train_end
    dates = pd.DatetimeIndex(dates)
    idx = dates.searchsorted(train_end)
    return dates[max(idx - horizon_days, 0)]


def load_feature_frame(feature_set_id: str, symbols: list[str]) -> pd.DataFrame:
    """
    Pulls features + prices and pivots features long->wide. Shared by
    load_training_frame (below, which adds a forward-return target) and
    models.screener.load_latest_features (which scores the most recent row
    per symbol — one that by definition has no forward return yet, since
    that would require future data).
    """
    engine = get_engine()
    symbol_list = symbol_in_clause(symbols)

    features = pd.read_sql(
        "SELECT symbol, ts, feature_name, value FROM features "  # noqa: S608 — symbols validated via symbol_in_clause; feature_set_id is a bind param
        f"WHERE feature_set_id = %(feature_set_id)s AND symbol IN ({symbol_list})",
        engine,
        params={"feature_set_id": feature_set_id},
    )
    prices = pd.read_sql(
        f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts",  # noqa: S608 — symbols validated via symbol_in_clause
        engine,
    )
    if features.empty or prices.empty:
        raise ValueError("No features or prices found — run ingestion + build_features first.")

    wide = features.pivot_table(index=["symbol", "ts"], columns="feature_name", values="value").reset_index()
    return wide.merge(prices, on=["symbol", "ts"], how="inner").sort_values(["symbol", "ts"])


NON_FEATURE_COLUMNS = ("symbol", "ts", "close", "fwd_return", "target")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything in a training frame that is a model input, in frame order."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def load_training_frame(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int,
    target_mode: str | None = None,
) -> pd.DataFrame:
    """
    Adds the forward-return target on top of load_feature_frame. Target is
    built from *future* prices relative to each row's ts — this is the one
    place look-ahead would sneak in if the join were done wrong, so keep
    this function small and test it directly.

    Always produces TWO return columns:

      fwd_return  the stock's absolute forward return. Never transformed.
                  Everything that measures money — the trade metrics, the
                  buy-and-hold benchmark — reads this, whatever the model
                  was trained on.
      target      what the model is trained to predict.

    target_mode (None = settings.target_mode):
      "absolute"  target == fwd_return. The original behaviour.
      "relative"  target = fwd_return minus the equal-weight mean fwd_return
                  of the universe on the same date, and every feature is
                  replaced by its cross-sectional z-score within its date.
                  The market move cancels out of both label and features, so
                  the model can only earn its keep by ranking stocks against
                  each other.

    The cross-sectional transforms are computed on the whole frame before
    the walk-forward splits it. That is not leakage: both use only rows
    sharing the SAME ts, so a training row's target and features depend on
    other stocks on that same day and never on any later date. The purge
    gap still handles the genuine leakage vector, which is the forward
    shift.
    """
    target_mode = settings.target_mode if target_mode is None else target_mode
    if target_mode not in ("absolute", "relative"):
        raise ValueError(f"target_mode must be 'absolute' or 'relative', got {target_mode!r}")

    # .copy() so the label columns are never written through a view onto a
    # caller's frame — pandas raises SettingWithCopyWarning for exactly this,
    # and a silent no-op write here would produce a frame with no labels.
    merged = load_feature_frame(feature_set_id, symbols).copy()
    merged["fwd_return"] = merged.groupby("symbol")["close"].transform(
        lambda s: s.shift(-target_horizon_days) / s - 1
    )
    merged = merged.dropna(subset=["fwd_return"])

    if target_mode == "absolute":
        merged["target"] = merged["fwd_return"]
        return merged

    merged["target"] = cross_sectional_excess(merged, "fwd_return", "ts")
    return cross_sectional_zscore(merged, feature_columns(merged), "ts")


def run_walk_forward(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int | None = None,
    n_folds: int = 10,
    model_name: str = "forecast_lgbm",
    n_ensemble_models: int = 5,
    confident_agreement_threshold: float = 0.8,
    purge_days: int | None = None,
    fold_dates: pd.DatetimeIndex | None = None,
    target_mode: str | None = None,
) -> pd.DataFrame:
    """
    n_folds defaults to 10 (was 6, and long before that 3): a mean over a
    handful of folds hides how unstable the metric is — report the per-fold
    spread, not just the average (main() prints both).

    target_horizon_days: None = the configured TARGET_HORIZON_DAYS.

    target_mode: None = the configured TARGET_MODE ("absolute" | "relative",
    see load_training_frame). The model is fitted on the `target` column,
    but every money metric — the trade returns, the buy-and-hold benchmark,
    the excess — is computed from absolute `fwd_return` in both modes, so
    the two are directly comparable.

    purge_days: how many trading days to cut off the end of each training
    window so no training label overlaps the test period (see module
    docstring). None = target_horizon_days (the correct value); 0 disables
    purging, which exists only to measure what the leakage was worth.

    fold_dates: when given, fold boundaries are built from THIS date index
    instead of the frame's own dates. A longer horizon loses its last
    `horizon` days to the label shift, so two horizons run back-to-back
    would otherwise get slightly different fold windows — passing one shared
    index (scripts/compare_horizons.py does) keeps a fold-by-fold
    comparison genuinely paired.
    """
    if target_horizon_days is None:
        target_horizon_days = settings.target_horizon_days
    target_mode = settings.target_mode if target_mode is None else target_mode
    purge_days = target_horizon_days if purge_days is None else purge_days
    df = load_training_frame(feature_set_id, symbols, target_horizon_days, target_mode)
    feature_cols = feature_columns(df)

    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(df["ts"]).unique()))
    folds = make_expanding_folds(fold_dates if fold_dates is not None else dates, n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough history to build any walk-forward folds — reduce n_folds or backfill more data.")

    # The hurdle a prediction must beat after paying to get in and out.
    # No per-symbol ADV here, so this is the spread-only floor — see
    # backtest/cost_model.py.
    round_trip_cost = round_trip_cost_fraction()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(model_name)

    results = []
    for fold in folds:
        cutoff = purged_train_cutoff(dates, fold.train_end, purge_days)
        train_df = df[(df["ts"] >= fold.train_start) & (df["ts"] < cutoff)]
        test_df = df[(df["ts"] >= fold.test_start) & (df["ts"] < fold.test_end)]
        if train_df.empty or test_df.empty:
            continue

        with mlflow.start_run(run_name=f"fold_{fold.fold_id}"):
            mlflow.log_params(
                {
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start),
                    "train_end": str(fold.train_end),
                    "test_start": str(fold.test_start),
                    "test_end": str(fold.test_end),
                    "feature_set_id": feature_set_id,
                    "target_horizon_days": target_horizon_days,
                    "target_mode": target_mode,
                    "n_features": len(feature_cols),
                    "n_ensemble_models": n_ensemble_models,
                    "confident_agreement_threshold": confident_agreement_threshold,
                    "purge_days": purge_days,
                    "purged_train_end": str(cutoff),
                    "round_trip_cost_fraction": round_trip_cost,
                }
            )

            model = EnsembleForecastModel(n_models=n_ensemble_models)
            model.fit(train_df[feature_cols], train_df["target"])
            ensemble_out = model.predict(test_df[feature_cols])
            preds = ensemble_out["mean_prediction"].to_numpy()
            # `actual` is whatever the model was trained to predict, so MAE,
            # RMSE and directional accuracy stay measured against the model's
            # own objective. `market` is the absolute forward return, always
            # — money is measured in money regardless of the target mode.
            actual = test_df["target"].to_numpy()
            market = test_df["fwd_return"].to_numpy()

            mae = mean_absolute_error(actual, preds)
            rmse = np.sqrt(mean_squared_error(actual, preds))
            # Directional accuracy matters at least as much as magnitude error
            # for a system that ultimately just takes long/short/flat decisions.
            # Two versions, because in relative mode they answer different
            # questions: did the stock beat the market (the model's own
            # objective), and did it go up at all (what a long actually needs).
            # In absolute mode the two are identical by construction.
            directional_acc = float(np.mean(np.sign(preds) == np.sign(actual)))
            directional_acc_absolute = float(np.mean(np.sign(preds) == np.sign(market)))

            # Does ensemble agreement actually correlate with being right?
            # This is what calibrates the screener's confidence threshold —
            # if accuracy on the "confident" subset isn't meaningfully better
            # than the overall accuracy above, agreement isn't a useful filter.
            confident_mask = (ensemble_out["direction_agreement"] >= confident_agreement_threshold).to_numpy()
            pct_confident = float(confident_mask.mean())
            directional_acc_confident = (
                float(np.mean(np.sign(preds[confident_mask]) == np.sign(actual[confident_mask])))
                if confident_mask.any()
                else float("nan")
            )

            # What trading the confident calls would have paid, AND what
            # doing nothing would have paid over the identical window.
            # `market` is every candidate row in the test window — the
            # equal-weight buy-and-hold baseline. Reporting the model's
            # return without it is how this project spent months mistaking
            # market drift for skill (see models/evaluation.py).
            metrics = trade_metrics(
                preds[confident_mask],
                market[confident_mask],
                market,
                round_trip_cost,
            )
            mlflow.log_metrics(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "directional_accuracy": directional_acc,
                    "directional_accuracy_absolute": directional_acc_absolute,
                    "directional_accuracy_when_confident": directional_acc_confident,
                    "pct_rows_confident": pct_confident,
                    "mean_ensemble_std": float(ensemble_out["std_prediction"].mean()),
                    # Historical names, still logged so runs from before the
                    # benchmark existed remain comparable in the MLflow UI.
                    # Nothing in this repo PRINTS them any more — a bare
                    # return figure with no benchmark beside it is the exact
                    # mistake this change exists to stop.
                    "mean_return_confident_gross": metrics["model_return_gross"],
                    "mean_return_confident_net": metrics["model_return_net"],
                    **{k: v for k, v in metrics.items() if pd.notna(v)},
                }
            )

            results.append(
                {
                    "fold_id": fold.fold_id,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "mae": mae,
                    "rmse": rmse,
                    "directional_accuracy": directional_acc,
                    "directional_accuracy_absolute": directional_acc_absolute,
                    "directional_accuracy_when_confident": directional_acc_confident,
                    "pct_rows_confident": pct_confident,
                    **metrics,
                }
            )

    return pd.DataFrame(results)


# Reported for every fold, in this order. excess_return sits directly under
# the two numbers it is the difference of, so the comparison can't be read
# past by accident.
SPREAD_COLUMNS = (
    "directional_accuracy",
    "directional_accuracy_absolute",
    "directional_accuracy_when_confident",
    "benchmark_return",
    "model_return_net",
    "excess_return",
    "long_return_net",
    "long_win_rate",
    "short_return_net",
    "short_win_rate",
)


def spread_summary(results: pd.DataFrame) -> str:
    """
    The per-fold spread, because a mean over folds hides instability: a
    strategy that's right 60% in three folds and 40% in three others is not
    a 50% strategy anyone should size positions on.

    Columns absent from `results` are skipped, so this still reads an older
    results frame (or a hand-built one in a test) that predates the
    benchmark metrics.
    """
    lines = []
    for col in SPREAD_COLUMNS:
        if col not in results.columns:
            continue
        vals = results[col].dropna()
        if vals.empty:
            lines.append(f"{col}: no folds produced a value")
            continue
        lines.append(
            f"{col}: mean {vals.mean():.4f} | std {vals.std():.4f} | "
            f"min {vals.min():.4f} | max {vals.max():.4f} | n_folds {len(vals)}"
        )
    return "\n".join(lines)


def headline_verdict(results: pd.DataFrame) -> str:
    """
    The one paragraph a reader should not be able to skip: what the model
    paid, what buying everything paid, and how many folds the model actually
    won. Written so a positive-but-losing result — the historical case here —
    reads as a failure rather than a success.
    """
    if "excess_return" not in results.columns:
        return "No benchmark columns in these results — rerun the harness."
    excess = results["excess_return"].dropna()
    if excess.empty:
        return "No fold produced a tradeable result, so there is nothing to compare."

    model = results["model_return_net"].dropna()
    bench = results["benchmark_return"].dropna()
    wins = int((excess > 0).sum())
    verdict = "BEATS" if excess.mean() > 0 else "LOSES TO"
    long_share = results["pct_long"].dropna()

    lines = [
        "=== Verdict vs doing nothing ===",
        f"model, net of costs : {model.mean():+.4%} per trade",
        f"buy-and-hold        : {bench.mean():+.4%} per candidate row, same windows, gross",
        f"EXCESS (the number) : {excess.mean():+.4%}  -> the model {verdict} buy-and-hold",
        f"folds with positive excess: {wins}/{len(excess)}",
    ]
    if not long_share.empty:
        lines.append(f"trades that were long: {long_share.mean():.1%}")
    lines.append(
        "A positive model return with a negative excess means the strategy made "
        "money and would have made more doing nothing. Excess is the only one of "
        "these three numbers that measures skill."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward train/evaluate the forecast model.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument(
        "--target-horizon-days", type=int, default=None,
        help=f"Forward-return horizon in trading days (default: TARGET_HORIZON_DAYS = {settings.target_horizon_days}).",
    )
    parser.add_argument(
        "--target-mode", choices=("absolute", "relative"), default=None,
        help=f"What the model predicts: 'absolute' forward return, or 'relative' "
             f"cross-sectional excess over the same-day universe mean with per-date "
             f"z-scored features (default: TARGET_MODE = {settings.target_mode}).",
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--confident-agreement-threshold", type=float, default=0.8)
    parser.add_argument(
        "--purge-days", type=int, default=None,
        help="Trading days purged from the end of each training window (default: the target horizon). "
             "0 disables purging — only useful to measure what the leakage was worth.",
    )
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    results = run_walk_forward(
        args.feature_set_id,
        symbols,
        args.target_horizon_days,
        args.n_folds,
        n_ensemble_models=args.n_ensemble_models,
        confident_agreement_threshold=args.confident_agreement_threshold,
        purge_days=args.purge_days,
        target_mode=args.target_mode,
    )

    print(results.to_string(index=False))
    print("\nPer-fold spread (the mean alone is not the story):")
    print(spread_summary(results))
    print("\n" + headline_verdict(results))
    print(
        "\nPer the plan: do not touch position sizing or live logic until this is "
        "stable across multiple folds, not just one lucky split. Check the spread "
        "above, not just the mean. If directional_accuracy_when_confident isn't "
        "meaningfully better than directional_accuracy, ensemble agreement isn't a "
        "useful confidence filter yet."
    )


if __name__ == "__main__":
    main()
