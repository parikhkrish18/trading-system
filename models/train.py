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

Usage:
    python -m models.train --feature-set-id v1 --symbols SPY,QQQ,TQQQ,SQQQ \\
        --target-horizon-days 5 --n-folds 6
"""
from __future__ import annotations

import argparse
import dataclasses

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.settings import settings
from data.ingest.db import get_engine
from models.forecast.lgbm_forecast import ForecastModel


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


def load_training_frame(feature_set_id: str, symbols: list[str], target_horizon_days: int) -> pd.DataFrame:
    """
    Pulls features + prices, pivots features long->wide, and builds the
    forward-return target. Target is built from *future* prices relative to
    each row's ts — this is the one place look-ahead would sneak in if the
    join were done wrong, so keep this function small and test it directly.
    """
    engine = get_engine()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)

    features = pd.read_sql(
        f"""SELECT symbol, ts, feature_name, value FROM features
            WHERE feature_set_id = '{feature_set_id}' AND symbol IN ({symbol_list})""",
        engine,
    )
    prices = pd.read_sql(
        f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts",
        engine,
    )
    if features.empty or prices.empty:
        raise ValueError("No features or prices found — run ingestion + build_features first.")

    wide = features.pivot_table(index=["symbol", "ts"], columns="feature_name", values="value").reset_index()
    merged = wide.merge(prices, on=["symbol", "ts"], how="inner").sort_values(["symbol", "ts"])

    merged["fwd_return"] = merged.groupby("symbol")["close"].transform(
        lambda s: s.shift(-target_horizon_days) / s - 1
    )
    return merged.dropna(subset=["fwd_return"])


def run_walk_forward(
    feature_set_id: str,
    symbols: list[str],
    target_horizon_days: int = 5,
    n_folds: int = 6,
    model_name: str = "forecast_lgbm",
) -> pd.DataFrame:
    df = load_training_frame(feature_set_id, symbols, target_horizon_days)
    feature_cols = [c for c in df.columns if c not in ("symbol", "ts", "close", "fwd_return")]

    folds = make_expanding_folds(pd.DatetimeIndex(df["ts"]), n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough history to build any walk-forward folds — reduce n_folds or backfill more data.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(model_name)

    results = []
    for fold in folds:
        train_df = df[(df["ts"] >= fold.train_start) & (df["ts"] < fold.train_end)]
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
                    "n_features": len(feature_cols),
                }
            )

            model = ForecastModel()
            model.fit(train_df[feature_cols], train_df["fwd_return"])
            preds = model.predict(test_df[feature_cols])

            mae = mean_absolute_error(test_df["fwd_return"], preds)
            rmse = np.sqrt(mean_squared_error(test_df["fwd_return"], preds))
            # Directional accuracy matters at least as much as magnitude error
            # for a system that ultimately just takes long/short/flat decisions.
            directional_acc = float(np.mean(np.sign(preds) == np.sign(test_df["fwd_return"])))

            mlflow.log_metrics({"mae": mae, "rmse": rmse, "directional_accuracy": directional_acc})

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
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward train/evaluate the forecast model.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--target-horizon-days", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=6)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    results = run_walk_forward(args.feature_set_id, symbols, args.target_horizon_days, args.n_folds)

    print(results.to_string(index=False))
    print(
        "\nPer the plan: do not touch position sizing or live logic until this is "
        "stable across multiple folds, not just one lucky split. Check "
        "directional_accuracy and rmse variance across folds above, not just the mean."
    )


if __name__ == "__main__":
    main()
