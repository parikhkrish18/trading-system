"""
Return-forecast model. Gradient-boosted trees per the plan: interpretable,
robust at modest data volumes, easy to inspect feature importance — a
reasonable first model before reaching for anything fancier.
"""
from __future__ import annotations

from typing import ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd


class ForecastModel:
    """Thin wrapper around LightGBM so train.py doesn't need to know library details."""

    DEFAULT_PARAMS: ClassVar[dict] = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,          # kept small deliberately — modest data volume, avoid overfit
        "learning_rate": 0.05,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }

    def __init__(self, params: dict | None = None, num_boost_round: int = 200):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> None:
        self.feature_names = list(X.columns)
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")

        callbacks = [lgb.log_evaluation(period=0)]
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet — call fit() first.")
        return self.model.predict(X, num_iteration=self.model.best_iteration or None)

    def predict_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Per-row, per-feature SHAP contributions for this model's predictions —
        LightGBM's `pred_contrib=True`. Returns shape (n_rows, n_features + 1);
        the trailing column is the base value (the model's expected output
        before any feature moves it), so each row sums exactly to that row's
        prediction. That exact-sum property is why this is used for the "why
        this pick" evidence rather than feature_importance(), which is a
        model-wide average and can't say anything about one symbol.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet — call fit() first.")
        contributions = self.model.predict(
            X, num_iteration=self.model.best_iteration or None, pred_contrib=True
        )
        return np.asarray(contributions)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        importances = self.model.feature_importance(importance_type="gain")
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str, feature_names: list[str]) -> ForecastModel:
        instance = cls()
        instance.model = lgb.Booster(model_file=path)
        instance.feature_names = feature_names
        return instance
