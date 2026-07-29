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

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        importances = self.model.feature_importance(importance_type="gain")
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def predict_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Per-prediction, per-feature contribution values (Tree SHAP, via
        LightGBM's pred_contrib) — genuine "why did the model predict this"
        for a single row, not just global feature importance. Columns are
        X's features plus a trailing "base_value" column; each row's
        contributions + base_value sum to that row's raw prediction.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        contrib = self.model.predict(X, pred_contrib=True, num_iteration=self.model.best_iteration or None)
        columns = [*self.feature_names, "base_value"]
        return pd.DataFrame(contrib, columns=columns, index=X.index)

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
