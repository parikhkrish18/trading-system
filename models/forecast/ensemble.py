"""
Bagged ensemble around ForecastModel — a way to get a genuine confidence
signal out of an otherwise point-estimate regressor. Trains K models with
different bagging/feature-sampling seeds; a symbol only counts as
"confident" (see models/screener.py) if the members agree on direction and
clear a magnitude threshold, not just because one point prediction happens
to be large. A single LightGBM model can't tell you "how sure" it is —
disagreement across an ensemble is the honest proxy for that.

Diversity modes: with diversity="seed" (the original behavior) members
differ only by bagging/feature-sampling seed, which turned out to produce
near-identical models — the v4 walk-forward showed ~96% of predictions
clearing the 0.8 agreement bar, so "confident" carried no information.
diversity="structural" additionally varies tree depth, feature fraction,
and (when `ts` is passed to fit) the recency of each member's training
window, so disagreement measures genuine model uncertainty instead of
seed noise. models/confidence_eval.py measures whether either mode's
disagreement actually predicts being right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.forecast.lgbm_forecast import ForecastModel

# Per-member structural overrides, cycled modulo len() when n_models > 5:
# (num_leaves, feature_fraction, train_window_frac). The first entry is the
# production single-model config so the ensemble mean stays anchored to it.
# train_window_frac < 1.0 trains that member on only the most recent
# fraction of distinct trading days — a member that has never seen 2023
# will genuinely disagree with one that has when the regime shifts.
_STRUCTURAL_GRID: list[tuple[int, float, float]] = [
    (15, 0.8, 1.0),
    (31, 0.6, 1.0),
    (7, 0.8, 1.0),
    (15, 0.5, 0.6),
    (31, 0.8, 0.4),
]


def recent_window_mask(ts: pd.Series, window_frac: float) -> np.ndarray:
    """
    Boolean mask keeping rows in the most recent `window_frac` of distinct
    dates in ts. Date-based, not positional: training frames are sorted by
    symbol first, so positional truncation would drop whole symbols
    instead of old history.
    """
    dates = pd.DatetimeIndex(pd.unique(ts)).sort_values()
    cutoff = dates[int(len(dates) * (1.0 - window_frac))]
    return np.asarray(pd.DatetimeIndex(ts) >= cutoff)


class EnsembleForecastModel:
    def __init__(
        self,
        n_models: int = 5,
        params: dict | None = None,
        num_boost_round: int = 200,
        base_seed: int = 42,
        diversity: str = "seed",
    ):
        if diversity not in ("seed", "structural"):
            raise ValueError(f"diversity must be 'seed' or 'structural', got {diversity!r}")
        self.n_models = n_models
        self.base_seed = base_seed
        self.diversity = diversity
        self._params = params or {}
        self._num_boost_round = num_boost_round
        self.models: list[ForecastModel] = []

    def _member_overrides(self, i: int) -> tuple[dict, float]:
        """Per-member (param overrides, train_window_frac) for the diversity mode."""
        if self.diversity == "seed":
            return {}, 1.0
        num_leaves, feature_fraction, window_frac = _STRUCTURAL_GRID[i % len(_STRUCTURAL_GRID)]
        return {"num_leaves": num_leaves, "feature_fraction": feature_fraction}, window_frac

    def fit(self, X: pd.DataFrame, y: pd.Series, ts: pd.Series | None = None) -> None:
        """
        ts: per-row timestamps aligned with X, only consulted when
        diversity="structural" — members with train_window_frac < 1.0 train
        on the most recent fraction of distinct dates. Without ts those
        members silently fall back to the full window (rows are sorted by
        symbol first in the training frame, so positional truncation would
        drop whole symbols, not old history).
        """
        self.models = []
        for i in range(self.n_models):
            seed = self.base_seed + i
            overrides, window_frac = self._member_overrides(i)
            params = {
                **self._params,
                **overrides,
                "bagging_seed": seed,
                "feature_fraction_seed": seed,
                "data_random_seed": seed,
            }
            X_i, y_i = X, y
            if window_frac < 1.0 and ts is not None:
                mask = recent_window_mask(ts, window_frac)
                X_i, y_i = X.loc[mask], y.loc[mask]
            model = ForecastModel(params=params, num_boost_round=self._num_boost_round)
            model.fit(X_i, y_i)
            self.models.append(model)

    def predict_members(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Raw per-member predictions, one column per member (member_0..member_K-1),
        indexed like X. This is what confidence_eval.py persists so candidate
        confidence definitions can be compared offline on identical predictions
        without retraining.
        """
        if not self.models:
            raise RuntimeError("Ensemble not trained yet — call fit() first.")
        preds = np.column_stack([m.predict(X) for m in self.models])  # (n_rows, n_models)
        return pd.DataFrame(preds, columns=[f"member_{i}" for i in range(len(self.models))], index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame indexed like X with columns:
          mean_prediction    — average predicted forward return across members
          std_prediction     — spread across members (higher = less agreement)
          direction_agreement — fraction of members agreeing with the
                                 majority sign, in [0.5, 1.0]. 1.0 means every
                                 member predicts the same direction.
        """
        return summarize_members(self.predict_members(X))

    def feature_importance(self) -> pd.Series:
        """Average gain-based feature importance across ensemble members."""
        if not self.models:
            raise RuntimeError("Ensemble not trained yet.")
        importances = pd.concat([m.feature_importance() for m in self.models], axis=1)
        return importances.mean(axis=1).sort_values(ascending=False)

    def predict_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions (see ForecastModel.predict_contributions), averaged across members."""
        if not self.models:
            raise RuntimeError("Ensemble not trained yet — call fit() first.")
        contribs = [m.predict_contributions(X) for m in self.models]
        return sum(contribs) / len(contribs)


def summarize_members(members: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a (n_rows x n_members) prediction matrix into the ensemble
    summary columns predict() has always returned. Split out so
    confidence_eval.py can recompute the summary from persisted member
    predictions without retraining anything.
    """
    preds = members.to_numpy()
    signs = np.sign(preds)
    positive_frac = (signs > 0).mean(axis=1)
    agreement = np.maximum(positive_frac, 1 - positive_frac)

    return pd.DataFrame(
        {
            "mean_prediction": preds.mean(axis=1),
            "std_prediction": preds.std(axis=1),
            "direction_agreement": agreement,
        },
        index=members.index,
    )
