"""
Regime classifier: trend vs. chop. The screener uses this per symbol
(models/screener.py:per_symbol_regimes) to damp position sizes for names
in a choppy, range-bound state — see risk.sizing.regime_adjusted_size.
(Originally written to gate the leveraged-ETF book, where chop additionally
causes daily-reset decay — see backtest/decay_sim.py, kept but inactive.)

Ships two implementations:
  - RuleBasedRegime: ADX-threshold rule, zero training required, good enough
    to unblock Phase 3/4 development immediately.
  - TrainableRegime: a real classifier (gradient-boosted trees) once you
    have labeled regime history and want something that adapts.

Swap which one train.py / execution use via the `regime_mode` config — don't
delete the rule-based version, it's a useful sanity-check baseline even
after the trained model ships.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

TREND = "trend"
CHOP = "chop"


class RuleBasedRegime:
    """ADX above threshold => trending; below => choppy. No training needed."""

    def __init__(self, adx_threshold: float = 25.0):
        self.adx_threshold = adx_threshold

    def predict(self, adx_values: pd.Series) -> pd.Series:
        return np.where(adx_values >= self.adx_threshold, TREND, CHOP)


class TrainableRegime:
    """
    Gradient-boosted classifier over regime features (ADX, vol-of-vol, etc).
    Labels for training should be constructed via a forward-looking rule at
    train time only (e.g. "was realized vol of a buy-and-hold position over
    the next N days low relative to trend strength") — never leak forward
    information into the *features* themselves.
    """

    def __init__(self, **kwargs):
        defaults = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        self.model = GradientBoostingClassifier(**{**defaults, **kwargs})

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
