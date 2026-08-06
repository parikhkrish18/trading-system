"""
Per-pick evidence: which features actually moved the model's forecast for
one specific symbol, and by how much.

models/forecast/ensemble.py can already say "here is the predicted return"
and "here is how much the members agreed", but neither answers *why* a
symbol was picked. feature_importance() doesn't either — it's a model-wide
average, identical for every symbol scored by that model. The row-level
contributions from LightGBM's `pred_contrib=True` do answer it: each
feature gets a signed number saying how far it pushed this row's
prediction away from the model's base value, and those numbers sum exactly
to the prediction the screener acted on.

This module turns that raw matrix into a small, ranked, persistable record
per symbol. It is deliberately free of presentation concerns — the
plain-English phrasing lives in monitoring/dashboard/evidence.py, so the
stored evidence stays the model's numbers and only the wording around it
can change.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

# Six is enough to explain a pick without turning the panel into a data
# dump: past that the contributions are typically an order of magnitude
# below the leaders and add noise rather than insight.
DEFAULT_TOP_N = 6


@dataclasses.dataclass
class FeatureContribution:
    """One feature's signed push on one symbol's forecast."""

    feature: str
    value: float | None  # the feature's own value for this symbol; None when missing/NaN
    contribution: float  # signed: positive pushed the forecast up, negative pushed it down
    rank: int  # 1 = largest absolute contribution for this symbol


@dataclasses.dataclass
class PickEvidence:
    """The ranked shortlist of what moved one symbol's forecast."""

    symbol: str
    base_value: float  # the model's expected output before any feature moved it
    contributions: list[FeatureContribution]

    @property
    def total_contribution(self) -> float:
        """Sum of the shown contributions — deliberately not the full forecast (only the top N are kept)."""
        return float(sum(c.contribution for c in self.contributions))


def _clean_value(value) -> float | None:
    """Feature values reach here as NaN when a symbol has no data for them; store NULL, not NaN."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def top_contributions(
    contribution_row: np.ndarray,
    feature_cols: list[str],
    feature_values: pd.Series | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[FeatureContribution]:
    """
    Rank one row of a `pred_contrib=True` matrix by absolute contribution and
    keep the strongest `top_n`.

    `contribution_row` is one row of shape (n_features + 1,) — the trailing
    base-value column is the caller's business (see extract_evidence) and is
    not treated as a feature here. Features that contributed exactly zero are
    dropped rather than padded in: a tree that never split on a feature says
    nothing about it, and "0.0000 → no effect" is noise in an evidence panel.
    """
    contributions = np.asarray(contribution_row, dtype=float)[: len(feature_cols)]
    order = np.argsort(-np.abs(contributions))

    rows: list[FeatureContribution] = []
    for position in order:
        contribution = float(contributions[position])
        if contribution == 0.0:
            continue
        if len(rows) >= top_n:
            break
        name = feature_cols[position]
        value = _clean_value(feature_values.get(name)) if feature_values is not None else None
        rows.append(
            FeatureContribution(
                feature=name, value=value, contribution=contribution, rank=len(rows) + 1
            )
        )
    return rows


def extract_evidence(
    ensemble,
    latest_features: pd.DataFrame,
    feature_cols: list[str],
    symbols: list[str] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, PickEvidence]:
    """
    Evidence for every symbol in `latest_features` (or just `symbols`, when
    given — the shortlist is a handful of names out of ~500, so scoring
    contributions for all of them and discarding most is wasted work).

    `ensemble` needs only a `predict_contributions(X)` returning
    (n_rows, n_features + 1); EnsembleForecastModel provides it. An ensemble
    without that method (an older pickle, a stub in a test) yields no
    evidence rather than an error — evidence is an explanation of a decision,
    never a precondition for making one.

    `latest_features` is one row per symbol, as produced by
    models.screener.load_latest_features.
    """
    if latest_features.empty or not feature_cols:
        return {}
    if not hasattr(ensemble, "predict_contributions"):
        return {}

    frame = latest_features
    if symbols is not None:
        frame = frame.loc[frame["symbol"].isin(list(symbols))]
        if frame.empty:
            return {}

    X = frame.reindex(columns=feature_cols)
    contributions = np.asarray(ensemble.predict_contributions(X), dtype=float)
    if contributions.ndim != 2 or contributions.shape[0] != len(frame):
        raise ValueError(
            f"predict_contributions returned shape {contributions.shape}, "
            f"expected ({len(frame)}, {len(feature_cols) + 1})"
        )

    evidence: dict[str, PickEvidence] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        symbol = str(row["symbol"])
        contribution_row = contributions[position]
        # Trailing column is the base value when present; a caller-supplied
        # matrix without it just means no base value is known.
        base_value = (
            float(contribution_row[len(feature_cols)])
            if contribution_row.shape[0] > len(feature_cols)
            else 0.0
        )
        evidence[symbol] = PickEvidence(
            symbol=symbol,
            base_value=base_value,
            contributions=top_contributions(
                contribution_row, feature_cols, feature_values=row, top_n=top_n
            ),
        )
    return evidence


def evidence_rows(
    evidence: PickEvidence, ts, feature_set_id: str, model_version: str
) -> list[dict]:
    """
    One PickEvidence flattened into `decision_evidence` rows (see
    data/schema/004_decision_evidence.sql). `ts` is the batch timestamp the
    matching `decisions` rows were written with — that plus symbol is what
    links a piece of evidence back to the decision it explains.
    """
    return [
        {
            "ts": ts,
            "symbol": evidence.symbol,
            "feature_set_id": feature_set_id,
            "model_version": model_version,
            "feature_name": contribution.feature,
            "feature_value": contribution.value,
            "contribution": contribution.contribution,
            "contribution_rank": contribution.rank,
            "base_value": evidence.base_value,
        }
        for contribution in evidence.contributions
    ]
