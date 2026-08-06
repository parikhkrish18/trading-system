import datetime as dt

import numpy as np
import pandas as pd
import pytest

from models.evidence import (
    FeatureContribution,
    PickEvidence,
    evidence_rows,
    extract_evidence,
    top_contributions,
)
from models.forecast.ensemble import EnsembleForecastModel
from models.screener import TradeCandidate, attach_evidence, log_evidence

FEATURE_COLS = ["mom_ret_20d", "adx_14", "meanrev_rsi_14", "vol_realized_20d"]


class _FakeContribEnsemble:
    """
    Stands in for EnsembleForecastModel: returns a preset contribution matrix
    of shape (n_rows, n_features + 1), base value in the trailing column —
    exactly what LightGBM's pred_contrib=True produces.
    """

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        self.seen_columns = None

    def predict_contributions(self, X):
        self.seen_columns = list(X.columns)
        return self.matrix[: len(X)]


def _latest_features():
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "TSLA"],
            "mom_ret_20d": [0.08, -0.04],
            "adx_14": [31.0, 12.0],
            "meanrev_rsi_14": [64.0, 41.0],
            "vol_realized_20d": [0.19, np.nan],
        }
    )


# --- top_contributions ----------------------------------------------------


def test_top_contributions_ranks_by_absolute_size_not_signed_size():
    row = np.array([0.001, -0.009, 0.004, 0.0002, 0.05])  # last entry is the base value

    contributions = top_contributions(row, FEATURE_COLS)

    assert [c.feature for c in contributions] == ["adx_14", "meanrev_rsi_14", "mom_ret_20d", "vol_realized_20d"]
    assert [c.rank for c in contributions] == [1, 2, 3, 4]
    # a big negative push outranks a smaller positive one, and keeps its sign
    assert contributions[0].contribution == pytest.approx(-0.009)


def test_top_contributions_keeps_only_top_n():
    row = np.array([0.001, -0.009, 0.004, 0.0002, 0.05])

    contributions = top_contributions(row, FEATURE_COLS, top_n=2)

    assert [c.feature for c in contributions] == ["adx_14", "meanrev_rsi_14"]


def test_top_contributions_drops_features_the_model_never_used():
    """A feature no tree split on contributed exactly 0 — that's not evidence, it's noise."""
    row = np.array([0.001, 0.0, 0.0, 0.004, 0.05])

    contributions = top_contributions(row, FEATURE_COLS)

    assert [c.feature for c in contributions] == ["vol_realized_20d", "mom_ret_20d"]


def test_top_contributions_attaches_the_feature_values():
    row = np.array([0.01, 0.005, 0.002, 0.001, 0.05])
    values = _latest_features().iloc[0]

    contributions = top_contributions(row, FEATURE_COLS, feature_values=values)

    by_feature = {c.feature: c.value for c in contributions}
    assert by_feature["mom_ret_20d"] == pytest.approx(0.08)
    assert by_feature["adx_14"] == pytest.approx(31.0)


def test_top_contributions_stores_missing_feature_values_as_none_not_nan():
    """NaN would land in the DB as NaN rather than NULL, and reads as a number downstream."""
    row = np.array([0.001, 0.002, 0.003, 0.01, 0.05])
    values = _latest_features().iloc[1]  # TSLA has no vol_realized_20d

    contributions = top_contributions(row, FEATURE_COLS, feature_values=values)

    assert contributions[0].feature == "vol_realized_20d"
    assert contributions[0].value is None


# --- extract_evidence -----------------------------------------------------


def test_extract_evidence_returns_ranked_evidence_per_symbol():
    matrix = [
        [0.010, 0.004, -0.001, 0.0005, 0.02],  # AAPL
        [-0.008, 0.001, 0.003, 0.0002, 0.02],  # TSLA
    ]
    ensemble = _FakeContribEnsemble(matrix)

    evidence = extract_evidence(ensemble, _latest_features(), FEATURE_COLS)

    assert set(evidence) == {"AAPL", "TSLA"}
    assert [c.feature for c in evidence["AAPL"].contributions][0] == "mom_ret_20d"
    assert evidence["AAPL"].base_value == pytest.approx(0.02)
    # the trailing base-value column is not itself reported as a feature
    assert all(c.feature in FEATURE_COLS for c in evidence["AAPL"].contributions)


def test_extract_evidence_scores_only_the_requested_symbols():
    """The shortlist is a handful of names out of ~500 — scoring the rest is wasted work."""
    ensemble = _FakeContribEnsemble([[0.01, 0.004, -0.001, 0.0005, 0.02]])

    evidence = extract_evidence(ensemble, _latest_features(), FEATURE_COLS, symbols=["TSLA"])

    assert set(evidence) == {"TSLA"}


def test_extract_evidence_passes_features_in_declared_column_order():
    ensemble = _FakeContribEnsemble([[0.01, 0.004, -0.001, 0.0005, 0.02]] * 2)

    extract_evidence(ensemble, _latest_features(), FEATURE_COLS)

    assert ensemble.seen_columns == FEATURE_COLS


def test_extract_evidence_without_contribution_support_returns_empty():
    """An older model that can't explain itself must not break the screener."""

    class _NoContribEnsemble:
        def predict(self, X):
            return pd.DataFrame(index=X.index)

    assert extract_evidence(_NoContribEnsemble(), _latest_features(), FEATURE_COLS) == {}


def test_extract_evidence_on_empty_features_returns_empty():
    ensemble = _FakeContribEnsemble([[0.01, 0.0, 0.0, 0.0, 0.02]])
    assert extract_evidence(ensemble, pd.DataFrame(), FEATURE_COLS) == {}


def test_extract_evidence_rejects_a_mismatched_contribution_matrix():
    ensemble = _FakeContribEnsemble([[0.01, 0.004, -0.001, 0.0005, 0.02]])  # 1 row, 2 symbols

    with pytest.raises(ValueError, match="expected"):
        extract_evidence(ensemble, _latest_features(), FEATURE_COLS)


# --- the real ensemble ----------------------------------------------------


def _synthetic_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})
    y = 0.5 * X["f1"] - 0.3 * X["f2"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_ensemble_contributions_sum_to_the_prediction_it_reports():
    """
    The property the whole evidence panel rests on: what's shown as the
    reasons must add up to the number the screener acted on, not to some
    other model's answer.
    """
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    contributions = ensemble.predict_contributions(X)
    predictions = ensemble.predict(X)["mean_prediction"].to_numpy()

    assert contributions.shape == (len(X), X.shape[1] + 1)
    np.testing.assert_allclose(contributions.sum(axis=1), predictions, rtol=1e-6, atol=1e-9)


def test_ensemble_predict_contributions_before_fit_raises():
    with pytest.raises(RuntimeError, match="not trained"):
        EnsembleForecastModel().predict_contributions(pd.DataFrame({"f1": [1.0]}))


# --- persistence ----------------------------------------------------------


def _evidence(symbol="AAPL"):
    return PickEvidence(
        symbol=symbol,
        base_value=0.02,
        contributions=[
            FeatureContribution(feature="mom_ret_20d", value=0.08, contribution=0.01, rank=1),
            FeatureContribution(feature="vol_realized_20d", value=None, contribution=-0.004, rank=2),
        ],
    )


def test_evidence_rows_flattens_one_pick_for_the_decision_evidence_table():
    ts = dt.datetime(2026, 8, 7, 13, 0, tzinfo=dt.UTC)

    rows = evidence_rows(_evidence(), ts, feature_set_id="v3", model_version="ensemble_v1")

    assert len(rows) == 2
    assert rows[0] == {
        "ts": ts,
        "symbol": "AAPL",
        "feature_set_id": "v3",
        "model_version": "ensemble_v1",
        "feature_name": "mom_ret_20d",
        "feature_value": 0.08,
        "contribution": 0.01,
        "contribution_rank": 1,
        "base_value": 0.02,
    }
    assert rows[1]["feature_value"] is None


def _candidate(symbol="AAPL", evidence=None):
    return TradeCandidate(
        symbol=symbol, side="long", predicted_return=0.03, direction_agreement=1.0,
        conviction_score=0.03, target_position_pct=0.1, evidence=evidence,
    )


def test_log_evidence_writes_one_row_per_contribution(monkeypatch):
    captured = {}

    def fake_to_sql(self, name, engine, **kwargs):
        captured["table"] = name
        captured["df"] = self

    monkeypatch.setattr("models.screener.get_engine", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    ts = dt.datetime(2026, 8, 7, 13, 0, tzinfo=dt.UTC)
    n = log_evidence([_candidate(evidence=_evidence())], feature_set_id="v3", ts=ts)

    assert n == 2
    assert captured["table"] == "decision_evidence"
    written = captured["df"]
    assert list(written["symbol"]) == ["AAPL", "AAPL"]
    assert list(written["contribution_rank"]) == [1, 2]
    # the (ts, symbol) pair is the only link back to the decisions rows
    assert set(written["ts"]) == {ts}


def test_log_evidence_writes_nothing_when_no_candidate_has_evidence(monkeypatch):
    def fail_to_sql(self, name, engine, **kwargs):
        raise AssertionError("should not have written anything")

    monkeypatch.setattr("models.screener.get_engine", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fail_to_sql)

    assert log_evidence([_candidate()], feature_set_id="v3") == 0
    assert log_evidence([], feature_set_id="v3") == 0


def test_log_candidates_records_direction_agreement(monkeypatch):
    captured = {}

    def fake_to_sql(self, name, engine, **kwargs):
        captured["df"] = self

    monkeypatch.setattr("models.screener.get_engine", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    from models.screener import log_candidates

    log_candidates([_candidate()], feature_set_id="v3")

    assert list(captured["df"]["direction_agreement"]) == [1.0]


# --- attach_evidence ------------------------------------------------------


def test_attach_evidence_fills_in_each_candidates_evidence():
    ensemble = _FakeContribEnsemble([[0.010, 0.004, -0.001, 0.0005, 0.02]])
    candidates = [_candidate(symbol="TSLA")]

    attach_evidence(candidates, ensemble, _latest_features(), FEATURE_COLS)

    assert candidates[0].evidence is not None
    assert candidates[0].evidence.symbol == "TSLA"
    assert candidates[0].evidence.contributions[0].feature == "mom_ret_20d"


def test_attach_evidence_never_drops_a_pick_when_explanation_fails():
    """Evidence explains a decision; it must not be able to change or block one."""

    class _BrokenEnsemble:
        def predict_contributions(self, X):
            raise RuntimeError("model blew up")

    candidates = [_candidate()]

    result = attach_evidence(candidates, _BrokenEnsemble(), _latest_features(), FEATURE_COLS)

    assert result == candidates
    assert candidates[0].evidence is None
