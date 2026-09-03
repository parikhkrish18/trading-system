import numpy as np
import pandas as pd
import pytest

from models.regime.trend_chop_classifier import TREND
from models.screener import (
    TradeCandidate,
    _attach_reasoning,
    _bounded_conviction_weights,
    apply_short_preference,
    attach_exit_levels,
    build_correlation_matrix,
    daily_volatility,
    score_universe,
    select_concentrated_trades,
    select_trades,
)


class _FakeEnsemble:
    """Fakes EnsembleForecastModel.predict with a preset mapping of row index -> stats."""

    def __init__(self, mean_prediction, direction_agreement, std_prediction=None, contributions=None):
        self.mean_prediction = mean_prediction
        self.direction_agreement = direction_agreement
        self.std_prediction = std_prediction if std_prediction is not None else [0.0] * len(mean_prediction)
        self._contributions = contributions

    def predict(self, X):
        return pd.DataFrame(
            {
                "mean_prediction": self.mean_prediction,
                "std_prediction": self.std_prediction,
                "direction_agreement": self.direction_agreement,
            },
            index=X.index,
        )

    def predict_contributions(self, X):
        return self._contributions.loc[X.index]


def test_score_universe_confidence_is_only_the_cost_hurdle():
    """
    One bar: is the predicted move bigger than what trading it costs. The
    agreement bar is gone — it admitted ~96% of predictions and the rows it
    called confident were no more accurate than the rest.
    """
    latest = pd.DataFrame({"symbol": ["AAPL", "TSLA", "MMM"], "f1": [0.1, 0.2, 0.3]})
    ensemble = _FakeEnsemble(mean_prediction=[0.05, -0.03, 0.01], direction_agreement=[1.0, 0.6, 0.9])

    result = score_universe(ensemble, latest, feature_cols=["f1"], min_abs_return=0.02)

    by_symbol = result.set_index("symbol")
    assert by_symbol.loc["AAPL", "confident"]  # |return| 0.05 >= 0.02
    assert by_symbol.loc["TSLA", "confident"]  # 0.6 agreement no longer disqualifies a 3% move
    assert not by_symbol.loc["MMM", "confident"]  # |return| 0.01 < 0.02


def test_conviction_score_is_the_size_of_the_move_alone():
    """
    Ranking used to be agreement x |move|, which mixed a noise term into
    who got the most capital.
    """
    latest = pd.DataFrame({"symbol": ["BIG", "SMALL"], "f1": [0.1, 0.2]})
    ensemble = _FakeEnsemble(mean_prediction=[0.05, 0.01], direction_agreement=[0.6, 1.0])

    result = score_universe(ensemble, latest, feature_cols=["f1"], min_abs_return=0.0)

    by_symbol = result.set_index("symbol")
    assert by_symbol.loc["BIG", "conviction_score"] == pytest.approx(0.05)
    assert by_symbol.loc["SMALL", "conviction_score"] == pytest.approx(0.01)
    # The bigger move ranks first despite the weaker agreement.
    assert list(result["symbol"]) == ["BIG", "SMALL"]


def test_score_universe_empty_input_returns_empty_with_columns():
    ensemble = _FakeEnsemble(mean_prediction=[], direction_agreement=[])
    result = score_universe(ensemble, pd.DataFrame(), feature_cols=["f1"])
    assert result.empty
    assert list(result.columns) == ["symbol", "predicted_return", "direction_agreement", "conviction_score", "confident"]


def _scored_df(rows):
    df = pd.DataFrame(rows)
    df["conviction_score"] = df["direction_agreement"] * df["predicted_return"].abs()
    return df


def test_select_trades_sizes_via_target_position_size_and_respects_top_k():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.04, "direction_agreement": 0.9, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.03, "direction_agreement": 0.85, "confident": True},
        ]
    )
    corr = pd.DataFrame(
        {"AAPL": [1.0, 0.0, 0.0], "TSLA": [0.0, 1.0, 0.0], "MMM": [0.0, 0.0, 1.0]},
        index=["AAPL", "TSLA", "MMM"],
    )

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
        top_k=2,
    )

    assert len(candidates) == 2
    assert [c.symbol for c in candidates] == ["AAPL", "TSLA"]  # highest conviction first
    assert all(c.side == "long" for c in candidates)


def test_select_trades_picks_short_side_for_negative_forecast():
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True}]
    )
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
    )

    assert len(candidates) == 1
    assert candidates[0].side == "short"
    assert candidates[0].target_position_pct < 0
    assert candidates[0].target_position_pct >= -0.15  # short cap, not the (looser) long cap


def test_select_trades_drops_unconfident_rows():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MMM", "predicted_return": 0.20, "direction_agreement": 0.55, "confident": False},
        ]
    )
    corr = pd.DataFrame({"AAPL": [1.0, 0.0], "MMM": [0.0, 1.0]}, index=["AAPL", "MMM"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
    )

    assert [c.symbol for c in candidates] == ["AAPL"]


def test_select_trades_skips_unshortable_candidates():
    scored = _scored_df(
        [{"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 0.9, "confident": True}]
    )
    corr = pd.DataFrame({"TSLA": [1.0]}, index=["TSLA"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        is_shortable_fn=lambda symbol: False,
    )

    assert candidates == []


def test_select_trades_shortable_check_does_not_affect_longs():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        is_shortable_fn=lambda symbol: False,  # would block a short, must not block this long
    )

    assert len(candidates) == 1


def test_select_trades_enforces_correlated_exposure_cap_across_its_own_picks():
    """
    Regression for the bug where select_trades only checked a new pick
    against the externally-passed current_positions and never folded its
    OWN newly-sized picks into that exposure as it went — so four
    pairwise-correlated symbols, each individually under the cap versus an
    empty starting book, could each get sized to the full single-position
    cap independently and land the combined book at 2x the correlated-
    exposure limit (100% instead of the 50% cap).

    With the fix, exposure accumulates across the loop: the first two picks
    fill the entire 50% correlated-exposure cap between them (0.25 each),
    which leaves exactly zero headroom for a third or fourth — so they size
    to 0 and get filtered out by the ordinary "no signal" skip-check, same
    as any other zero-sized candidate. The combined book stays at, never
    over, the cap.
    """
    scored = _scored_df(
        [
            {"symbol": "A", "predicted_return": 0.20, "direction_agreement": 1.0, "confident": True},
            {"symbol": "B", "predicted_return": 0.19, "direction_agreement": 1.0, "confident": True},
            {"symbol": "C", "predicted_return": 0.18, "direction_agreement": 1.0, "confident": True},
            {"symbol": "D", "predicted_return": 0.17, "direction_agreement": 1.0, "confident": True},
        ]
    )
    symbols = ["A", "B", "C", "D"]
    # All four pairwise-correlated at 0.9 — well above the 0.7 threshold.
    corr = pd.DataFrame(0.9, index=symbols, columns=symbols)
    for s in symbols:
        corr.loc[s, s] = 1.0

    candidates = select_trades(
        scored,
        regime=TREND,
        forecast_scale=0.05,  # every forecast saturates the confidence scaling
        max_position_pct=0.25,
        max_short_position_pct=0.15,
        max_correlated_exposure_pct=0.50,
        correlation_matrix=corr,
        top_k=10,
        current_positions={},  # empty starting book
    )

    total_exposure = sum(abs(c.target_position_pct) for c in candidates)
    assert total_exposure <= 0.50 + 1e-9  # the whole point: never past the cap, unlike the buggy version (1.0)
    assert total_exposure == pytest.approx(0.50)  # and it's not left needlessly under-deployed either
    assert {c.symbol for c in candidates} == {"A", "B"}  # first two by rank fill the cap; C/D get zero headroom
    for c in candidates:
        assert c.target_position_pct == pytest.approx(0.25)


def test_select_trades_does_not_mutate_the_callers_current_positions_dict():
    """current_positions is folded into internally as each pick is sized, but the caller's own dict must be untouched."""
    scored = _scored_df(
        [{"symbol": "A", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"A": [1.0]}, index=["A"])
    caller_positions = {"EXISTING": 0.1}

    select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
        current_positions=caller_positions,
    )

    assert caller_positions == {"EXISTING": 0.1}


def test_select_trades_skips_nan_sized_candidates(monkeypatch):
    """
    A NaN target_position_pct must never survive into a candidate, even if
    it originates somewhere upstream that isn't itself NaN-guarded — the
    `abs(size) < 1e-9` skip-check alone doesn't catch NaN (NaN comparisons
    are always False in Python), so screener.py needs its own explicit
    isnan check as defense in depth, independent of risk.sizing's own guard.
    """
    import models.screener as scr

    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])
    monkeypatch.setattr(scr, "target_position_size", lambda **kwargs: float("nan"))

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
    )

    assert candidates == []


def test_select_trades_nan_forecast_scale_yields_no_candidates():
    """Integration-level: a NaN forecast_scale (e.g. std() of a too-small training frame) must not produce a candidate."""
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    )
    corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])

    candidates = select_trades(
        scored, regime=TREND, forecast_scale=float("nan"), max_position_pct=0.25,
        max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
    )

    assert candidates == []


class TestApplyShortPreference:
    """
    apply_short_preference adds a `rank_score` column used only to decide
    WHICH candidates get selected — the long/short preference the user asked
    for (slight edge to longs, waived for a confident short with contained
    downside). It must never touch conviction_score itself, which sizing
    still reads unmodified.
    """

    def test_no_op_when_penalty_is_zero(self):
        scored = _scored_df(
            [
                {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
                {"symbol": "TSLA", "predicted_return": -0.05, "direction_agreement": 1.0, "confident": True},
            ]
        )
        out = apply_short_preference(scored, vol_by_symbol={}, horizon_days=20, penalty=0.0, low_risk_stop_loss_pct=0.06)
        assert (out["rank_score"] == out["conviction_score"]).all()
        # conviction_score itself is untouched
        assert out["conviction_score"].tolist() == scored["conviction_score"].tolist()

    def test_longs_are_never_handicapped(self):
        scored = _scored_df(
            [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
        )
        out = apply_short_preference(scored, vol_by_symbol={}, horizon_days=20, penalty=0.15, low_risk_stop_loss_pct=0.06)
        assert out.loc[0, "rank_score"] == pytest.approx(out.loc[0, "conviction_score"])

    def test_risky_short_is_handicapped(self):
        """High volatility -> wide derived stop-loss -> above the low-risk bar -> full penalty applies."""
        scored = _scored_df(
            [{"symbol": "TSLA", "predicted_return": -0.05, "direction_agreement": 1.0, "confident": True}]
        )
        out = apply_short_preference(
            scored, vol_by_symbol={"TSLA": 0.08}, horizon_days=20, penalty=0.15, low_risk_stop_loss_pct=0.06,
        )
        assert out.loc[0, "rank_score"] == pytest.approx(out.loc[0, "conviction_score"] * 0.85)
        # conviction_score itself is untouched — sizing later reads this, not rank_score
        assert out.loc[0, "conviction_score"] == pytest.approx(0.05)

    def test_low_risk_confident_short_is_exempted(self):
        """Low volatility -> tight derived stop-loss -> at/below the low-risk bar -> no handicap."""
        scored = _scored_df(
            [{"symbol": "KO", "predicted_return": -0.05, "direction_agreement": 1.0, "confident": True}]
        )
        # A calm stock: small daily vol -> small horizon-sigma -> stop-loss lands under the 0.06 bar.
        out = apply_short_preference(
            scored, vol_by_symbol={"KO": 0.003}, horizon_days=20, penalty=0.15, low_risk_stop_loss_pct=0.06,
        )
        assert out.loc[0, "rank_score"] == pytest.approx(out.loc[0, "conviction_score"])

    def test_unmeasurable_volatility_does_not_qualify_for_the_exemption(self):
        """No vol data -> exit_levels_for falls back to the global default stop-loss (wider than 0.06) -> handicapped."""
        scored = _scored_df(
            [{"symbol": "NEWCO", "predicted_return": -0.05, "direction_agreement": 1.0, "confident": True}]
        )
        out = apply_short_preference(
            scored, vol_by_symbol={}, horizon_days=20, penalty=0.15, low_risk_stop_loss_pct=0.06,
        )
        assert out.loc[0, "rank_score"] == pytest.approx(out.loc[0, "conviction_score"] * 0.85)

    def test_empty_input(self):
        out = apply_short_preference(pd.DataFrame(), vol_by_symbol={}, horizon_days=20, penalty=0.15, low_risk_stop_loss_pct=0.06)
        assert out.empty


class TestSelectionRespectsRankScoreCol:
    def test_select_trades_uses_rank_score_when_given(self):
        """
        Without a preference, TSLA (conviction 0.06) would outrank AAPL
        (0.05). A rank_score that reverses that ordering must change who
        gets the single top_k=1 slot — proving the ranking column, not
        conviction_score, decides selection order when provided.
        """
        scored = _scored_df(
            [
                {"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
                {"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 1.0, "confident": True},
            ]
        )
        scored["rank_score"] = [0.05, 0.03]  # TSLA's short handicapped below AAPL's long
        corr = pd.DataFrame({"AAPL": [1.0, 0.0], "TSLA": [0.0, 1.0]}, index=["AAPL", "TSLA"])

        candidates = select_trades(
            scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
            max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
            top_k=1, rank_score_col="rank_score",
        )
        assert [c.symbol for c in candidates] == ["AAPL"]

    def test_select_trades_falls_back_to_conviction_score_when_column_missing(self):
        scored = _scored_df(
            [{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
        )
        corr = pd.DataFrame({"AAPL": [1.0]}, index=["AAPL"])
        candidates = select_trades(
            scored, regime=TREND, forecast_scale=0.05, max_position_pct=0.25,
            max_short_position_pct=0.15, max_correlated_exposure_pct=0.50, correlation_matrix=corr,
            rank_score_col="rank_score",  # not present on `scored` -> must not raise, falls back
        )
        assert len(candidates) == 1

    def test_select_concentrated_trades_uses_rank_score_when_given(self):
        scored = _scored_df(
            [
                {"symbol": "AAPL", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
                {"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 1.0, "confident": True},
                {"symbol": "MMM", "predicted_return": 0.015, "direction_agreement": 1.0, "confident": True},
            ]
        )
        # Raw conviction order would be TSLA(0.06) > AAPL(0.02) > MMM(0.015) -> top 2 = TSLA, AAPL.
        # Handicap TSLA below MMM so the top 2 becomes AAPL, MMM instead.
        scored["rank_score"] = [0.02, 0.01, 0.015]
        candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6, rank_score_col="rank_score")
        assert {c.symbol for c in candidates} == {"AAPL", "MMM"}
        # sizing weight still uses the real conviction_score (0.02 vs 0.015), not rank_score
        by_symbol = {c.symbol: c for c in candidates}
        assert by_symbol["AAPL"].target_position_pct > by_symbol["MMM"].target_position_pct


def test_select_concentrated_trades_splits_by_relative_conviction():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    # conviction_score: AAPL=0.04, TSLA=0.02 -> raw split 2:1 -> 0.667/0.333, within [0.30, 0.70] bounds.
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6)

    assert len(candidates) == 2
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].target_position_pct == pytest.approx(2 / 3, rel=1e-6)
    assert by_symbol["TSLA"].target_position_pct == pytest.approx(1 / 3, rel=1e-6)
    # fully deployed
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(1.0)


def test_select_concentrated_trades_clamps_dominant_leg_at_bound():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.50, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.01, "direction_agreement": 1.0, "confident": True},
        ]
    )
    # Raw split would be ~98/2 — clamped to 70/30 so one pick can't swallow the whole deployment.
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6)

    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].target_position_pct == pytest.approx(0.70)
    assert by_symbol["TSLA"].target_position_pct == pytest.approx(0.30)


def test_select_concentrated_trades_signs_match_forecast_direction():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": -0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6)
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["AAPL"].side == "long"
    assert by_symbol["AAPL"].target_position_pct > 0
    assert by_symbol["TSLA"].side == "short"
    assert by_symbol["TSLA"].target_position_pct < 0


def test_select_concentrated_trades_single_confident_candidate_goes_all_in():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True}]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6)

    assert len(candidates) == 1
    assert candidates[0].target_position_pct == pytest.approx(1.0)


def test_select_concentrated_trades_no_confident_candidates_returns_empty():
    scored = _scored_df(
        [{"symbol": "AAPL", "predicted_return": 0.20, "direction_agreement": 0.55, "confident": False}]
    )
    assert select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6) == []


def test_select_concentrated_trades_respects_total_deploy_pct():
    """Regime-based damping (see run_screen) scales both legs down together."""
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6, total_deploy_pct=0.35)
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(0.35)


def test_select_concentrated_trades_skips_unshortable_and_falls_through_ranking():
    scored = _scored_df(
        [
            {"symbol": "TSLA", "predicted_return": -0.06, "direction_agreement": 1.0, "confident": True},  # unshortable
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MSFT", "predicted_return": 0.03, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(
        scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6,
        is_shortable_fn=lambda symbol: symbol != "TSLA",
    )

    symbols = {c.symbol for c in candidates}
    assert symbols == {"AAPL", "MSFT"}  # TSLA skipped, next two ranked candidates picked instead


def test_select_concentrated_trades_respects_max_positions_cap():
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.06, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MSFT", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True},
            {"symbol": "NVDA", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "TSLA", "predicted_return": 0.03, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6, max_positions=3)
    assert {c.symbol for c in candidates} == {"AAPL", "MSFT", "NVDA"}  # top 3 by conviction, TSLA dropped
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(1.0)


def test_select_concentrated_trades_splits_three_legs_weighted_with_a_floor():
    """
    Regression coverage for the min 2 / max 3 concentrated book: the raw
    proportional split (60/30/10) would squeeze the third pick to a token
    10% sliver -- the floor (0.6 * 1/3 = 20%) rescues it to a real size,
    and the freed 10% redistributes across the other two by their own
    relative conviction (2:1), not evenly.
    """
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.06, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MSFT", "predicted_return": 0.03, "direction_agreement": 1.0, "confident": True},
            {"symbol": "NVDA", "predicted_return": 0.01, "direction_agreement": 1.0, "confident": True},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6, max_positions=3)

    assert len(candidates) == 3
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["NVDA"].target_position_pct == pytest.approx(0.20)
    assert by_symbol["AAPL"].target_position_pct == pytest.approx(0.5333333, rel=1e-4)
    assert by_symbol["MSFT"].target_position_pct == pytest.approx(0.2666667, rel=1e-4)
    assert sum(abs(c.target_position_pct) for c in candidates) == pytest.approx(1.0)


def test_select_concentrated_trades_never_forces_a_third_pick_to_hit_max_positions():
    """max_positions=3 but only 2 clear the confidence bar -- the cap is a ceiling, never a floor to force-fill."""
    scored = _scored_df(
        [
            {"symbol": "AAPL", "predicted_return": 0.04, "direction_agreement": 1.0, "confident": True},
            {"symbol": "MSFT", "predicted_return": 0.02, "direction_agreement": 1.0, "confident": True},
            {"symbol": "NVDA", "predicted_return": 0.001, "direction_agreement": 0.55, "confident": False},
        ]
    )
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6, max_positions=3)
    assert {c.symbol for c in candidates} == {"AAPL", "MSFT"}


class TestBoundedConvictionWeights:
    """
    Regression coverage for the water-filling bug: when a dominant leg needs
    capping AND the other legs need flooring in the SAME iteration, the old
    code fixed every violating leg to its bound without checking whether
    those bounds summed to more than the remaining budget — e.g. with
    production defaults (max_leg_pct=0.70, min_leg_floor_fraction=0.6) and a
    3-way conviction spread this dominant, [0.70, 0.20, 0.20] summed to
    1.0999999999999999, a 10% over-deployment.
    """

    def test_multi_leg_simultaneous_violation_never_exceeds_the_budget(self):
        # Exact reproduction: one leg wants ~98% of the split (violates the
        # 0.70 cap) while the other two want ~1% each (violate the 0.20
        # floor) — ALL THREE violate in the very first water-filling pass.
        weights = _bounded_conviction_weights([100.0, 1.0, 1.0], max_leg_pct=0.70, min_leg_floor_fraction=0.6)

        assert sum(weights) <= 1.0 + 1e-9
        assert weights[0] <= 0.70 + 1e-9
        for w in weights:
            assert w >= 0.20 - 1e-9  # this feasible case can fully honor the floor too
        assert sum(weights) == pytest.approx(1.0)  # and does so with the whole book deployed

    def test_multi_leg_violation_regardless_of_which_leg_is_dominant(self):
        """Same shape, dominant leg in a different position — order shouldn't matter."""
        weights = _bounded_conviction_weights([1.0, 100.0, 1.0], max_leg_pct=0.70, min_leg_floor_fraction=0.6)
        assert sum(weights) <= 1.0 + 1e-9
        assert all(w <= 0.70 + 1e-9 for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_never_exceeds_the_cap_or_the_budget_across_many_conviction_spreads(self):
        """Broader sweep at production bounds: no spread should ever push a leg over its cap or the total over 1.0."""
        max_leg_pct, min_leg_floor_fraction = 0.70, 0.6
        spreads = [
            [100.0, 1.0, 1.0],
            [1.0, 1.0, 100.0],
            [50.0, 49.0, 1.0],
            [0.04, 0.02, 0.001],
            [10.0, 10.0],
            [10.0, 0.01],
            [1.0, 1.0, 1.0, 1.0],
        ]
        for scores in spreads:
            weights = _bounded_conviction_weights(scores, max_leg_pct=max_leg_pct, min_leg_floor_fraction=min_leg_floor_fraction)
            assert sum(weights) <= 1.0 + 1e-9, scores
            assert all(w <= max_leg_pct + 1e-9 for w in weights), scores

    def test_logs_a_warning_when_the_floor_cannot_be_honored_within_the_cap(self, caplog):
        """
        Genuinely infeasible bounds (max_leg_pct below what an equal split
        would need) can't honor every floor without breaching the cap or the
        1.0 total — both of those are the hard invariants, so the floor
        yields, and that under-deployment must be logged, not silent.
        """
        with caplog.at_level("WARNING"):
            weights = _bounded_conviction_weights([1.0, 1.0], max_leg_pct=0.3, min_leg_floor_fraction=0.6)

        assert sum(weights) <= 1.0 + 1e-9
        assert sum(weights) < 1.0 - 1e-9  # genuinely under-deployed, not just capped exactly at 1.0
        assert any("under-deployed" in r.message for r in caplog.records)


def test_attach_reasoning_picks_top_features_by_absolute_contribution():
    scored = _scored_df([{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}])
    candidates = select_concentrated_trades(scored, max_leg_pct=0.70, min_leg_floor_fraction=0.6)
    latest = pd.DataFrame({"symbol": ["AAPL"], "f1": [1.5], "f2": [-0.5], "f3": [0.1]})
    # Indexed by symbol, matching what _attach_reasoning's X (latest.set_index("symbol")) actually looks like.
    contributions = pd.DataFrame(
        {"f1": [0.03], "f2": [-0.08], "f3": [0.001], "base_value": [0.0]}, index=pd.Index(["AAPL"], name="symbol")
    )
    ensemble = _FakeEnsemble(mean_prediction=[0.05], direction_agreement=[1.0], contributions=contributions)

    _attach_reasoning(
        candidates, ensemble, latest, feature_cols=["f1", "f2", "f3"], scored=scored, regime=TREND,
        max_leg_pct=0.70, min_leg_floor_fraction=0.6, top_n=2,
    )

    phases = {p["phase"]: p for p in candidates[0].reasoning}
    assert set(phases) == {2, 3, 4}
    lines = " ".join(phases[2]["lines"]).lower()
    assert lines.index("f2") < lines.index("f1")  # ranked by |contribution|, f3 excluded (top_n=2)
    assert "f3" not in lines


def test_attach_reasoning_empty_candidates_is_noop():
    ensemble = _FakeEnsemble(mean_prediction=[], direction_agreement=[])
    _attach_reasoning(
        [], ensemble, pd.DataFrame(), feature_cols=[], scored=pd.DataFrame({"confident": []}),
        regime=TREND, max_leg_pct=0.70, min_leg_floor_fraction=0.6,    )  # should not raise


# --- strategy dispatch ----------------------------------------------------


def _run_screen_harness(monkeypatch, mode, *, full_deployment=False, diversified_result=None):
    """
    Wire run_screen's heavy dependencies to fakes so the dispatch itself can
    be exercised: which selector ran, with which settings-derived arguments.
    """
    import models.screener as scr

    monkeypatch.setattr(scr.settings, "strategy_mode", mode)
    monkeypatch.setattr(scr.settings, "screener_top_k", 7)
    monkeypatch.setattr(scr.settings, "full_deployment", full_deployment)
    monkeypatch.setattr(scr.settings, "max_single_position_pct", 0.25)
    monkeypatch.setattr(scr.settings, "max_short_position_pct", 0.15)
    monkeypatch.setattr(scr.settings, "max_correlated_exposure_pct", 0.50)

    dates = pd.bdate_range("2026-01-01", periods=3, tz="UTC")
    # `target` is what the model is fitted on (absolute forward return here,
    # cross-sectional excess under TARGET_MODE=relative); `fwd_return` stays
    # absolute so money is always measured in money.
    train_df = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "ts": dates,
            "close": [1.0, 1.1, 1.2],
            "fwd_return": [0.01, 0.02, 0.03],
            "target": [0.01, 0.02, 0.03],
            "f1": [1, 2, 3],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    class _NoopEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y): ...

    monkeypatch.setattr(scr, "EnsembleForecastModel", _NoopEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["A"], "f1": [3]}))
    monkeypatch.setattr(scr, "score_universe", lambda *a, **k: _scored_df(
        [{"symbol": "A", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    ))
    monkeypatch.setattr(scr, "build_correlation_matrix", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)

    calls = {}

    def fake_select_trades(scored, **kwargs):
        calls["diversified"] = kwargs
        return list(diversified_result or [])

    def fake_select_concentrated(scored, **kwargs):
        calls["concentrated"] = kwargs
        return []

    monkeypatch.setattr(scr, "select_trades", fake_select_trades)
    monkeypatch.setattr(scr, "select_concentrated_trades", fake_select_concentrated)
    return scr, calls


def test_run_screen_diversified_mode_uses_the_topk_book_with_conservative_caps(monkeypatch):
    scr, calls = _run_screen_harness(monkeypatch, "diversified")

    scr.run_screen("v3", ["A"])

    assert "diversified" in calls and "concentrated" not in calls
    assert calls["diversified"]["top_k"] == 7  # settings.screener_top_k
    assert calls["diversified"]["max_position_pct"] == 0.25
    assert calls["diversified"]["max_short_position_pct"] == 0.15
    assert calls["diversified"]["max_correlated_exposure_pct"] == 0.50
    assert calls["diversified"]["forecast_scale"] == pytest.approx(0.01)  # std of fwd_return


def test_run_screen_concentrated_mode_uses_the_two_trade_split(monkeypatch):
    scr, calls = _run_screen_harness(monkeypatch, "concentrated")

    scr.run_screen("v3", ["A"])

    assert "concentrated" in calls and "diversified" not in calls


def test_run_screen_concentrated_mode_passes_position_count_and_split_settings(monkeypatch):
    scr, calls = _run_screen_harness(monkeypatch, "concentrated")
    monkeypatch.setattr(scr.settings, "max_concentrated_position_pct", 0.70)
    monkeypatch.setattr(scr.settings, "min_concentrated_leg_floor_fraction", 0.6)
    monkeypatch.setattr(scr.settings, "max_concentrated_positions", 3)

    scr.run_screen("v3", ["A"])

    assert calls["concentrated"]["max_leg_pct"] == 0.70
    assert calls["concentrated"]["min_leg_floor_fraction"] == 0.6
    assert calls["concentrated"]["max_positions"] == 3


def test_run_screen_concentrated_mode_max_positions_override_wins_over_setting(monkeypatch):
    """execution/contradiction_monitor.py's reactivation passes this to cap new picks at the open slot count."""
    scr, calls = _run_screen_harness(monkeypatch, "concentrated")
    monkeypatch.setattr(scr.settings, "max_concentrated_positions", 3)

    scr.run_screen("v3", ["A"], max_positions_override=1)

    assert calls["concentrated"]["max_positions"] == 1


def test_run_screen_diversified_scales_sizes_by_the_freed_capital_fraction(monkeypatch):
    from models.screener import TradeCandidate

    candidate = TradeCandidate(
        symbol="A", side="long", predicted_return=0.05, direction_agreement=1.0,
        conviction_score=0.05, target_position_pct=0.20,
    )
    scr, _calls = _run_screen_harness(monkeypatch, "diversified", diversified_result=[candidate])

    result = scr.run_screen("v3", ["A"], total_deploy_pct=0.5)

    assert result[0].target_position_pct == pytest.approx(0.10)  # 0.20 * 0.5


def test_run_screen_diversified_honors_full_deployment(monkeypatch):
    from models.screener import TradeCandidate

    candidate = TradeCandidate(
        symbol="A", side="long", predicted_return=0.05, direction_agreement=1.0,
        conviction_score=0.05, target_position_pct=0.10,
    )
    scr, _calls = _run_screen_harness(
        monkeypatch, "diversified", full_deployment=True, diversified_result=[candidate]
    )

    result = scr.run_screen("v3", ["A"])

    # One pick under a 25% cap: scaled up to the cap, shortfall logged, never past it.
    assert result[0].target_position_pct == pytest.approx(0.25)


def test_run_screen_diversified_threads_current_positions_to_select_trades(monkeypatch):
    """run_screen_with_scores must pass its caller's current_positions through to select_trades, not default it away."""
    scr, calls = _run_screen_harness(monkeypatch, "diversified")

    scr.run_screen("v3", ["A"], current_positions={"EXISTING": 0.1})

    assert calls["diversified"]["current_positions"] == {"EXISTING": 0.1}


def test_run_screen_diversified_defaults_current_positions_to_none_when_not_given(monkeypatch):
    """No broker/portfolio context available (e.g. offline scoring) -> select_trades sees None, which it treats as {}."""
    scr, calls = _run_screen_harness(monkeypatch, "diversified")

    scr.run_screen("v3", ["A"])

    assert calls["diversified"]["current_positions"] is None


def test_run_screen_diversified_skips_screening_on_nan_forecast_scale(monkeypatch):
    """
    A training frame with <=1 usable row makes std() (forecast_scale) NaN —
    every downstream size would be NaN too, so the cycle must be skipped
    rather than shortlisting against a meaningless scale.
    """
    import models.screener as scr

    monkeypatch.setattr(scr.settings, "strategy_mode", "diversified")
    monkeypatch.setattr(scr.settings, "screener_top_k", 7)
    monkeypatch.setattr(scr.settings, "full_deployment", False)
    monkeypatch.setattr(scr.settings, "max_single_position_pct", 0.25)
    monkeypatch.setattr(scr.settings, "max_short_position_pct", 0.15)
    monkeypatch.setattr(scr.settings, "max_correlated_exposure_pct", 0.50)

    dates = pd.bdate_range("2026-01-01", periods=1, tz="UTC")
    train_df = pd.DataFrame(
        {
            "symbol": ["A"], "ts": dates, "close": [1.0], "fwd_return": [0.01], "target": [0.01], "f1": [1],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    class _NoopEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y): ...

    monkeypatch.setattr(scr, "EnsembleForecastModel", _NoopEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["A"], "f1": [3]}))
    monkeypatch.setattr(scr, "score_universe", lambda *a, **k: _scored_df(
        [{"symbol": "A", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}]
    ))
    called = {}
    monkeypatch.setattr(scr, "select_trades", lambda *a, **k: called.setdefault("ran", True))
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)

    result = scr.run_screen("v3", ["A"])

    assert result == []
    assert "ran" not in called  # select_trades never even called


def test_attach_reasoning_diversified_wording_tells_the_topk_story():
    from models.screener import TradeCandidate

    scored = _scored_df([{"symbol": "AAPL", "predicted_return": 0.05, "direction_agreement": 1.0, "confident": True}])
    candidates = [
        TradeCandidate(
            symbol="AAPL", side="long", predicted_return=0.05, direction_agreement=1.0,
            conviction_score=0.05, target_position_pct=0.10,
        )
    ]
    latest = pd.DataFrame({"symbol": ["AAPL"], "f1": [1.5]})
    contributions = pd.DataFrame({"f1": [0.03], "base_value": [0.0]}, index=pd.Index(["AAPL"], name="symbol"))
    ensemble = _FakeEnsemble(mean_prediction=[0.05], direction_agreement=[1.0], contributions=contributions)

    _attach_reasoning(
        candidates, ensemble, latest, feature_cols=["f1"], scored=scored, regime=TREND,
        max_leg_pct=0.70, min_leg_floor_fraction=0.6,        strategy="diversified", top_k=10,
    )

    phase4 = next(p for p in candidates[0].reasoning if p["phase"] == 4)
    text = " ".join(phase4["lines"])
    assert "diversified book" in text
    assert "up to 10" in text
    assert "two picks" not in text  # the concentrated split story must not leak in


def test_build_correlation_matrix_from_prices():
    dates = pd.bdate_range("2026-01-01", periods=30, tz="UTC")
    rng = np.random.default_rng(0)
    shared = rng.normal(size=30)
    prices = pd.DataFrame(
        {
            "symbol": ["A"] * 30 + ["B"] * 30,
            "ts": list(dates) * 2,
            "close": list(100 * np.cumprod(1 + 0.001 * shared)) + list(100 * np.cumprod(1 + 0.001 * shared)),
        }
    )
    corr = build_correlation_matrix(prices)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)  # identical return series


# --------------------------------------------------------------------------
# Per-pick exit levels
# --------------------------------------------------------------------------


def _price_history(symbol, daily_moves):
    """A price series with the given daily returns, starting at 100."""
    closes, price = [], 100.0
    for move in daily_moves:
        price *= 1 + move
        closes.append(price)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": pd.bdate_range("2026-01-01", periods=len(closes)),
            "close": closes,
        }
    )


def test_daily_volatility_separates_a_calm_stock_from_a_jumpy_one():
    calm = _price_history("KO", [0.001, -0.001] * 30)
    jumpy = _price_history("MRNA", [0.05, -0.05] * 30)

    vols = daily_volatility(pd.concat([calm, jumpy]))

    assert vols["MRNA"] > vols["KO"]


def test_daily_volatility_omits_symbols_without_enough_history():
    """
    Absent, not zero. A missing measurement has to stay distinguishable
    from a genuinely motionless stock — they call for opposite fallbacks.
    """
    vols = daily_volatility(_price_history("NEW", [0.01] * 5), window=20)

    assert vols == {}


def test_attach_exit_levels_gives_each_candidate_its_own_pair():
    calm = TradeCandidate("KO", "long", 0.04, 1.0, 0.04, 0.1)
    jumpy = TradeCandidate("MRNA", "long", 0.04, 1.0, 0.04, 0.1)

    attach_exit_levels([calm, jumpy], {"KO": 0.004, "MRNA": 0.04})

    assert jumpy.exit_levels.stop_loss_pct > calm.exit_levels.stop_loss_pct


def test_attach_exit_levels_falls_back_when_a_symbol_has_no_volatility():
    candidate = TradeCandidate("NEW", "long", 0.04, 1.0, 0.04, 0.1)

    attach_exit_levels([candidate], {})

    assert candidate.exit_levels.derived is False
