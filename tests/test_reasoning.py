from monitoring import reasoning
from risk.circuit_breakers import BreakerResult


def test_explain_feature_formats_percentage_features_and_direction():
    line = reasoning.explain_feature("mom_ret_5d", 0.081, contribution=0.02)
    assert "8.1%" in line
    assert "pushed the forecast up" in line


def test_explain_feature_negative_contribution_reads_down():
    line = reasoning.explain_feature("sentiment_mean_3d", -0.5, contribution=-0.01)
    assert "pushed the forecast down" in line


def test_explain_feature_unknown_feature_falls_back_to_readable_name():
    line = reasoning.explain_feature("some_new_feature", 1.0, contribution=0.01)
    assert "Some new feature" in line


def test_explain_feature_handles_missing_value():
    line = reasoning.explain_feature("mom_ret_5d", None, contribution=0.01)
    assert "unavailable" in line


def test_phase_pretrade_risk_clear():
    phase = reasoning.phase_pretrade_risk([])
    assert phase["phase"] == 1
    assert "clear" in phase["summary"].lower()


def test_phase_pretrade_risk_triggered():
    phase = reasoning.phase_pretrade_risk([BreakerResult(True, "drawdown breach")])
    assert "drawdown breach" in phase["summary"]


def test_phase_signals_includes_regime_and_features():
    top_features = [{"feature_name": "mom_ret_5d", "value": 0.05, "contribution": 0.02}]
    phase = reasoning.phase_signals("trend", top_features)
    assert phase["phase"] == 2
    assert "TREND" in phase["lines"][0]
    assert any("momentum" in line for line in phase["lines"])


def test_phase_forecast_reports_agreement_against_threshold():
    phase = reasoning.phase_forecast(0.03, 0.9, 0.027, min_direction_agreement=0.8)
    assert phase["phase"] == 3
    assert "90%" in phase["summary"]
    assert "80%" in " ".join(phase["lines"])


def test_phase_selection_single_candidate_notes_fallback():
    phase = reasoning.phase_selection("AAPL", "long", 1.0, n_confident=1, n_selected=1, max_leg_pct=0.7, min_leg_pct=0.3)
    assert "alone" in " ".join(phase["lines"])


def test_phase_selection_at_bound_notes_the_cap():
    phase = reasoning.phase_selection("AAPL", "long", 0.7, n_confident=2, n_selected=2, max_leg_pct=0.7, min_leg_pct=0.3)
    assert "cap" in " ".join(phase["lines"]).lower()


def test_phase_selection_closed_explains_why():
    phase = reasoning.phase_selection_closed("OLD1")
    assert phase["phase"] == 4
    assert "OLD1" in phase["summary"]


def test_phase_execution_open_vs_close():
    opened = reasoning.phase_execution("AAPL", "opened", 10.0, "market")
    closed = reasoning.phase_execution("AAPL", "closed", None, "market")
    assert "10" in opened["summary"]
    assert "closed" in closed["summary"]


def test_phase_reconciliation_flags_divergence():
    ok = reasoning.phase_reconciliation("AAPL", 10.0, 10.0, flagged=False)
    bad = reasoning.phase_reconciliation("AAPL", 10.0, 5.0, flagged=True)
    assert "reconciled cleanly" in ok["summary"]
    assert "FLAGGED" in bad["summary"]


def test_phase_ongoing_monitoring_open_vs_closed():
    open_phase = reasoning.phase_ongoing_monitoring(closed=False)
    closed_phase = reasoning.phase_ongoing_monitoring(closed=True)
    assert "hourly" in open_phase["summary"].lower()
    assert "nothing to monitor" in closed_phase["summary"].lower()


def test_phase_contradiction_lists_each_reason():
    reasons = [
        {"signal": "news_sentiment", "detail": "sentiment turned negative"},
        {"signal": "price_momentum", "detail": "price dropped 6%"},
    ]
    phase = reasoning.phase_contradiction(reasons)
    assert phase["lines"] == ["sentiment turned negative", "price dropped 6%"]
    assert "news_sentiment" in phase["summary"]


def test_combine_phases_sorts_by_phase_number():
    p3 = {"phase": 3}
    p1 = {"phase": 1}
    p2 = {"phase": 2}
    assert reasoning.combine_phases(p3, p1, p2) == [p1, p2, p3]
