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


def test_explain_feature_momentum_narrative_reflects_magnitude():
    sharp_drop = reasoning.explain_feature("mom_ret_20d", -0.195, contribution=-0.01)
    assert "sold off sharply" in sharp_drop
    flat = reasoning.explain_feature("mom_ret_5d", 0.01, contribution=0.01)
    assert "roughly flat" in flat


def test_explain_feature_rsi_narrative_reflects_zone():
    oversold = reasoning.explain_feature("meanrev_rsi_14", 22.0, contribution=0.01)
    assert "oversold" in oversold
    overbought = reasoning.explain_feature("meanrev_rsi_14", 78.0, contribution=-0.01)
    assert "overbought" in overbought
    neutral = reasoning.explain_feature("meanrev_rsi_14", 50.0, contribution=0.01)
    assert "neutral" in neutral


def test_explain_feature_adx_narrative_reflects_trend_strength():
    weak = reasoning.explain_feature("adx_14", 12.0, contribution=-0.01)
    assert "choppy" in weak
    strong = reasoning.explain_feature("adx_14", 45.0, contribution=0.01)
    assert "strong" in strong.lower()


def test_explain_feature_volatility_narrative_reflects_level():
    calm = reasoning.explain_feature("vol_realized_20d", 0.10, contribution=0.01)
    assert "calmly" in calm
    turbulent = reasoning.explain_feature("vol_realized_20d", 1.11, contribution=-0.01)
    assert "unstable" in turbulent


def test_explain_feature_event_narrative_reflects_urgency():
    imminent = reasoning.explain_feature("days_to_next_cpi", 2.0, contribution=-0.01)
    assert "imminent" in imminent
    distant = reasoning.explain_feature("days_to_next_fomc", 40.0, contribution=0.01)
    assert "not yet a near-term factor" in distant


def test_explain_feature_event_narrative_preserves_acronym_capitalization():
    """Regression test: str.capitalize() lowercases the rest of the string, turning "CPI" into "cpi"."""
    line = reasoning.explain_feature("days_to_next_cpi", 10.0, contribution=-0.01)
    assert "CPI" in line
    assert "cpi" not in line


def test_explain_feature_sentiment_narrative_reflects_polarity():
    negative = reasoning.explain_feature("sentiment_mean_3d", -0.5, contribution=-0.01)
    assert "clearly negative" in negative
    positive = reasoning.explain_feature("sentiment_mean_10d", 0.5, contribution=0.01)
    assert "clearly positive" in positive


def test_explain_feature_all_known_features_produce_narratives_for_typical_values():
    """Every feature the model actually uses should get a real narrative, not the generic fallback."""
    sample_values = {
        "mom_ret_5d": 0.02, "mom_ret_20d": -0.05, "adx_14": 22.0, "vol_realized_20d": 0.3,
        "vol_atr_14": 3.5, "vol_of_vol": 0.02, "meanrev_zscore_20d": 0.5, "meanrev_bollinger_pctb": 0.5,
        "meanrev_rsi_14": 50.0, "sentiment_mean_10d": 0.1, "sentiment_mean_3d": 0.1,
        "sentiment_momentum_3v10": 0.05, "news_volume_3d": 4.0, "days_to_next_fomc": 10.0,
        "days_to_next_cpi": 10.0, "days_to_next_jobs": 10.0, "fund_eps_actual_latest": 2.5,
        "fund_revenue_actual_latest": 1_000_000.0, "fund_net_income_latest": 500_000.0,
        "fund_gross_profit_latest": 700_000.0, "fund_total_assets_latest": 5_000_000.0,
        "fund_total_liabilities_latest": 2_000_000.0,
    }
    assert set(sample_values) == set(reasoning._NARRATIVE_FNS)
    for feature_name, value in sample_values.items():
        line = reasoning.explain_feature(feature_name, value, contribution=0.01)
        assert "This pushed the forecast up" in line
        assert len(line) > 40  # a real sentence, not a bare number dump


def test_phase_pretrade_risk_clear():
    phase = reasoning.phase_pretrade_risk([])
    assert phase["phase"] == 1
    assert "clear" in phase["summary"].lower()


def test_phase_pretrade_risk_triggered():
    phase = reasoning.phase_pretrade_risk([BreakerResult(True, "drawdown breach")])
    assert "drawdown breach" in phase["summary"]


def test_phase_signals_includes_regime_and_features():
    top_features = [{"feature_name": "mom_ret_5d", "value": 0.08, "contribution": 0.02}]
    phase = reasoning.phase_signals("trend", top_features)
    assert phase["phase"] == 2
    assert "TREND" in phase["lines"][0]
    assert any("climbed steadily" in line for line in phase["lines"])


def test_phase_forecast_reports_the_move_against_what_trading_it_costs():
    """
    The bar a pick has to clear is the round-trip cost, so that is what the
    forecast has to be reported against.
    """
    phase = reasoning.phase_forecast(0.03, 0.03)

    assert phase["phase"] == 3
    assert "+3.00%" in phase["summary"]
    assert "cost" in " ".join(phase["lines"]).lower()


def test_phase_forecast_never_mentions_model_agreement():
    """
    Agreement was measured to predict nothing. Printing it beside a trade
    proposal invited exactly the confidence it could not support.
    """
    text = " ".join([reasoning.phase_forecast(0.03, 0.03)["summary"], *reasoning.phase_forecast(0.03, 0.03)["lines"]])

    assert "agree" not in text.lower()


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
    bad = reasoning.phase_reconciliation("AAPL", 10.0, 5.0, flagged=True, outcome="diverged")
    assert "reconciled cleanly" in ok["summary"]
    assert "diverged" in bad["summary"]
    assert "off by" in bad["summary"]


def test_phase_reconciliation_does_not_call_a_queued_order_a_problem():
    """
    An order waiting for the next open holds nothing yet and has nothing
    wrong with it. Reporting that as a divergence is what trained everyone
    to ignore this phase.
    """
    queued = reasoning.phase_reconciliation("AAPL", 10.0, 0.0, flagged=False, outcome="queued")

    assert "queued" in queued["summary"]
    assert "nothing is wrong" in " ".join(queued["lines"])


def test_phase_reconciliation_prints_share_counts_a_human_can_read():
    phase = reasoning.phase_reconciliation("AAPL", 32.063831455312794, 0.0, flagged=True, outcome="rejected")

    assert "32.063831455312794" not in " ".join([phase["summary"], *phase["lines"]])
    assert "32.06" in " ".join(phase["lines"])


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


def test_phase_selection_summary_does_not_repeat_the_symbol():
    """
    Phase 4's summary used to start with the symbol, and its caller prefixed
    the symbol too — producing "SNDK: SNDK: short". The summary is always
    shown somewhere that already names the position, so it says the symbol
    once, elsewhere.
    """
    diversified = reasoning.phase_selection_diversified(
        "SNDK", "short", 0.052, n_confident=20, n_selected=10, top_k=10
    )
    concentrated = reasoning.phase_selection(
        "SNDK", "short", 0.7, n_confident=2, n_selected=2, max_leg_pct=0.7, min_leg_pct=0.3
    )

    assert not diversified["summary"].startswith("SNDK")
    assert not concentrated["summary"].startswith("SNDK")
    assert "short" in diversified["summary"]


def test_phase_execution_prints_share_counts_a_human_can_read():
    phase = reasoning.phase_execution("AAPL", "opened", 32.063831455312794, "market")

    assert "32.063831455312794" not in phase["summary"]
    assert "32.06" in phase["summary"]
