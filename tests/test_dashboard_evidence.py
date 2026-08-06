import numpy as np
import pandas as pd

from models.regime.trend_chop_classifier import CHOP, TREND
from monitoring.dashboard.evidence import (
    COL_EFFECT,
    COL_FACTOR,
    COL_OBSERVED,
    NO_NEWS_DATA,
    PUSHED_DOWN,
    PUSHED_UP,
    confidence_note,
    describe_effect,
    describe_value,
    evidence_table,
    feature_label,
    news_sentiment_note,
    regime_note,
)


def _evidence(rows):
    return pd.DataFrame(
        rows, columns=["feature_name", "feature_value", "contribution", "contribution_rank"]
    )


# --- labels and values ----------------------------------------------------


def test_feature_label_uses_plain_english_for_known_features():
    assert feature_label("mom_ret_20d") == "Momentum over the last month"
    assert "ADX" in feature_label("adx_14")


def test_feature_label_handles_fundamentals_and_unknown_features_readably():
    assert feature_label("fund_pe_ratio_latest") == "Company financials: pe ratio"
    assert feature_label("some_new_thing") == "Some new thing"


def test_describe_value_uses_each_features_own_units():
    assert describe_value("mom_ret_20d", 0.084) == "+8.4%"
    assert describe_value("adx_14", 31.0) == "31 (trending)"
    assert describe_value("adx_14", 12.0) == "12 (drifting sideways)"
    assert describe_value("meanrev_rsi_14", 78.0).endswith("(overbought — has run up hard)")
    assert describe_value("days_to_next_fomc", 12.0) == "12 day(s) away"


def test_describe_value_says_not_available_for_missing_data():
    assert describe_value("vol_realized_20d", None) == "not available"
    assert describe_value("vol_realized_20d", np.nan) == "not available"


# --- effect wording -------------------------------------------------------


def test_describe_effect_names_the_direction_it_pushed():
    assert PUSHED_UP in describe_effect(0.01, strongest=0.01)
    assert PUSHED_DOWN in describe_effect(-0.01, strongest=0.01)


def test_describe_effect_scales_strength_against_the_biggest_factor():
    """
    Raw contributions are fractions of a percent — meaningless in isolation.
    They only inform a reader relative to the other reasons for the same pick.
    """
    assert describe_effect(0.010, strongest=0.010).startswith("strongly")
    assert describe_effect(0.005, strongest=0.010).startswith("moderately")
    assert describe_effect(0.001, strongest=0.010).startswith("slightly")


# --- the panel table ------------------------------------------------------


def test_evidence_table_renders_strongest_factor_first():
    rows = _evidence(
        [
            ("adx_14", 31.0, 0.004, 2),
            ("mom_ret_20d", 0.08, 0.010, 1),
        ]
    )

    table = evidence_table(rows)

    assert list(table.columns) == [COL_FACTOR, COL_OBSERVED, COL_EFFECT]
    assert table.iloc[0][COL_FACTOR] == "Momentum over the last month"
    assert table.iloc[0][COL_OBSERVED] == "+8.0%"
    assert table.iloc[0][COL_EFFECT] == f"strongly {PUSHED_UP}"
    assert table.iloc[1][COL_OBSERVED] == "31 (trending)"


def test_evidence_table_on_no_stored_evidence_is_empty_with_columns():
    table = evidence_table(_evidence([]))
    assert table.empty
    assert list(table.columns) == [COL_FACTOR, COL_OBSERVED, COL_EFFECT]


def test_evidence_table_shows_a_missing_feature_value_without_crashing():
    table = evidence_table(_evidence([("vol_realized_20d", None, -0.01, 1)]))
    assert table.iloc[0][COL_OBSERVED] == "not available"
    assert PUSHED_DOWN in table.iloc[0][COL_EFFECT]


# --- news sentiment -------------------------------------------------------


def test_news_sentiment_note_says_no_data_when_no_sentiment_feature_is_present():
    note = news_sentiment_note(_evidence([("mom_ret_20d", 0.08, 0.01, 1)]))
    assert NO_NEWS_DATA in note


def test_news_sentiment_note_says_no_data_when_sentiment_is_present_but_null():
    """The feature column can exist with nothing in it — that's still "no news ingested"."""
    note = news_sentiment_note(_evidence([("sentiment_mean_10d", None, 0.001, 1)]))
    assert NO_NEWS_DATA in note


def test_news_sentiment_note_on_empty_evidence_says_no_data():
    assert NO_NEWS_DATA in news_sentiment_note(_evidence([]))


def test_news_sentiment_note_reports_the_score_once_news_exists():
    note = news_sentiment_note(
        _evidence(
            [
                ("mom_ret_20d", 0.08, 0.010, 1),
                ("sentiment_mean_10d", 0.42, 0.004, 2),
            ]
        )
    )

    assert NO_NEWS_DATA not in note
    assert "+0.42" in note
    assert "positive" in note
    assert PUSHED_UP in note


# --- regime and confidence ------------------------------------------------


def test_regime_note_explains_what_the_regime_did_to_the_size():
    assert "sized normally" in regime_note(TREND)
    assert "smaller" in regime_note(CHOP)
    assert "unknown" in regime_note(None).lower()


def test_confidence_note_reads_as_agreement_not_as_a_success_probability():
    assert "unanimous" in confidence_note(1.0)
    assert "80%" in confidence_note(0.8)
    assert "not recorded" in confidence_note(None)
    assert "not recorded" in confidence_note(np.nan)
