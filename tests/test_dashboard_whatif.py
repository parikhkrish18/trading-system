import numpy as np
import pandas as pd

from monitoring.dashboard.picks import COL_DIRECTION, COL_FORECAST, COL_SYMBOL, LONG, SHORT
from monitoring.dashboard.whatif import (
    filter_by_thresholds,
    shortlist_summary,
    whatif_table,
)


def _batch(rows):
    """Rows as (symbol, forecast) — the column the panel filters on."""
    return pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-08-07T09:14:27Z"),
                "symbol": symbol,
                "forecast": forecast,
                "regime": "trend",
                "target_position": abs(forecast) if forecast >= 0 else -abs(forecast),
                "executed_position": None,
            }
            for symbol, forecast in rows
        ]
    )


# --- filtering ------------------------------------------------------------


def test_default_threshold_keeps_everything_the_screener_already_shortlisted():
    batch = _batch([("AAA", 0.04), ("BBB", 0.01), ("CCC", -0.001)])
    assert len(filter_by_thresholds(batch, min_abs_move=0.0)) == 3


def test_raising_the_move_bar_drops_the_smaller_forecasts():
    batch = _batch([("AAA", 0.047), ("BBB", 0.011), ("CCC", 0.006)])
    kept = filter_by_thresholds(batch, min_abs_move=0.01)
    assert list(kept["symbol"]) == ["AAA", "BBB"]


def test_move_bar_measures_size_not_direction_so_a_predicted_fall_can_clear_it():
    batch = _batch([("UP", 0.03), ("DOWN", -0.03), ("FLATISH", 0.001)])
    kept = filter_by_thresholds(batch, min_abs_move=0.02)
    assert sorted(kept["symbol"]) == ["DOWN", "UP"]


def test_unrecorded_forecast_survives_only_at_the_move_floor():
    batch = _batch([("NOFC", np.nan), ("OK", 0.04)])
    assert len(filter_by_thresholds(batch, min_abs_move=0.0)) == 2
    assert list(filter_by_thresholds(batch, min_abs_move=0.001)["symbol"]) == ["OK"]


def test_impossible_settings_return_an_empty_batch_rather_than_raising():
    batch = _batch([("AAA", 0.04)])
    assert filter_by_thresholds(batch, min_abs_move=0.5).empty


def test_empty_batch_filters_to_empty():
    assert filter_by_thresholds(pd.DataFrame(), min_abs_move=0.01).empty


def test_filtering_never_mutates_the_batch_it_was_given():
    batch = _batch([("AAA", 0.04), ("BBB", 0.001)])
    before = batch.copy()
    filter_by_thresholds(batch, min_abs_move=0.01)
    pd.testing.assert_frame_equal(batch, before)


def test_filtering_does_not_need_an_agreement_column_at_all():
    """
    The panel must work on rows that never carried agreement — it is no
    longer part of the question being asked.
    """
    batch = _batch([("AAA", 0.04)])
    assert "direction_agreement" not in batch.columns

    assert list(filter_by_thresholds(batch, min_abs_move=0.01)["symbol"]) == ["AAA"]


# --- display table --------------------------------------------------------


def test_table_reranks_by_size_of_predicted_move():
    batch = _batch([("SMALL", 0.01), ("BIG", -0.05), ("MID", 0.03)])
    table = whatif_table(filter_by_thresholds(batch, min_abs_move=0.0))

    assert list(table[COL_SYMBOL]) == ["BIG", "MID", "SMALL"]
    assert table[COL_FORECAST].iloc[0] == -0.05


def test_table_no_longer_shows_model_agreement():
    """
    Presenting agreement beside a pick invited reading confidence into a
    number measured to predict nothing.
    """
    batch = _batch([("AAA", 0.04)])

    assert not any("agreement" in str(c).lower() for c in whatif_table(batch).columns)


def test_table_keeps_direction_readable_for_shorts():
    batch = _batch([("DOWN", -0.05), ("UP", 0.04)])
    table = whatif_table(batch)
    assert table.set_index(COL_SYMBOL).loc["DOWN", COL_DIRECTION] == SHORT
    assert table.set_index(COL_SYMBOL).loc["UP", COL_DIRECTION] == LONG


def test_empty_shortlist_is_an_empty_table_not_an_error():
    assert whatif_table(pd.DataFrame()).empty


# --- the count line -------------------------------------------------------


def test_summary_reads_as_a_before_and_after_count():
    assert shortlist_summary(9, 4) == "9 picks → 4 picks at these settings"


def test_summary_says_when_the_slider_is_not_filtering():
    assert "nothing is filtered out" in shortlist_summary(9, 9)


def test_summary_singularises_one_pick():
    assert shortlist_summary(3, 1) == "3 picks → 1 pick at these settings"


def test_summary_words_the_empty_and_no_run_cases():
    assert shortlist_summary(9, 0) == "9 picks → no picks at these settings"
    assert "No picks in the latest run" in shortlist_summary(0, 0)
