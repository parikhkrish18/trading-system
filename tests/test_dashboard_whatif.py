import numpy as np
import pandas as pd

from monitoring.dashboard.picks import COL_DIRECTION, COL_FORECAST, COL_SYMBOL, LONG, SHORT
from monitoring.dashboard.whatif import (
    COL_AGREEMENT,
    filter_by_thresholds,
    shortlist_summary,
    whatif_table,
)


def _batch(rows):
    """Rows as (symbol, forecast, direction_agreement) — the columns the panel filters on."""
    return pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-08-07T09:14:27Z"),
                "symbol": symbol,
                "forecast": forecast,
                "direction_agreement": agreement,
                "regime": "trend",
                "target_position": abs(forecast) if forecast >= 0 else -abs(forecast),
                "executed_position": None,
            }
            for symbol, forecast, agreement in rows
        ]
    )


# --- filtering ------------------------------------------------------------


def test_default_thresholds_keep_everything_the_screener_already_shortlisted():
    batch = _batch([("AAA", 0.04, 1.0), ("BBB", 0.01, 0.8), ("CCC", -0.001, 0.8)])
    assert len(filter_by_thresholds(batch, min_agreement=0.8, min_abs_move=0.0)) == 3


def test_raising_the_agreement_bar_drops_the_less_agreed_picks():
    batch = _batch([("AAA", 0.04, 1.0), ("BBB", 0.01, 0.8), ("CCC", 0.02, 0.6)])
    kept = filter_by_thresholds(batch, min_agreement=1.0, min_abs_move=0.0)
    assert list(kept["symbol"]) == ["AAA"]


def test_raising_the_move_bar_drops_the_smaller_forecasts():
    batch = _batch([("AAA", 0.047, 1.0), ("BBB", 0.011, 1.0), ("CCC", 0.006, 1.0)])
    kept = filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.01)
    assert list(kept["symbol"]) == ["AAA", "BBB"]


def test_move_bar_measures_size_not_direction_so_a_predicted_fall_can_clear_it():
    batch = _batch([("UP", 0.03, 1.0), ("DOWN", -0.03, 1.0), ("FLATISH", 0.001, 1.0)])
    kept = filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.02)
    assert sorted(kept["symbol"]) == ["DOWN", "UP"]


def test_both_bars_apply_together():
    batch = _batch([("AAA", 0.04, 1.0), ("BBB", 0.04, 0.6), ("CCC", 0.001, 1.0)])
    kept = filter_by_thresholds(batch, min_agreement=0.9, min_abs_move=0.01)
    assert list(kept["symbol"]) == ["AAA"]


def test_unrecorded_agreement_survives_only_while_the_slider_sits_at_its_floor():
    # Picks logged before the screener stored direction_agreement — can't be
    # shown to clear a bar, so they drop as soon as one is asked for.
    batch = _batch([("OLD", 0.04, np.nan), ("NEW", 0.04, 1.0)])
    assert sorted(filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.0)["symbol"]) == ["NEW", "OLD"]
    assert list(filter_by_thresholds(batch, min_agreement=0.55, min_abs_move=0.0)["symbol"]) == ["NEW"]


def test_unrecorded_forecast_survives_only_at_the_move_floor():
    batch = _batch([("NOFC", np.nan, 1.0), ("OK", 0.04, 1.0)])
    assert len(filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.0)) == 2
    assert list(filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.001)["symbol"]) == ["OK"]


def test_impossible_settings_return_an_empty_batch_rather_than_raising():
    batch = _batch([("AAA", 0.04, 1.0)])
    assert filter_by_thresholds(batch, min_agreement=1.0, min_abs_move=0.5).empty


def test_empty_batch_filters_to_empty():
    assert filter_by_thresholds(pd.DataFrame(), min_agreement=0.9, min_abs_move=0.01).empty


def test_filtering_never_mutates_the_batch_it_was_given():
    batch = _batch([("AAA", 0.04, 1.0), ("BBB", 0.001, 0.6)])
    before = batch.copy()
    filter_by_thresholds(batch, min_agreement=0.9, min_abs_move=0.01)
    pd.testing.assert_frame_equal(batch, before)


# --- display table --------------------------------------------------------


def test_table_reranks_by_size_of_predicted_move_and_carries_agreement():
    batch = _batch([("SMALL", 0.01, 0.8), ("BIG", -0.05, 1.0), ("MID", 0.03, 0.9)])
    table = whatif_table(filter_by_thresholds(batch, min_agreement=0.5, min_abs_move=0.0))

    assert list(table[COL_SYMBOL]) == ["BIG", "MID", "SMALL"]
    assert table[COL_AGREEMENT].tolist() == [1.0, 0.9, 0.8]
    assert table[COL_FORECAST].iloc[0] == -0.05


def test_table_keeps_direction_readable_for_shorts():
    batch = _batch([("DOWN", -0.05, 1.0), ("UP", 0.04, 1.0)])
    table = whatif_table(batch)
    assert table.set_index(COL_SYMBOL).loc["DOWN", COL_DIRECTION] == SHORT
    assert table.set_index(COL_SYMBOL).loc["UP", COL_DIRECTION] == LONG


def test_empty_shortlist_still_has_the_agreement_column():
    table = whatif_table(pd.DataFrame())
    assert table.empty
    assert COL_AGREEMENT in table.columns


# --- the count line -------------------------------------------------------


def test_summary_reads_as_a_before_and_after_count():
    assert shortlist_summary(9, 4) == "9 picks → 4 picks at these settings"


def test_summary_says_when_the_sliders_are_not_filtering():
    assert "nothing is filtered out" in shortlist_summary(9, 9)


def test_summary_singularises_one_pick():
    assert shortlist_summary(3, 1) == "3 picks → 1 pick at these settings"


def test_summary_words_the_empty_and_no_run_cases():
    assert shortlist_summary(9, 0) == "9 picks → no picks at these settings"
    assert "No picks in the latest run" in shortlist_summary(0, 0)
