import numpy as np
import pandas as pd

from monitoring.dashboard.report_card import (
    FOLD_LABEL,
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MODERATE,
    LEVEL_UNKNOWN,
    SERIES_ALL,
    SERIES_CONFIDENT,
    accuracy_chart_frame,
    agreement_edge_note,
    confidence_callout,
    confidence_level,
    fold_metrics_frame,
    headline_metrics,
)


def _run(fold_id, accuracy=0.52, confident_accuracy=0.53, pct_confident=0.88, mae=0.03, rmse=0.04):
    return {
        "run_name": f"fold_{fold_id}",
        "fold_id": str(fold_id),
        "metrics": {
            "directional_accuracy": accuracy,
            "directional_accuracy_when_confident": confident_accuracy,
            "pct_rows_confident": pct_confident,
            "mae": mae,
            "rmse": rmse,
            "mean_ensemble_std": 0.0035,
        },
    }


# --- shaping the fold table -----------------------------------------------


def test_folds_come_back_oldest_first_however_mlflow_ordered_them():
    # MLflow hands back newest-first; the chart reads left to right in time.
    frame = fold_metrics_frame([_run(2), _run(1), _run(0)])
    assert list(frame[FOLD_LABEL]) == ["Fold 0", "Fold 1", "Fold 2"]


def test_retraining_does_not_double_up_a_fold():
    # Two training runs of the same folds leave two runs per fold in MLflow.
    newest, older = _run(0, accuracy=0.55), _run(0, accuracy=0.11)
    frame = fold_metrics_frame([newest, older, _run(1)])
    assert len(frame) == 2
    assert frame.loc[frame[FOLD_LABEL] == "Fold 0", "directional_accuracy"].item() == 0.55


def test_runs_without_a_fold_id_still_appear_named_after_their_run():
    run = _run(0)
    run["fold_id"] = None
    run["run_name"] = "adhoc_eval"
    frame = fold_metrics_frame([run])
    assert list(frame[FOLD_LABEL]) == ["adhoc_eval"]


def test_missing_metrics_become_nan_rather_than_raising():
    run = _run(0)
    run["metrics"] = {}
    frame = fold_metrics_frame([run])
    assert pd.isna(frame["directional_accuracy"].item())


def test_no_runs_gives_an_empty_frame_with_the_expected_columns():
    frame = fold_metrics_frame([])
    assert frame.empty
    assert FOLD_LABEL in frame.columns
    assert "pct_rows_confident" in frame.columns


# --- chart frame ----------------------------------------------------------


def test_chart_frame_has_both_series_for_every_fold():
    frame = accuracy_chart_frame(fold_metrics_frame([_run(0), _run(1)]))
    assert len(frame) == 4
    assert set(frame["series"]) == {SERIES_ALL, SERIES_CONFIDENT}


def test_a_fold_where_nothing_was_confident_drops_that_bar_instead_of_drawing_zero():
    frame = accuracy_chart_frame(
        fold_metrics_frame([_run(0, confident_accuracy=float("nan"), pct_confident=0.0), _run(1)])
    )
    fold_0 = frame[frame[FOLD_LABEL] == "Fold 0"]
    assert list(fold_0["series"]) == [SERIES_ALL]
    assert len(frame[frame[FOLD_LABEL] == "Fold 1"]) == 2


def test_chart_frame_of_no_folds_is_empty_not_an_error():
    assert accuracy_chart_frame(fold_metrics_frame([])).empty


# --- headline numbers -----------------------------------------------------


def test_headline_metrics_average_across_folds():
    headline = headline_metrics(fold_metrics_frame([_run(0, accuracy=0.50), _run(1, accuracy=0.54)]))
    assert headline["n_folds"] == 2
    assert headline["directional_accuracy"] == 0.52


def test_headline_metrics_ignore_folds_missing_a_number():
    frame = fold_metrics_frame([_run(0, confident_accuracy=np.nan), _run(1, confident_accuracy=0.60)])
    assert headline_metrics(frame)["directional_accuracy_when_confident"] == 0.60


def test_headline_metrics_report_none_rather_than_nan_when_nothing_was_recorded():
    headline = headline_metrics(fold_metrics_frame([]))
    assert headline["n_folds"] == 0
    assert headline["directional_accuracy"] is None


# --- the confidence callout -----------------------------------------------


def test_agreement_that_keeps_almost_everything_is_flagged_as_high():
    assert confidence_level(0.88) == LEVEL_HIGH
    assert confidence_level(0.80) == LEVEL_HIGH


def test_moderate_and_low_agreement_bands():
    assert confidence_level(0.60) == LEVEL_MODERATE
    assert confidence_level(0.10) == LEVEL_LOW
    assert confidence_level(None) == LEVEL_UNKNOWN
    assert confidence_level(float("nan")) == LEVEL_UNKNOWN


def test_high_callout_says_the_filter_is_barely_filtering_and_what_to_do():
    text = confidence_callout(0.88)
    assert "88%" in text
    assert "barely filters" in text
    assert "diversif" in text


def test_low_callout_warns_the_shortlist_stays_short():
    assert "strict" in confidence_callout(0.12)


def test_unknown_callout_does_not_invent_a_percentage():
    assert "%" not in confidence_callout(None)


# --- does agreement buy accuracy? -----------------------------------------


def test_a_real_gain_from_agreement_is_called_real():
    assert "real accuracy" in agreement_edge_note(0.52, 0.58)


def test_a_tiny_gain_is_called_close_to_nothing():
    note = agreement_edge_note(0.5209, 0.5269)
    assert "close to nothing" in note
    assert "isn't a useful confidence filter" in note


def test_a_modest_gain_is_called_a_small_edge():
    assert "small but positive edge" in agreement_edge_note(0.52, 0.535)


def test_agreement_pointing_the_wrong_way_is_stated_outright():
    note = agreement_edge_note(0.55, 0.48)
    assert "worse" in note
    assert "noise, not confidence" in note


def test_edge_note_handles_missing_numbers():
    assert "Not enough recorded folds" in agreement_edge_note(None, 0.55)
    assert "Not enough recorded folds" in agreement_edge_note(0.55, float("nan"))
