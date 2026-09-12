"""
The dashboard's "Model report card": what the walk-forward training run in
models/train.py actually scored, fold by fold.

Those numbers live in the walk_forward_folds Postgres table (train.py writes
one row per fold there — see data/schema/017_walk_forward_folds.sql) — this
module is the one place in the dashboard that reads it. The fetch is kept to
a single thin function; everything the panel actually renders is shaped by
pure functions below it, so the wording and the maths can be tested without
a database running.

The reader this is written for doesn't know what a walk-forward fold is. The
framing throughout: each fold is one honest re-run of "train on the past,
predict the next stretch, check what happened" — so several folds in a row is
the evidence, and one good fold is a coincidence.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine

# models/train.py's default `model_name` — this table's rows are scoped by it,
# so training under a different --model-name writes a separate set of folds
# this panel won't see.
DEFAULT_EXPERIMENT = "forecast_lgbm"

FOLD_LABEL = "fold_label"
METRIC_COLUMNS = (
    "directional_accuracy",
    "directional_accuracy_when_confident",
    "pct_rows_confident",
    "mae",
    "rmse",
)

# Series names for the grouped accuracy chart. These are the legend text a
# non-finance reader sees, so they say what the bar counts, not which metric
# key it came from.
SERIES_ALL = "All predictions"
SERIES_CONFIDENT = "Only when the models agreed"

# Above this share of rows clearing the agreement bar, "confident" stops being a
# meaningful subset — it's very nearly the whole population. Matches the
# threshold train.py calibrates against (confident_agreement_threshold=0.8).
CROWDED_AGREEMENT = 0.80
SPARSE_AGREEMENT = 0.40

LEVEL_HIGH = "high"
LEVEL_MODERATE = "moderate"
LEVEL_LOW = "low"
LEVEL_UNKNOWN = "unknown"


def fetch_fold_runs(model_name: str = DEFAULT_EXPERIMENT) -> list[dict]:
    """
    Every fold from the latest walk-forward run for this model, as plain
    dicts — same shape this has always returned, so fold_metrics_frame and
    everything downstream of it needs no changes. train.py deletes a
    model's old rows before writing a new run's, so this table only ever
    holds one run's worth of folds per model_name — no de-duplication
    needed here the way multiple MLflow runs once required.
    """
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT fold_id, mae, rmse, directional_accuracy, "
            "directional_accuracy_when_confident, pct_rows_confident, mean_ensemble_std "
            "FROM walk_forward_folds WHERE model_name = :model_name ORDER BY fold_id"
        ),
        engine,
        params={"model_name": model_name},
    )
    return [
        {
            "run_name": f"fold_{row['fold_id']}",
            "fold_id": str(row["fold_id"]),
            "metrics": row.drop("fold_id").to_dict(),
        }
        for _, row in df.iterrows()
    ]


def _fold_key(run: dict) -> tuple[int, str]:
    """Sort key: numeric fold_id when train.py logged one, else the run name."""
    raw = run.get("fold_id")
    try:
        return (int(raw), "")
    except (TypeError, ValueError):
        return (10**6, str(run.get("run_name") or ""))


def fold_metrics_frame(runs: list[dict]) -> pd.DataFrame:
    """
    One row per fold, oldest fold first, with a column per metric.

    Runs are expected newest-first (MLflow's default order). Training the same
    feature set twice leaves two runs per fold in the experiment, so folds are
    de-duplicated keeping the first occurrence — the latest training run wins
    rather than the chart sprouting a second bar per fold.
    """
    columns = [FOLD_LABEL, *METRIC_COLUMNS]
    if not runs:
        return pd.DataFrame(columns=columns)

    rows, seen = [], set()
    for run in sorted(runs, key=_fold_key):
        key = _fold_key(run)
        if key in seen:
            continue
        seen.add(key)
        metrics = run.get("metrics") or {}
        label = f"Fold {run['fold_id']}" if run.get("fold_id") is not None else str(run.get("run_name") or "run")
        rows.append(
            {
                FOLD_LABEL: label,
                **{name: pd.to_numeric(metrics.get(name), errors="coerce") for name in METRIC_COLUMNS},
            }
        )

    return pd.DataFrame(rows, columns=columns)


def accuracy_chart_frame(folds: pd.DataFrame) -> pd.DataFrame:
    """
    The two accuracy series in long form — one row per (fold, series) — ready to
    hand to a grouped bar chart.

    A fold where nothing cleared the agreement bar has no confident accuracy to
    plot (train.py logs NaN there); that bar is dropped rather than drawn as
    zero, which would read as "it got everything wrong".
    """
    columns = [FOLD_LABEL, "series", "accuracy"]
    if folds.empty:
        return pd.DataFrame(columns=columns)

    pairs = [(SERIES_ALL, "directional_accuracy"), (SERIES_CONFIDENT, "directional_accuracy_when_confident")]
    parts = [
        pd.DataFrame({FOLD_LABEL: folds[FOLD_LABEL], "series": name, "accuracy": folds[column]})
        for name, column in pairs
    ]
    return pd.concat(parts, ignore_index=True).dropna(subset=["accuracy"]).reset_index(drop=True)


def _mean(folds: pd.DataFrame, column: str) -> float | None:
    if folds.empty or column not in folds.columns:
        return None
    value = folds[column].mean(skipna=True)
    return None if pd.isna(value) else float(value)


def headline_metrics(folds: pd.DataFrame) -> dict[str, float | None]:
    """Averages across folds, for the stat tiles above the chart."""
    return {
        "n_folds": len(folds),
        "directional_accuracy": _mean(folds, "directional_accuracy"),
        "directional_accuracy_when_confident": _mean(folds, "directional_accuracy_when_confident"),
        "pct_rows_confident": _mean(folds, "pct_rows_confident"),
        "mae": _mean(folds, "mae"),
    }


def confidence_level(pct_confident: float | None) -> str:
    """How much of the population "confident" actually covers, as a band."""
    if pct_confident is None or pd.isna(pct_confident):
        return LEVEL_UNKNOWN
    if pct_confident >= CROWDED_AGREEMENT:
        return LEVEL_HIGH
    if pct_confident < SPARSE_AGREEMENT:
        return LEVEL_LOW
    return LEVEL_MODERATE


def confidence_callout(pct_confident: float | None) -> str:
    """
    What that band means, in words. The high case is the one that matters: a
    filter that keeps almost everything isn't a filter, and the models agreeing
    that often is a fact about the ensemble (same data, same architecture,
    different seeds) rather than evidence that they're right.
    """
    level = confidence_level(pct_confident)
    if level == LEVEL_UNKNOWN:
        return "How often the models agreed wasn't recorded for these folds."

    share = f"{float(pct_confident):.0%}"
    if level == LEVEL_HIGH:
        return (
            f"{share} of predictions clear the agreement bar — the models agree so often that "
            "agreement barely filters anything. Consider raising the bar or diversifying the "
            "ensemble (different model types, not just different random seeds), so that "
            '"the models agreed" actually singles something out.'
        )
    if level == LEVEL_LOW:
        return (
            f"Only {share} of predictions clear the agreement bar — the filter is strict, so "
            "the shortlist stays short. That's a real filter, but check there are still enough "
            "confident rows per fold for the accuracy beside it to mean anything."
        )
    return (
        f"{share} of predictions clear the agreement bar — the filter throws out a real share "
        "of predictions without discarding most of them, which is roughly what you want."
    )


def agreement_edge_note(accuracy: float | None, accuracy_when_confident: float | None) -> str:
    """
    Whether waiting for the models to agree actually bought any accuracy. This
    is the question the whole confidence filter rests on: if the confident
    subset isn't more accurate than everything else, agreement is measuring
    something other than being right.
    """
    if accuracy is None or accuracy_when_confident is None or pd.isna(accuracy) or pd.isna(accuracy_when_confident):
        return "Not enough recorded folds to compare the two accuracies yet."

    gain = float(accuracy_when_confident) - float(accuracy)
    head = (
        f"When the models agreed, the direction was right {accuracy_when_confident:.1%} of the time, "
        f"against {accuracy:.1%} across all predictions"
    )
    if gain >= 0.03:
        return f"{head} — a gain of {gain * 100:.1f} points, so agreement is buying real accuracy."
    if gain >= 0.01:
        return f"{head} — a gain of {gain * 100:.1f} points, a small but positive edge."
    if gain >= 0:
        return (
            f"{head} — a gain of {gain * 100:.1f} points, which is close to nothing. "
            "Agreement isn't a useful confidence filter at this threshold yet."
        )
    return (
        f"{head} — {abs(gain) * 100:.1f} points *worse* than the overall figure. "
        "Agreement is currently pointing the wrong way; treat it as noise, not confidence."
    )
