"""
Read-only model-drift diagnostics for the dashboard.

Decision (2026-08-28): the model does NOT change itself automatically based
on live results. The system has logged only a handful of live paper
decisions so far, and the walk-forward result in BENCHMARK_RESULTS.md is
~50% directional accuracy with no demonstrated edge after correcting for
multiplicity -- an automated loop nudging the model from live outcomes at
this sample size would very likely be fitting noise, not learning. This
module exists instead: it surfaces two signals a human can act on --

    1. Has live directional accuracy fallen below the walk-forward
       baseline for several straight weeks (one bad week is noise, several
       in a row is worth a look) -- see accuracy_drift_flag.
    2. Which recent top-driver features have been showing up in decisions
       that turned out wrong more often than right -- see feature_drag.

Neither function retrains, reweights, or otherwise touches the model.
Acting on what they report (dropping a feature, triggering models/train.py
early, tightening the confidence bar) is still a human decision, same as
every other change to what the model does.

Kept apart from monitoring/dashboard/server.py for the same reason as
report_card.py/whatif.py/picks.py: the logic is worth testing without an
HTTP or DB context.
"""
from __future__ import annotations

import json

import pandas as pd

# Consecutive below-baseline weeks required before flagging. Mirrors the
# philosophy behind HOLD_MAX_MISSED_CYCLES elsewhere in this repo: don't
# react to a single data point, several in a row is the signal.
DEFAULT_DRIFT_WEEKS = 3

# A feature needs at least this many matured, scoreable decisions behind it
# before its hit rate is reported -- two unlucky trades is not evidence a
# feature is broken.
MIN_FEATURE_SAMPLES = 5

# How many top-driver features a feature_drag() caller typically wants to
# see rendered -- kept here so the dashboard and any script agree on it.
DEFAULT_TOP_N = 10


def weekly_hit_rate(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets `scored` (columns: ts, hit -- the output of
    monitoring.forecast_accuracy.compute_forecast_accuracy) into calendar
    weeks. Returns one row per week with decisions in it: week_start, n,
    hit_rate -- oldest week first.
    """
    columns = ["week_start", "n", "hit_rate"]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    df = scored.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["week_start"] = df["ts"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    grouped = df.groupby("week_start")["hit"].agg(["count", "mean"]).reset_index()
    grouped.columns = ["week_start", "n", "hit_rate"]
    return grouped.sort_values("week_start").reset_index(drop=True)


def accuracy_drift_flag(
    weekly: pd.DataFrame,
    baseline_accuracy: float | None,
    consecutive_weeks: int = DEFAULT_DRIFT_WEEKS,
) -> dict:
    """
    Whether the most recent `consecutive_weeks` weeks of live directional
    accuracy have ALL sat below the walk-forward baseline. A description for
    a human to read, never a number meant to drive an automatic action.

    Returns {flagged, message, weeks_checked, worst_week_hit_rate?}.
    """
    if baseline_accuracy is None:
        return {
            "flagged": False,
            "message": "No walk-forward baseline available yet (needs at least one MLflow training run) to compare live accuracy against.",
            "weeks_checked": 0,
        }
    if weekly.empty or len(weekly) < consecutive_weeks:
        return {
            "flagged": False,
            "message": f"Only {len(weekly)} week(s) of matured live decisions so far -- need {consecutive_weeks} before a drift check means anything.",
            "weeks_checked": len(weekly),
        }

    recent = weekly.tail(consecutive_weeks)
    below = bool((recent["hit_rate"] < baseline_accuracy).all())
    worst = float(recent["hit_rate"].min())
    if below:
        message = (
            f"Live directional accuracy has been below the {baseline_accuracy:.1%} walk-forward "
            f"baseline for {consecutive_weeks} straight week(s), as low as {worst:.1%}. Worth a "
            "human look before the next weekly cycle -- this alone should not trigger a retrain."
        )
    else:
        message = (
            f"Live accuracy over the last {consecutive_weeks} week(s) has stayed at or above the "
            f"{baseline_accuracy:.1%} walk-forward baseline. No drift flagged."
        )
    return {
        "flagged": below,
        "message": message,
        "weeks_checked": len(recent),
        "worst_week_hit_rate": worst,
    }


def _parse_reasoning(raw) -> list | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    return raw if isinstance(raw, list) else None


def feature_drag(decisions: pd.DataFrame, scored: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """
    For each feature that showed up as a top-5 SHAP driver (phase 2 of the
    reasoning log, see monitoring/reasoning.py::phase_signals), the hit rate
    of the decisions it drove -- among decisions old enough to have matured.

    `decisions`: columns symbol, ts, reasoning (JSON string or list of phase
        dicts, as stored in the `decisions` table).
    `scored`: monitoring.forecast_accuracy.compute_forecast_accuracy's
        output for the SAME decisions (columns symbol, ts, hit) -- joined
        here on (symbol, ts).

    Returns up to `top_n` rows with >= MIN_FEATURE_SAMPLES matured decisions
    behind them, worst hit rate first. This is a rear-view mirror, not a
    diagnosis: a feature showing a low hit rate might be genuinely
    unhelpful, or might just be correlated with a regime the model handles
    badly for other reasons. It's a prompt for a human to look, not a
    verdict to act on automatically.
    """
    if decisions.empty or scored.empty:
        return []

    hit_by_key = {
        (row["symbol"], pd.Timestamp(row["ts"])): bool(row["hit"]) for _, row in scored.iterrows()
    }

    feature_hits: dict[str, list[bool]] = {}
    for _, row in decisions.iterrows():
        key = (row["symbol"], pd.Timestamp(row["ts"]))
        hit = hit_by_key.get(key)
        if hit is None:
            continue
        reasoning = _parse_reasoning(row.get("reasoning"))
        if reasoning is None:
            continue
        phase2 = next((p for p in reasoning if p.get("phase") == 2), None)
        if not phase2:
            continue
        for f in phase2.get("top_features", []):
            name = f.get("feature_name")
            if not name:
                continue
            feature_hits.setdefault(name, []).append(hit)

    rows = [
        {"feature_name": name, "n": len(hits), "hit_rate": sum(hits) / len(hits)}
        for name, hits in feature_hits.items()
        if len(hits) >= MIN_FEATURE_SAMPLES
    ]
    rows.sort(key=lambda r: r["hit_rate"])
    return rows[:top_n]
