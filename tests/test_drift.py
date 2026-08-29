import json

import pandas as pd
import pytest

from monitoring import drift


def _scored(rows):
    """rows: list of (symbol, ts_str, hit)."""
    return pd.DataFrame(
        [{"symbol": s, "ts": pd.Timestamp(ts, tz="UTC"), "hit": h} for s, ts, h in rows]
    )


class TestWeeklyHitRate:
    def test_empty_input(self):
        result = drift.weekly_hit_rate(pd.DataFrame(columns=["symbol", "ts", "hit"]))
        assert result.empty
        assert list(result.columns) == ["week_start", "n", "hit_rate"]

    def test_buckets_by_calendar_week(self):
        scored = _scored(
            [
                ("AAA", "2026-08-03", True),  # week of Aug 3
                ("BBB", "2026-08-04", False),  # same week
                ("AAA", "2026-08-10", True),  # next week
            ]
        )
        result = drift.weekly_hit_rate(scored)
        assert len(result) == 2
        assert result.iloc[0]["n"] == 2
        assert result.iloc[0]["hit_rate"] == pytest.approx(0.5)
        assert result.iloc[1]["n"] == 1
        assert result.iloc[1]["hit_rate"] == pytest.approx(1.0)
        # oldest week first
        assert result.iloc[0]["week_start"] < result.iloc[1]["week_start"]


class TestAccuracyDriftFlag:
    def test_no_baseline_available(self):
        weekly = drift.weekly_hit_rate(_scored([("AAA", "2026-08-03", True)]))
        result = drift.accuracy_drift_flag(weekly, baseline_accuracy=None)
        assert result["flagged"] is False
        assert "baseline" in result["message"].lower()

    def test_not_enough_weeks_yet(self):
        weekly = drift.weekly_hit_rate(_scored([("AAA", "2026-08-03", True)]))
        result = drift.accuracy_drift_flag(weekly, baseline_accuracy=0.52, consecutive_weeks=3)
        assert result["flagged"] is False
        assert result["weeks_checked"] == 1

    def test_flags_when_all_recent_weeks_below_baseline(self):
        rows = []
        for week_offset, hit_rate in enumerate([0.3, 0.2, 0.4]):  # 3 straight bad weeks
            day = pd.Timestamp("2026-08-03") + pd.Timedelta(weeks=week_offset)
            n_hits = round(hit_rate * 10)
            for i in range(10):
                rows.append((f"S{i}", (day + pd.Timedelta(days=i % 5)).isoformat(), i < n_hits))
        weekly = drift.weekly_hit_rate(_scored(rows))
        result = drift.accuracy_drift_flag(weekly, baseline_accuracy=0.52, consecutive_weeks=3)
        assert result["flagged"] is True
        assert result["worst_week_hit_rate"] < 0.52

    def test_not_flagged_when_one_recent_week_clears_baseline(self):
        rows = []
        for week_offset, hit_rate in enumerate([0.3, 0.2, 0.9]):  # last week is strong
            day = pd.Timestamp("2026-08-03") + pd.Timedelta(weeks=week_offset)
            n_hits = round(hit_rate * 10)
            for i in range(10):
                rows.append((f"S{i}", (day + pd.Timedelta(days=i % 5)).isoformat(), i < n_hits))
        weekly = drift.weekly_hit_rate(_scored(rows))
        result = drift.accuracy_drift_flag(weekly, baseline_accuracy=0.52, consecutive_weeks=3)
        assert result["flagged"] is False


class TestFeatureDrag:
    def _decisions(self, rows):
        """rows: list of (symbol, ts_str, feature_names)."""
        out = []
        for symbol, ts, features in rows:
            phase2 = {
                "phase": 2, "title": "x", "summary": "x",
                "lines": [], "top_features": [{"feature_name": f, "value": 1.0, "contribution": 0.01} for f in features],
            }
            out.append({"symbol": symbol, "ts": pd.Timestamp(ts, tz="UTC"), "reasoning": json.dumps([phase2])})
        return pd.DataFrame(out)

    def test_empty_inputs(self):
        assert drift.feature_drag(pd.DataFrame(), pd.DataFrame()) == []

    def test_below_min_sample_is_excluded(self):
        decisions = self._decisions([("AAA", "2026-08-01", ["mom_ret_5d"])])
        scored = _scored([("AAA", "2026-08-01", False)])
        assert drift.feature_drag(decisions, scored) == []

    def test_computes_hit_rate_per_feature_and_sorts_worst_first(self):
        decisions_rows = []
        scored_rows = []
        # "bad_feature": present in 5 decisions, all misses -> hit_rate 0.0
        for i in range(5):
            ts = f"2026-08-{i + 1:02d}"
            decisions_rows.append((f"BAD{i}", ts, ["bad_feature"]))
            scored_rows.append((f"BAD{i}", ts, False))
        # "good_feature": present in 5 decisions, all hits -> hit_rate 1.0
        for i in range(5):
            ts = f"2026-08-{i + 10:02d}"
            decisions_rows.append((f"GOOD{i}", ts, ["good_feature"]))
            scored_rows.append((f"GOOD{i}", ts, True))

        decisions = self._decisions(decisions_rows)
        scored = _scored(scored_rows)
        result = drift.feature_drag(decisions, scored)

        by_name = {r["feature_name"]: r for r in result}
        assert by_name["bad_feature"]["hit_rate"] == pytest.approx(0.0)
        assert by_name["bad_feature"]["n"] == 5
        assert by_name["good_feature"]["hit_rate"] == pytest.approx(1.0)
        # worst hit rate first
        assert result[0]["feature_name"] == "bad_feature"

    def test_unscored_decisions_are_skipped(self):
        decisions = self._decisions([("AAA", "2026-08-01", ["f1"])])
        scored = _scored([("BBB", "2026-08-01", True)])  # different symbol -- no match
        assert drift.feature_drag(decisions, scored) == []

    def test_respects_top_n(self):
        decisions_rows, scored_rows = [], []
        for feat_i in range(15):
            for i in range(5):
                ts = f"2026-{(feat_i % 9) + 1:02d}-{i + 1:02d}"
                decisions_rows.append((f"S{feat_i}_{i}", ts, [f"feature_{feat_i}"]))
                scored_rows.append((f"S{feat_i}_{i}", ts, i % 2 == 0))
        decisions = self._decisions(decisions_rows)
        scored = _scored(scored_rows)
        result = drift.feature_drag(decisions, scored, top_n=4)
        assert len(result) == 4
