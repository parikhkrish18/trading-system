"""
Real-DB integration test for scripts/refresh_model_report_card.py -- proven
against the actual decisions/prices/model_report_card_history schema
(data/schema/018_model_report_card.sql), not a mock.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest.db import get_engine
from scripts.refresh_model_report_card import refresh_model_report_card

_AS_OF = dt.date(2026, 9, 20)


@pytest.fixture
def _report_card_cleanup():
    engine = get_engine()
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM decisions WHERE symbol = 'ZZZTEST'"))
        conn.execute(text("DELETE FROM model_report_card_history WHERE as_of_date = :d"), {"d": _AS_OF})


def test_refresh_model_report_card_counts_only_real_trades(_report_card_cleanup):
    """
    Delta-based rather than an absolute count: this reads the WHOLE
    decisions table cumulatively by design, so a real deployment already
    has other genuine paper/live rows in it. Only the CHANGE this fixture's
    two rows cause is what's actually under test here.
    """
    engine = _report_card_cleanup
    baseline = refresh_model_report_card(as_of_date=_AS_OF)

    decisions = pd.DataFrame(
        [
            {
                # The row that SHOULD count: a real paper decision.
                "symbol": "ZZZTEST", "ts": pd.Timestamp("2026-08-01T00:00:00Z"),
                "feature_set_id": "v4", "model_version": "test",
                "mode": "paper", "forecast": 0.05, "target_position": 0.3,
            },
            {
                # Same symbol/ts, backfill mode with an enormous forecast --
                # must never be counted, or this test would fail by showing
                # up as a hit instead of being excluded outright.
                "symbol": "ZZZTEST", "ts": pd.Timestamp("2026-08-01T00:00:00Z"),
                "feature_set_id": "v4", "model_version": "test",
                "mode": "backfill", "forecast": 0.99, "target_position": 0.9,
            },
        ]
    )
    decisions.to_sql("decisions", engine, if_exists="append", index=False)

    updated = refresh_model_report_card(as_of_date=_AS_OF)

    assert updated["n_trades_taken"] == baseline["n_trades_taken"] + 1
    assert updated["capital_deployed_pct"] == pytest.approx(baseline["capital_deployed_pct"] + 0.3)

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM model_report_card_history WHERE as_of_date = :d"), {"d": _AS_OF}
        ).mappings().one()
    assert row["n_trades_taken"] == updated["n_trades_taken"]


def test_refresh_model_report_card_rerunning_the_same_day_replaces_the_row(_report_card_cleanup):
    engine = _report_card_cleanup
    refresh_model_report_card(as_of_date=_AS_OF)
    second = refresh_model_report_card(as_of_date=_AS_OF)

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM model_report_card_history WHERE as_of_date = :d"), {"d": _AS_OF}
        ).scalar()
    assert count == 1
    assert second["as_of_date"] == "2026-09-20"
