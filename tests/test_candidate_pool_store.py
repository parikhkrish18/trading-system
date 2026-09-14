from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine

from models import candidate_pool_store as store


@pytest.fixture
def engine(monkeypatch):
    eng = create_engine("sqlite://")
    monkeypatch.setattr(store, "get_engine", lambda: eng)
    return eng


def _row(symbol, confidence, side="long"):
    return {
        "symbol": symbol,
        "side": side,
        "predicted_return": 0.05,
        "direction_agreement": 0.9,
        "conviction_score": confidence,
        "confidence": confidence,
        "take_profit_pct": 0.08,
        "stop_loss_pct": 0.04,
        "reasoning": [{"phase": 2, "title": "x", "summary": "s", "lines": ["a"]}],
    }


def test_save_then_load_round_trips_every_field(engine):
    store.save_candidate_pool("v4", [_row("AAPL", 0.9), _row("MSFT", 0.6)])

    pool = store.load_recent_pool("v4")

    assert {r["symbol"] for r in pool} == {"AAPL", "MSFT"}
    aapl = next(r for r in pool if r["symbol"] == "AAPL")
    assert aapl["side"] == "long"
    assert aapl["confidence"] == pytest.approx(0.9)
    assert aapl["take_profit_pct"] == pytest.approx(0.08)
    assert aapl["stop_loss_pct"] == pytest.approx(0.04)
    assert aapl["reasoning"] == [{"phase": 2, "title": "x", "summary": "s", "lines": ["a"]}]


def test_load_orders_by_confidence_descending(engine):
    store.save_candidate_pool("v4", [_row("LOW", 0.2), _row("HIGH", 0.95), _row("MID", 0.5)])

    pool = store.load_recent_pool("v4")

    assert [r["symbol"] for r in pool] == ["HIGH", "MID", "LOW"]


def test_load_is_empty_when_nothing_was_ever_saved(engine):
    assert store.load_recent_pool("v4") == []


def test_save_with_an_empty_pool_is_a_noop(engine):
    store.save_candidate_pool("v4", [])
    assert store.load_recent_pool("v4") == []


def test_load_ignores_a_different_feature_set_id(engine):
    store.save_candidate_pool("v3", [_row("OLD_FS", 0.9)])
    store.save_candidate_pool("v4", [_row("THIS_FS", 0.9)])

    pool = store.load_recent_pool("v4")

    assert [r["symbol"] for r in pool] == ["THIS_FS"]


def test_load_only_returns_the_latest_batch_not_every_batch_ever_saved(engine):
    store.save_candidate_pool("v4", [_row("OLD_BATCH", 0.9)])
    store.save_candidate_pool("v4", [_row("NEW_BATCH_A", 0.5), _row("NEW_BATCH_B", 0.4)])

    pool = store.load_recent_pool("v4")

    assert {r["symbol"] for r in pool} == {"NEW_BATCH_A", "NEW_BATCH_B"}


def test_load_treats_a_stale_batch_as_unusable(engine):
    """
    Regression test: 'this week's analysis' should not mean 'whatever the
    last save happened to be, no matter how old' -- a batch older than
    MAX_POOL_AGE must be treated the same as no pool at all, so the caller
    falls back to a fresh screen instead of redeploying against a stale plan.
    """
    from sqlalchemy import text

    store.save_candidate_pool("v4", [_row("STALE", 0.9)])
    stale_ts = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=10)
    with engine.begin() as conn:
        conn.execute(text("UPDATE llm_candidate_pool SET ts = :ts"), {"ts": stale_ts})

    assert store.load_recent_pool("v4") == []


def test_load_respects_a_custom_max_age(engine):
    store.save_candidate_pool("v4", [_row("RECENT", 0.9)])

    assert store.load_recent_pool("v4", max_age=dt.timedelta(seconds=0)) == []
    assert len(store.load_recent_pool("v4", max_age=dt.timedelta(days=1))) == 1


def test_save_failure_is_caught_not_raised(monkeypatch):
    """Best-effort: a persistence failure must never break the live screen that produced this pool."""

    class _BoomEngine:
        pass

    monkeypatch.setattr(store, "get_engine", lambda: _BoomEngine())

    store.save_candidate_pool("v4", [_row("A", 0.9)])  # must not raise


def test_load_failure_returns_empty_not_raises(monkeypatch):
    class _BoomEngine:
        pass

    monkeypatch.setattr(store, "get_engine", lambda: _BoomEngine())

    assert store.load_recent_pool("v4") == []
