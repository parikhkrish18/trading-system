"""
scripts/rebackfill_prices.py -- the one-time remediation for prices/features
rows written under data/ingest/prices.py's old, unadjusted fetch (the
"+252.7% in 20 days" incident). Covers the wiring: the right symbols and
date range reach ingest_prices, and the same symbols (not padded with the
regime proxy) reach build_and_store afterward.
"""
from __future__ import annotations

import datetime as dt

from scripts import rebackfill_prices as rbp


def test_rebackfill_pulls_the_full_lookback_window_including_the_regime_proxy(monkeypatch):
    captured = {}
    monkeypatch.setattr(rbp, "ingest_prices", lambda symbols, start, end, source: captured.update(symbols=symbols, start=start, end=end, source=source) or 0)
    monkeypatch.setattr(rbp, "build_and_store", lambda symbols, feature_set_id, lookback_years: 0)

    rbp.rebackfill(["AAPL", "TSLA"], backfill_years=5, feature_set_id="v4")

    assert set(captured["symbols"]) == {"AAPL", "TSLA", "SPY"}
    today = dt.datetime.now(tz=dt.UTC).date()
    assert captured["end"] == today
    assert captured["start"] == today.replace(year=today.year - 5)
    assert captured["source"] == "yfinance"


def test_rebackfill_rebuilds_features_for_exactly_the_requested_symbols(monkeypatch):
    captured = {}
    monkeypatch.setattr(rbp, "ingest_prices", lambda *a, **k: 0)
    monkeypatch.setattr(
        rbp, "build_and_store",
        lambda symbols, feature_set_id, lookback_years: captured.update(
            symbols=symbols, feature_set_id=feature_set_id, lookback_years=lookback_years
        ) or 0,
    )

    rbp.rebackfill(["AAPL", "TSLA"], backfill_years=3, feature_set_id="v4")

    # Not padded with the regime proxy -- that's an ingest_prices-only concern.
    assert captured["symbols"] == ["AAPL", "TSLA"]
    assert captured["feature_set_id"] == "v4"
    assert captured["lookback_years"] == 3


def test_rebackfill_respects_a_custom_source(monkeypatch):
    captured = {}
    monkeypatch.setattr(rbp, "ingest_prices", lambda symbols, start, end, source: captured.update(source=source) or 0)
    monkeypatch.setattr(rbp, "build_and_store", lambda symbols, feature_set_id, lookback_years: 0)

    rbp.rebackfill(["AAPL"], backfill_years=1, feature_set_id="v4", source="alpaca")

    assert captured["source"] == "alpaca"
