"""
execution/market_hours.py — the computed NYSE-hours fallback used by
IBKRBroker.get_clock() (Alpaca answers this from Alpaca's own clock
instead; see test_broker_alpaca.py::test_get_clock_maps_alpaca_clock_fields).
No holiday calendar by design — see the module docstring for why that's an
acceptable, one-directional gap for a dashboard label.
"""
from __future__ import annotations

import datetime as dt

from execution import market_hours


def test_open_during_regular_trading_hours():
    # Tuesday 2026-08-25, 10:00 America/New_York (14:00 UTC in August, EDT).
    now = dt.datetime(2026, 8, 25, 14, 0, tzinfo=dt.UTC)

    clock = market_hours.compute_clock(now)

    assert clock["is_open"] is True
    assert clock["source"] == "computed_no_holiday_calendar"


def test_closed_on_a_weekend():
    # Saturday 2026-08-29, 10:00 ET.
    now = dt.datetime(2026, 8, 29, 14, 0, tzinfo=dt.UTC)

    clock = market_hours.compute_clock(now)

    assert clock["is_open"] is False


def test_closed_after_the_close_on_a_weekday():
    # Tuesday 2026-08-25, 19:00 ET — well past the 16:00 close.
    now = dt.datetime(2026, 8, 25, 23, 0, tzinfo=dt.UTC)

    clock = market_hours.compute_clock(now)

    assert clock["is_open"] is False


def test_closed_before_the_open_on_a_weekday():
    # Tuesday 2026-08-25, 06:00 ET — before the 9:30 open.
    now = dt.datetime(2026, 8, 25, 10, 0, tzinfo=dt.UTC)

    clock = market_hours.compute_clock(now)

    assert clock["is_open"] is False


def test_next_open_during_rth_is_the_following_session_not_today():
    """While the market is open, next_open should point at tomorrow's open, not today's (already passed)."""
    now = dt.datetime(2026, 8, 25, 14, 0, tzinfo=dt.UTC)  # Tue 10:00 ET

    clock = market_hours.compute_clock(now)
    next_open = dt.datetime.fromisoformat(clock["next_open"])

    assert next_open.date() == dt.date(2026, 8, 26)
    assert next_open.time() == dt.time(9, 30)


def test_next_close_during_rth_is_todays_close():
    now = dt.datetime(2026, 8, 25, 14, 0, tzinfo=dt.UTC)  # Tue 10:00 ET

    clock = market_hours.compute_clock(now)
    next_close = dt.datetime.fromisoformat(clock["next_close"])

    assert next_close.date() == dt.date(2026, 8, 25)
    assert next_close.time() == dt.time(16, 0)


def test_next_open_from_a_weekend_skips_to_monday():
    now = dt.datetime(2026, 8, 29, 14, 0, tzinfo=dt.UTC)  # Sat 10:00 ET

    clock = market_hours.compute_clock(now)
    next_open = dt.datetime.fromisoformat(clock["next_open"])

    assert next_open.date() == dt.date(2026, 8, 31)  # Monday
    assert next_open.weekday() == 0


def test_next_open_from_friday_after_close_skips_to_monday():
    # Friday 2026-08-28, 19:00 ET — after close, should skip the weekend.
    now = dt.datetime(2026, 8, 28, 23, 0, tzinfo=dt.UTC)

    clock = market_hours.compute_clock(now)
    next_open = dt.datetime.fromisoformat(clock["next_open"])

    assert next_open.date() == dt.date(2026, 8, 31)  # Monday
    assert next_open.weekday() == 0


def test_return_shape_matches_alpaca_broker_get_clock():
    """Same keys as AlpacaBroker.get_clock() so the dashboard's /api/market_clock works with either broker."""
    clock = market_hours.compute_clock()

    assert set(clock.keys()) == {"is_open", "timestamp", "next_open", "next_close", "source"}
