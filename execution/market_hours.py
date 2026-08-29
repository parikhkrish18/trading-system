"""
Fallback NYSE regular-trading-hours clock — used only when the active
broker has no clock endpoint of its own.

broker_alpaca.py answers AlpacaBroker.get_clock() straight from Alpaca's
own /v2/clock, which already accounts for market holidays and early
closes. ib_insync has no equivalent single call, so broker_ibkr.py falls
back to this: a plain 9:30-16:00 America/New_York, Monday-Friday check
with NO holiday calendar.

That means this fallback can only ever be wrong in one direction: it may
call a market holiday "closed" a little early or "open" on a day that's
actually a holiday it doesn't know about — but it will never say "open"
outside 9:30-16:00 on a real weekday, and never say "closed" during actual
regular trading hours. Good enough for a dashboard label; swap in a real
NYSE calendar (e.g. pandas_market_calendars) if that residual holiday gap
ever needs closing.
"""
from __future__ import annotations

import datetime as dt

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback, not expected here
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

_NY = ZoneInfo("America/New_York")
_OPEN_TIME = dt.time(9, 30)
_CLOSE_TIME = dt.time(16, 0)


def _next_occurrence(now_ny: dt.datetime, target_time: dt.time) -> dt.datetime:
    """Next Mon-Fri datetime at `target_time` (NY-local) strictly after `now_ny`."""
    candidate = dt.datetime.combine(now_ny.date(), target_time, tzinfo=_NY)
    if candidate <= now_ny:
        candidate = dt.datetime.combine(candidate.date() + dt.timedelta(days=1), target_time, tzinfo=_NY)
    while candidate.weekday() >= 5:  # Sat=5, Sun=6
        candidate = dt.datetime.combine(candidate.date() + dt.timedelta(days=1), target_time, tzinfo=_NY)
    return candidate


def compute_clock(now: dt.datetime | None = None) -> dict:
    """
    Returns the same shape as AlpacaBroker.get_clock(): is_open, timestamp,
    next_open, next_close (all ISO 8601), plus `source` so a caller can
    tell this apart from the real Alpaca clock and caveat accordingly.
    """
    now = now or dt.datetime.now(tz=dt.UTC)
    now_ny = now.astimezone(_NY)
    is_open = now_ny.weekday() < 5 and _OPEN_TIME <= now_ny.time() < _CLOSE_TIME

    next_open = _next_occurrence(now_ny, _OPEN_TIME)
    # The close that pairs with next_open's session: today's close if the
    # market is open right now, otherwise the close of whatever session
    # next_open starts.
    close_date = now_ny.date() if is_open else next_open.date()
    next_close = dt.datetime.combine(close_date, _CLOSE_TIME, tzinfo=_NY)

    return {
        "is_open": is_open,
        "timestamp": now_ny.isoformat(),
        "next_open": next_open.isoformat(),
        "next_close": next_close.isoformat(),
        "source": "computed_no_holiday_calendar",
    }
