import datetime as dt

import pandas as pd
import pytest

from data.ingest import macro_calendar

# Minimal fixture mirroring federalreserve.gov/monetarypolicy/fomccalendars.htm's
# real markup: one past meeting (has a statement link -> exact date wins) and
# one future meeting (no statement link yet -> falls back to the date range).
_FOMC_HTML = """
<html><body>
<div class="panel panel-default">
  <div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month"><strong>January</strong></div>
    <div class="fomc-meeting__date">27-28</div>
    <div class="col-xs-12">
      <strong>Statement:</strong><br>
      <a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF</a> |
      <a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a>
    </div>
  </div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month"><strong>September</strong></div>
    <div class="fomc-meeting__date">15-16*</div>
  </div>
</div>
<div class="panel panel-default">
  <div class="panel-heading"><h4><a id="2">2025 FOMC Meetings</a></h4></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month"><strong>December</strong></div>
    <div class="fomc-meeting__date">9-10*</div>
    <div class="col-xs-12">
      <strong>Statement:</strong><br>
      <a href="/newsevents/pressreleases/monetary20251210a.htm">HTML</a>
    </div>
  </div>
</div>
</body></html>
"""


class _FakeResponse:
    def __init__(self, payload=None, text=None):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_fomc_events_uses_statement_link_when_available(monkeypatch):
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=_FOMC_HTML))

    df = macro_calendar.fetch_fomc_events()

    assert list(df.columns) == ["event_name", "ts", "category", "notes"]
    assert (df["event_name"] == "FOMC Decision").all()
    assert (df["category"] == "FOMC").all()
    assert len(df) == 3

    # Both January and December fall in EST (UTC-5): 2pm ET is 19:00 UTC,
    # not the 18:00 UTC that would only be correct in EDT (summer).
    jan_2026 = df.loc[df["ts"] == pd.Timestamp("2026-01-28T19:00:00Z")]
    assert len(jan_2026) == 1  # exact date came from the statement link (day 28, not the range's last "8")

    dec_2025 = df.loc[df["ts"] == pd.Timestamp("2025-12-10T19:00:00Z")]
    assert len(dec_2025) == 1


def test_fetch_fomc_events_falls_back_to_date_range_for_future_meetings(monkeypatch):
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=_FOMC_HTML))

    df = macro_calendar.fetch_fomc_events()

    # September 2026 has no statement link -> falls back to last day of "15-16*" -> the 16th.
    sep_2026 = df.loc[df["ts"] == pd.Timestamp("2026-09-16T18:00:00Z")]
    assert len(sep_2026) == 1


def test_fetch_fomc_events_future_meeting_spanning_a_month_boundary(monkeypatch):
    """
    Regression test: a two-day meeting can straddle a month boundary (e.g.
    April 30-May 1) -- the Fed's markup gives only the START month
    ("April"), with the date range itself ("30-1*") showing the wrap. The
    old code always built the date from `month` (April) + the range's last
    number (1), landing on April 1st -- three weeks early -- instead of
    May 1st.
    """
    html = """
    <html><body>
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>April</strong></div>
        <div class="fomc-meeting__date">30-1*</div>
      </div>
    </div>
    </body></html>
    """
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=html))

    df = macro_calendar.fetch_fomc_events()

    # Must resolve to May 1st (the following month), not April 1st.
    assert (df["ts"] == pd.Timestamp("2026-05-01T18:00:00Z")).all()


def test_fetch_fomc_events_future_meeting_within_one_month_is_unaffected(monkeypatch):
    """A normal same-month range ("15-16*") must still resolve to the last day within `month` -- the month-boundary fix must not change this ordinary case."""
    html = """
    <html><body>
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>June</strong></div>
        <div class="fomc-meeting__date">15-16*</div>
      </div>
    </div>
    </body></html>
    """
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=html))

    df = macro_calendar.fetch_fomc_events()

    assert (df["ts"] == pd.Timestamp("2026-06-16T18:00:00Z")).all()


def test_fetch_fred_events_requires_api_key(monkeypatch):
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "")
    with pytest.raises(RuntimeError):
        macro_calendar.fetch_fred_events()


def test_fetch_fred_events_resolves_release_ids_and_filters_dates(monkeypatch):
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "test-key")

    # A fixed `today` (rather than dt.date.today()) makes this deterministic
    # regardless of what time the test happens to run.
    today = dt.date(2026, 6, 1)
    in_window = today + dt.timedelta(days=10)
    out_of_window = today + dt.timedelta(days=400)

    def fake_get(url, params=None, timeout=None):
        if url == macro_calendar.FRED_RELEASES_URL:
            return _FakeResponse(
                {
                    "releases": [
                        {"id": 10, "name": "Consumer Price Index"},
                        {"id": 50, "name": "Employment Situation"},
                        {"id": 99, "name": "Something Unrelated"},
                    ]
                }
            )
        if url == macro_calendar.FRED_RELEASE_DATES_URL:
            release_id = params["release_id"]
            dates = {
                10: [in_window.isoformat(), out_of_window.isoformat()],
                50: [in_window.isoformat()],
            }[release_id]
            return _FakeResponse({"release_dates": [{"date": d} for d in dates]})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(macro_calendar.requests, "get", fake_get)

    df = macro_calendar.fetch_fred_events(months_ahead=6, today=today)

    assert set(df["category"]) == {"CPI", "JOBS"}
    # The out-of-window CPI date (400 days out) should be excluded by the horizon filter.
    assert len(df) == 2
    assert (df["ts"].dt.date == in_window).all()


def test_fetch_fred_events_today_defaults_to_eastern_not_server_local(monkeypatch):
    """
    Regression test: `today` used to be dt.date.today() (server-local --
    UTC on the real deployment), inconsistent with every other date in this
    file being Eastern-anchored via _eastern_to_utc. Near UTC midnight
    (~7-8pm ET), that would already read as tomorrow while it's still
    today in Eastern time, silently excluding a same-day release.

    Simulates the server clock reading 2026-01-02 02:00 UTC -- already
    Jan 2 in UTC, but still 2026-01-01 21:00 in Eastern (EST, UTC-5). A
    release scheduled for 2026-01-01 must still be included: the old
    dt.date.today() (UTC) would read "today" as Jan 2 and exclude it
    (Jan 1 < Jan 2); the fix reads "today" as Jan 1 via the Eastern clock
    and includes it.
    """
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "test-key")

    fake_now = dt.datetime(2026, 1, 2, 2, 0, tzinfo=dt.UTC)

    class _FakeDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now.astimezone(tz) if tz else fake_now

    monkeypatch.setattr(macro_calendar.dt, "datetime", _FakeDatetime)

    def fake_get(url, params=None, timeout=None):
        if url == macro_calendar.FRED_RELEASES_URL:
            return _FakeResponse({"releases": [{"id": 10, "name": "Consumer Price Index"}]})
        if url == macro_calendar.FRED_RELEASE_DATES_URL:
            return _FakeResponse({"release_dates": [{"date": "2026-01-01"}]})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(macro_calendar.requests, "get", fake_get)

    df = macro_calendar.fetch_fred_events(months_ahead=1)  # no `today` override -- exercises the real default

    assert len(df) == 1  # the Jan-1 release must survive the window filter, not be excluded as "in the past"


def test_eastern_to_utc_uses_edt_offset_in_summer():
    """Second Sunday in March to first Sunday in November is EDT (UTC-4) --
    2pm ET on a July date must be 18:00 UTC, not the EST-season 19:00."""
    ts = macro_calendar._eastern_to_utc(dt.date(2026, 7, 15), 14, 0)
    assert ts == dt.datetime(2026, 7, 15, 18, 0, tzinfo=dt.UTC)


def test_eastern_to_utc_uses_est_offset_in_winter():
    """Outside EDT season is EST (UTC-5) -- 2pm ET on a January date must be
    19:00 UTC. A hardcoded 18:00 UTC (only correct in EDT) would be an hour
    early for every winter event."""
    ts = macro_calendar._eastern_to_utc(dt.date(2026, 1, 15), 14, 0)
    assert ts == dt.datetime(2026, 1, 15, 19, 0, tzinfo=dt.UTC)


def test_fetch_fomc_events_future_meeting_uses_dst_aware_offset(monkeypatch):
    """The no-statement-link fallback path (a future meeting) must also use
    the DST-aware conversion, not a hardcoded 18:00 UTC -- a December
    meeting (EST) should land at 19:00 UTC."""
    html = """
    <html><body>
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>December</strong></div>
        <div class="fomc-meeting__date">9-10*</div>
      </div>
    </div>
    </body></html>
    """
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=html))

    df = macro_calendar.fetch_fomc_events()

    assert (df["ts"] == pd.Timestamp("2026-12-10T19:00:00Z")).all()


def test_fetch_fred_events_uses_dst_aware_offset_in_winter(monkeypatch):
    """8:30am ET during EST (winter) must be 13:30 UTC, not the EDT-season
    12:30 UTC that dt.time(12, 30) hardcoded."""
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "test-key")

    # A date guaranteed to be both in the future (fetch_fred_events filters
    # to [today, horizon]) and in EST season (winter), regardless of what
    # today actually is when this test runs: the next Jan 13 from now, at
    # least 90 days out so a 6-month months_ahead window always covers it.
    today = dt.date.today()
    winter_date = dt.date(today.year, 1, 13)
    while winter_date < today + dt.timedelta(days=90):
        winter_date = dt.date(winter_date.year + 1, 1, 13)

    def fake_get(url, params=None, timeout=None):
        if url == macro_calendar.FRED_RELEASES_URL:
            return _FakeResponse({"releases": [{"id": 10, "name": "Consumer Price Index"}]})
        return _FakeResponse({"release_dates": [{"date": winter_date.isoformat()}]})

    monkeypatch.setattr(macro_calendar.requests, "get", fake_get)

    df = macro_calendar.fetch_fred_events(months_ahead=12)

    assert len(df) == 1
    expected = pd.Timestamp(dt.datetime(winter_date.year, 1, 13, 13, 30, tzinfo=dt.UTC))
    assert df.iloc[0]["ts"] == expected


def test_scrape_macro_calendar_skips_fred_when_no_api_key(monkeypatch, caplog):
    """
    A blank FRED_API_KEY means CPI/Jobs macro-calendar rows never populate
    at all, every single week -- days_to_next_cpi/days_to_next_jobs
    (features/build_features.py) then silently never exist as features.
    That must show up as a real log warning (not a print() only stdout
    catches) since this call runs unattended.
    """
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "")
    monkeypatch.setattr(macro_calendar, "fetch_fomc_events", lambda: pd.DataFrame(
        {"event_name": ["FOMC Decision"], "ts": [pd.Timestamp("2026-01-28T18:00:00Z")], "category": ["FOMC"], "notes": [""]}
    ))

    with caplog.at_level("WARNING"):
        df = macro_calendar.scrape_macro_calendar()

    assert (df["category"] == "FOMC").all()
    assert "FRED_API_KEY" in caplog.text
