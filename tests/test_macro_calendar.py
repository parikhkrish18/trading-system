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

    jan_2026 = df.loc[df["ts"] == pd.Timestamp("2026-01-28T18:00:00Z")]
    assert len(jan_2026) == 1  # exact date came from the statement link (day 28, not the range's last "8")

    dec_2025 = df.loc[df["ts"] == pd.Timestamp("2025-12-10T18:00:00Z")]
    assert len(dec_2025) == 1


def test_fetch_fomc_events_falls_back_to_date_range_for_future_meetings(monkeypatch):
    monkeypatch.setattr(macro_calendar.requests, "get", lambda *a, **k: _FakeResponse(text=_FOMC_HTML))

    df = macro_calendar.fetch_fomc_events()

    # September 2026 has no statement link -> falls back to last day of "15-16*" -> the 16th.
    sep_2026 = df.loc[df["ts"] == pd.Timestamp("2026-09-16T18:00:00Z")]
    assert len(sep_2026) == 1


def test_fetch_fred_events_requires_api_key(monkeypatch):
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "")
    with pytest.raises(RuntimeError):
        macro_calendar.fetch_fred_events()


def test_fetch_fred_events_resolves_release_ids_and_filters_dates(monkeypatch):
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "test-key")

    today = dt.date.today()
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

    df = macro_calendar.fetch_fred_events(months_ahead=6)

    assert set(df["category"]) == {"CPI", "JOBS"}
    # The out-of-window CPI date (400 days out) should be excluded by the horizon filter.
    assert len(df) == 2
    assert (df["ts"].dt.date == in_window).all()


def test_scrape_macro_calendar_skips_fred_when_no_api_key(monkeypatch):
    monkeypatch.setattr(macro_calendar.settings, "fred_api_key", "")
    monkeypatch.setattr(macro_calendar, "fetch_fomc_events", lambda: pd.DataFrame(
        {"event_name": ["FOMC Decision"], "ts": [pd.Timestamp("2026-01-28T18:00:00Z")], "category": ["FOMC"], "notes": [""]}
    ))

    df = macro_calendar.scrape_macro_calendar()

    assert (df["category"] == "FOMC").all()
