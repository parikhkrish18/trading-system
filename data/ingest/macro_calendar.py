"""
Phase 1 macro calendar puller.

FOMC dates are scraped directly from the Fed's own calendar page with
BeautifulSoup. CPI/jobs dates are NOT scraped from bls.gov — that site
returns HTTP 403 ("bot activity ... is prohibited") to any automated
request, including a plain robots.txt fetch, so scraping it would mean
working around an explicit anti-bot policy. Instead, CPI/jobs release
dates come from FRED (Federal Reserve Bank of St. Louis)'s public API,
which republishes the same BLS release calendar and explicitly allows
programmatic access with a free API key:
https://fred.stlouisfed.org/docs/api/api_key.html -> set FRED_API_KEY.

Usage:
    python -m data.ingest.macro_calendar --scrape                 # FOMC + (if FRED_API_KEY set) CPI/Jobs
    python -m data.ingest.macro_calendar --scrape --months-ahead 3
    python -m data.ingest.macro_calendar --seed                   # static fallback, no network calls
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import settings
from data.ingest.db import upsert_dataframe

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_EASTERN = ZoneInfo("America/New_York")


def _eastern_to_utc(date: dt.date, hour: int, minute: int) -> dt.datetime:
    """
    Build a UTC timestamp for a wall-clock Eastern time on `date`, DST-aware.

    A hardcoded UTC offset (e.g. "2pm ET is 18:00 UTC") is only true for
    half the year -- EDT (UTC-4), roughly mid-March to early November. The
    other half (EST, UTC-5) it's 19:00 UTC, and hand-rolling "second Sunday
    in March to first Sunday in November" is exactly the kind of date-math
    stdlib's zoneinfo already gets right (including the rare edge cases,
    like a transition falling on the event date itself).
    """
    local = dt.datetime.combine(date, dt.time(hour, minute), tzinfo=_EASTERN)
    return local.astimezone(dt.UTC)
FRED_RELEASES_URL = "https://api.stlouisfed.org/fred/releases"
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
_USER_AGENT = "trading-system-macro-calendar/1.0 (personal research bot)"

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

# FRED release names -> (event_name, category) written into macro_calendar.
_FRED_TARGETS = {
    "Consumer Price Index": ("CPI Release", "CPI"),
    "Employment Situation": ("Jobs Report", "JOBS"),
}

# Example seed only — replace/extend with real, maintained dates before
# using event-risk features in anything but a smoke test.
SEED_EVENTS: list[dict] = [
    {"event_name": "FOMC Decision", "ts": "2026-09-16T18:00:00Z", "category": "FOMC", "notes": ""},
    {"event_name": "FOMC Decision", "ts": "2026-10-28T18:00:00Z", "category": "FOMC", "notes": ""},
    {"event_name": "FOMC Decision", "ts": "2026-12-09T19:00:00Z", "category": "FOMC", "notes": ""},
    {"event_name": "CPI Release", "ts": "2026-08-12T12:30:00Z", "category": "CPI", "notes": ""},
    {"event_name": "CPI Release", "ts": "2026-09-11T12:30:00Z", "category": "CPI", "notes": ""},
    {"event_name": "Jobs Report", "ts": "2026-08-07T12:30:00Z", "category": "JOBS", "notes": ""},
    {"event_name": "Jobs Report", "ts": "2026-09-04T12:30:00Z", "category": "JOBS", "notes": ""},
]


def _parse_fomc_row(row, year: int) -> dict | None:
    month_el = row.find(class_="fomc-meeting__month")
    date_el = row.find(class_="fomc-meeting__date")
    if month_el is None or date_el is None:
        return None

    month = _MONTHS.get(month_el.get_text(strip=True))
    if month is None:
        return None

    # Once a meeting has happened, the statement link carries the exact
    # decision date (YYYYMMDD); use it when present since it's authoritative.
    statement_link = row.find("a", href=re.compile(r"monetary\d{8}a\.htm$"))
    if statement_link:
        date_str = re.search(r"(\d{8})", statement_link["href"]).group(1)
        decision_date = dt.datetime.strptime(date_str, "%Y%m%d").date()
    else:
        # Future meeting, no statement yet: fall back to the last day of the
        # published date range (e.g. "17-18*" -> 18th), approximate the
        # statement release as 2pm ET (converted to UTC below, DST-aware).
        digits_only = re.sub(r"[^\d-]", "", date_el.get_text(strip=True))
        try:
            last_day = int(digits_only.split("-")[-1])
        except ValueError:
            return None
        decision_date = dt.date(year, month, last_day)

    return {
        "event_name": "FOMC Decision",
        "ts": _eastern_to_utc(decision_date, 14, 0),
        "category": "FOMC",
        "notes": "",
    }


def fetch_fomc_events() -> pd.DataFrame:
    """Scrape the Fed's FOMC meeting calendar. Returns event_name, ts, category, notes."""
    resp = requests.get(FOMC_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows: list[dict] = []
    for panel in soup.find_all("div", class_="panel-default"):
        heading = panel.find("h4")
        if heading is None:
            continue
        year_match = re.search(r"(\d{4})\s+FOMC Meetings", heading.get_text())
        if not year_match:
            continue
        year = int(year_match.group(1))

        for meeting_row in panel.find_all("div", class_="fomc-meeting"):
            parsed = _parse_fomc_row(meeting_row, year)
            if parsed:
                rows.append(parsed)

    return pd.DataFrame(rows, columns=["event_name", "ts", "category", "notes"])


def fetch_fred_events(months_ahead: int = 6) -> pd.DataFrame:
    """
    Pull upcoming CPI / Employment Situation release dates from FRED.
    Requires FRED_API_KEY. Returns event_name, ts, category, notes.
    """
    if not settings.fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env."
        )

    resp = requests.get(
        FRED_RELEASES_URL, params={"api_key": settings.fred_api_key, "file_type": "json"}, timeout=30
    )
    resp.raise_for_status()
    releases = resp.json().get("releases", [])
    release_id_by_name = {r["name"].strip().lower(): r["id"] for r in releases}

    today = dt.date.today()
    horizon = today + dt.timedelta(days=30 * months_ahead)
    rows: list[dict] = []

    for release_name, (event_name, category) in _FRED_TARGETS.items():
        release_id = release_id_by_name.get(release_name.lower())
        if release_id is None:
            continue

        dates_resp = requests.get(
            FRED_RELEASE_DATES_URL,
            params={
                "release_id": release_id,
                "api_key": settings.fred_api_key,
                "file_type": "json",
                "realtime_start": today.isoformat(),
                "realtime_end": horizon.isoformat(),
                "include_release_dates_with_no_data": "true",
            },
            timeout=30,
        )
        dates_resp.raise_for_status()

        for entry in dates_resp.json().get("release_dates", []):
            release_date = dt.date.fromisoformat(entry["date"])
            if not (today <= release_date <= horizon):
                continue
            rows.append(
                {
                    "event_name": event_name,
                    # 8:30am ET, DST-aware (see _eastern_to_utc) -- not a
                    # hardcoded 12:30 UTC, which is only correct during EDT.
                    "ts": _eastern_to_utc(release_date, 8, 30),
                    "category": category,
                    "notes": "",
                }
            )

    return pd.DataFrame(rows, columns=["event_name", "ts", "category", "notes"])


def scrape_macro_calendar(months_ahead: int = 6) -> pd.DataFrame:
    """FOMC (always) + CPI/Jobs (only if FRED_API_KEY is configured)."""
    frames = [fetch_fomc_events()]
    if settings.fred_api_key:
        frames.append(fetch_fred_events(months_ahead=months_ahead))
    else:
        print("FRED_API_KEY not set — skipping CPI/Jobs dates, scraping FOMC only.")
    return pd.concat(frames, ignore_index=True)


def refresh_macro_calendar(months_ahead: int = 6) -> int:
    df = scrape_macro_calendar(months_ahead=months_ahead)
    if df.empty:
        return 0
    return upsert_dataframe(df, table="macro_calendar", conflict_cols=["event_name", "ts"])


def seed_macro_calendar() -> int:
    df = pd.DataFrame(SEED_EVENTS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return upsert_dataframe(df, table="macro_calendar", conflict_cols=["event_name", "ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the macro calendar table.")
    parser.add_argument("--scrape", action="store_true", help="Scrape live FOMC + FRED dates (recommended).")
    parser.add_argument("--months-ahead", type=int, default=6, help="How far ahead to pull CPI/Jobs dates.")
    parser.add_argument("--seed", action="store_true", help="Load the static built-in seed list instead.")
    args = parser.parse_args()

    if args.scrape:
        n = refresh_macro_calendar(months_ahead=args.months_ahead)
        print(f"Upserted {n} macro calendar row(s) from live sources.")
    elif args.seed:
        n = seed_macro_calendar()
        print(f"Seeded {n} macro calendar rows. Prefer --scrape for real use.")
    else:
        print("Nothing to do — pass --scrape (recommended) or --seed.")


if __name__ == "__main__":
    main()
