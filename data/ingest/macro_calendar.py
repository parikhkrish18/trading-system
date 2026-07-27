"""
Phase 1 macro calendar puller.

FOMC/CPI/jobs dates are published on fixed public schedules, so this doesn't
need a paid vendor — but it does need to be kept current. This module ships
with a small seed list so the pipeline runs end-to-end out of the box;
replace `SEED_EVENTS` with a scraped/maintained feed (e.g. the Fed's own
FOMC schedule page, BLS release calendar) before relying on it.

Usage:
    python -m data.ingest.macro_calendar --seed
"""
from __future__ import annotations

import argparse

import pandas as pd

from data.ingest.db import upsert_dataframe

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


def seed_macro_calendar() -> int:
    df = pd.DataFrame(SEED_EVENTS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return upsert_dataframe(df, table="macro_calendar", conflict_cols=["event_name", "ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed/refresh the macro calendar table.")
    parser.add_argument("--seed", action="store_true", help="Load the built-in seed list.")
    args = parser.parse_args()
    if args.seed:
        n = seed_macro_calendar()
        print(f"Seeded {n} macro calendar rows. Replace SEED_EVENTS with a maintained feed for real use.")
    else:
        print("Nothing to do — pass --seed, or extend this module with a real scraper.")


if __name__ == "__main__":
    main()
