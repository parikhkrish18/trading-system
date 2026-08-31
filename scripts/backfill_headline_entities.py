"""
One-time backfill: decodes HTML entities that were stored literally in
news_events.headline before the ingestion fix in data/ingest/news.py and
data/ingest/news_stream.py (both now call html.unescape() on the raw
headline/title text before it's written).

Benzinga (via Alpaca's news stream) and, less often, Polygon deliver
headline text that's already been through an HTML renderer somewhere
upstream, so an apostrophe sometimes arrives as the literal 5-character
string "&#39;" rather than an actual apostrophe. Nothing previously decoded
that before writing to news_events, so it was stored -- and then displayed
-- exactly as broken as it arrived (e.g. "Designates ChatGPT As &#39;Very
Large Online Search Engine&#39;" instead of "...As 'Very Large Online
Search Engine'").

The ingestion fix only affects headlines written from here on. This script
is what cleans up everything already sitting in the table.

Safe to run more than once: html.unescape() is a no-op on text that has no
entities in it, and only rows whose decoded value actually differs from
the stored one are written.

Usage:
    python -m scripts.backfill_headline_entities            # applies the fix
    python -m scripts.backfill_headline_entities --dry-run  # reports only
"""
from __future__ import annotations

import argparse
import html
import logging

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine

logger = logging.getLogger(__name__)

_UPDATE_BATCH_SIZE = 1000


def find_and_fix(dry_run: bool = False) -> int:
    engine = get_engine()
    # "&...;" is the shape of every HTML entity (&#39; &amp; &quot; &nbsp;
    # &#x27; ...) -- this is just a cheap pre-filter to skip the vast
    # majority of headlines that were never affected. It doesn't decide
    # correctness: html.unescape() below is what actually determines
    # whether a row's text changes, so a false-positive match here (a
    # headline that legitimately contains a bare "&" followed by "...;"
    # text that isn't really an entity) just costs an unnecessary compare,
    # not a wrong write.
    df = pd.read_sql(text("SELECT id, ts, headline FROM news_events WHERE headline ~ '&#?[a-zA-Z0-9]+;'"), engine)
    if df.empty:
        logger.info("No headlines with entity-shaped text found — nothing to do.")
        return 0

    df["fixed"] = df["headline"].map(lambda h: html.unescape(h) if h else h)
    changed = df[df["fixed"] != df["headline"]]
    logger.info(
        "%d headline(s) matched the entity-shaped pre-filter; %d actually decode to something different.",
        len(df),
        len(changed),
    )

    if dry_run:
        for _, row in changed.head(20).iterrows():
            logger.info("WOULD FIX id=%s: %r -> %r", row["id"], row["headline"], row["fixed"])
        if len(changed) > 20:
            logger.info("...and %d more.", len(changed) - 20)
        return len(changed)

    rows = changed[["id", "ts", "fixed"]].to_dict("records")
    with engine.begin() as conn:
        for start in range(0, len(rows), _UPDATE_BATCH_SIZE):
            batch = rows[start : start + _UPDATE_BATCH_SIZE]
            conn.execute(text("UPDATE news_events SET headline = :fixed WHERE id = :id AND ts = :ts"), batch)
    logger.info("Fixed %d headline(s).", len(changed))
    return len(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    find_and_fix(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
