"""Regression coverage for the historical news_events duplicate repair."""

from __future__ import annotations

from pathlib import Path

from data.ingest.db import get_engine


_MIGRATION = Path("data/schema/014_news_id_unique.sql")


def test_news_id_migration_deduplicates_and_preserves_scoring_fields():
    """A corrected timestamp must not make the id-only PK migration fail."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    engine = get_engine()

    with engine.begin() as conn:
        # A temp table shadows production/public.news_events for this one
        # connection, so the migration can be exercised against the exact
        # pre-014 shape without disturbing the schema initialized by CI.
        conn.exec_driver_sql(
            """
            CREATE TEMP TABLE news_events (
                id BIGINT NOT NULL,
                symbol TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                headline TEXT,
                source TEXT NOT NULL DEFAULT 'unknown',
                sentiment DOUBLE PRECISION,
                surprise DOUBLE PRECISION,
                ingested_at TIMESTAMPTZ NOT NULL,
                sentiment_reason TEXT,
                sentiment_relevant BOOLEAN,
                CONSTRAINT news_events_pkey PRIMARY KEY (id, ts)
            )
            """
        )
        conn.exec_driver_sql(
            """
            INSERT INTO news_events
                (id, symbol, ts, headline, source, sentiment, surprise,
                 ingested_at, sentiment_reason, sentiment_relevant)
            VALUES
                (42, 'AAPL', '2026-09-01T12:00:00Z', 'older delivery', 'polygon',
                 0.8, NULL, '2026-09-01T12:01:00Z', 'already scored', TRUE),
                (42, 'AAPL', '2026-09-01T12:05:00Z', 'corrected timestamp', 'polygon',
                 NULL, 0.25, '2026-09-01T12:06:00Z', NULL, NULL),
                (99, 'MSFT', '2026-09-01T13:00:00Z', 'unrelated row', 'polygon',
                 -0.2, NULL, '2026-09-01T13:01:00Z', 'keep me', FALSE)
            """
        )

        conn.exec_driver_sql(sql)
        # migrate.py replays every migration on every deploy, so a second
        # execution must be a no-op rather than another destructive pass.
        conn.exec_driver_sql(sql)

        rows = conn.exec_driver_sql(
            """
            SELECT id, symbol, ts, headline, sentiment, surprise,
                   sentiment_reason, sentiment_relevant
            FROM news_events
            ORDER BY id
            """
        ).mappings().all()

        assert len(rows) == 2
        repaired = rows[0]
        assert repaired["id"] == 42
        assert repaired["headline"] == "corrected timestamp"
        assert repaired["ts"].isoformat().startswith("2026-09-01T12:05:00")
        assert repaired["sentiment"] == 0.8
        assert repaired["surprise"] == 0.25
        assert repaired["sentiment_reason"] == "already scored"
        assert repaired["sentiment_relevant"] is True

        pk_columns = conn.exec_driver_sql(
            """
            SELECT a.attname
            FROM pg_constraint c
            JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
            JOIN pg_attribute a
              ON a.attrelid = c.conrelid
             AND a.attnum = k.attnum
            WHERE c.conrelid = 'news_events'::regclass
              AND c.contype = 'p'
            ORDER BY k.ord
            """
        ).scalars().all()
        assert pk_columns == ["id"]
