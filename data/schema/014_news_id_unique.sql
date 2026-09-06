-- news_events' PK was (id, ts): article id + published timestamp. That
-- doesn't survive a vendor redelivering the same article with a corrected
-- published_utc -- the upsert's old ON CONFLICT (id, ts) target couldn't
-- match the existing row by id alone, so a corrected timestamp inserted a
-- SECOND row instead of updating the first, double-counting the story in
-- sentiment aggregation windows.
--
-- Move to id alone as the row's identity. id is already a stable hash of
-- (article_id, symbol) -- see data/ingest/news.py::_stable_id -- so it is the
-- intended natural key. Before changing the PK, clean up any historical
-- duplicates created under the old (id, ts) key. Keep the newest delivery
-- (latest ingested_at, then ts), but preserve any non-null scoring fields
-- from the duplicate set so the repair does not throw away already-computed
-- sentiment/relevance annotations.
--
-- This migration is deliberately idempotent because migrate.py executes
-- every SQL file on every run. Once the PK is id-only, the guarded block is
-- a no-op.
--
-- NOTE: per 001_init.sql's own framing, TimescaleDB is optional here and
-- nothing in this codebase depends on hypertable-specific behavior. If
-- news_events is ever turned into a real hypertable, a single-column PK on
-- id violates TimescaleDB's partitioning-column requirement; this migration
-- should fail loudly rather than silently weakening the constraint.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'news_events'::regclass
          AND conname = 'news_events_pkey'
          AND cardinality(conkey) > 1
    ) THEN
        -- Preserve useful annotations before removing duplicate deliveries.
        -- The survivor is the most recently ingested copy; nullable scoring
        -- fields come from the newest non-null value found across the set.
        WITH duplicate_ids AS (
            SELECT id
            FROM news_events
            GROUP BY id
            HAVING COUNT(*) > 1
        ),
        survivors AS (
            SELECT DISTINCT ON (n.id)
                n.id,
                n.ctid AS keep_ctid
            FROM news_events n
            JOIN duplicate_ids d USING (id)
            ORDER BY n.id, n.ingested_at DESC, n.ts DESC, n.ctid DESC
        ),
        merged AS (
            SELECT
                n.id,
                (array_agg(n.sentiment ORDER BY (n.sentiment IS NULL), n.ingested_at DESC, n.ts DESC))[1]
                    AS sentiment,
                (array_agg(n.surprise ORDER BY (n.surprise IS NULL), n.ingested_at DESC, n.ts DESC))[1]
                    AS surprise,
                (array_agg(n.sentiment_reason ORDER BY (n.sentiment_reason IS NULL), n.ingested_at DESC, n.ts DESC))[1]
                    AS sentiment_reason,
                (array_agg(n.sentiment_relevant ORDER BY (n.sentiment_relevant IS NULL), n.ingested_at DESC, n.ts DESC))[1]
                    AS sentiment_relevant
            FROM news_events n
            JOIN duplicate_ids d USING (id)
            GROUP BY n.id
        )
        UPDATE news_events n
        SET
            sentiment = COALESCE(n.sentiment, m.sentiment),
            surprise = COALESCE(n.surprise, m.surprise),
            sentiment_reason = COALESCE(n.sentiment_reason, m.sentiment_reason),
            sentiment_relevant = COALESCE(n.sentiment_relevant, m.sentiment_relevant)
        FROM survivors s
        JOIN merged m USING (id)
        WHERE n.ctid = s.keep_ctid;

        -- Now remove only the redundant older deliveries. ctid is used only
        -- inside this one transaction to identify physical duplicate rows.
        WITH ranked AS (
            SELECT
                ctid,
                ROW_NUMBER() OVER (
                    PARTITION BY id
                    ORDER BY ingested_at DESC, ts DESC, ctid DESC
                ) AS rn
            FROM news_events
        )
        DELETE FROM news_events n
        USING ranked r
        WHERE n.ctid = r.ctid
          AND r.rn > 1;

        ALTER TABLE news_events DROP CONSTRAINT news_events_pkey;
        ALTER TABLE news_events ADD CONSTRAINT news_events_pkey PRIMARY KEY (id);
    END IF;
END $$;
