-- news_events' PK was (id, ts): article id + published timestamp. That
-- doesn't survive a vendor redelivering the same article with a corrected
-- published_utc -- the upsert's ON CONFLICT (id, ts) target in
-- data/ingest/news.py::ingest_news (and news_stream.py's flush) can't match
-- the existing row by id alone, so a corrected timestamp inserted a SECOND
-- row instead of updating the first, double-counting the story in sentiment
-- aggregation windows.
--
-- Move to id alone as the row's identity: a redelivery with a different ts
-- now UPDATEs the existing row (ts included) instead of inserting a
-- duplicate. id is already a stable hash of (article_id, symbol) -- see
-- data/ingest/news.py::_stable_id -- so it was already the intended natural
-- key; ts only rode along in the original PK so the table could become a
-- TimescaleDB hypertable (which requires the partitioning column in every
-- unique constraint) if the extension happened to be installed.
--
-- NOTE: per 001_init.sql's own framing, TimescaleDB is optional here and
-- nothing in this codebase depends on hypertable-specific behavior -- these
-- stay plain Postgres tables whenever the extension isn't installed (true
-- of every environment this system currently runs in). If news_events ever
-- IS turned into a real hypertable, a single-column PK on id violates
-- TimescaleDB's partitioning-column requirement and this migration would
-- fail loudly at that ALTER TABLE -- the right failure mode (refuse and
-- alert) rather than silently corrupting the constraint.
DO $$
BEGIN
    -- Guard on "the existing pkey still covers more than one column" rather
    -- than a fixed before/after constraint name, so re-running this
    -- migration (migrate.py applies every file on every run) is a no-op the
    -- second time instead of erroring on an already-dropped constraint.
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'news_events_pkey' AND cardinality(conkey) > 1
    ) THEN
        ALTER TABLE news_events DROP CONSTRAINT news_events_pkey;
        ALTER TABLE news_events ADD CONSTRAINT news_events_pkey PRIMARY KEY (id);
    END IF;
END $$;
