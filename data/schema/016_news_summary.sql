-- The article's short summary/dek, alongside the existing bare headline --
-- both ingestion paths (data/ingest/news.py's Polygon pull and
-- data/ingest/news_stream.py's Alpaca websocket) now capture it, so
-- features/qualitative/sentiment.py has more than a one-line headline to
-- judge a story from. A headline alone is often ambiguous or even
-- misleading about direction; the summary is there specifically to resolve
-- that.
--
-- Nullable, same pattern as `sentiment_reason` (009): existing rows simply
-- have no summary, and nothing downstream requires it to be present --
-- sentiment.py falls back to scoring off the headline alone for those, same
-- as it always has.

ALTER TABLE news_events ADD COLUMN IF NOT EXISTS summary TEXT;
