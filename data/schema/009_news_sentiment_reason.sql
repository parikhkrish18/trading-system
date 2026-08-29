-- A one-sentence "why" alongside the numeric sentiment score, so the
-- dashboard's Live News tab can answer "why was this ticker affected by
-- this headline" on click instead of just showing a Positive/Negative
-- label with no explanation behind it.
--
-- Nullable, same pattern as `sentiment` itself: news_events rows start
-- with sentiment_reason unset at ingest time and features/qualitative/
-- sentiment.py::backfill_unscored_news fills both columns together in its
-- next pass. Rows scored before this column existed simply have no reason
-- text and the dashboard falls back to "not yet scored" copy for them too.

ALTER TABLE news_events ADD COLUMN IF NOT EXISTS sentiment_reason TEXT;
