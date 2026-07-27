"""
Phase 2 qualitative features — STUB.

Turns raw headlines/filing text (from `news_events`) into a numeric
sentiment/surprise score per (symbol, day). Two reasonable approaches:

  1. LLM-based: prompt an LLM per headline/article for a sentiment score in
     [-1, 1], batched, with a cheap model — this is a high-volume, low-value-
     per-call task, so cost matters more than for the core forecast model.
  2. Fine-tuned classifier: cheaper at scale once you have labeled data
     (e.g. label by subsequent short-horizon return as a proxy target).

Either way, keep this as a separate pass from ingestion (see
data/ingest/news.py) so you can re-score historical news with a better
model without re-pulling raw data.
"""
from __future__ import annotations

import pandas as pd

from data.ingest.db import get_engine


def score_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with at least ['id', 'ts', 'symbol', 'headline'].
    Output: same rows plus a 'sentiment' column in [-1, 1].

    Replace this with a real call to your chosen scorer (LLM API or
    fine-tuned classifier inference).
    """
    raise NotImplementedError(
        "Wire up an LLM call or fine-tuned classifier here. Keep this function "
        "pure (dataframe in, dataframe out) so it's easy to unit test with a "
        "handful of hand-labeled examples before trusting it in the pipeline."
    )


def backfill_unscored_news(batch_size: int = 500) -> int:
    """Pull rows from news_events where sentiment IS NULL, score them, write back."""
    engine = get_engine()
    query = f"""
        SELECT id, ts, symbol, headline FROM news_events
        WHERE sentiment IS NULL
        ORDER BY ts
        LIMIT {batch_size}
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return 0

    scored = score_sentiment(df)
    with engine.begin() as conn:
        for _, row in scored.iterrows():
            conn.exec_driver_sql(
                "UPDATE news_events SET sentiment = %s WHERE id = %s AND ts = %s",
                (row["sentiment"], row["id"], row["ts"]),
            )
    return len(scored)
