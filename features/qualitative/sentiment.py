"""
Phase 2 qualitative features.

Turns raw headlines/filing text (from `news_events`) into a numeric
sentiment score per (symbol, day), via batched calls to Claude (cheap model —
this is a high-volume, low-value-per-call task, so cost matters more than
for the core forecast model).

Kept as a separate pass from ingestion (see data/ingest/news.py) so you can
re-score historical news with a better model without re-pulling raw data.
"""
from __future__ import annotations

import json

import pandas as pd
from anthropic import Anthropic

from config.settings import settings
from data.ingest.db import get_engine

_MODEL = "claude-haiku-4-5"
_BATCH_SIZE = 20

_SYSTEM_PROMPT = (
    "You are scoring financial news headlines for sentiment. For each "
    "headline, assign a sentiment score from -1.0 (very negative for the "
    "stock) to 1.0 (very positive for the stock), 0.0 for neutral/mixed. "
    "Respond with ONLY a JSON array of objects: "
    '[{"id": <id>, "sentiment": <float>}, ...], one entry per headline, '
    "in the same order given. No other text."
)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _score_batch(client: Anthropic, batch: pd.DataFrame) -> dict[int, float]:
    items = [{"id": int(row["id"]), "symbol": row["symbol"], "headline": row["headline"]} for _, row in batch.iterrows()]
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(items)}],
    )
    text = _strip_code_fence(resp.content[0].text)
    scores = json.loads(text)
    return {int(s["id"]): float(s["sentiment"]) for s in scores}


def score_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with at least ['id', 'ts', 'symbol', 'headline'].
    Output: same rows plus a 'sentiment' column in [-1, 1].
    """
    if headlines.empty:
        return headlines.assign(sentiment=pd.Series(dtype=float))

    client = Anthropic(api_key=settings.anthropic_api_key)
    scored = headlines.copy()
    scored["sentiment"] = pd.NA

    for start in range(0, len(headlines), _BATCH_SIZE):
        batch = headlines.iloc[start : start + _BATCH_SIZE]
        id_to_score = _score_batch(client, batch)
        for row_id, score in id_to_score.items():
            scored.loc[scored["id"] == row_id, "sentiment"] = score

    return scored


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
