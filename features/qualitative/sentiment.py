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
    "Also give a one-sentence, plain-English reason a trader could read at "
    "a glance explaining why THIS symbol is affected by THIS headline (under "
    "25 words) -- e.g. 'Direct competitor's product launch threatens market "
    "share' or 'Company beat EPS estimates by a wide margin'. "
    "Respond with ONLY a JSON array of objects: "
    '[{"id": <id>, "sentiment": <float>, "reason": <string>}, ...], one '
    "entry per headline, in the same order given. No other text."
)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _score_batch(client: Anthropic, batch: pd.DataFrame) -> dict[int, tuple[float, str]]:
    items = [{"id": int(row["id"]), "symbol": row["symbol"], "headline": row["headline"]} for _, row in batch.iterrows()]
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(items)}],
    )
    text = _strip_code_fence(resp.content[0].text)
    scores = json.loads(text)
    # .get(..., "") rather than a required key: an older prompt version or a
    # model that drops the field on a given call should degrade to "no
    # reason text" rather than take the whole batch down with a KeyError --
    # the sentiment score itself is the part everything else (contradiction
    # monitor, hold rules) actually depends on.
    return {int(s["id"]): (float(s["sentiment"]), str(s.get("reason", "") or "")) for s in scores}


def score_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with at least ['id', 'ts', 'symbol', 'headline'].
    Output: same rows plus 'sentiment' (float, [-1, 1]) and 'sentiment_reason'
    (a short plain-English explanation of why that symbol is affected) columns.
    """
    if headlines.empty:
        return headlines.assign(sentiment=pd.Series(dtype=float), sentiment_reason=pd.Series(dtype=object))

    client = Anthropic(api_key=settings.anthropic_api_key)
    scored = headlines.copy()
    scored["sentiment"] = pd.NA
    scored["sentiment_reason"] = pd.NA

    for start in range(0, len(headlines), _BATCH_SIZE):
        batch = headlines.iloc[start : start + _BATCH_SIZE]
        id_to_result = _score_batch(client, batch)
        for row_id, (score, reason) in id_to_result.items():
            scored.loc[scored["id"] == row_id, "sentiment"] = score
            scored.loc[scored["id"] == row_id, "sentiment_reason"] = reason

    return scored


def backfill_unscored_news(batch_size: int = 500) -> int:
    """Pull rows from news_events where sentiment IS NULL, score them, write back."""
    engine = get_engine()
    query = """
        SELECT id, ts, symbol, headline FROM news_events
        WHERE sentiment IS NULL
        ORDER BY ts
        LIMIT %(limit)s
    """
    df = pd.read_sql(query, engine, params={"limit": int(batch_size)})
    if df.empty:
        return 0

    scored = score_sentiment(df)
    with engine.begin() as conn:
        for _, row in scored.iterrows():
            conn.exec_driver_sql(
                "UPDATE news_events SET sentiment = %s, sentiment_reason = %s WHERE id = %s AND ts = %s",
                (row["sentiment"], row["sentiment_reason"], row["id"], row["ts"]),
            )
    return len(scored)
