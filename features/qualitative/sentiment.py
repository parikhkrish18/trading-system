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
import logging

import pandas as pd
from anthropic import Anthropic

from config.settings import settings
from data.ingest.db import get_engine

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"
_BATCH_SIZE = 20

_SYSTEM_PROMPT = (
    "You are scoring financial news headlines for sentiment, one headline "
    "paired with one ticker symbol it was tagged with by a news vendor. "
    "News vendors sometimes mistag a symbol onto a story that isn't "
    "actually about that company -- check this first: is the headline "
    "genuinely, substantively about THIS symbol's company, or is the tag "
    "wrong/incidental (a passing mention, an unrelated company with a "
    "similar name, or a clear vendor tagging error)? Set \"relevant\" to "
    "false for the latter case. "
    "For each headline, assign a sentiment score from -1.0 (very negative for "
    "the stock) to 1.0 (very positive for the stock), 0.0 for neutral/mixed -- "
    "if relevant is false, still give your best-guess sentiment, it just won't "
    "be used. "
    "Also give a one-sentence, plain-English reason a trader could read at "
    "a glance explaining why THIS symbol is affected by THIS headline (under "
    "25 words) -- e.g. 'Direct competitor's product launch threatens market "
    "share' or 'Company beat EPS estimates by a wide margin'; if relevant is "
    "false, the reason should say what the story is actually about instead. "
    "Respond with ONLY a JSON array of objects: "
    '[{"id": <id>, "sentiment": <float>, "reason": <string>, "relevant": '
    "<bool>}, ...], one entry per headline, in the same order given. No "
    "other text."
)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _score_batch(client: Anthropic, batch: pd.DataFrame) -> dict[int, tuple[float, str, bool]]:
    items = [{"id": int(row["id"]), "symbol": row["symbol"], "headline": row["headline"]} for _, row in batch.iterrows()]
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(items)}],
    )
    text = _strip_code_fence(resp.content[0].text)
    scores = json.loads(text)
    # .get(..., default) rather than a required key: an older prompt version
    # or a model that drops a field on a given call should degrade gracefully
    # rather than take the whole batch down with a KeyError -- the sentiment
    # score itself is the part everything else (contradiction monitor, hold
    # rules) actually depends on. relevant defaults to True (assume the
    # vendor's tag is fine) rather than False, since a missing field must
    # never silently start excluding real data from the model/contradiction
    # check that a prior prompt version's rows never had a chance to set.
    return {
        int(s["id"]): (float(s["sentiment"]), str(s.get("reason", "") or ""), bool(s.get("relevant", True)))
        for s in scores
    }


def score_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with at least ['id', 'ts', 'symbol', 'headline'].
    Output: same rows plus 'sentiment' (float, [-1, 1]), 'sentiment_reason'
    (a short plain-English explanation of why that symbol is affected), and
    'sentiment_relevant' (False when the vendor's symbol tag doesn't
    actually fit the story -- see data/schema/010_news_sentiment_relevance.sql)
    columns.
    """
    if headlines.empty:
        return headlines.assign(
            sentiment=pd.Series(dtype=float),
            sentiment_reason=pd.Series(dtype=object),
            sentiment_relevant=pd.Series(dtype=object),
        )

    client = Anthropic(api_key=settings.anthropic_api_key)
    scored = headlines.copy()
    scored["sentiment"] = pd.NA
    scored["sentiment_reason"] = pd.NA
    scored["sentiment_relevant"] = pd.NA

    for start in range(0, len(headlines), _BATCH_SIZE):
        batch = headlines.iloc[start : start + _BATCH_SIZE]
        id_to_result = _score_batch(client, batch)
        for row_id, (score, reason, relevant) in id_to_result.items():
            scored.loc[scored["id"] == row_id, "sentiment"] = score
            scored.loc[scored["id"] == row_id, "sentiment_reason"] = reason
            scored.loc[scored["id"] == row_id, "sentiment_relevant"] = relevant

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
    written = 0
    with engine.begin() as conn:
        for _, row in scored.iterrows():
            # A row whose id the LLM's JSON response omitted (a malformed/
            # truncated response, or the model just dropping one) keeps the
            # pd.NA that score_sentiment initializes every row to. bool(pd.NA)
            # raises TypeError -- which used to abort this entire batch
            # transaction, including every row that scored fine, and since
            # the next run re-selects the same oldest unscored batch it hit
            # the same missing id and crashed again forever. Skip just this
            # row instead: it stays sentiment IS NULL, so it's naturally
            # retried by the next backfill run rather than permanently
            # skipped, and every other row in the batch still gets written.
            if pd.isna(row["sentiment"]) or pd.isna(row["sentiment_relevant"]):
                logger.warning(
                    "No sentiment score came back for news_events id=%s (ts=%s) -- "
                    "leaving it unscored for the next backfill run.",
                    row["id"], row["ts"],
                )
                continue
            conn.exec_driver_sql(
                "UPDATE news_events SET sentiment = %s, sentiment_reason = %s, sentiment_relevant = %s "
                "WHERE id = %s AND ts = %s",
                (row["sentiment"], row["sentiment_reason"], bool(row["sentiment_relevant"]), row["id"], row["ts"]),
            )
            written += 1
    return written
