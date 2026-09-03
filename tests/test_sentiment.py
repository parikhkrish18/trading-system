import json

import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest.db import get_engine, upsert_dataframe
from features.qualitative import sentiment


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_fn):
        self._response_fn = response_fn

    def create(self, model, max_tokens, system, messages):
        return _FakeMessage(self._response_fn(messages))


class _FakeAnthropic:
    def __init__(self, response_fn, api_key=None):
        self.messages = _FakeMessages(response_fn)


def test_score_sentiment_merges_scores_back_onto_rows(monkeypatch):
    def respond(messages):
        items = json.loads(messages[0]["content"])
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "sentiment": 0.5 if "good" in item["headline"] else -0.5,
                    "reason": "good news" if "good" in item["headline"] else "bad news",
                    "relevant": True,
                }
                for item in items
            ]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    headlines = pd.DataFrame(
        {
            "id": [1, 2],
            "ts": pd.to_datetime(["2026-07-27", "2026-07-27"], utc=True),
            "symbol": ["SPY", "SPY"],
            "headline": ["good news for SPY", "bad news for SPY"],
        }
    )

    scored = sentiment.score_sentiment(headlines)

    assert list(scored["sentiment"]) == [0.5, -0.5]
    assert list(scored["sentiment_reason"]) == ["good news", "bad news"]
    assert list(scored["sentiment_relevant"]) == [True, True]
    assert set(scored.columns) >= {
        "id",
        "ts",
        "symbol",
        "headline",
        "sentiment",
        "sentiment_reason",
        "sentiment_relevant",
    }


def test_score_sentiment_flags_a_mistagged_symbol_as_not_relevant(monkeypatch):
    """A news vendor mistagging a symbol onto a story (e.g. an MSFT story tagged NYT)
    should come back with relevant=False rather than a fabricated sentiment reading."""

    def respond(messages):
        items = json.loads(messages[0]["content"])
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "sentiment": 0.1,
                    "reason": "story is actually about a different company",
                    "relevant": False,
                }
                for item in items
            ]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    headlines = pd.DataFrame(
        {
            "id": [1],
            "ts": pd.to_datetime(["2026-08-29"], utc=True),
            "symbol": ["NYT"],
            "headline": ["Steve Ballmer is $65 Billion Richer than Bill Gates. Here's Why."],
        }
    )

    scored = sentiment.score_sentiment(headlines)

    assert list(scored["sentiment_relevant"]) == [False]


def test_score_sentiment_missing_relevant_field_defaults_to_true(monkeypatch):
    """An older prompt/model response that omits "relevant" must not silently start
    excluding real data -- default to assuming the vendor's tag is fine."""

    def respond(messages):
        items = json.loads(messages[0]["content"])
        return json.dumps([{"id": item["id"], "sentiment": 0.2} for item in items])

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    headlines = pd.DataFrame(
        {
            "id": [1],
            "ts": pd.to_datetime(["2026-07-27"], utc=True),
            "symbol": ["SPY"],
            "headline": ["headline"],
        }
    )

    scored = sentiment.score_sentiment(headlines)

    assert list(scored["sentiment_relevant"]) == [True]


def test_score_sentiment_missing_reason_field_degrades_to_empty_string(monkeypatch):
    """An older prompt/model response that omits "reason" entirely must not crash the batch --
    the sentiment score is the part everything downstream actually depends on."""

    def respond(messages):
        items = json.loads(messages[0]["content"])
        return json.dumps([{"id": item["id"], "sentiment": 0.2} for item in items])

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    headlines = pd.DataFrame(
        {
            "id": [1],
            "ts": pd.to_datetime(["2026-07-27"], utc=True),
            "symbol": ["SPY"],
            "headline": ["headline"],
        }
    )

    scored = sentiment.score_sentiment(headlines)

    assert list(scored["sentiment"]) == [0.2]
    assert list(scored["sentiment_reason"]) == [""]


def test_score_sentiment_handles_markdown_code_fence(monkeypatch):
    """Claude sometimes wraps JSON in ```json ... ``` despite being told not to."""

    def respond(messages):
        items = json.loads(messages[0]["content"])
        payload = json.dumps([{"id": item["id"], "sentiment": 0.3} for item in items])
        return f"```json\n{payload}\n```"

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    headlines = pd.DataFrame(
        {
            "id": [1],
            "ts": pd.to_datetime(["2026-07-27"], utc=True),
            "symbol": ["SPY"],
            "headline": ["headline"],
        }
    )
    scored = sentiment.score_sentiment(headlines)
    assert list(scored["sentiment"]) == [0.3]


def test_score_sentiment_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(lambda messages: "[]"))
    headlines = pd.DataFrame(columns=["id", "ts", "symbol", "headline"])
    scored = sentiment.score_sentiment(headlines)
    assert scored.empty


def test_score_sentiment_batches_in_groups_of_batch_size(monkeypatch):
    calls = []

    def respond(messages):
        items = json.loads(messages[0]["content"])
        calls.append(len(items))
        return json.dumps([{"id": item["id"], "sentiment": 0.0} for item in items])

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    monkeypatch.setattr(sentiment, "_BATCH_SIZE", 2)

    headlines = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "ts": pd.to_datetime(["2026-07-27"] * 3, utc=True),
            "symbol": ["SPY"] * 3,
            "headline": ["h1", "h2", "h3"],
        }
    )
    sentiment.score_sentiment(headlines)

    assert calls == [2, 1]


# --------------------------------------------------------------------------
# backfill_unscored_news: a batch with one id missing from the LLM's
# response must not crash and must not take the rest of the batch down.
# --------------------------------------------------------------------------


@pytest.fixture
def _unscored_news_rows():
    """Two unscored news_events rows on the real DB, cleaned up after."""
    engine = get_engine()
    rows = pd.DataFrame(
        {
            "id": [900001, 900002],
            "symbol": ["ZZZTEST", "ZZZTEST"],
            "ts": pd.to_datetime(["2026-07-27T12:00:00Z", "2026-07-27T13:00:00Z"], utc=True),
            "headline": ["headline scores fine", "headline the LLM response omits"],
            "source": ["test-fixture", "test-fixture"],
        }
    )
    upsert_dataframe(rows, table="news_events", conflict_cols=["id"])
    yield rows
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM news_events WHERE source = 'test-fixture'"))


def test_backfill_unscored_news_survives_a_response_missing_one_id(monkeypatch, _unscored_news_rows):
    """
    Regression test, finding #3: score_sentiment leaves sentiment_relevant as
    pd.NA for any row whose id the LLM's JSON response omits.
    backfill_unscored_news used to do a raw bool(row["sentiment_relevant"])
    per row -- bool(pd.NA) raises TypeError, aborting the WHOLE batch
    transaction (including rows that scored fine), and since the next run
    re-selects the same oldest batch it hit the same missing id and crashed
    again forever. The row with a real score must still get written; the
    row the response omitted must not crash the batch, and should stay
    unscored (sentiment IS NULL) so it's retried, not permanently skipped.
    """
    engine = get_engine()
    scored_row_id = _unscored_news_rows.iloc[0]["id"]
    missing_row_id = _unscored_news_rows.iloc[1]["id"]

    def respond(messages):
        items = json.loads(messages[0]["content"])
        # Omit the second id entirely, simulating a malformed/truncated
        # LLM response that doesn't cover every headline it was sent.
        return json.dumps(
            [
                {"id": item["id"], "sentiment": 0.4, "reason": "fine", "relevant": True}
                for item in items
                if item["id"] == int(scored_row_id)
            ]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    n = sentiment.backfill_unscored_news(batch_size=500)

    assert n == 1  # only the row that actually scored was written

    with engine.connect() as conn:
        scored = conn.execute(
            text("SELECT sentiment, sentiment_relevant FROM news_events WHERE id = :id"), {"id": int(scored_row_id)}
        ).fetchone()
        missing = conn.execute(
            text("SELECT sentiment, sentiment_relevant FROM news_events WHERE id = :id"), {"id": int(missing_row_id)}
        ).fetchone()

    assert scored.sentiment == 0.4
    assert scored.sentiment_relevant is True
    assert missing.sentiment is None  # left unscored, not crashed on and not fabricated
    assert missing.sentiment_relevant is None
