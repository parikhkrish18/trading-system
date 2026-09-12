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


def test_score_sentiment_includes_the_summary_in_the_request_when_present(monkeypatch):
    seen_items = []

    def respond(messages):
        items = json.loads(messages[0]["content"])
        seen_items.extend(items)
        return json.dumps([{"id": item["id"], "sentiment": 0.0, "reason": "", "relevant": True} for item in items])

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    headlines = pd.DataFrame(
        {
            "id": [1, 2],
            "ts": pd.to_datetime(["2026-07-27", "2026-07-27"], utc=True),
            "symbol": ["SPY", "AAPL"],
            "headline": ["headline with a summary", "headline with no summary"],
            "summary": ["Revenue beat estimates by a wide margin.", None],
        }
    )

    sentiment.score_sentiment(headlines)

    by_id = {item["id"]: item for item in seen_items}
    assert by_id[1]["summary"] == "Revenue beat estimates by a wide margin."
    assert by_id[2]["summary"] == ""  # NaN in the DataFrame must not reach the API as NaN/None


def test_score_sentiment_works_without_a_summary_column_at_all(monkeypatch):
    """Callers that never pass 'summary' (older code paths, tests) must not break."""

    def respond(messages):
        items = json.loads(messages[0]["content"])
        assert all(item["summary"] == "" for item in items)
        return json.dumps([{"id": item["id"], "sentiment": 0.0, "reason": "", "relevant": True} for item in items])

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
    assert list(scored["sentiment"]) == [0.0]


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


def test_score_sentiment_survives_a_batch_whose_response_is_not_valid_json(monkeypatch):
    """
    Regression test, hit live: one batch's response failing to parse at all
    (a stray unescaped character mid-JSON -- distinct from the
    already-covered case of one row's optional field missing from an
    otherwise-valid response) used to raise straight out of
    score_sentiment and discard every OTHER batch's already-successful
    results from the same call, along with the Claude spend that produced
    them, since backfill_unscored_news never got to write anything. The
    bad batch's rows must stay unscored for a retry; every sibling batch's
    real results must still come back.
    """

    def respond(messages):
        items = json.loads(messages[0]["content"])
        if items[0]["id"] == 2:
            return '[{"id": 2, "sentiment": 0.5,'  # deliberately invalid/truncated JSON
        return json.dumps(
            [{"id": item["id"], "sentiment": 0.7, "reason": "fine", "relevant": True} for item in items]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    monkeypatch.setattr(sentiment, "_BATCH_SIZE", 1)

    headlines = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "ts": pd.to_datetime(["2026-07-27"] * 3, utc=True),
            "symbol": ["SPY"] * 3,
            "headline": ["h1", "h2", "h3"],
        }
    )

    scored = sentiment.score_sentiment(headlines)

    by_id = {row["id"]: row for _, row in scored.iterrows()}
    assert by_id[1]["sentiment"] == 0.7
    assert by_id[3]["sentiment"] == 0.7
    assert pd.isna(by_id[2]["sentiment"])  # the bad batch stays unscored, not crashed on


def test_score_sentiment_retries_a_batch_that_fails_to_parse_and_succeeds(monkeypatch):
    """
    A parse failure is usually just sampling variance -- re-asking the same
    batch often works. The batch must not be left unscored if a later
    attempt succeeds.
    """
    attempts = {"count": 0}

    def respond(messages):
        items = json.loads(messages[0]["content"])
        attempts["count"] += 1
        if attempts["count"] < 3:
            return "not valid json at all"
        return json.dumps(
            [{"id": item["id"], "sentiment": 0.8, "reason": "fine", "relevant": True} for item in items]
        )

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

    assert attempts["count"] == 3  # failed twice, succeeded on the third attempt
    assert list(scored["sentiment"]) == [0.8]


def test_score_sentiment_gives_up_on_a_batch_after_max_attempts(monkeypatch):
    """A batch that never parses, across every retry, must still leave its rows
    unscored rather than crash -- same outcome as before retries existed, just
    reached after _MAX_SCORE_ATTEMPTS tries instead of one."""
    attempts = {"count": 0}

    def respond(messages):
        attempts["count"] += 1
        return "not valid json at all"

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

    assert attempts["count"] == sentiment._MAX_SCORE_ATTEMPTS
    assert pd.isna(scored.iloc[0]["sentiment"])


def test_backfill_unscored_news_writes_the_good_batches_when_one_batch_fails_to_parse(monkeypatch):
    """
    Same regression as above, exercised through backfill_unscored_news's
    real DB write path: a batch that fails to parse must not stop the
    other batches' scores from actually being written.
    """
    engine = get_engine()
    rows = pd.DataFrame(
        {
            "id": [900101, 900102, 900103],
            "symbol": ["ZZZTEST", "ZZZTEST", "ZZZTEST"],
            "ts": pd.to_datetime(
                ["2026-07-27T12:00:00Z", "2026-07-27T13:00:00Z", "2026-07-27T14:00:00Z"], utc=True
            ),
            "headline": ["good headline one", "headline that fails to parse", "good headline two"],
            "source": ["test-fixture"] * 3,
        }
    )
    upsert_dataframe(rows, table="news_events", conflict_cols=["id"])

    def respond(messages):
        items = json.loads(messages[0]["content"])
        if items[0]["id"] == 900102:
            return "not valid json at all"
        return json.dumps(
            [{"id": item["id"], "sentiment": 0.6, "reason": "fine", "relevant": True} for item in items]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))
    monkeypatch.setattr(sentiment, "_BATCH_SIZE", 1)

    try:
        n = sentiment.backfill_unscored_news(batch_size=500)
        assert n == 2  # the two good headlines were written despite the bad batch

        with engine.connect() as conn:
            good1 = conn.execute(
                text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": 900101}
            ).fetchone()
            bad = conn.execute(
                text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": 900102}
            ).fetchone()
            good2 = conn.execute(
                text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": 900103}
            ).fetchone()
        assert good1.sentiment == 0.6
        assert good2.sentiment == 0.6
        assert bad.sentiment is None  # left unscored, retried next run -- not crashed on
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM news_events WHERE source = 'test-fixture'"))


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


def test_backfill_unscored_news_scores_the_newest_rows_first_when_backlog_exceeds_batch_size(monkeypatch):
    """
    The dashboard, contradiction monitor, and screener only ever look at
    recent news -- a big backlog of older unscored rows must never sit
    ahead of today's headlines in the queue. With batch_size smaller than
    the unscored backlog, only the newest rows should be sent to Claude at
    all; the oldest row must be left untouched (still sentiment IS NULL).
    """
    engine = get_engine()
    rows = pd.DataFrame(
        {
            "id": [900201, 900202, 900203],
            "symbol": ["ZZZTEST"] * 3,
            "ts": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-27T00:00:00Z", "2026-08-15T00:00:00Z"], utc=True
            ),
            "headline": ["oldest headline", "middle headline", "newest headline"],
            "source": ["test-fixture"] * 3,
        }
    )
    upsert_dataframe(rows, table="news_events", conflict_cols=["id"])

    seen_ids = []

    def respond(messages):
        items = json.loads(messages[0]["content"])
        seen_ids.extend(item["id"] for item in items)
        return json.dumps(
            [{"id": item["id"], "sentiment": 0.1, "reason": "fine", "relevant": True} for item in items]
        )

    monkeypatch.setattr(sentiment, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    try:
        n = sentiment.backfill_unscored_news(batch_size=2)
        assert n == 2
        assert set(seen_ids) == {900203, 900202}  # newest two, oldest never even sent to Claude

        with engine.connect() as conn:
            oldest = conn.execute(
                text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": 900201}
            ).fetchone()
        assert oldest.sentiment is None  # left for a later run
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM news_events WHERE source = 'test-fixture'"))
