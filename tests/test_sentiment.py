import json

import pandas as pd

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
    assert set(scored.columns) >= {"id", "ts", "symbol", "headline", "sentiment", "sentiment_reason"}


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
