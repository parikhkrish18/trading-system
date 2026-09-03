import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest import news
from data.ingest.db import get_engine, upsert_dataframe


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_stable_id_is_deterministic_and_fits_bigint():
    a = news._stable_id("polygon-article-123", "AAPL")
    b = news._stable_id("polygon-article-123", "AAPL")
    c = news._stable_id("polygon-article-456", "AAPL")
    assert a == b
    assert a != c
    assert 0 <= a < 2**63  # must fit Postgres BIGINT


def test_stable_id_differs_by_symbol_for_the_same_article():
    """
    A story tagged to multiple tickers must not collide on (id, ts) within
    one insert batch — Postgres's ON CONFLICT DO UPDATE can't affect the
    same target row twice in a single statement.
    """
    aapl_id = news._stable_id("shared-article-1", "AAPL")
    msft_id = news._stable_id("shared-article-1", "MSFT")
    assert aapl_id != msft_id


def test_fetch_news_shapes_polygon_response(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert params["ticker"] in ("SPY", "QQQ")
        return _FakeResponse(
            {
                "results": [
                    {
                        "id": f"article-{params['ticker']}-1",
                        "published_utc": "2026-07-27T12:00:00Z",
                        "title": f"{params['ticker']} headline",
                    }
                ]
            }
        )

    monkeypatch.setattr(news, "polygon_get",fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY", "QQQ"], since_hours=24)

    assert list(df.columns) == ["id", "symbol", "ts", "headline", "source"]
    assert len(df) == 2
    assert set(df["symbol"]) == {"SPY", "QQQ"}
    assert (df["source"] == "polygon").all()
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])


def test_fetch_news_decodes_html_entities_in_the_title(monkeypatch):
    """Polygon (like Benzinga via the news stream) sometimes hands back
    title text with literal HTML entities in it -- an apostrophe as "&#39;"
    rather than an actual apostrophe -- which must be decoded at ingest."""

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            {
                "results": [
                    {
                        "id": "article-entities-1",
                        "published_utc": "2026-08-30T12:00:00Z",
                        "title": "Designates ChatGPT As &#39;Very Large Online Search Engine&#39;",
                    }
                ]
            }
        )

    monkeypatch.setattr(news, "polygon_get", fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY"], since_hours=24)

    assert df.iloc[0]["headline"] == "Designates ChatGPT As 'Very Large Online Search Engine'"


def test_fetch_news_same_article_shared_across_symbols_produces_two_rows(monkeypatch):
    """Regression test: a story tagged to two tickers must not collide on (id, ts)."""

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            {
                "results": [
                    {
                        "id": "shared-article-1",  # same underlying Polygon article id for both tickers
                        "published_utc": "2026-07-27T12:00:00Z",  # same timestamp too
                        "title": "Tech stocks rally",
                    }
                ]
            }
        )

    monkeypatch.setattr(news, "polygon_get", fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["AAPL", "MSFT"], since_hours=24)

    assert len(df) == 2
    assert df["id"].nunique() == 2  # distinct ids despite identical article id + ts
    assert set(df["symbol"]) == {"AAPL", "MSFT"}


def test_fetch_news_skips_a_symbol_that_errors_instead_of_losing_the_whole_batch(monkeypatch):
    """
    Regression test, hit live: a single transient network error (read
    timeout, DNS blip) on one symbol used to propagate out of fetch_news
    entirely, discarding every other symbol already successfully fetched --
    losing ~2 hours of work for a 500-symbol universe pull. One bad symbol
    must not take down the whole batch.
    """
    import requests

    def fake_get(url, params=None, timeout=None):
        if params["ticker"] == "BAD":
            raise requests.exceptions.ConnectionError("read timed out")
        return _FakeResponse(
            {"results": [{"id": f"article-{params['ticker']}", "published_utc": "2026-07-27T12:00:00Z", "title": "ok"}]}
        )

    monkeypatch.setattr(news, "polygon_get", fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY", "BAD", "QQQ"], since_hours=24, sleep_seconds=0)

    assert set(df["symbol"]) == {"SPY", "QQQ"}  # BAD skipped, the rest survived


def test_fetch_news_follows_next_url_pagination(monkeypatch):
    """
    Regression test: requesting limit=1000 with no next_url handling meant a
    heavily-covered symbol with a wide --since-hours window silently
    truncated past the first 1000 articles. next_url must be followed.
    """
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url == news.POLYGON_NEWS_URL:
            assert params["ticker"] == "SPY"
            return _FakeResponse(
                {
                    "results": [{"id": "article-page1", "published_utc": "2026-07-27T12:00:00Z", "title": "page 1"}],
                    "next_url": "https://api.polygon.io/v2/reference/news?cursor=abc123",
                }
            )
        # Page 2: fetched via the next_url Polygon returned, with just the
        # API key re-appended (next_url carries its own cursor/limit already).
        assert url == "https://api.polygon.io/v2/reference/news?cursor=abc123"
        assert params == {"apiKey": "test-key"}
        return _FakeResponse(
            {"results": [{"id": "article-page2", "published_utc": "2026-07-27T13:00:00Z", "title": "page 2"}]}
        )

    monkeypatch.setattr(news, "polygon_get", fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY"], since_hours=24, sleep_seconds=0)

    assert calls == [news.POLYGON_NEWS_URL, "https://api.polygon.io/v2/reference/news?cursor=abc123"]
    assert len(df) == 2
    assert set(df["headline"]) == {"page 1", "page 2"}


def test_fetch_news_page_cap_stops_an_endless_next_url_loop(monkeypatch):
    """A malformed/looping next_url must not fetch forever -- capped at
    MAX_PAGES_PER_SYMBOL pages."""
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(
            {
                "results": [{"id": f"article-{calls['n']}", "published_utc": "2026-07-27T12:00:00Z", "title": "x"}],
                "next_url": "https://api.polygon.io/v2/reference/news?cursor=loop",  # same every time
            }
        )

    monkeypatch.setattr(news, "polygon_get", fake_get)
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY"], since_hours=24, sleep_seconds=0)

    assert calls["n"] == news.MAX_PAGES_PER_SYMBOL
    assert len(df) == news.MAX_PAGES_PER_SYMBOL


def test_fetch_news_empty_results_returns_empty_dataframe(monkeypatch):
    monkeypatch.setattr(news, "polygon_get",lambda *a, **k: _FakeResponse({"results": []}))
    monkeypatch.setattr(news.settings, "polygon_api_key", "test-key")

    df = news.fetch_news(["SPY"], since_hours=24)
    assert df.empty
    assert list(df.columns) == ["id", "symbol", "ts", "headline", "source"]


def test_ingest_news_adds_null_sentiment_and_surprise(monkeypatch):
    monkeypatch.setattr(
        news, "fetch_news", lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [1],
                "symbol": ["SPY"],
                "ts": pd.to_datetime(["2026-07-27T12:00:00Z"], utc=True),
                "headline": ["headline"],
                "source": ["polygon"],
            }
        )
    )
    captured = {}

    def fake_upsert(df, table, conflict_cols, preserve_cols=None):
        captured["df"] = df
        captured["table"] = table
        captured["conflict_cols"] = conflict_cols
        captured["preserve_cols"] = preserve_cols
        return len(df)

    monkeypatch.setattr(news, "upsert_dataframe", fake_upsert)

    n = news.ingest_news(["SPY"], since_hours=24)

    assert n == 1
    assert captured["table"] == "news_events"
    assert captured["conflict_cols"] == ["id"]
    assert captured["preserve_cols"] == ["sentiment", "surprise"]
    assert captured["df"]["sentiment"].isna().all()
    assert captured["df"]["surprise"].isna().all()


# --------------------------------------------------------------------------
# Real-DB integration tests: preserve_cols and the id-only conflict target,
# proven against the actual news_events schema/upsert, not a mock.
# --------------------------------------------------------------------------


@pytest.fixture
def _news_events_cleanup():
    engine = get_engine()
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM news_events WHERE source = 'test-fixture'"))


def test_ingest_news_preserves_an_already_scored_sentiment_on_re_ingest(monkeypatch, _news_events_cleanup):
    """
    Regression test, finding #1: re-ingesting news (e.g. a routine re-poll
    of an overlapping time window) used to wipe an already-computed
    sentiment score back to NULL, because ingest_news always writes NaN and
    the upsert had no way to exclude a column from its ON CONFLICT UPDATE.
    """
    engine = _news_events_cleanup
    article_id = news._stable_id("zzztest-article-1", "ZZZTEST")

    # Simulate: article already ingested and scored by a prior sentiment pass.
    already_scored = pd.DataFrame(
        {
            "id": [article_id],
            "symbol": ["ZZZTEST"],
            "ts": pd.to_datetime(["2026-07-27T12:00:00Z"], utc=True),
            "headline": ["original headline"],
            "source": ["test-fixture"],
            "sentiment": [0.6],
            "surprise": [float("nan")],
        }
    )
    upsert_dataframe(already_scored, table="news_events", conflict_cols=["id"])

    # Re-ingest: same article, sentiment/surprise NaN again -- ingest_news
    # never carries a real score, that's a separate pass.
    monkeypatch.setattr(
        news,
        "fetch_news",
        lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [article_id],
                "symbol": ["ZZZTEST"],
                "ts": pd.to_datetime(["2026-07-27T12:00:00Z"], utc=True),
                "headline": ["original headline"],
                "source": ["test-fixture"],
            }
        ),
    )

    news.ingest_news(["ZZZTEST"], since_hours=24)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": article_id}).fetchone()
    assert row.sentiment == 0.6  # preserved, not wiped back to NULL


def test_ingest_news_same_article_id_with_a_corrected_ts_updates_not_duplicates(monkeypatch, _news_events_cleanup):
    """
    Regression test, finding #8: the upsert's conflict target used to be
    (id, ts), so a vendor redelivering the same article with a corrected
    published_utc couldn't match the existing row and inserted a SECOND one
    instead of updating the first -- double-counting the story in sentiment
    aggregation windows. id alone is now the conflict target (see
    data/schema/014_news_id_unique.sql), so a correction updates in place.
    """
    engine = _news_events_cleanup
    article_id = news._stable_id("zzztest-article-2", "ZZZTEST")

    def _fake_fetch(ts):
        return lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [article_id],
                "symbol": ["ZZZTEST"],
                "ts": pd.to_datetime([ts], utc=True),
                "headline": ["headline"],
                "source": ["test-fixture"],
            }
        )

    monkeypatch.setattr(news, "fetch_news", _fake_fetch("2026-07-27T12:00:00Z"))
    news.ingest_news(["ZZZTEST"], since_hours=24)

    monkeypatch.setattr(news, "fetch_news", _fake_fetch("2026-07-27T15:30:00Z"))  # corrected published_utc
    news.ingest_news(["ZZZTEST"], since_hours=24)

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ts FROM news_events WHERE id = :id"), {"id": article_id}).fetchall()
    assert len(rows) == 1  # updated in place, not duplicated
    assert rows[0].ts == pd.Timestamp("2026-07-27T15:30:00Z", tz="UTC")
