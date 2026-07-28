import pandas as pd

from data.ingest import news


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

    def fake_upsert(df, table, conflict_cols):
        captured["df"] = df
        captured["table"] = table
        captured["conflict_cols"] = conflict_cols
        return len(df)

    monkeypatch.setattr(news, "upsert_dataframe", fake_upsert)

    n = news.ingest_news(["SPY"], since_hours=24)

    assert n == 1
    assert captured["table"] == "news_events"
    assert captured["conflict_cols"] == ["id", "ts"]
    assert captured["df"]["sentiment"].isna().all()
    assert captured["df"]["surprise"].isna().all()
