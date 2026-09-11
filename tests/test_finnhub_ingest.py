import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest import finnhub
from data.ingest.db import get_engine, upsert_dataframe


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_stable_id_is_deterministic_and_fits_bigint():
    a = finnhub._stable_id("25286", "AAPL")
    b = finnhub._stable_id("25286", "AAPL")
    c = finnhub._stable_id("25287", "AAPL")
    assert a == b
    assert a != c
    assert 0 <= a < 2**63  # must fit Postgres BIGINT


def test_stable_id_differs_by_symbol_for_the_same_key():
    aapl_id = finnhub._stable_id("shared-key-1", "AAPL")
    msft_id = finnhub._stable_id("shared-key-1", "MSFT")
    assert aapl_id != msft_id


def test_fetch_company_news_shapes_finnhub_response(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert url == finnhub.FINNHUB_COMPANY_NEWS_URL
        assert params["symbol"] in ("SPY", "QQQ")
        return _FakeResponse(
            [
                {
                    "id": 25286,
                    "datetime": 1758000000,
                    "headline": f"{params['symbol']} headline",
                    "summary": "Some summary.",
                    "category": "company news",
                    "source": "Reuters",
                }
            ]
        )

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_company_news(["SPY", "QQQ"], since_hours=24 * 365)

    assert list(df.columns) == ["id", "symbol", "ts", "headline", "summary", "source"]
    assert len(df) == 2
    assert set(df["symbol"]) == {"SPY", "QQQ"}
    assert (df["source"] == "finnhub").all()
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert (df["summary"] == "Some summary.").all()


def test_fetch_company_news_filters_out_anything_older_than_since_hours(monkeypatch):
    """
    Finnhub's from/to are whole dates -- a --since-hours 1 call still gets a
    full day's worth back from the API and must filter down afterward.
    """
    import datetime as dt

    now = dt.datetime.now(tz=dt.UTC)
    old = now - dt.timedelta(hours=5)
    recent = now - dt.timedelta(minutes=10)

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            [
                {"id": 1, "datetime": int(old.timestamp()), "headline": "too old", "summary": ""},
                {"id": 2, "datetime": int(recent.timestamp()), "headline": "recent enough", "summary": ""},
            ]
        )

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_company_news(["AAPL"], since_hours=1)

    assert list(df["headline"]) == ["recent enough"]


def test_fetch_company_news_decodes_html_entities(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            [
                {
                    "id": 1,
                    "datetime": 1758000000,
                    "headline": "Designates ChatGPT As &#39;Very Large Online Search Engine&#39;",
                    "summary": "The CEO said it &#39;exceeded expectations&#39;.",
                }
            ]
        )

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_company_news(["AAPL"], since_hours=24 * 365)

    assert df.iloc[0]["headline"] == "Designates ChatGPT As 'Very Large Online Search Engine'"
    assert df.iloc[0]["summary"] == "The CEO said it 'exceeded expectations'."


def test_fetch_company_news_skips_a_symbol_that_errors_instead_of_losing_the_whole_batch(monkeypatch):
    import requests

    def fake_get(url, params=None, timeout=None):
        if params["symbol"] == "BAD":
            raise requests.exceptions.ConnectionError("read timed out")
        return _FakeResponse([{"id": 1, "datetime": 1758000000, "headline": "ok", "summary": ""}])

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_company_news(["SPY", "BAD", "QQQ"], since_hours=24 * 365, sleep_seconds=0)

    assert set(df["symbol"]) == {"SPY", "QQQ"}


def test_fetch_company_news_without_a_key_returns_empty_and_does_not_call_out(monkeypatch):
    called = []
    monkeypatch.setattr(finnhub, "finnhub_get", lambda *a, **k: called.append(1))
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "")

    df = finnhub.fetch_company_news(["AAPL"], since_hours=24)

    assert df.empty
    assert called == []


def test_fetch_sec_filings_shapes_finnhub_response(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert url == finnhub.FINNHUB_FILINGS_URL
        return _FakeResponse(
            [
                {
                    "accessNumber": "0001193125-20-050884",
                    "symbol": params["symbol"],
                    "cik": "320193",
                    "form": "8-K",
                    "filedDate": "2026-09-08 06:14:21",
                    "reportUrl": "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/report.htm",
                    "filingUrl": "https://www.sec.gov/Archives/edgar/data/320193/index.html",
                }
            ]
        )

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_sec_filings(["AAPL"], since_hours=24 * 365)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["headline"] == "AAPL filed Form 8-K"
    assert row["summary"] == "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/report.htm"
    assert row["source"] == "finnhub_filing"


def test_fetch_sec_filings_falls_back_to_filing_url_when_no_report_url(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            [
                {
                    "accessNumber": "0001193125-20-050885",
                    "form": "NT 10-K",
                    "filedDate": "2026-09-08 06:14:21",
                    "filingUrl": "https://www.sec.gov/Archives/edgar/data/320193/other-index.html",
                }
            ]
        )

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_sec_filings(["AAPL"], since_hours=24 * 365)

    assert df.iloc[0]["summary"] == "https://www.sec.gov/Archives/edgar/data/320193/other-index.html"


def test_fetch_sec_filings_skips_a_row_missing_the_access_number(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse([{"form": "8-K", "filedDate": "2026-09-08 06:14:21"}])

    monkeypatch.setattr(finnhub, "finnhub_get", fake_get)
    monkeypatch.setattr(finnhub.settings, "finnhub_api_key", "test-key")

    df = finnhub.fetch_sec_filings(["AAPL"], since_hours=24 * 365)

    assert df.empty


def test_ingest_finnhub_combines_news_and_filings_with_null_sentiment(monkeypatch):
    monkeypatch.setattr(
        finnhub, "fetch_company_news",
        lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [1],
                "symbol": ["SPY"],
                "ts": pd.to_datetime(["2026-09-08T12:00:00Z"], utc=True),
                "headline": ["a headline"],
                "summary": [""],
                "source": ["finnhub"],
            }
        ),
    )
    monkeypatch.setattr(
        finnhub, "fetch_sec_filings",
        lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [2],
                "symbol": ["SPY"],
                "ts": pd.to_datetime(["2026-09-08T13:00:00Z"], utc=True),
                "headline": ["SPY filed Form 8-K"],
                "summary": ["https://sec.gov/..."],
                "source": ["finnhub_filing"],
            }
        ),
    )
    captured = {}

    def fake_upsert(df, table, conflict_cols, preserve_cols=None):
        captured["df"] = df
        captured["table"] = table
        captured["conflict_cols"] = conflict_cols
        captured["preserve_cols"] = preserve_cols
        return len(df)

    monkeypatch.setattr(finnhub, "upsert_dataframe", fake_upsert)

    n = finnhub.ingest_finnhub(["SPY"], since_hours=24)

    assert n == 2
    assert captured["table"] == "news_events"
    assert captured["conflict_cols"] == ["id"]
    assert captured["preserve_cols"] == ["sentiment", "surprise"]
    assert captured["df"]["sentiment"].isna().all()
    assert captured["df"]["surprise"].isna().all()
    assert set(captured["df"]["source"]) == {"finnhub", "finnhub_filing"}


def test_ingest_finnhub_with_nothing_from_either_endpoint_does_not_call_upsert(monkeypatch):
    monkeypatch.setattr(finnhub, "fetch_company_news", lambda *a, **k: pd.DataFrame(columns=finnhub._NEWS_COLUMNS))
    monkeypatch.setattr(finnhub, "fetch_sec_filings", lambda *a, **k: pd.DataFrame(columns=finnhub._NEWS_COLUMNS))
    called = []
    monkeypatch.setattr(finnhub, "upsert_dataframe", lambda *a, **k: called.append(1))

    n = finnhub.ingest_finnhub(["SPY"], since_hours=24)

    assert n == 0
    assert called == []


# --------------------------------------------------------------------------
# Real-DB integration tests: preserve_cols and the id-only conflict target,
# proven against the actual news_events schema/upsert, not a mock.
# --------------------------------------------------------------------------


@pytest.fixture
def _news_events_cleanup():
    engine = get_engine()
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM news_events WHERE source IN ('test-fixture')"))


def test_ingest_finnhub_preserves_an_already_scored_sentiment_on_re_ingest(monkeypatch, _news_events_cleanup):
    engine = _news_events_cleanup
    article_id = finnhub._stable_id("zzztest-article-1", "ZZZTEST")

    already_scored = pd.DataFrame(
        {
            "id": [article_id],
            "symbol": ["ZZZTEST"],
            "ts": pd.to_datetime(["2026-09-08T12:00:00Z"], utc=True),
            "headline": ["original headline"],
            "source": ["test-fixture"],
            "sentiment": [0.6],
            "surprise": [float("nan")],
        }
    )
    upsert_dataframe(already_scored, table="news_events", conflict_cols=["id"])

    monkeypatch.setattr(
        finnhub, "fetch_company_news",
        lambda symbols, since_hours, sleep_seconds=0: pd.DataFrame(
            {
                "id": [article_id],
                "symbol": ["ZZZTEST"],
                "ts": pd.to_datetime(["2026-09-08T12:00:00Z"], utc=True),
                "headline": ["original headline"],
                "summary": [""],
                "source": ["test-fixture"],
            }
        ),
    )
    monkeypatch.setattr(finnhub, "fetch_sec_filings", lambda *a, **k: pd.DataFrame(columns=finnhub._NEWS_COLUMNS))

    finnhub.ingest_finnhub(["ZZZTEST"], since_hours=24)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT sentiment FROM news_events WHERE id = :id"), {"id": article_id}).fetchone()
    assert row.sentiment == 0.6  # preserved, not wiped back to NULL
