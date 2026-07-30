import pandas as pd

from data.ingest import fundamentals


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _polygon_report(end_date: str, eps: float, revenue: float) -> dict:
    return {
        "end_date": end_date,
        "financials": {
            "income_statement": {
                "diluted_earnings_per_share": {"value": eps},
                "revenues": {"value": revenue},
            }
        },
    }


def test_fetch_fundamentals_reshapes_to_long_format(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert params["ticker"] == "SPY"
        return _FakeResponse({"results": [_polygon_report("2026-06-30", eps=2.5, revenue=1_000_000.0)]})

    monkeypatch.setattr(fundamentals, "polygon_get",fake_get)
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert list(df.columns) == ["symbol", "ts", "metric", "value", "source"]
    assert set(df["metric"]) == {"eps_actual", "revenue_actual"}
    assert (df["symbol"] == "SPY").all()
    assert (df["source"] == "polygon").all()

    eps_row = df.loc[df["metric"] == "eps_actual"].iloc[0]
    assert eps_row["value"] == 2.5


def test_fetch_fundamentals_prefers_filing_date_over_end_date(monkeypatch):
    """
    filing_date is when a report actually became public; end_date is just the
    fiscal period it covers. Using end_date would let the model "see" a
    quarter's numbers weeks before they were filed — a look-ahead bug.
    """
    report = {
        "end_date": "2026-06-30",
        "filing_date": "2026-08-04",
        "financials": {"income_statement": {"diluted_earnings_per_share": {"value": 2.5}}},
    }
    monkeypatch.setattr(fundamentals, "polygon_get",lambda *a, **k: _FakeResponse({"results": [report]}))
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert (df["ts"] == pd.Timestamp("2026-08-04", tz="UTC")).all()


def test_fetch_fundamentals_falls_back_to_end_date_when_filing_date_missing(monkeypatch):
    monkeypatch.setattr(fundamentals, "polygon_get",lambda *a, **k: _FakeResponse({"results": [_polygon_report("2026-06-30", eps=2.5, revenue=1_000_000.0)]}))
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert (df["ts"] == pd.Timestamp("2026-06-30", tz="UTC")).all()


def test_fetch_fundamentals_skips_missing_metrics(monkeypatch):
    report = {"end_date": "2026-06-30", "financials": {"income_statement": {}}}
    monkeypatch.setattr(fundamentals, "polygon_get",lambda *a, **k: _FakeResponse({"results": [report]}))
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])
    assert df.empty


def test_fetch_fundamentals_skips_reports_without_a_date(monkeypatch):
    report = {"financials": {"income_statement": {"revenues": {"value": 100.0}}}}
    monkeypatch.setattr(fundamentals, "polygon_get",lambda *a, **k: _FakeResponse({"results": [report]}))
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])
    assert df.empty


def test_fetch_fundamentals_dedupes_same_key_across_reports(monkeypatch):
    """
    Regression test, hit live on the full S&P 500 universe: Polygon can
    return more than one report resolving to the same (symbol, filing_date,
    metric) key (e.g. a restated filing published the same day as the
    original). Two such rows in one batch make upsert_dataframe's
    ON CONFLICT DO UPDATE fail with a CardinalityViolation — the exact bug
    class already fixed for news.py's shared-story case.
    """
    reports = [
        {
            "end_date": "2026-06-30",
            "filing_date": "2026-08-04",
            "financials": {"income_statement": {"diluted_earnings_per_share": {"value": 2.5}}},
        },
        {
            "end_date": "2026-06-30",
            "filing_date": "2026-08-04",
            "financials": {"income_statement": {"diluted_earnings_per_share": {"value": 2.6}}},
        },
    ]
    monkeypatch.setattr(fundamentals, "polygon_get", lambda *a, **k: _FakeResponse({"results": reports}))
    monkeypatch.setattr(fundamentals.settings, "polygon_api_key", "test-key")

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert len(df) == 1  # deduped, not one row per report
    assert df.iloc[0]["value"] == 2.6  # keeps the later (revised) value


def test_ingest_fundamentals_upserts_with_correct_conflict_cols(monkeypatch):
    monkeypatch.setattr(
        fundamentals,
        "fetch_fundamentals",
        lambda symbols, sleep_seconds=0: pd.DataFrame(
            {
                "symbol": ["SPY"],
                "ts": pd.to_datetime(["2026-06-30"], utc=True),
                "metric": ["eps_actual"],
                "value": [2.5],
                "source": ["polygon"],
            }
        ),
    )
    captured = {}

    def fake_upsert(df, table, conflict_cols):
        captured["table"] = table
        captured["conflict_cols"] = conflict_cols
        return len(df)

    monkeypatch.setattr(fundamentals, "upsert_dataframe", fake_upsert)

    n = fundamentals.ingest_fundamentals(["SPY"])

    assert n == 1
    assert captured["table"] == "fundamentals"
    assert captured["conflict_cols"] == ["symbol", "ts", "metric"]
