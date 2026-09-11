import pandas as pd

from data.ingest import fundamentals


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_UNSET = object()


def _finnhub_report(end_date: str, *, eps=None, revenue=None, filed_date=_UNSET) -> dict:
    ic = []
    if eps is not None:
        ic.append({"concept": "EarningsPerShareDiluted", "label": "Diluted EPS", "unit": "usd/shares", "value": eps})
    if revenue is not None:
        ic.append({"concept": "Revenues", "label": "Revenues", "unit": "usd", "value": revenue})
    return {
        "endDate": end_date,
        # Defaults to end_date so tests that aren't specifically about the
        # filedDate/endDate distinction get a normally-dated, ingestible
        # report -- pass filed_date=None explicitly to simulate a report
        # missing it entirely.
        "filedDate": end_date if filed_date is _UNSET else filed_date,
        "report": {"ic": ic},
    }


def test_fetch_fundamentals_reshapes_to_long_format(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert params["symbol"] == "SPY"
        return _FakeResponse({"data": [_finnhub_report("2026-06-30", eps=2.5, revenue=1_000_000.0)]})

    monkeypatch.setattr(fundamentals, "finnhub_get", fake_get)
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert list(df.columns) == ["symbol", "ts", "metric", "value", "source"]
    assert set(df["metric"]) == {"eps_actual", "revenue_actual"}
    assert (df["symbol"] == "SPY").all()
    assert (df["source"] == "finnhub").all()

    eps_row = df.loc[df["metric"] == "eps_actual"].iloc[0]
    assert eps_row["value"] == 2.5


def test_fetch_fundamentals_prefers_filed_date_over_end_date(monkeypatch):
    """
    filedDate is when a report actually became public; endDate is just the
    fiscal period it covers. Using endDate would let the model "see" a
    quarter's numbers weeks before they were filed — a look-ahead bug.
    """
    report = _finnhub_report("2026-06-30", eps=2.5, filed_date="2026-08-04")
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": [report]}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert (df["ts"] == pd.Timestamp("2026-08-04", tz="UTC")).all()


def test_fetch_fundamentals_excludes_reports_missing_filed_date(monkeypatch):
    """
    Regression test: a report with no filedDate must be dropped, not
    silently mis-dated by falling back to endDate — exactly the look-ahead
    bias the filedDate-over-endDate preference above exists to prevent.
    """
    report = _finnhub_report("2026-06-30", eps=2.5, revenue=1_000_000.0, filed_date=None)
    assert report["filedDate"] is None
    assert report["endDate"] == "2026-06-30"

    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": [report]}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert df.empty  # excluded entirely, not silently dated by endDate


def test_fetch_fundamentals_skips_missing_metrics(monkeypatch):
    report = {"endDate": "2026-06-30", "filedDate": "2026-06-30", "report": {"ic": []}}
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": [report]}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])
    assert df.empty


def test_fetch_fundamentals_skips_reports_without_a_date(monkeypatch):
    report = {"report": {"ic": [{"concept": "Revenues", "value": 100.0}]}}
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": [report]}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])
    assert df.empty


def test_fetch_fundamentals_matches_metrics_regardless_of_statement_section(monkeypatch):
    """Line items are matched by concept across bs/ic/cf, not a fixed section."""
    report = {
        "endDate": "2026-06-30",
        "filedDate": "2026-06-30",
        "report": {
            "bs": [
                {"concept": "Assets", "label": "Total assets", "value": 500.0},
                {"concept": "Liabilities", "label": "Total liabilities", "value": 200.0},
            ],
            "ic": [{"concept": "NetIncomeLoss", "label": "Net income", "value": 42.0}],
        },
    }
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": [report]}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])

    assert set(df["metric"]) == {"total_assets", "total_liabilities", "net_income"}


def test_fetch_fundamentals_skips_a_symbol_that_errors_instead_of_losing_the_whole_batch(monkeypatch):
    """
    Regression test: a single transient network error on one symbol must not
    propagate out of fetch_fundamentals entirely, discarding every other
    symbol already successfully fetched.
    """
    import requests

    def fake_get(url, params=None, timeout=None):
        if params["symbol"] == "BAD":
            raise requests.exceptions.ConnectionError("read timed out")
        return _FakeResponse({"data": [_finnhub_report("2026-06-30", eps=2.5, revenue=1_000_000.0)]})

    monkeypatch.setattr(fundamentals, "finnhub_get", fake_get)
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY", "BAD", "MSFT"], sleep_seconds=0)

    assert set(df["symbol"]) == {"SPY", "MSFT"}  # BAD skipped, the rest survived


def test_fetch_fundamentals_dedupes_same_key_across_reports(monkeypatch):
    """
    Regression test: Finnhub can return more than one report resolving to the
    same (symbol, filedDate, metric) key (e.g. a restated filing published
    the same day as the original). Two such rows in one batch make
    upsert_dataframe's ON CONFLICT DO UPDATE fail with a CardinalityViolation.
    """
    reports = [
        _finnhub_report("2026-06-30", eps=2.5, filed_date="2026-08-04"),
        _finnhub_report("2026-06-30", eps=2.6, filed_date="2026-08-04"),
    ]
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: _FakeResponse({"data": reports}))
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)

    df = fundamentals.fetch_fundamentals(["SPY"])

    eps_rows = df.loc[df["metric"] == "eps_actual"]
    assert len(eps_rows) == 1  # deduped, not one row per report
    assert eps_rows.iloc[0]["value"] == 2.6  # keeps the later (revised) value


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
                "source": ["finnhub"],
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


# --------------------------------------------------------------------------
# An unset key must fail once, not 401 times slowly
# --------------------------------------------------------------------------


def test_no_finnhub_key_skips_immediately_instead_of_making_requests(monkeypatch):
    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: False)

    def _must_not_be_called(*a, **k):
        raise AssertionError("no request should be made without a key")

    monkeypatch.setattr(fundamentals, "finnhub_get", _must_not_be_called)
    monkeypatch.setattr(fundamentals.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not sleep")))

    df = fundamentals.fetch_fundamentals(["AAPL", "MSFT", "TSLA"])

    assert df.empty
    assert list(df.columns) == ["symbol", "ts", "metric", "value", "source"]


def test_a_configured_key_still_fetches_normally(monkeypatch):
    """The guard must not disable ingestion for anyone who has a key."""
    calls = []

    class _Resp:
        def json(self):
            return {"data": []}

    monkeypatch.setattr(fundamentals, "finnhub_configured", lambda: True)
    monkeypatch.setattr(fundamentals, "finnhub_get", lambda *a, **k: calls.append(1) or _Resp())

    fundamentals.fetch_fundamentals(["AAPL", "MSFT"], sleep_seconds=0)

    assert len(calls) == 2
