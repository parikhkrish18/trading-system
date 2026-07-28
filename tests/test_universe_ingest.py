import pandas as pd
import pytest

from data.ingest import universe

_WIKI_HTML = """
<html><body>
<table class="wikitable" id="constituents">
<tbody>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td><a href="#">MMM</a></td><td>3M</td><td>Industrials</td></tr>
<tr><td><a href="#">TSLA</a></td><td>Tesla, Inc.</td><td>Consumer Discretionary</td></tr>
</tbody>
</table>
</body></html>
"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fetch_sp500_constituents_parses_rows(monkeypatch):
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: _FakeResponse(_WIKI_HTML))

    df = universe.fetch_sp500_constituents()

    assert list(df.columns) == ["symbol", "name", "gics_sector"]
    assert len(df) == 2
    tsla = df.loc[df["symbol"] == "TSLA"].iloc[0]
    assert tsla["name"] == "Tesla, Inc."
    assert tsla["gics_sector"] == "Consumer Discretionary"


def test_fetch_sp500_constituents_skips_header_row(monkeypatch):
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: _FakeResponse(_WIKI_HTML))
    df = universe.fetch_sp500_constituents()
    assert "Symbol" not in set(df["symbol"])


def test_fetch_sp500_constituents_missing_table_returns_empty(monkeypatch):
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: _FakeResponse("<html><body>no table here</body></html>"))
    df = universe.fetch_sp500_constituents()
    assert df.empty
    assert list(df.columns) == ["symbol", "name", "gics_sector"]


def test_refresh_universe_deactivates_removed_symbols(monkeypatch):
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: pd.DataFrame(
        {"symbol": ["MMM", "TSLA"], "name": ["3M", "Tesla, Inc."], "gics_sector": ["Industrials", "Consumer Discretionary"]}
    ))

    upsert_calls = {}

    def fake_upsert(df, table, conflict_cols):
        upsert_calls["df"] = df
        upsert_calls["table"] = table
        upsert_calls["conflict_cols"] = conflict_cols
        return len(df)

    executed_sql = {}

    class _FakeConn:
        def execute(self, stmt):
            executed_sql["stmt"] = str(stmt)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeEngine:
        def begin(self):
            return _FakeConn()

    monkeypatch.setattr(universe, "upsert_dataframe", fake_upsert)
    monkeypatch.setattr(universe, "get_engine", lambda: _FakeEngine())

    n = universe.refresh_universe()

    assert n == 2
    assert upsert_calls["table"] == "universe"
    assert upsert_calls["conflict_cols"] == ["symbol"]
    assert (upsert_calls["df"]["is_active"]).all()
    assert "MMM" in executed_sql["stmt"] and "TSLA" in executed_sql["stmt"]
    assert "is_active = FALSE" in executed_sql["stmt"]


def test_refresh_universe_empty_scrape_is_noop(monkeypatch):
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: pd.DataFrame(columns=["symbol", "name", "gics_sector"]))
    n = universe.refresh_universe()
    assert n == 0


def test_resolve_symbols_uses_universe_flag(monkeypatch):
    monkeypatch.setattr(universe, "load_active_universe", lambda: ["AAPL", "TSLA"])
    assert universe.resolve_symbols(None, use_universe=True) == ["AAPL", "TSLA"]


def test_resolve_symbols_errors_on_empty_universe(monkeypatch):
    monkeypatch.setattr(universe, "load_active_universe", lambda: [])
    with pytest.raises(SystemExit, match="universe table is empty"):
        universe.resolve_symbols(None, use_universe=True)


def test_resolve_symbols_parses_comma_list():
    assert universe.resolve_symbols("spy, qqq", use_universe=False) == ["SPY", "QQQ"]


def test_resolve_symbols_errors_when_neither_given():
    with pytest.raises(SystemExit, match="Pass --symbols"):
        universe.resolve_symbols(None, use_universe=False)
