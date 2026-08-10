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


class _FakeConn:
    def __init__(self, calls: dict):
        self._calls = calls

    def execute(self, stmt, params=None):
        self._calls["stmt"] = str(stmt)
        self._calls["params"] = params

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeEngine:
    def __init__(self, calls: dict):
        self._calls = calls

    def begin(self):
        return _FakeConn(self._calls)


def _scrape_of(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "name": symbols, "gics_sector": ["Industrials"] * len(symbols)})


def test_refresh_universe_deactivates_removed_symbols(monkeypatch):
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: _scrape_of(["MMM", "TSLA"]))
    monkeypatch.setattr(universe, "MIN_EXPECTED_CONSTITUENTS", 2)

    upsert_calls = {}

    def fake_upsert(df, table, conflict_cols):
        upsert_calls[table] = {"df": df, "conflict_cols": conflict_cols}
        return len(df)

    executed = {}
    monkeypatch.setattr(universe, "upsert_dataframe", fake_upsert)
    monkeypatch.setattr(universe, "get_engine", lambda: _FakeEngine(executed))

    n = universe.refresh_universe()

    assert n == 2
    assert upsert_calls["universe"]["conflict_cols"] == ["symbol"]
    assert (upsert_calls["universe"]["df"]["is_active"]).all()
    # Every refresh also appends a dated membership snapshot, so future
    # backtests can use point-in-time membership instead of today's winners.
    snapshot = upsert_calls["universe_snapshot"]
    assert snapshot["conflict_cols"] == ["snapshot_date", "symbol"]
    assert set(snapshot["df"]["symbol"]) == {"MMM", "TSLA"}
    assert "snapshot_date" in snapshot["df"].columns
    # Deactivation is parameterized now — symbols travel as bind params, not
    # spliced into the SQL string.
    assert "is_active = FALSE" in executed["stmt"]
    assert executed["params"] == {"symbols": ["MMM", "TSLA"]}
    assert "MMM" not in executed["stmt"]


def test_refresh_universe_aborts_on_empty_scrape(monkeypatch):
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: pd.DataFrame(columns=["symbol", "name", "gics_sector"]))
    with pytest.raises(ValueError, match="refusing to refresh"):
        universe.refresh_universe()


def test_refresh_universe_aborts_on_suspiciously_small_scrape(monkeypatch):
    """A 50-row scrape means the page changed, not that 450 companies vanished —
    accepting it would deactivate most of the universe and the next cycle
    would propose closing every position."""
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: _scrape_of([f"S{i}" for i in range(50)]))
    upsert_called = []
    monkeypatch.setattr(universe, "upsert_dataframe", lambda *a, **k: upsert_called.append(a))
    with pytest.raises(ValueError, match="refusing to refresh"):
        universe.refresh_universe()
    assert not upsert_called  # nothing touched the table


def test_refresh_universe_aborts_on_non_ticker_garbage(monkeypatch):
    symbols = [f"S{i}" for i in range(460)]
    symbols[3] = "x'); DROP TABLE universe;--"
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: _scrape_of(symbols))
    upsert_called = []
    monkeypatch.setattr(universe, "upsert_dataframe", lambda *a, **k: upsert_called.append(a))
    with pytest.raises(ValueError, match="don't look like tickers"):
        universe.refresh_universe()
    assert not upsert_called


def test_refresh_universe_accepts_a_full_realistic_scrape(monkeypatch):
    symbols = [f"S{i}" for i in range(500)] + ["BRK.B", "BF-B", "MMM"]
    monkeypatch.setattr(universe, "fetch_sp500_constituents", lambda: _scrape_of(symbols))
    monkeypatch.setattr(universe, "upsert_dataframe", lambda df, **k: len(df))
    executed = {}
    monkeypatch.setattr(universe, "get_engine", lambda: _FakeEngine(executed))
    assert universe.refresh_universe() == len(symbols)
    assert executed["params"] == {"symbols": symbols}


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
