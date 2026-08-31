import pandas as pd

from data.ingest import prices
from data.ingest.prices import _fetch_yfinance, ingest_prices


def _fake_multi_symbol_frame():
    """
    Mimics yfinance's group_by="ticker" multi-symbol return shape: a
    DatetimeIndex with a MultiIndex of columns (symbol, field).
    """
    idx = pd.bdate_range("2026-01-02", periods=3)
    frames = {}
    frames["SPY"] = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Close": [1, 2, 3], "Volume": [10, 20, 30]},
        index=idx,
    )
    # BADCO has a null Close on one bar (real vendor glitch hit live) but
    # otherwise-valid OHLV — must not reach the DB, which has a NOT NULL
    # constraint on close, or it poisons the whole batch insert.
    frames["BADCO"] = pd.DataFrame(
        {"Open": [5, 6, 7], "High": [5, 6, 7], "Low": [5, 6, 7], "Close": [5, None, 7], "Volume": [50, 60, 70]},
        index=idx,
    )
    return pd.concat(frames, axis=1)


def test_fetch_yfinance_drops_rows_with_any_null_ohlcv_field(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _fake_multi_symbol_frame())

    df = _fetch_yfinance(["SPY", "BADCO"], pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date())

    assert df["close"].isna().sum() == 0
    assert len(df[df["symbol"] == "SPY"]) == 3  # all 3 SPY bars kept
    assert len(df[df["symbol"] == "BADCO"]) == 2  # the null-close bar dropped, other 2 kept


def test_fetch_yfinance_translates_dotted_tickers_to_dashes(monkeypatch):
    """
    Regression test, hit live: Wikipedia/Polygon/Alpaca all use "BRK.B" (the
    SEC's own convention), but yfinance only recognizes "BRK-B" and silently
    reports the dotted form as delisted. Symbol in our DB stays dotted.
    """
    import yfinance

    idx = pd.bdate_range("2026-01-02", periods=2)
    # Single-symbol yfinance response is flat (no MultiIndex columns).
    flat = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [10, 20]}, index=idx
    )
    captured_symbols = {}

    def fake_download(symbols, **kwargs):
        captured_symbols["requested"] = symbols
        return flat

    monkeypatch.setattr(yfinance, "download", fake_download)

    df = _fetch_yfinance(["BRK.B"], pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date())

    assert captured_symbols["requested"] == ["BRK-B"]  # yfinance got the dashed form
    assert set(df["symbol"]) == {"BRK.B"}  # our data stays in canonical dotted form
    assert len(df) == 2


def test_fetch_yfinance_single_symbol_still_drops_null_rows(monkeypatch):
    import yfinance

    idx = pd.bdate_range("2026-01-02", periods=2)
    single = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, None], "Volume": [10, 20]}, index=idx
    )
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: single)

    df = _fetch_yfinance(["SPY"], pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date())

    assert len(df) == 1
    assert df["close"].isna().sum() == 0


def test_fetch_yfinance_single_symbol_with_multiindex_columns(monkeypatch):
    """
    Regression test, hit live: current yfinance returns (ticker, field)
    MultiIndex columns for group_by="ticker" even with ONE ticker, and the
    old flat-column selection crashed — which is how the SPY regime proxy
    went unfilled and the regime check fell back to CHOP every cycle.
    """
    import yfinance

    idx = pd.bdate_range("2026-01-02", periods=3)
    frame = pd.concat(
        {
            "SPY": pd.DataFrame(
                {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Close": [1, 2, 3], "Volume": [10, 20, 30]},
                index=idx,
            )
        },
        axis=1,
    )
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)

    df = _fetch_yfinance(["SPY"], pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-06").date())

    assert set(df["symbol"]) == {"SPY"}
    assert len(df) == 3


def test_ingest_prices_drops_nonpositive_price_rows_before_writing(monkeypatch):
    """
    Regression test for the "-118.7% in 5 days" bug: a vendor row with a
    zero/negative close previously reached `prices` unfiltered (only NaN was
    dropped, in _fetch_yfinance) and, via rolling_return()'s pct_change(),
    turned into a physically-impossible return once mom_ret_5d picked it up.
    """
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "BADCO", "BADCO"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"], utc=True),
            "open": [500.0, 10.0, 11.0], "high": [505.0, 10.0, 11.0],
            "low": [495.0, 10.0, 11.0], "close": [502.0, 0.0, 11.0],
            "volume": [1000, 100, 100], "source": ["yfinance"] * 3,
        }
    )
    monkeypatch.setattr(prices, "_fetch_yfinance", lambda *a, **k: df)

    captured = {}

    def fake_upsert(written_df, table, conflict_cols):
        captured["df"] = written_df
        return len(written_df)

    monkeypatch.setattr(prices, "upsert_dataframe", fake_upsert)

    alerted = []
    monkeypatch.setattr(prices, "alert_pipeline_failure", lambda job, detail: alerted.append((job, detail)))

    n = ingest_prices(["SPY", "BADCO"], pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date())

    assert n == 2  # SPY row + BADCO's valid second row; the zero-close row dropped
    assert (captured["df"]["close"] <= 0).sum() == 0
    assert len(alerted) == 1
    assert alerted[0][0] == "price_ingest"
