import pandas as pd

from data.ingest.prices import _fetch_yfinance


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
