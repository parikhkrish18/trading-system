import pandas as pd

from data.validators.checks import (
    check_duplicates,
    check_extreme_single_day_moves,
    check_gaps,
    check_nonpositive_prices,
    check_nulls,
    check_staleness,
)


def test_check_duplicates_flags_repeated_key():
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "QQQ"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-02"], utc=True),
            "close": [500.0, 501.0, 400.0],
        }
    )
    issues = check_duplicates(df, key_cols=["symbol", "ts"])
    assert len(issues) == 1
    assert "SPY" in issues[0]


def test_check_duplicates_clean_data_has_no_issues():
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "QQQ"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
        }
    )
    assert check_duplicates(df, key_cols=["symbol", "ts"]) == []


def test_check_gaps_detects_missing_business_day():
    dates = pd.bdate_range("2026-01-02", "2026-01-09")
    dates = dates.delete(2)  # drop one weekday in the middle
    df = pd.DataFrame({"symbol": ["SPY"] * len(dates), "ts": dates})
    issues = check_gaps(df, expect_daily=True)
    assert len(issues) == 1
    assert "SPY" in issues[0]


def test_check_gaps_clean_series_has_no_issues():
    dates = pd.bdate_range("2026-01-02", "2026-01-09")
    df = pd.DataFrame({"symbol": ["SPY"] * len(dates), "ts": dates})
    assert check_gaps(df, expect_daily=True) == []


def test_check_staleness_flags_old_data():
    old_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    df = pd.DataFrame({"symbol": ["SPY"], "ts": [old_ts]})
    issues = check_staleness(df, max_age_days=5)
    assert len(issues) == 1
    assert "SPY" in issues[0]


def test_check_staleness_fresh_data_has_no_issues():
    fresh_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)
    df = pd.DataFrame({"symbol": ["SPY"], "ts": [fresh_ts]})
    assert check_staleness(df, max_age_days=5) == []


def test_check_nulls_flags_missing_required_column():
    df = pd.DataFrame({"symbol": ["SPY"], "close": [None]})
    issues = check_nulls(df, required_cols=["symbol", "close", "volume"])
    assert any("close" in i for i in issues)
    assert any("volume" in i for i in issues)


def test_check_nonpositive_prices_flags_zero_close():
    df = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "open": [500.0], "high": [501.0], "low": [499.0], "close": [0.0],
        }
    )
    issues = check_nonpositive_prices(df)
    assert len(issues) == 1
    assert "SPY" in issues[0]


def test_check_nonpositive_prices_flags_negative_price():
    df = pd.DataFrame(
        {
            "symbol": ["BADCO"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "open": [10.0], "high": [10.0], "low": [-3.0], "close": [10.0],
        }
    )
    issues = check_nonpositive_prices(df)
    assert len(issues) == 1
    assert "BADCO" in issues[0]


def test_check_nonpositive_prices_flags_missing_close():
    df = pd.DataFrame(
        {
            "symbol": ["BADCO"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "open": [10.0], "high": [10.0], "low": [9.0], "close": [None],
        }
    )
    issues = check_nonpositive_prices(df)
    assert len(issues) == 1


def test_check_nonpositive_prices_clean_data_has_no_issues():
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "QQQ"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "open": [500.0, 400.0], "high": [505.0, 405.0], "low": [495.0, 395.0], "close": [502.0, 402.0],
        }
    )
    assert check_nonpositive_prices(df) == []


def test_check_nonpositive_prices_only_flags_the_bad_row():
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "BADCO"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "open": [500.0, 10.0], "high": [505.0, 10.0], "low": [495.0, 10.0], "close": [502.0, 0.0],
        }
    )
    issues = check_nonpositive_prices(df)
    assert len(issues) == 1
    assert "BADCO" in issues[0]
    assert "SPY" not in issues[0]


def test_check_extreme_single_day_moves_flags_a_split_shaped_jump():
    """
    A 1-for-4 reverse split reads as a fake ~+300% single day in an
    unadjusted price series -- the exact class of bug data/ingest/prices.py's
    auto_adjust fix targets, and this validator is the early-warning layer
    for whatever slips through anyway (a vendor bug, a mixed-adjustment
    history) rather than the primary fix.
    """
    dates = pd.bdate_range("2026-08-17", periods=4)
    df = pd.DataFrame({"symbol": ["SPLITCO"] * 4, "ts": dates, "close": [55.0, 56.0, 220.0, 222.0]})
    issues = check_extreme_single_day_moves(df)
    assert len(issues) == 1
    assert "SPLITCO" in issues[0]
    assert "+293" in issues[0] or "+292" in issues[0]  # (220-56)/56 ~= +292.9%


def test_check_extreme_single_day_moves_ignores_ordinary_daily_noise():
    dates = pd.bdate_range("2026-08-17", periods=5)
    df = pd.DataFrame({"symbol": ["SPY"] * 5, "ts": dates, "close": [500.0, 503.0, 498.0, 502.0, 505.0]})
    assert check_extreme_single_day_moves(df) == []


def test_check_extreme_single_day_moves_respects_a_custom_threshold():
    dates = pd.bdate_range("2026-08-17", periods=2)
    df = pd.DataFrame({"symbol": ["VOLCO"] * 2, "ts": dates, "close": [100.0, 130.0]})  # +30%
    assert check_extreme_single_day_moves(df, max_abs_move=0.60) == []
    assert len(check_extreme_single_day_moves(df, max_abs_move=0.20)) == 1


def test_check_extreme_single_day_moves_checks_each_symbol_independently():
    dates = pd.bdate_range("2026-08-17", periods=2)
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPLITCO", "SPLITCO"],
            "ts": list(dates) * 2,
            "close": [500.0, 503.0, 55.0, 220.0],
        }
    )
    issues = check_extreme_single_day_moves(df)
    assert len(issues) == 1
    assert "SPLITCO" in issues[0]


def test_check_extreme_single_day_moves_no_prior_bar_is_not_a_move():
    df = pd.DataFrame({"symbol": ["SPY"], "ts": [pd.Timestamp("2026-08-17")], "close": [500.0]})
    assert check_extreme_single_day_moves(df) == []
