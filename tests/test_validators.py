import pandas as pd

from data.validators.checks import (
    check_duplicates,
    check_gaps,
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
