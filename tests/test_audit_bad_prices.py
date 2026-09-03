import pandas as pd

from scripts import audit_bad_prices as audit


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def execute(self, stmt, params=None):
        self._log.append(params)
        return None


class _FakeBeginCtx:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return _FakeConn(self._log)

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self):
        self.executed: list = []

    def begin(self):
        return _FakeBeginCtx(self.executed)


_BAD_PRICES = pd.DataFrame(
    [
        {"symbol": "BADCO", "ts": "2026-08-25T00:00:00Z", "open": 10.0, "high": 10.0, "low": 10.0, "close": 0.0, "volume": 100},
    ]
)

_BAD_FEATURES = pd.DataFrame(
    [
        {"symbol": "BADCO", "ts": "2026-08-30T00:00:00Z", "feature_name": "mom_ret_5d", "value": -1.187},
    ]
)

_EXTREME_JUMPS = pd.DataFrame(
    [
        {
            "symbol": "SPLITCO", "ts": "2026-08-20T00:00:00Z",
            "close": 220.0, "prev_close": 55.0, "pct_change": 3.0,
        }
    ]
)


def _fake_read_sql(query, engine, params=None):
    # find_bad_prices/find_bad_return_features/find_extreme_price_jumps all
    # call pd.read_sql with a sqlalchemy text() clause -- inspect it for
    # which one it is so this one fake can answer every call the audit
    # function makes. find_extreme_price_jumps' query is checked first: its
    # SQL text also contains "FROM prices" (in a nested subquery), so it
    # would otherwise be misidentified as find_bad_prices' query.
    sql = str(query)
    if "LAG(close)" in sql:
        return _EXTREME_JUMPS.copy()
    if "FROM prices" in sql:
        return _BAD_PRICES.copy()
    if "FROM features" in sql:
        return _BAD_FEATURES.copy()
    raise AssertionError(f"unexpected query: {sql}")


def test_audit_reports_bad_prices_and_bad_features_without_deleting(monkeypatch):
    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql)
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    result = audit.audit(delete=False)

    assert result == {"bad_prices": 1, "bad_return_features": 1, "extreme_price_jumps": 1}
    assert engine.executed == []  # report-only: nothing written


def test_audit_delete_removes_bad_prices_rows(monkeypatch):
    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql)
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    result = audit.audit(delete=True)

    assert result == {"bad_prices": 1, "bad_return_features": 1, "extreme_price_jumps": 1}
    assert len(engine.executed) == 1
    assert engine.executed[0] == {"symbol": "BADCO", "ts": "2026-08-25T00:00:00Z"}


def test_audit_delete_never_touches_extreme_jump_rows(monkeypatch):
    """
    Unlike bad_prices, an extreme jump is report-only even with --delete --
    deleting the jump row would leave a gap rather than fix the stale
    unadjusted history around it; the real fix is scripts.rebackfill_prices.
    """
    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql)
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    audit.audit(delete=True)

    deleted_keys = engine.executed
    assert {"symbol": "SPLITCO", "ts": "2026-08-20T00:00:00Z"} not in deleted_keys


def test_audit_clean_data_finds_nothing(monkeypatch):
    def _fake_read_sql_clean(query, engine, params=None):
        sql = str(query)
        if "LAG(close)" in sql:
            return pd.DataFrame(columns=["symbol", "ts", "close", "prev_close", "pct_change"])
        if "FROM prices" in sql:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql_clean)
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    result = audit.audit(delete=True)

    assert result == {"bad_prices": 0, "bad_return_features": 0, "extreme_price_jumps": 0}
    assert engine.executed == []


def test_find_extreme_price_jumps_passes_the_threshold_as_a_query_param(monkeypatch):
    captured = {}

    def _fake_read_sql_capture(query, engine, params=None):
        captured["params"] = params
        return _EXTREME_JUMPS.copy()

    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql_capture)

    result = audit.find_extreme_price_jumps(engine=object(), max_abs_move=0.75)

    assert captured["params"] == {"max_abs_move": 0.75}
    assert len(result) == 1
