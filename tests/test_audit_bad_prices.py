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


def _fake_read_sql(query, engine, params=None):
    # Both find_bad_prices/find_bad_return_features call pd.read_sql with a
    # sqlalchemy text() clause -- inspect it for which table it targets so
    # this one fake can answer both calls the audit function makes.
    sql = str(query)
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

    assert result == {"bad_prices": 1, "bad_return_features": 1}
    assert engine.executed == []  # report-only: nothing written


def test_audit_delete_removes_bad_prices_rows(monkeypatch):
    monkeypatch.setattr(audit.pd, "read_sql", _fake_read_sql)
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    result = audit.audit(delete=True)

    assert result == {"bad_prices": 1, "bad_return_features": 1}
    assert len(engine.executed) == 1
    assert engine.executed[0] == {"symbol": "BADCO", "ts": "2026-08-25T00:00:00Z"}


def test_audit_clean_data_finds_nothing(monkeypatch):
    monkeypatch.setattr(
        audit.pd, "read_sql",
        lambda query, engine, params=None: pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        if "FROM prices" in str(query)
        else pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"]),
    )
    engine = _FakeEngine()
    monkeypatch.setattr(audit, "get_engine", lambda: engine)

    result = audit.audit(delete=True)

    assert result == {"bad_prices": 0, "bad_return_features": 0}
    assert engine.executed == []
