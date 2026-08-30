"""
Tests for the client-management admin endpoints (/api/clients/*) and the
client portal (/portal, /portal/login, /api/portal/*) added in
monitoring/dashboard/server.py -- see execution/client_fanout.py and
execution/client_crypto.py for the trading/encryption logic these drive.
"""
from __future__ import annotations

import pandas as pd
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from execution import client_crypto
from monitoring.dashboard import server


@pytest.fixture(autouse=True)
def _local_dev_bind(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_host", "127.0.0.1")


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setattr(client_crypto.settings, "client_key_encryption_key", Fernet.generate_key().decode())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "")
    return TestClient(server.app, base_url="http://127.0.0.1")


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def exec_driver_sql(self, sql, params=None):
        self._log.append((sql, params))
        return _FakeResult()


class _FakeResult:
    def scalar_one(self):
        return 42


class _FakeBeginCtx:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return _FakeConn(self._log)

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self):
        self.executed: list[tuple] = []

    def begin(self):
        return _FakeBeginCtx(self.executed)


class _FakeVerifyBroker:
    """Stands in for AlpacaBroker during the "does this key work" check in create_client."""

    def __init__(self, mode, confirm_live, api_key, secret_key, should_fail=False):
        self._should_fail = should_fail

    def get_account(self):
        if self._should_fail:
            raise RuntimeError("invalid API key")
        return {}


# ---------------------------------------------------------------------
# POST /api/clients — add a client
# ---------------------------------------------------------------------


def test_create_client_rejects_bad_alpaca_credentials(monkeypatch, client):
    monkeypatch.setattr(server, "AlpacaBroker", lambda *a, **k: _FakeVerifyBroker(*a, **k, should_fail=True))
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post(
        "/api/clients",
        json={"name": "alice", "alpaca_api_key": "bad", "alpaca_api_secret": "bad", "password": "pw12345"},
    )

    assert resp.status_code == 400
    assert "Could not connect" in resp.json()["detail"]
    assert engine.executed == []  # never even tried to write to the DB


def test_create_client_requires_name_and_password(monkeypatch, client):
    resp = client.post("/api/clients", json={"name": "", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": ""})
    assert resp.status_code == 400


def test_create_client_succeeds_and_reports_trading_disabled(monkeypatch, client):
    """With CLIENT_TRADING_ENABLED off (the default), a client can be added
    but the immediate buy-in must not fire -- the response says so."""
    monkeypatch.setattr(server, "AlpacaBroker", lambda *a, **k: _FakeVerifyBroker(*a, **k))
    monkeypatch.setattr(server.settings, "client_trading_enabled", False)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post(
        "/api/clients",
        json={
            "name": "alice", "alpaca_api_key": "PKGOODKEY", "alpaca_api_secret": "goodsecret",
            "margin_enabled": True, "password": "pw12345",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 42
    assert body["name"] == "alice"
    assert body["margin_enabled"] is True
    assert "off" in body["buy_in"].lower()
    # One INSERT happened, and neither the plaintext key nor secret is in it.
    assert len(engine.executed) == 1
    insert_params = engine.executed[0][1]
    assert "PKGOODKEY" not in insert_params
    assert "goodsecret" not in insert_params


def test_create_client_buys_in_immediately_when_trading_enabled(monkeypatch, client):
    monkeypatch.setattr(server, "AlpacaBroker", lambda *a, **k: _FakeVerifyBroker(*a, **k))
    monkeypatch.setattr(server.settings, "client_trading_enabled", True)
    monkeypatch.setattr(server, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(server, "get_broker", lambda: object())  # master broker, opaque here

    onboard_calls = []
    monkeypatch.setattr(server, "onboard_client", lambda client_id, master_broker, engine: onboard_calls.append(client_id))

    resp = client.post(
        "/api/clients",
        json={"name": "bob", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "pw12345"},
    )

    assert resp.status_code == 200
    assert onboard_calls == [42]
    assert "buy-in submitted" in resp.json()["buy_in"].lower()


# ---------------------------------------------------------------------
# GET /api/clients — list, masked
# ---------------------------------------------------------------------


def test_list_clients_masks_the_api_key(monkeypatch, client):
    encrypted_key = client_crypto.encrypt_credential("PKLIVEKEY123456")
    row = pd.DataFrame(
        [
            {
                "id": 1, "name": "alice", "alpaca_api_key_encrypted": encrypted_key,
                "margin_enabled": True, "active": True, "created_at": pd.Timestamp("2026-08-01", tz="UTC"),
            }
        ]
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: row)

    resp = client.get("/api/clients")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["name"] == "alice"
    assert "PKLIVEKEY123456" not in body[0]["api_key_preview"]
    assert body[0]["api_key_preview"] == "PKLI…3456"


# ---------------------------------------------------------------------
# deactivate / reactivate / reset password
# ---------------------------------------------------------------------


def test_deactivate_client_writes_active_false(monkeypatch, client):
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/clients/7/deactivate")

    assert resp.status_code == 200
    assert resp.json() == {"id": 7, "active": False}
    assert engine.executed[0][1] == (7,)


def test_reset_client_password_requires_new_password(monkeypatch, client):
    resp = client.post("/api/clients/7/reset_password", json={"new_password": ""})
    assert resp.status_code == 400


def test_reset_client_password_hashes_before_storing(monkeypatch, client):
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/clients/7/reset_password", json={"new_password": "new-secret-pw"})

    assert resp.status_code == 200
    stored_hash = engine.executed[0][1][0]
    assert stored_hash != "new-secret-pw"
    assert client_crypto.verify_password("new-secret-pw", stored_hash) is True


# ---------------------------------------------------------------------
# Client portal — login and gated read endpoints
# ---------------------------------------------------------------------


def _log_in_as(client, client_id: int, password_hash: str) -> None:
    """
    Sets the client-portal session cookie directly, computed from the SAME
    password_hash a test's mocked DB row carries -- rather than relying on
    the TestClient's cookie jar to carry a Secure cookie forward from a
    real /portal/login POST. The operator-login tests in
    test_dashboard_server.py use the exact same direct-cookie-set pattern
    for the same reason (Secure cookies don't reliably round-trip through
    httpx's test transport the way a real browser would). Note this takes
    the HASH, not the plaintext password: PBKDF2 is salted, so re-hashing
    the same password here would produce a different string than whatever
    hash the mocked row already contains, and the token would never match.
    """
    client.cookies.set(server._CLIENT_SESSION_COOKIE, f"{client_id}.{server._client_session_token(password_hash)}")


# Hashed once at import time: PBKDF2 is salted, so hashing "clientpass123"
# fresh in every fixture call would produce a different string each time --
# tests that need to log in via _log_in_as (which needs the exact hash a
# mocked row carries, not a freshly re-salted one) share this constant.
_ALICE_PASSWORD = "clientpass123"
_ALICE_PASSWORD_HASH = client_crypto.hash_password(_ALICE_PASSWORD)


def _client_row_df(client_id=1, name="alice", password_hash=_ALICE_PASSWORD_HASH, active=True):
    return pd.DataFrame([{"id": client_id, "name": name, "password_hash": password_hash, "active": active}])


def test_portal_login_wrong_password_returns_401(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())

    resp = client.post("/portal/login", data={"name": "alice", "password": "wrong"})

    assert resp.status_code == 401
    assert "client_portal_session" not in resp.cookies


def test_portal_login_inactive_client_rejected(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df(active=False))

    resp = client.post("/portal/login", data={"name": "alice", "password": _ALICE_PASSWORD})

    assert resp.status_code == 401


def test_portal_endpoints_require_a_session(client):
    resp = client.get("/api/portal/positions")
    assert resp.status_code == 401
    resp = client.get("/api/portal/account")
    assert resp.status_code == 401
    resp = client.get("/api/portal/trades")
    assert resp.status_code == 401


def test_portal_login_sets_a_cookie(monkeypatch, client):
    """The login response itself, independent of whether it round-trips
    through a cookie jar afterwards (see _log_in_as for why later tests
    don't rely on that)."""
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())

    login_resp = client.post(
        "/portal/login", data={"name": "alice", "password": _ALICE_PASSWORD}, follow_redirects=False
    )

    assert login_resp.status_code == 303
    assert "client_portal_session" in login_resp.cookies


def test_portal_positions_authenticated(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    fake_positions = [{"symbol": "TSLA", "side": "long", "qty": 5.0, "market_value": 1000.0, "unrealized_pl": 50.0}]

    class _FakeClientBroker:
        def get_positions_detailed(self):
            return fake_positions

    monkeypatch.setattr(server, "_client_broker", lambda client_id: _FakeClientBroker())

    resp = client.get("/api/portal/positions")
    assert resp.status_code == 200
    assert resp.json() == fake_positions


def test_portal_trades_excludes_reasoning_and_forecast_fields(monkeypatch, client):
    """The 'results only' decision: a client's own trade history carries no
    sentiment/reasoning/forecast data, only what happened to their capital."""
    monkeypatch.setattr(server, "get_engine", lambda: None)

    def fake_read_sql(query, engine, params=None):
        sql = str(query)
        if "client_orders" in sql:
            return pd.DataFrame(
                [
                    {
                        "symbol": "TSLA", "side": "long", "target_position_pct": 0.4, "target_shares": 20.0,
                        "status": "submitted", "ts": pd.Timestamp("2026-08-28T15:00:00Z"),
                    }
                ]
            )
        return _client_row_df()

    monkeypatch.setattr(server.pd, "read_sql", fake_read_sql)
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    resp = client.get("/api/portal/trades")

    assert resp.status_code == 200
    row = resp.json()[0]
    assert set(row.keys()) == {"symbol", "side", "target_position_pct", "target_shares", "status", "ts"}


def test_password_reset_invalidates_the_old_session_cookie(monkeypatch, client):
    """Resetting a client's password must sign them out everywhere -- the
    session token is an HMAC of the password hash, so it changes the moment
    the hash does (same self-invalidating design as the operator session)."""
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())

    class _FakeClientBroker:
        def get_positions_detailed(self):
            return []

    monkeypatch.setattr(server, "_client_broker", lambda client_id: _FakeClientBroker())

    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)
    assert client.get("/api/portal/positions").status_code == 200  # authenticated

    # Now the password changes underneath the same stored hash.
    monkeypatch.setattr(
        server.pd, "read_sql", lambda *a, **k: _client_row_df(password_hash=client_crypto.hash_password("a-different-password"))
    )

    resp = client.get("/api/portal/positions")
    assert resp.status_code == 401
