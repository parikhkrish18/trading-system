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
        json={"name": "alice", "alpaca_api_key": "bad", "alpaca_api_secret": "bad", "password": "pw123456"},
    )

    assert resp.status_code == 400
    assert "Could not connect" in resp.json()["detail"]
    assert engine.executed == []  # never even tried to write to the DB


def test_create_client_requires_name_and_password(monkeypatch, client):
    resp = client.post("/api/clients", json={"name": "", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": ""})
    assert resp.status_code == 400


def test_create_client_rejects_a_too_short_password(monkeypatch, client):
    resp = client.post(
        "/api/clients",
        json={"name": "alice", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "short1"},
    )
    assert resp.status_code == 400
    assert "8 characters" in resp.json()["detail"]


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
            "margin_enabled": True, "password": "pw123456",
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
        json={"name": "bob", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "pw123456"},
    )

    assert resp.status_code == 200
    assert onboard_calls == [42]
    assert "buy-in submitted" in resp.json()["buy_in"].lower()


# ---------------------------------------------------------------------
# Leverage: creation-time validation and the standalone update endpoint
# ---------------------------------------------------------------------


def test_create_client_rejects_leverage_above_the_hard_cap(monkeypatch, client):
    resp = client.post(
        "/api/clients",
        json={
            "name": "pat", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "pw123456",
            "margin_enabled": True, "leverage_multiplier": 4,
        },
    )
    assert resp.status_code == 400
    assert "leverage_multiplier" in resp.json()["detail"]


def test_create_client_rejects_leverage_without_margin(monkeypatch, client):
    resp = client.post(
        "/api/clients",
        json={
            "name": "quinn", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "pw123456",
            "margin_enabled": False, "leverage_multiplier": 2,
        },
    )
    assert resp.status_code == 400
    assert "margin" in resp.json()["detail"].lower()


def test_create_client_accepts_valid_leverage(monkeypatch, client):
    monkeypatch.setattr(server, "AlpacaBroker", lambda *a, **k: _FakeVerifyBroker(*a, **k))
    monkeypatch.setattr(server.settings, "client_trading_enabled", False)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post(
        "/api/clients",
        json={
            "name": "rae", "alpaca_api_key": "k", "alpaca_api_secret": "s", "password": "pw123456",
            "margin_enabled": True, "leverage_multiplier": 2,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["leverage_multiplier"] == 2
    assert engine.executed[0][1][4] == 2  # leverage_multiplier is the 5th INSERT param


def test_set_client_leverage_rejects_above_the_hard_cap(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: object())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"margin_enabled": True}]))

    resp = client.post("/api/clients/7/leverage", json={"leverage_multiplier": 5})
    assert resp.status_code == 400


def test_set_client_leverage_rejects_when_client_is_not_margin_enabled(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: object())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"margin_enabled": False}]))

    resp = client.post("/api/clients/7/leverage", json={"leverage_multiplier": 2})
    assert resp.status_code == 400
    assert "margin" in resp.json()["detail"].lower()


def test_set_client_leverage_404s_for_an_unknown_client(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: object())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["margin_enabled"]))

    resp = client.post("/api/clients/999/leverage", json={"leverage_multiplier": 2})
    assert resp.status_code == 404


def test_set_client_leverage_updates_when_valid(monkeypatch, client):
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"margin_enabled": True}]))

    resp = client.post("/api/clients/7/leverage", json={"leverage_multiplier": 3})

    assert resp.status_code == 200
    assert resp.json() == {"id": 7, "leverage_multiplier": 3}
    assert engine.executed[0][1] == (3, 7)


# ---------------------------------------------------------------------
# GET /api/clients — list, masked
# ---------------------------------------------------------------------


def test_list_clients_masks_the_api_key(monkeypatch, client):
    encrypted_key = client_crypto.encrypt_credential("PKLIVEKEY123456")
    row = pd.DataFrame(
        [
            {
                "id": 1, "name": "alice", "alpaca_api_key_encrypted": encrypted_key,
                "margin_enabled": True, "leverage_multiplier": 1, "active": True,
                "trading_paused": False, "pause_reason": None,
                "created_at": pd.Timestamp("2026-08-01", tz="UTC"),
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
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"id": 7}]))

    resp = client.post("/api/clients/7/deactivate")

    assert resp.status_code == 200
    assert resp.json() == {"id": 7, "active": False}
    assert engine.executed[0][1] == (7,)


def test_deactivate_client_404s_for_an_unknown_client(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["id"]))

    resp = client.post("/api/clients/999/deactivate")

    assert resp.status_code == 404


def test_reactivate_client_writes_active_true(monkeypatch, client):
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"id": 7}]))

    resp = client.post("/api/clients/7/reactivate")

    assert resp.status_code == 200
    assert resp.json() == {"id": 7, "active": True}
    assert engine.executed[0][1] == (7,)


def test_reactivate_client_404s_for_an_unknown_client(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["id"]))

    resp = client.post("/api/clients/999/reactivate")

    assert resp.status_code == 404


def test_reset_client_password_requires_new_password(monkeypatch, client):
    resp = client.post("/api/clients/7/reset_password", json={"new_password": ""})
    assert resp.status_code == 400


def test_reset_client_password_rejects_a_too_short_password(monkeypatch, client):
    resp = client.post("/api/clients/7/reset_password", json={"new_password": "short1"})
    assert resp.status_code == 400
    assert "8 characters" in resp.json()["detail"]


def test_reset_client_password_hashes_before_storing(monkeypatch, client):
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame([{"id": 7}]))

    resp = client.post("/api/clients/7/reset_password", json={"new_password": "new-secret-pw"})

    assert resp.status_code == 200
    stored_hash = engine.executed[0][1][0]
    assert stored_hash != "new-secret-pw"
    assert client_crypto.verify_password("new-secret-pw", stored_hash) is True


def test_reset_client_password_404s_for_an_unknown_client(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["id"]))

    resp = client.post("/api/clients/999/reset_password", json={"new_password": "new-secret-pw"})

    assert resp.status_code == 404


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


def _client_row_df(client_id=1, name="alice", password_hash=_ALICE_PASSWORD_HASH, active=True, trading_paused=False):
    return pd.DataFrame(
        [{"id": client_id, "name": name, "password_hash": password_hash, "active": active, "trading_paused": trading_paused}]
    )


def test_portal_page_escapes_a_malicious_client_name(monkeypatch, client):
    """
    Stored-XSS regression guard: the client name is operator-entered (see
    execution/client_fanout.py) but still lands directly in the portal
    page's markup as __CLIENT_NAME__ -- a name containing a script tag or
    an onerror handler must never reach the response unescaped.
    """
    evil_name = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df(name=evil_name))
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    resp = client.get("/portal")

    assert resp.status_code == 200
    # The page's own legitimate <script> block is expected -- what must
    # never appear is the *injected* markup surviving as real HTML rather
    # than inert escaped text.
    assert "<script>alert(1)" not in resp.text
    assert "<img src=x onerror=alert(2)>" not in resp.text
    # The escaped form is present -- proof this actually rendered the name
    # (inert, as plain text) rather than stripping it silently.
    assert "&lt;script&gt;alert(1)" in resp.text
    assert "&lt;img src=x onerror=alert(2)&gt;" in resp.text


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


def test_portal_login_calls_verify_password_even_for_a_nonexistent_name(monkeypatch, client):
    """
    Timing side-channel guard: verify_password (a ~600,000-iteration PBKDF2
    check) must run for a name that isn't in `clients` at all, not just for
    a real one -- short-circuiting before ever calling it is exactly the
    timing gap that would let an attacker enumerate real client names by
    how long /portal/login takes to respond.
    """
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["id", "password_hash", "active"]))
    calls = []
    monkeypatch.setattr(server, "verify_password", lambda password, stored: calls.append((password, stored)) or False)

    resp = client.post("/portal/login", data={"name": "ghost", "password": "whatever"})

    assert resp.status_code == 401
    assert len(calls) == 1
    assert calls[0] == ("whatever", server._DUMMY_PASSWORD_HASH)


def test_portal_login_compares_against_the_real_hash_when_the_name_exists(monkeypatch, client):
    """The flip side: a real client's own stored hash is what gets compared, not the dummy."""
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    calls = []
    monkeypatch.setattr(server, "verify_password", lambda password, stored: calls.append((password, stored)) or False)

    resp = client.post("/portal/login", data={"name": "alice", "password": "whatever"})

    assert resp.status_code == 401
    assert len(calls) == 1
    assert calls[0] == ("whatever", _ALICE_PASSWORD_HASH)


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


def test_portal_positions_does_not_leak_the_raw_exception_to_the_client(monkeypatch, client, caplog):
    """
    A broker failure (e.g. a credential-decryption error from
    _client_broker()) must be logged server-side, not forwarded verbatim to
    the client -- internal state (a decryption error's own message, an
    Alpaca error string) has no business crossing that trust boundary.
    """
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    secret_detail = "decryption failed: CLIENT_KEY_ENCRYPTION_KEY rotated, ciphertext abc123==)"

    class _FakeClientBroker:
        def get_positions_detailed(self):
            raise RuntimeError(secret_detail)

    monkeypatch.setattr(server, "_client_broker", lambda client_id: _FakeClientBroker())

    with caplog.at_level("ERROR"):
        resp = client.get("/api/portal/positions")

    assert resp.status_code == 502
    assert secret_detail not in resp.text
    assert "Could not reach your Alpaca account" in resp.json()["detail"]
    # The real detail is still findable server-side, just not in the response.
    assert secret_detail in caplog.text


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


# ---------------------------------------------------------------------
# Client portal — self-service risk controls (liquidate / resume / risk_settings)
# ---------------------------------------------------------------------


class _FakeRiskBroker:
    """Stands in for _client_broker in the liquidate/resume/risk_settings tests."""

    def __init__(self, equity=10_000.0, flatten_fails=False, account_fails=False):
        self.equity = equity
        self.flatten_fails = flatten_fails
        self.account_fails = account_fails
        self.flatten_calls = 0

    def flatten_all(self):
        self.flatten_calls += 1
        if self.flatten_fails:
            raise RuntimeError("Alpaca is down")

    def get_account(self):
        if self.account_fails:
            raise RuntimeError("Alpaca is down")
        return {"equity": self.equity}


def test_portal_liquidate_and_resume_require_a_session(client):
    assert client.post("/api/portal/liquidate").status_code == 401
    assert client.post("/api/portal/resume").status_code == 401
    assert client.get("/api/portal/risk_settings").status_code == 401
    assert client.post("/api/portal/risk_settings", json={}).status_code == 401


def test_portal_liquidate_flattens_and_pauses(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    broker = _FakeRiskBroker()
    monkeypatch.setattr(server, "_client_broker", lambda client_id: broker)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/portal/liquidate")

    assert resp.status_code == 200
    assert resp.json() == {"status": "liquidated", "trading_paused": True}
    assert broker.flatten_calls == 1
    update_params = next(p for sql, p in engine.executed if "UPDATE clients" in sql)
    assert update_params == ("client_liquidate", 1)
    insert_params = next(p for sql, p in engine.executed if "INSERT INTO client_orders" in sql)
    assert insert_params == (1, "ALL", "client_liquidate")


def test_portal_liquidate_is_idempotent_when_already_paused(monkeypatch, client):
    """
    A double-click, or a race with an auto-triggered max_drawdown/
    profit_target pause landing between two clicks, must not fire
    flatten_all() twice or write two client_orders audit rows for one
    logical liquidation -- the real harm a missing idempotency check causes.
    Simulates the already-paused state directly (a stateless
    already-paused-before-this-request read) rather than a literal
    double-call, per the same "simulate it" pattern used for the timing
    guard above.
    """
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df(trading_paused=True))
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    broker = _FakeRiskBroker()
    monkeypatch.setattr(server, "_client_broker", lambda client_id: broker)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/portal/liquidate")

    assert resp.status_code == 200
    assert resp.json() == {"status": "already_paused", "trading_paused": True}
    assert broker.flatten_calls == 0  # never touched the broker
    assert engine.executed == []  # no duplicate UPDATE, no duplicate client_orders row


def test_portal_liquidate_502s_when_broker_unreachable_and_does_not_pause(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    broker = _FakeRiskBroker(flatten_fails=True)
    monkeypatch.setattr(server, "_client_broker", lambda client_id: broker)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/portal/liquidate")

    assert resp.status_code == 502
    assert engine.executed == []  # never got as far as pausing


def test_portal_resume_clears_pause_and_reseeds_baselines(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    broker = _FakeRiskBroker(equity=12_345.0)
    monkeypatch.setattr(server, "_client_broker", lambda client_id: broker)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/portal/resume")

    assert resp.status_code == 200
    assert resp.json() == {"status": "resumed", "trading_paused": False}
    sql, params = engine.executed[0]
    assert "trading_paused = FALSE" in sql
    assert params[0] == 12_345.0  # equity_peak reseeded
    assert params[1] == 12_345.0  # profit_target_period_start_equity reseeded
    assert params[3] == 1  # client_id


def test_portal_resume_502s_when_broker_unreachable(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)
    monkeypatch.setattr(server, "_client_broker", lambda client_id: _FakeRiskBroker(account_fails=True))

    resp = client.post("/api/portal/resume")
    assert resp.status_code == 502


def _risk_settings_row(**overrides) -> pd.DataFrame:
    row = {
        "trading_paused": False, "pause_reason": None,
        "max_drawdown_pct": None, "profit_target_pct": None, "profit_target_window_days": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_get_portal_risk_settings_returns_current_values(monkeypatch, client):
    def fake_read_sql(query, engine, params=None):
        sql = str(query)
        if "trading_paused" in sql and "max_drawdown_pct" in sql:
            return _risk_settings_row(trading_paused=True, pause_reason="max_drawdown", max_drawdown_pct=0.1)
        return _client_row_df()

    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", fake_read_sql)
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    resp = client.get("/api/portal/risk_settings")

    assert resp.status_code == 200
    assert resp.json() == {
        "trading_paused": True, "pause_reason": "max_drawdown",
        "max_drawdown_pct": 0.1, "profit_target_pct": None, "profit_target_window_days": None,
    }


@pytest.mark.parametrize(
    "body",
    [
        {"max_drawdown_pct": 0.9},  # above the 50% hard cap
        {"max_drawdown_pct": 0.0},  # 0% is not a valid limit
        {"profit_target_pct": 0.05},  # window missing
        {"profit_target_window_days": 7},  # target missing
        {"profit_target_pct": 0.05, "profit_target_window_days": 400},  # window above 365
    ],
)
def test_set_portal_risk_settings_rejects_invalid_input(monkeypatch, client, body):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    resp = client.post("/api/portal/risk_settings", json=body)
    assert resp.status_code == 400


def test_set_portal_risk_settings_updates_and_reseeds_baselines(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    broker = _FakeRiskBroker(equity=20_000.0)
    monkeypatch.setattr(server, "_client_broker", lambda client_id: broker)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    body = {"max_drawdown_pct": 0.15, "profit_target_pct": 0.05, "profit_target_window_days": 7}
    resp = client.post("/api/portal/risk_settings", json=body)

    assert resp.status_code == 200
    assert resp.json() == body
    sql, params = engine.executed[0]
    assert "UPDATE clients SET max_drawdown_pct" in sql
    assert params[0] == 0.15  # max_drawdown_pct
    assert params[1] == 20_000.0  # equity_peak reseeded
    assert params[2] == 0.05  # profit_target_pct
    assert params[3] == 7  # profit_target_window_days
    assert params[4] == 20_000.0  # profit_target_period_start_equity reseeded
    assert params[6] == 1  # client_id


def test_set_portal_risk_settings_turning_everything_off_skips_the_broker_call(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _client_row_df())
    _log_in_as(client, client_id=1, password_hash=_ALICE_PASSWORD_HASH)

    def _boom(client_id):
        raise AssertionError("should not need the broker when every threshold is being turned off")

    monkeypatch.setattr(server, "_client_broker", _boom)
    engine = _FakeEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)

    resp = client.post("/api/portal/risk_settings", json={})

    assert resp.status_code == 200
    assert resp.json() == {"max_drawdown_pct": None, "profit_target_pct": None, "profit_target_window_days": None}
    _sql, params = engine.executed[0]
    assert params[:6] == (None, None, None, None, None, None)
