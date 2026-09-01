"""
Tests for execution/client_risk_controls.py -- the client self-service
max-drawdown / profit-target auto-close checks (see that module's docstring
for the two different pause postures being tested here). The portal's own
liquidate/resume/risk_settings endpoints have their own tests in
tests/test_dashboard_clients.py; this file is only about the hourly
check_all_clients_risk pass itself.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from execution import client_risk_controls as crc


@pytest.fixture(autouse=True)
def _no_decryption(monkeypatch):
    # Credentials aren't the point of these tests -- keep the fake rows'
    # "encrypted" fields as plain markers and skip real Fernet round-trips.
    monkeypatch.setattr(crc, "decrypt_credential", lambda v: v)


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def exec_driver_sql(self, sql, params=None):
        self._log.append((sql, params))
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
        self.executed: list[tuple] = []

    def begin(self):
        return _FakeBeginCtx(self.executed)


class _FakeBroker:
    def __init__(self, equity=10_000.0, account_fails=False, flatten_fails=False):
        self.equity = equity
        self.account_fails = account_fails
        self.flatten_fails = flatten_fails
        self.flatten_calls = 0

    def get_account(self):
        if self.account_fails:
            raise RuntimeError("Alpaca is down")
        return {"equity": self.equity}

    def flatten_all(self):
        self.flatten_calls += 1
        if self.flatten_fails:
            raise RuntimeError("Alpaca is down")


def _install_fake_brokers(monkeypatch, brokers_by_key: dict[str, _FakeBroker]):
    def factory(mode, confirm_live, api_key, secret_key):
        assert mode == "live"
        assert confirm_live is True
        broker = brokers_by_key.get(api_key)
        if broker is None:
            raise AssertionError(f"no fake broker registered for {api_key}")
        return broker

    monkeypatch.setattr(crc, "AlpacaBroker", factory)


def _no_brokers(monkeypatch):
    """Registers a factory that fails the test if a broker is ever constructed."""

    def factory(*a, **k):
        raise AssertionError("should not have needed a broker for this client")

    monkeypatch.setattr(crc, "AlpacaBroker", factory)


def _row(
    client_id: int,
    name: str,
    *,
    trading_paused: bool = False,
    pause_reason: str | None = None,
    max_drawdown_pct: float | None = None,
    equity_peak: float | None = None,
    profit_target_pct: float | None = None,
    profit_target_window_days: int | None = None,
    profit_target_period_start_equity: float | None = None,
    profit_target_period_start_ts=None,
) -> dict:
    return {
        "id": client_id,
        "name": name,
        "alpaca_api_key_encrypted": f"key-{name}",
        "alpaca_api_secret_encrypted": f"secret-{name}",
        "trading_paused": trading_paused,
        "pause_reason": pause_reason,
        "max_drawdown_pct": max_drawdown_pct,
        "equity_peak": equity_peak,
        "profit_target_pct": profit_target_pct,
        "profit_target_window_days": profit_target_window_days,
        "profit_target_period_start_equity": profit_target_period_start_equity,
        "profit_target_period_start_ts": profit_target_period_start_ts,
    }


def test_noop_when_nothing_configured(monkeypatch):
    row = _row(1, "alice")
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _no_brokers(monkeypatch)
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == []


def test_noop_when_no_active_clients(monkeypatch):
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([]))
    _no_brokers(monkeypatch)
    engine = _FakeEngine()

    assert crc.check_all_clients_risk(engine) == []


def test_skips_a_client_already_paused_for_max_drawdown(monkeypatch):
    row = _row(2, "bob", trading_paused=True, pause_reason="max_drawdown", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _no_brokers(monkeypatch)  # a paused client must never even need a broker call
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == []


def test_skips_a_client_paused_for_client_liquidate(monkeypatch):
    """A client's own 'Liquidate now' click pauses them exactly like max_drawdown
    -- only their own Resume click (not this module) should ever clear it."""
    row = _row(3, "carol", trading_paused=True, pause_reason="client_liquidate", max_drawdown_pct=0.1)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _no_brokers(monkeypatch)
    engine = _FakeEngine()

    assert crc.check_all_clients_risk(engine) == []
    assert engine.executed == []


def test_updates_equity_peak_without_triggering(monkeypatch):
    row = _row(4, "dave", max_drawdown_pct=0.1, equity_peak=None)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _install_fake_brokers(monkeypatch, {"key-dave": _FakeBroker(equity=10_000.0)})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == [("UPDATE clients SET equity_peak = %s WHERE id = %s", (10_000.0, 4))]


def test_does_not_rewrite_equity_peak_when_it_has_not_moved(monkeypatch):
    row = _row(5, "erin", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    # Equity below the existing peak but well inside the 10% drawdown limit.
    _install_fake_brokers(monkeypatch, {"key-erin": _FakeBroker(equity=19_000.0)})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == []


def test_triggers_max_drawdown_flatten_and_pauses(monkeypatch):
    row = _row(6, "frank", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeBroker(equity=17_000.0)  # -15% off peak, past the -10% limit
    _install_fake_brokers(monkeypatch, {"key-frank": broker})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert broker.flatten_calls == 1
    assert len(actions) == 1
    assert actions[0].client_id == 6
    assert actions[0].action == "max_drawdown_flatten"
    pause_sql, pause_params = engine.executed[0]
    assert "trading_paused = TRUE" in pause_sql
    assert pause_params == ("max_drawdown", 6)
    insert_sql, insert_params = engine.executed[1]
    assert "INSERT INTO client_orders" in insert_sql
    assert insert_params == (6, "ALL", "max_drawdown")


def test_pauses_even_when_flatten_call_itself_fails(monkeypatch):
    """A triggered threshold must still pause the account -- an Alpaca-side
    failure on the flatten call is not a reason to leave it tradeable."""
    row = _row(7, "gina", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeBroker(equity=17_000.0, flatten_fails=True)
    _install_fake_brokers(monkeypatch, {"key-gina": broker})
    engine = _FakeEngine()

    crc.check_all_clients_risk(engine)

    assert broker.flatten_calls == 1
    assert any("trading_paused = TRUE" in sql for sql, _ in engine.executed)


def test_seeds_profit_target_baseline_on_first_run(monkeypatch):
    row = _row(8, "hank", profit_target_pct=0.05, profit_target_window_days=7)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _install_fake_brokers(monkeypatch, {"key-hank": _FakeBroker(equity=10_000.0)})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert len(engine.executed) == 1
    sql, params = engine.executed[0]
    assert "profit_target_period_start_equity" in sql
    assert params[0] == 10_000.0
    assert params[2] == 8


def test_triggers_profit_target_flatten_and_pauses(monkeypatch):
    started = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=1)
    row = _row(
        9, "ivan", profit_target_pct=0.05, profit_target_window_days=7,
        profit_target_period_start_equity=10_000.0, profit_target_period_start_ts=started,
    )
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeBroker(equity=10_600.0)  # +6%, past the 5% target
    _install_fake_brokers(monkeypatch, {"key-ivan": broker})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert broker.flatten_calls == 1
    assert len(actions) == 1
    assert actions[0].action == "profit_target_flatten"
    _pause_sql, pause_params = engine.executed[0]
    assert pause_params == ("profit_target", 9)


def test_profit_target_not_yet_reached_does_not_trigger(monkeypatch):
    started = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=1)
    row = _row(
        10, "jill", profit_target_pct=0.05, profit_target_window_days=7,
        profit_target_period_start_equity=10_000.0, profit_target_period_start_ts=started,
    )
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeBroker(equity=10_200.0)  # only +2%, below the 5% target
    _install_fake_brokers(monkeypatch, {"key-jill": broker})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert broker.flatten_calls == 0
    assert engine.executed == []


def test_rolls_an_elapsed_profit_target_window_and_auto_resumes(monkeypatch):
    started = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=8)
    row = _row(
        11, "kate", trading_paused=True, pause_reason="profit_target",
        profit_target_pct=0.05, profit_target_window_days=7,
        profit_target_period_start_equity=10_000.0, profit_target_period_start_ts=started,
    )
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _install_fake_brokers(monkeypatch, {"key-kate": _FakeBroker(equity=10_100.0)})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert len(actions) == 1
    assert actions[0].action == "profit_window_rolled"
    assert actions[0].client_id == 11
    unpause_sql, unpause_params = engine.executed[0]
    assert "trading_paused = FALSE" in unpause_sql
    assert unpause_params == (11,)
    # The new window's baseline is reseeded off a fresh equity read in the
    # same pass, not left stale until next hour.
    reseed_sql, reseed_params = engine.executed[1]
    assert "profit_target_period_start_equity" in reseed_sql
    assert reseed_params[0] == 10_100.0


def test_max_drawdown_pause_is_not_affected_by_profit_target_rolling(monkeypatch):
    """A client paused for max_drawdown who ALSO has a profit_target
    configured must stay paused -- only the client's own Resume clears a
    max_drawdown pause, an elapsed profit-target window doesn't touch it."""
    started = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=8)
    row = _row(
        12, "leo", trading_paused=True, pause_reason="max_drawdown",
        max_drawdown_pct=0.1, equity_peak=20_000.0,
        profit_target_pct=0.05, profit_target_window_days=7,
        profit_target_period_start_equity=10_000.0, profit_target_period_start_ts=started,
    )
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _no_brokers(monkeypatch)
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == []


def test_one_clients_broken_account_does_not_stop_the_rest(monkeypatch):
    broken = _row(13, "mona", max_drawdown_pct=0.1, equity_peak=20_000.0)
    healthy = _row(14, "nora", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([broken, healthy]))
    healthy_broker = _FakeBroker(equity=17_000.0)  # past the drawdown limit
    _install_fake_brokers(
        monkeypatch,
        {"key-mona": _FakeBroker(account_fails=True), "key-nora": healthy_broker},
    )
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert healthy_broker.flatten_calls == 1
    action_types = {(a.client_id, a.action) for a in actions}
    assert (13, "error") in action_types
    assert (14, "max_drawdown_flatten") in action_types


def test_zero_or_negative_equity_is_skipped_without_writes(monkeypatch):
    row = _row(15, "oscar", max_drawdown_pct=0.1, equity_peak=20_000.0)
    monkeypatch.setattr(crc.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    _install_fake_brokers(monkeypatch, {"key-oscar": _FakeBroker(equity=0.0)})
    engine = _FakeEngine()

    actions = crc.check_all_clients_risk(engine)

    assert actions == []
    assert engine.executed == []
