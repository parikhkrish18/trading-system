import pandas as pd
import pytest
from cryptography.fernet import Fernet

from execution import client_fanout


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setattr("execution.client_crypto.settings.client_key_encryption_key", Fernet.generate_key().decode())


@pytest.fixture(autouse=True)
def _client_trading_enabled(monkeypatch):
    # Every test here is specifically about what fan-out DOES once it's on;
    # the "off by default" behavior gets its own dedicated test below.
    monkeypatch.setattr(client_fanout.settings, "client_trading_enabled", True)


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def exec_driver_sql(self, sql, params=None):
        self._log.append(params)
        return None


class _FakeEngine:
    """Records every client_orders INSERT via _log_order — good enough for
    these tests, which never need to read client_orders back."""

    def __init__(self):
        self.inserted: list[tuple] = []

    def begin(self):
        return _FakeBeginCtx(self.inserted)


class _FakeBeginCtx:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return _FakeConn(self._log)

    def __exit__(self, *exc):
        return False


def _make_client_row(client_id: int, name: str, margin_enabled: bool, leverage_multiplier: int = 1) -> dict:
    from execution.client_crypto import encrypt_credential

    return {
        "id": client_id,
        "name": name,
        "alpaca_api_key_encrypted": encrypt_credential(f"key-{name}"),
        "alpaca_api_secret_encrypted": encrypt_credential(f"secret-{name}"),
        "margin_enabled": margin_enabled,
        "leverage_multiplier": leverage_multiplier,
    }


class _FakeAlpacaBroker:
    """Records every submit_target_position call and can fail a specific symbol's order on demand."""

    def __init__(self, mode, confirm_live, api_key, secret_key, portfolio_value=10_000.0, fail_symbols=None):
        assert mode == "live"
        assert confirm_live is True
        self.api_key = api_key
        self.secret_key = secret_key
        self.portfolio_value = portfolio_value
        self.fail_symbols = fail_symbols or set()
        self.submitted: list[tuple[str, float]] = []

    def get_portfolio_value(self):
        return self.portfolio_value

    def submit_target_position(self, symbol, target_shares):
        if symbol in self.fail_symbols:
            raise RuntimeError(f"order rejected for {symbol}")
        self.submitted.append((symbol, target_shares))
        return {"id": f"order-{symbol}"}


def _install_fake_broker_factory(monkeypatch, brokers_by_key: dict[str, _FakeAlpacaBroker]):
    """brokers_by_key maps the fake api_key string ("key-<name>") to the
    pre-built fake broker that construction should hand back."""

    def factory(mode, confirm_live, api_key, secret_key):
        broker = brokers_by_key.get(api_key)
        if broker is None:
            raise RuntimeError(f"no fake broker registered for {api_key}")
        if isinstance(broker, Exception):
            raise broker
        return broker

    monkeypatch.setattr(client_fanout, "AlpacaBroker", factory)


def test_replicate_to_clients_noop_when_trading_disabled(monkeypatch):
    monkeypatch.setattr(client_fanout.settings, "client_trading_enabled", False)
    calls = []
    monkeypatch.setattr(client_fanout, "load_active_clients", lambda *a, **k: calls.append(1))
    client_fanout.replicate_to_clients({"AAPL": 0.5}, {"AAPL": 100.0}, engine=object())
    assert calls == []  # load_active_clients was never even reached


def test_replicate_to_clients_sizes_proportionally_to_each_clients_own_equity(monkeypatch):
    row_a = _make_client_row(1, "alice", margin_enabled=True)
    row_b = _make_client_row(2, "bob", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row_a, row_b]))

    broker_a = _FakeAlpacaBroker("live", True, "key-alice", "secret-alice", portfolio_value=10_000.0)
    broker_b = _FakeAlpacaBroker("live", True, "key-bob", "secret-bob", portfolio_value=50_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-alice": broker_a, "key-bob": broker_b})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"TSLA": 0.40}, {"TSLA": 200.0}, engine)

    # $10,000 * 40% / $200 = 20 sh; $50,000 * 40% / $200 = 100 sh.
    assert broker_a.submitted == [("TSLA", pytest.approx(20.0))]
    assert broker_b.submitted == [("TSLA", pytest.approx(100.0))]


def test_replicate_to_clients_skips_short_leg_for_a_non_margin_client(monkeypatch):
    row = _make_client_row(1, "carol", margin_enabled=False)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-carol", "secret-carol")
    _install_fake_broker_factory(monkeypatch, {"key-carol": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"SNDK": -0.60}, {"SNDK": 50.0}, engine)

    assert broker.submitted == []  # the short was never submitted
    # _log_order's param tuple is (client_id, symbol, side, target_position_pct, target_shares, status, alpaca_order_id, error_message)
    assert engine.inserted[-1][5] == "skipped_no_margin"


def test_replicate_to_clients_submits_short_for_a_margin_client(monkeypatch):
    row = _make_client_row(1, "dave", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-dave", "secret-dave", portfolio_value=10_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-dave": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"SNDK": -0.60}, {"SNDK": 50.0}, engine)

    # $10,000 * -60% / $50 = -120 sh
    assert broker.submitted == [("SNDK", pytest.approx(-120.0))]


def test_replicate_to_clients_applies_leverage_multiplier(monkeypatch):
    row = _make_client_row(1, "leo", margin_enabled=True, leverage_multiplier=2)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-leo", "secret-leo", portfolio_value=10_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-leo": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"TSLA": 0.10}, {"TSLA": 200.0}, engine)

    # Unlevered would be $10,000 * 10% / $200 = 5 sh; 2x leverage doubles it to 10.
    assert broker.submitted == [("TSLA", pytest.approx(10.0))]
    # target_position_pct logged is the LEVERED figure actually used, not the master's raw 0.10.
    logged = engine.inserted[-1]
    assert logged[3] == pytest.approx(0.20)
    assert logged[4] == pytest.approx(10.0)


def test_replicate_to_clients_caps_leveraged_exposure_at_max_single_position_pct(monkeypatch):
    monkeypatch.setattr(client_fanout.settings, "max_single_position_pct", 0.20)
    row = _make_client_row(1, "mia", margin_enabled=True, leverage_multiplier=3)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-mia", "secret-mia", portfolio_value=10_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-mia": broker})

    engine = _FakeEngine()
    # 0.10 * 3x leverage = 0.30, which exceeds the 0.20 cap -- clamped to 0.20.
    client_fanout.replicate_to_clients({"TSLA": 0.10}, {"TSLA": 200.0}, engine)

    assert broker.submitted == [("TSLA", pytest.approx(10.0))]  # $10,000 * 20% (capped) / $200


def test_replicate_to_clients_caps_a_leveraged_short_symmetrically(monkeypatch):
    monkeypatch.setattr(client_fanout.settings, "max_single_position_pct", 0.20)
    row = _make_client_row(1, "nate", margin_enabled=True, leverage_multiplier=3)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-nate", "secret-nate", portfolio_value=10_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-nate": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"SNDK": -0.10}, {"SNDK": 200.0}, engine)

    assert broker.submitted == [("SNDK", pytest.approx(-10.0))]  # capped at -20%, not -30%


def test_replicate_to_clients_does_not_cap_an_unlevered_client(monkeypatch):
    """
    The cap is new behavior introduced specifically for leverage > 1x -- an
    unlevered (leverage=1, the default for every existing client) target
    weight must size exactly as before even if it's already above
    max_single_position_pct (e.g. one candidate getting the full deployable
    book), since that's an existing, intentional master-account sizing
    decision this module has never second-guessed.
    """
    monkeypatch.setattr(client_fanout.settings, "max_single_position_pct", 0.20)
    row = _make_client_row(1, "opal", margin_enabled=True, leverage_multiplier=1)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-opal", "secret-opal", portfolio_value=10_000.0)
    _install_fake_broker_factory(monkeypatch, {"key-opal": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"TSLA": 0.90}, {"TSLA": 200.0}, engine)

    assert broker.submitted == [("TSLA", pytest.approx(45.0))]  # $10,000 * 90% / $200, uncapped


def test_one_clients_construction_failure_does_not_block_the_next_client(monkeypatch):
    row_a = _make_client_row(1, "eve", margin_enabled=True)
    row_b = _make_client_row(2, "frank", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row_a, row_b]))

    broker_b = _FakeAlpacaBroker("live", True, "key-frank", "secret-frank")

    def factory(mode, confirm_live, api_key, secret_key):
        if api_key == "key-eve":
            raise RuntimeError("revoked API key")
        return broker_b

    monkeypatch.setattr(client_fanout, "AlpacaBroker", factory)

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"AAPL": 0.5}, {"AAPL": 100.0}, engine)

    assert broker_b.submitted == [("AAPL", pytest.approx(50.0))]  # frank still traded despite eve's broker failing


def test_one_symbols_order_failure_does_not_block_the_next_symbol(monkeypatch):
    row = _make_client_row(1, "gina", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-gina", "secret-gina", portfolio_value=10_000.0, fail_symbols={"AAPL"})
    _install_fake_broker_factory(monkeypatch, {"key-gina": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": 100.0, "MSFT": 100.0}, engine)

    assert ("MSFT", pytest.approx(50.0)) in broker.submitted
    assert not any(s == "AAPL" for s, _ in broker.submitted)


def test_a_full_close_needs_no_price(monkeypatch):
    """target_pct == 0.0 must still submit (target_shares=0) even if the
    symbol has no quote this cycle -- a client holding a position the master
    no longer prices must still be able to exit it."""
    row = _make_client_row(1, "henry", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-henry", "secret-henry")
    _install_fake_broker_factory(monkeypatch, {"key-henry": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"OLDPOS": 0.0}, {}, engine)  # no price for OLDPOS at all

    assert broker.submitted == [("OLDPOS", 0.0)]


def test_missing_price_for_a_nonzero_target_is_skipped(monkeypatch):
    row = _make_client_row(1, "ivy", margin_enabled=True)
    monkeypatch.setattr(client_fanout.pd, "read_sql", lambda *a, **k: pd.DataFrame([row]))
    broker = _FakeAlpacaBroker("live", True, "key-ivy", "secret-ivy")
    _install_fake_broker_factory(monkeypatch, {"key-ivy": broker})

    engine = _FakeEngine()
    client_fanout.replicate_to_clients({"NEWPOS": 0.3}, {}, engine)

    assert broker.submitted == []


def test_current_master_weights_computes_signed_percent_of_portfolio():
    class _FakeMasterBroker:
        def get_portfolio_value(self):
            return 100_000.0

        def get_positions_detailed(self):
            return [
                {"symbol": "TSLA", "market_value": 40_000.0},
                {"symbol": "SNDK", "market_value": -20_000.0},
            ]

    weights = client_fanout.current_master_weights(_FakeMasterBroker())
    assert weights == {"TSLA": pytest.approx(0.40), "SNDK": pytest.approx(-0.20)}


def test_current_master_weights_empty_when_portfolio_value_not_positive():
    class _FlatBroker:
        def get_portfolio_value(self):
            return 0.0

        def get_positions_detailed(self):
            raise AssertionError("should never be called when portfolio_value <= 0")

    assert client_fanout.current_master_weights(_FlatBroker()) == {}


def test_onboard_client_is_a_noop_for_a_non_alpaca_master_broker(caplog):
    class _IBKRLikeBroker:
        pass

    client_fanout.onboard_client(client_id=1, master_broker=_IBKRLikeBroker(), engine=_FakeEngine())
    assert "Alpaca master broker" in caplog.text
