"""
execution/broker.py::get_broker — the BROKER=ibkr/alpaca factory, and the
fast reachability pre-check that guards the ibkr path.
"""
from __future__ import annotations

import socket

import pytest

from execution import broker as broker_module
from execution.broker import get_broker


def _unused_port() -> int:
    """A TCP port on loopback that is (momentarily) free to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_get_broker_raises_a_clear_fast_error_when_nothing_listens_on_ibkr_port(monkeypatch):
    """
    The exact bug this guards against: BROKER left at its 'ibkr' default on
    a host with no TWS/IB Gateway (e.g. Railway) should fail immediately
    and clearly, not burn ~40s of IBKRBroker's 5-retry loop before the
    cycle silently no-ops.
    """
    monkeypatch.setattr(broker_module.settings, "broker", "ibkr")
    monkeypatch.setattr(broker_module.settings, "trading_mode", "paper")
    monkeypatch.setattr(broker_module.settings, "ibkr_host", "127.0.0.1")
    monkeypatch.setattr(broker_module.settings, "ibkr_paper_port", _unused_port())

    with pytest.raises(RuntimeError) as exc_info:
        get_broker()

    message = str(exc_info.value)
    assert "BROKER=ibkr" in message
    assert "127.0.0.1" in message
    assert "BROKER=alpaca" in message


def test_get_broker_precheck_uses_live_port_in_live_mode(monkeypatch):
    monkeypatch.setattr(broker_module.settings, "broker", "ibkr")
    monkeypatch.setattr(broker_module.settings, "ibkr_host", "127.0.0.1")
    monkeypatch.setattr(broker_module.settings, "ibkr_live_port", _unused_port())
    monkeypatch.setattr(broker_module.settings, "ibkr_paper_port", _unused_port())

    with pytest.raises(RuntimeError) as exc_info:
        get_broker(mode="live")

    assert str(broker_module.settings.ibkr_live_port) in str(exc_info.value)


def test_get_broker_precheck_passes_through_to_ibkr_broker_when_something_listens(monkeypatch):
    """
    A pre-check pass doesn't mean IBKRBroker succeeds (it still has to do
    the real IB protocol handshake) — it just means get_broker() doesn't
    fast-fail and instead proceeds to construct IBKRBroker as before.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        monkeypatch.setattr(broker_module.settings, "broker", "ibkr")
        monkeypatch.setattr(broker_module.settings, "trading_mode", "paper")
        monkeypatch.setattr(broker_module.settings, "ibkr_host", "127.0.0.1")
        monkeypatch.setattr(broker_module.settings, "ibkr_paper_port", port)

        constructed = {}

        class _FakeIBKRBroker:
            def __init__(self, mode=None, confirm_live=False):
                constructed["mode"] = mode
                constructed["confirm_live"] = confirm_live

        monkeypatch.setattr(broker_module, "IBKRBroker", _FakeIBKRBroker)

        result = get_broker()

        assert isinstance(result, _FakeIBKRBroker)
        assert constructed == {"mode": None, "confirm_live": False}
    finally:
        listener.close()


def test_get_broker_alpaca_path_never_touches_the_ibkr_precheck(monkeypatch):
    monkeypatch.setattr(broker_module.settings, "broker", "alpaca")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("IBKR reachability pre-check must not run for BROKER=alpaca")

    monkeypatch.setattr(broker_module, "_ibkr_reachable", _fail_if_called)

    class _FakeAlpacaBroker:
        def __init__(self, mode=None, confirm_live=False):
            pass

    monkeypatch.setattr(broker_module, "AlpacaBroker", _FakeAlpacaBroker)

    result = get_broker()

    assert isinstance(result, _FakeAlpacaBroker)


def test_get_broker_rejects_unknown_broker_name(monkeypatch):
    monkeypatch.setattr(broker_module.settings, "broker", "not-a-real-broker")

    with pytest.raises(ValueError, match="BROKER must be"):
        get_broker()
