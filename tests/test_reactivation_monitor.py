import contextlib

import pytest

from execution import reactivation_monitor as rm


@contextlib.contextmanager
def _free_lock(*a, **k):
    yield True


@contextlib.contextmanager
def _busy_lock(*a, **k):
    yield False


class _FakeClock:
    def __init__(self, is_open: bool):
        self.is_open = is_open


class _FakeClient:
    def __init__(self, is_open: bool = True):
        self._clock = _FakeClock(is_open)

    def get_clock(self):
        return self._clock


class _FakeBroker:
    def __init__(self, mode: str = "paper", is_open: bool = True, portfolio_value: float = 100_000.0):
        self.mode = mode
        self.client = _FakeClient(is_open)
        self._portfolio_value = portfolio_value
        self.flattened = False

    def get_positions(self):
        return {}

    def get_portfolio_value(self):
        return self._portfolio_value

    def flatten_all(self):
        self.flattened = True


@pytest.fixture(autouse=True)
def _lock_free_by_default(monkeypatch):
    """This file is about reactivation's own flow, not the overlapping-run guard (see the concurrency test below)."""
    monkeypatch.setattr(rm, "advisory_lock", _free_lock)


@pytest.fixture(autouse=True)
def _no_breaker_trip_by_default(monkeypatch):
    """This file is about reactivation, not the master-account circuit breakers (see the breaker test below)."""
    monkeypatch.setattr(rm, "_run_breaker_check", lambda broker, engine: [])


@pytest.fixture(autouse=True)
def _no_real_equity_writes(monkeypatch):
    monkeypatch.setattr(rm, "record_equity_snapshot", lambda *a, **k: None)


def test_market_closed_is_a_clean_noop(monkeypatch):
    broker = _FakeBroker(is_open=False)
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: calls.append(1))

    rm.run_reactivation()

    assert calls == []


def test_breaker_trip_flattens_and_skips_reactivation(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())

    class _Trigger:
        reason = "drawdown breach"

    monkeypatch.setattr(rm, "_run_breaker_check", lambda b, e: [_Trigger()])
    flatten_calls = []
    monkeypatch.setattr(rm, "_flatten_and_alert", lambda b, reasons: flatten_calls.append(reasons))
    reactivation_calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: reactivation_calls.append(1))

    rm.run_reactivation()

    assert flatten_calls == ["drawdown breach"]
    assert reactivation_calls == []


def test_normal_pass_attempts_reactivation_with_nothing_excluded(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: calls.append((a, k)))

    rm.run_reactivation()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is broker
    assert kwargs["excluded_symbols"] == set()


def test_equity_snapshot_recorded_on_a_normal_pass(monkeypatch):
    broker = _FakeBroker(portfolio_value=123_456.0)
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(rm, "record_equity_snapshot", lambda value, mode: calls.append((value, mode)))

    rm.run_reactivation()

    assert calls == [(123_456.0, "paper")]


def test_snapshot_failure_does_not_abort_reactivation(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")

    broker = _FakeBroker()
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    monkeypatch.setattr(rm, "record_equity_snapshot", _boom)
    calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: calls.append(1))

    rm.run_reactivation()

    assert calls == [1]


def test_concurrent_pass_is_skipped_not_double_processed(monkeypatch):
    """
    This job's own schedule already gives the retrain a wide window, but a
    run that somehow still spills past its own next scheduled fire must not
    overlap with itself -- same advisory_lock guard contradiction_monitor.py
    uses for its own hourly pass.
    """
    broker = _FakeBroker()
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    monkeypatch.setattr(rm, "advisory_lock", _busy_lock)
    calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: calls.append(1))

    rm.run_reactivation()

    assert calls == []


def test_request_fn_is_threaded_through_to_attempt_reactivation(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(rm, "get_broker", lambda: broker)
    monkeypatch.setattr(rm, "get_engine", lambda: object())
    sentinel = object()
    calls = []
    monkeypatch.setattr(rm, "_attempt_reactivation", lambda *a, **k: calls.append(k.get("request_fn")))

    rm.run_reactivation(request_fn=sentinel)

    assert calls == [sentinel]
