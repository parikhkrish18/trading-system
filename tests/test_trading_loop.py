import pytest

from execution import trading_loop
from models.screener import TradeCandidate
from risk.circuit_breakers import BreakerResult


class _FakeBroker:
    def __init__(self, mode="paper", positions=None, submit_error_for=None, portfolio_value=100_000.0):
        self.mode = mode
        self._positions = dict(positions or {})
        self._portfolio_value = portfolio_value
        self._submit_error_for = submit_error_for or set()
        self.submitted: list[tuple[str, float]] = []
        self.flattened = False

    def get_positions(self):
        return dict(self._positions)

    def get_portfolio_value(self):
        return self._portfolio_value

    def submit_target_position(self, symbol, target_shares):
        self.submitted.append((symbol, target_shares))
        if symbol in self._submit_error_for:
            raise RuntimeError(f"order rejected for {symbol}")
        self._positions[symbol] = target_shares
        return {"symbol": symbol, "qty": target_shares}

    def flatten_all(self):
        self.flattened = True
        self._positions = {}


def _candidate(symbol, side, target_pct, pred_return=0.02, agreement=0.9):
    return TradeCandidate(
        symbol=symbol,
        side=side,
        predicted_return=pred_return,
        direction_agreement=agreement,
        conviction_score=abs(pred_return) * agreement,
        target_position_pct=target_pct,
    )


@pytest.fixture(autouse=True)
def _quiet_alerts(monkeypatch):
    monkeypatch.setattr(trading_loop, "send_slack_alert", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_db_writes(monkeypatch):
    monkeypatch.setattr(trading_loop, "_log_decisions", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_equity_writes(monkeypatch):
    monkeypatch.setattr(trading_loop, "record_equity_snapshot", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(trading_loop.time, "sleep", lambda s: None)


def test_run_cycle_flattens_and_skips_trading_on_pretrade_breaker(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [BreakerResult(True, "drawdown breach")])

    screen_called = []
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: screen_called.append(1))

    result = trading_loop.run_cycle("v3", ["AAPL"])

    assert result.status == "flattened_pre_trade"
    assert broker.flattened
    assert screen_called == []  # never got to screening


def test_run_cycle_dry_run_never_touches_broker(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(
        trading_loop, "run_screen",
        lambda *a, **k: [_candidate("AAPL", "long", 0.1)],
    )

    result = trading_loop.run_cycle("v3", ["AAPL"], dry_run=True)

    assert result.status == "dry_run"
    assert result.candidates_screened == 1
    assert broker.submitted == []


def test_run_cycle_no_candidates_returns_early(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [])

    result = trading_loop.run_cycle("v3", ["AAPL"])

    assert result.status == "no_candidates"
    assert broker.submitted == []


def test_run_cycle_isolates_per_symbol_order_failures(monkeypatch):
    broker = _FakeBroker(submit_error_for={"BAD"})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(
        trading_loop, "run_screen",
        lambda *a, **k: [_candidate("BAD", "long", 0.1), _candidate("GOOD", "long", 0.05)],
    )
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"BAD": 100.0, "GOOD": 50.0})

    result = trading_loop.run_cycle("v3", ["BAD", "GOOD"])

    # BAD's order raised, GOOD's still got submitted — one bad symbol doesn't kill the cycle.
    assert result.status == "traded"
    assert result.orders_placed == 1
    submitted_symbols = [s for s, _ in broker.submitted]
    assert "BAD" in submitted_symbols
    assert "GOOD" in submitted_symbols


def test_run_cycle_flattens_on_posttrade_breaker(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)

    check_calls = []

    def fake_check(b, e):
        check_calls.append(1)
        # First call (pre-trade) clean, second call (post-trade) triggers.
        return [] if len(check_calls) == 1 else [BreakerResult(True, "single position breach")]

    monkeypatch.setattr(trading_loop, "_run_breaker_check", fake_check)
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("AAPL", "long", 0.1)])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"AAPL": 100.0})

    result = trading_loop.run_cycle("v3", ["AAPL"])

    assert result.status == "flattened_post_trade"
    assert broker.flattened


def test_run_cycle_never_passes_confirm_live(monkeypatch):
    """The one hard safety invariant: this module can never fire a live order."""
    captured = {}

    def fake_get_broker(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeBroker()

    monkeypatch.setattr(trading_loop, "get_broker", fake_get_broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [])

    trading_loop.run_cycle("v3", ["AAPL"])

    # get_broker() is called with zero arguments — meaning it relies entirely
    # on its own default confirm_live=False. There's no code path here that
    # could pass confirm_live=True even by accident.
    assert captured["args"] == ()
    assert captured["kwargs"] == {}
