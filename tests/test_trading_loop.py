import json

import pandas as pd
import pytest

from execution import trading_loop
from models.screener import TradeCandidate
from monitoring import reasoning
from risk.circuit_breakers import BreakerResult

# Grabbed before the autouse _no_real_db_writes fixture replaces the module
# attribute with a no-op, so tests can still exercise the real function.
_real_log_decisions = trading_loop._log_decisions


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


def test_log_decisions_builds_full_phase_reasoning_for_candidates_and_closures(monkeypatch):
    """
    Exercises the real _log_decisions (not the autouse no-op mock) to check
    that candidates get phases 1-7 merged in, and closed-out positions —
    previously not logged as decisions at all — get a phase-4 explanation
    of why they weren't picked, plus phases 1/5/6/7.
    """
    captured = {}
    monkeypatch.setattr(trading_loop, "get_engine", lambda: object())
    monkeypatch.setattr(
        pd.DataFrame, "to_sql", lambda self, *a, **k: captured.setdefault("rows", self.to_dict("records"))
    )

    candidate = TradeCandidate(
        symbol="AAPL", side="long", predicted_return=0.03, direction_agreement=0.9,
        conviction_score=0.027, target_position_pct=0.6,
        reasoning=[
            reasoning.phase_signals("trend", [{"feature_name": "mom_ret_5d", "value": 0.05, "contribution": 0.02}]),
            reasoning.phase_forecast(0.03, 0.9, 0.027),
            reasoning.phase_selection("AAPL", "long", 0.6, n_confident=2, n_selected=2, max_leg_pct=0.7, min_leg_pct=0.3),
        ],
    )
    phase1 = reasoning.phase_pretrade_risk([])

    _real_log_decisions(
        candidates=[candidate],
        closing_symbols=["OLD1"],
        executed={"AAPL": 10.0, "OLD1": 0.0},
        intended_shares={"AAPL": 10.0, "OLD1": 0.0},
        feature_set_id="v3",
        mode="paper",
        regime="trend",
        phase1=phase1,
        phase6_by_symbol={},
        order_type="market",
    )

    rows = {r["symbol"]: r for r in captured["rows"]}
    aapl_phases = [p["phase"] for p in json.loads(rows["AAPL"]["reasoning"])]
    old1_phases = [p["phase"] for p in json.loads(rows["OLD1"]["reasoning"])]
    assert aapl_phases == [1, 2, 3, 4, 5, 6, 7]
    assert old1_phases == [1, 4, 5, 6, 7]  # no 2/3 -- no fresh forecast for a symbol that wasn't screened as a pick


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


def test_run_cycle_closes_positions_not_in_new_candidate_set(monkeypatch):
    """
    The core rebalance fix: switching from a top-10 diversified book to a
    top-2 concentrated one means everything not in this cycle's shortlist
    must actually get closed, not just left open from a prior cycle.
    """
    broker = _FakeBroker(positions={"OLD1": 10.0, "OLD2": -5.0, "TSLA": 3.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("TSLA", "long", 0.5)])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"TSLA": 100.0})

    result = trading_loop.run_cycle("v3", ["TSLA", "OLD1", "OLD2"])

    submitted_symbols = {s for s, _ in broker.submitted}
    assert submitted_symbols == {"OLD1", "OLD2", "TSLA"}
    old1_order = next(shares for s, shares in broker.submitted if s == "OLD1")
    old2_order = next(shares for s, shares in broker.submitted if s == "OLD2")
    assert old1_order == 0.0
    assert old2_order == 0.0
    assert result.orders_placed == 3


def test_run_cycle_no_candidates_still_closes_stale_positions(monkeypatch):
    """Zero confidence this cycle should mean fully in cash, not 'leave whatever was open.'"""
    broker = _FakeBroker(positions={"OLD1": 10.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {})

    result = trading_loop.run_cycle("v3", ["OLD1"])

    assert result.status == "traded"
    assert broker.submitted == [("OLD1", 0.0)]


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
