import json

import pandas as pd
import pytest

from execution import trading_loop
from execution.approval_gate import ApprovalOutcome, number_proposals
from models.screener import TradeCandidate
from monitoring import reasoning
from risk.circuit_breakers import BreakerResult

# Grabbed before the autouse _no_real_db_writes fixture replaces the module
# attribute with a no-op, so tests can still exercise the real function.
_real_log_decisions = trading_loop._log_decisions


def _approve_all(proposals, *, context, **kwargs):
    """The gate with the human removed — what most tests want in the way."""
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        list(ordered), [], status="auto", statuses={p.index: "auto" for p in ordered}
    )


def _reject_all(proposals, *, context, **kwargs):
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        [], list(ordered), status="replied", statuses={p.index: "rejected" for p in ordered}
    )


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


@pytest.fixture(autouse=True)
def _gate_wide_open(monkeypatch):
    """
    Pre-existing tests exercise the engine, not the gate — give them a gate
    that approves everything. Gate-specific tests pass their own request_fn.
    """
    monkeypatch.setattr(trading_loop, "request_approval", _approve_all)


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


# --- the human gate at the weekly seam ------------------------------------


def _gate_returning(approved_symbols=()):
    """A request_fn whose verdict is fixed: approve listed symbols, reject the rest."""

    def gate(proposals, *, context, **kwargs):
        ordered = number_proposals(list(proposals))
        approved = [p for p in ordered if p.symbol in approved_symbols]
        rejected = [p for p in ordered if p.symbol not in approved_symbols]
        statuses = {
            p.index: ("approved" if p.symbol in approved_symbols else "rejected") for p in ordered
        }
        return ApprovalOutcome(approved, rejected, status="replied", statuses=statuses)

    return gate


def _timeout_gate(proposals, *, context, **kwargs):
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        [], list(ordered), status="timeout", statuses={p.index: "timeout" for p in ordered}
    )


def test_run_cycle_trades_only_the_approved_subset(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(
        trading_loop, "run_screen",
        lambda *a, **k: [_candidate("YES", "long", 0.1), _candidate("NOPE", "long", 0.1)],
    )
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"YES": 100.0, "NOPE": 100.0})

    result = trading_loop.run_cycle("v3", ["YES", "NOPE"], request_fn=_gate_returning({"YES"}))

    assert result.status == "traded"
    assert [s for s, _ in broker.submitted] == ["YES"]


def test_run_cycle_gates_closes_too(monkeypatch):
    """A rejected close means the position STAYS OPEN — nothing is submitted for it."""
    broker = _FakeBroker(positions={"KEEP": 10.0, "TSLA": 3.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("TSLA", "long", 0.5)])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"TSLA": 100.0})

    # Approve the TSLA open; reject closing KEEP.
    trading_loop.run_cycle("v3", ["TSLA", "KEEP"], request_fn=_gate_returning({"TSLA"}))

    submitted_symbols = [s for s, _ in broker.submitted]
    assert "KEEP" not in submitted_symbols  # the human said keep it
    assert "TSLA" in submitted_symbols
    assert broker.get_positions().get("KEEP") == 10.0


def test_run_cycle_timeout_rejects_everything_and_places_no_orders(monkeypatch):
    broker = _FakeBroker(positions={"OLD": 5.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("AAPL", "long", 0.1)])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"AAPL": 100.0})

    result = trading_loop.run_cycle("v3", ["AAPL", "OLD"], request_fn=_timeout_gate)

    assert result.status == "traded"  # the cycle completed; it just did nothing
    assert result.orders_placed == 0
    assert broker.submitted == []


def test_run_cycle_logs_rejected_proposals_with_their_status(monkeypatch):
    captured = {}
    broker = _FakeBroker(positions={"OLD": 5.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("AAPL", "long", 0.1)])
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"AAPL": 100.0})
    monkeypatch.setattr(trading_loop, "_log_decisions", lambda *a, **k: captured.update(kwargs=k, args=a))

    trading_loop.run_cycle("v3", ["AAPL", "OLD"], request_fn=_timeout_gate)

    assert [c.symbol for c in captured["kwargs"]["rejected_candidates"]] == ["AAPL"]
    assert captured["kwargs"]["rejected_close_symbols"] == ["OLD"]
    statuses = captured["kwargs"]["approval_status_by_symbol"]
    assert statuses == {"AAPL": "timeout", "OLD": "timeout"}
    assert captured["args"][0] == []  # no approved candidates
    assert captured["args"][1] == []  # no approved closes


def test_log_decisions_writes_rejected_rows_with_zero_executed_position(monkeypatch):
    captured = {}
    monkeypatch.setattr(trading_loop, "get_engine", lambda: object())
    monkeypatch.setattr(
        pd.DataFrame, "to_sql", lambda self, *a, **k: captured.setdefault("rows", self.to_dict("records"))
    )

    rejected = _candidate("NOPE", "long", 0.2, agreement=0.85)
    phase1 = reasoning.phase_pretrade_risk([])

    _real_log_decisions(
        candidates=[],
        closing_symbols=[],
        executed={},
        intended_shares={},
        feature_set_id="v3",
        mode="paper",
        regime="trend",
        phase1=phase1,
        phase6_by_symbol={},
        order_type="market",
        rejected_candidates=[rejected],
        rejected_close_symbols=["KEEP"],
        approval_status_by_symbol={"NOPE": "rejected", "KEEP": "timeout"},
    )

    rows = {r["symbol"]: r for r in captured["rows"]}
    assert rows["NOPE"]["executed_position"] == 0.0
    assert rows["NOPE"]["approval_status"] == "rejected"
    assert rows["NOPE"]["direction_agreement"] == 0.85
    assert rows["KEEP"]["executed_position"] == 0.0
    assert rows["KEEP"]["approval_status"] == "timeout"
    nope_phase5 = next(p for p in json.loads(rows["NOPE"]["reasoning"]) if p["phase"] == 5)
    assert "no order" in nope_phase5["summary"].lower() or "rejected" in nope_phase5["summary"].lower()


def test_log_decisions_writes_direction_agreement_and_status_for_approved_rows(monkeypatch):
    captured = {}
    monkeypatch.setattr(trading_loop, "get_engine", lambda: object())
    monkeypatch.setattr(
        pd.DataFrame, "to_sql", lambda self, *a, **k: captured.setdefault("rows", self.to_dict("records"))
    )

    approved = _candidate("AAPL", "long", 0.1, agreement=0.9)
    phase1 = reasoning.phase_pretrade_risk([])

    _real_log_decisions(
        candidates=[approved],
        closing_symbols=[],
        executed={"AAPL": 10.0},
        intended_shares={"AAPL": 10.0},
        feature_set_id="v3",
        mode="paper",
        regime="trend",
        phase1=phase1,
        phase6_by_symbol={},
        order_type="market",
        approval_status_by_symbol={"AAPL": "approved"},
    )

    row = captured["rows"][0]
    assert row["direction_agreement"] == 0.9
    assert row["approval_status"] == "approved"


def test_run_cycle_dry_run_never_asks_anyone(monkeypatch):
    """--dry-run returns before the gate — no Telegram message, no approval poll."""
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("AAPL", "long", 0.1)])

    def _never(*a, **k):
        raise AssertionError("dry run must never reach the approval gate")

    result = trading_loop.run_cycle("v3", ["AAPL"], dry_run=True, request_fn=_never)

    assert result.status == "dry_run"
    assert broker.submitted == []


def test_run_cycle_fetches_prices_after_the_gate_not_before(monkeypatch):
    """Sizing must use quotes from after the (possibly long) human wait."""
    order = []
    broker = _FakeBroker()
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen", lambda *a, **k: [_candidate("AAPL", "long", 0.1)])

    def fake_prices(symbols):
        order.append("prices")
        return {"AAPL": 100.0}

    def gate(proposals, *, context, **kwargs):
        order.append("gate")
        return _approve_all(proposals, context=context)

    monkeypatch.setattr(trading_loop, "_latest_prices", fake_prices)
    trading_loop.run_cycle("v3", ["AAPL"], request_fn=gate)

    assert order.index("gate") < order.index("prices")
