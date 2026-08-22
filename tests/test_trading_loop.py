import json

import pandas as pd
import pytest

from execution import trading_loop
from execution.approval_gate import ApprovalOutcome, number_proposals
from models.screener import ScreenResult, TradeCandidate
from monitoring import reasoning
from risk.circuit_breakers import BreakerResult

# Grabbed before the autouse _no_real_db_writes fixture replaces the module
# attribute with a no-op, so tests can still exercise the real function.
_real_log_decisions = trading_loop._log_decisions


def _screen(candidates, scored=None):
    """Wrap a candidate list the way run_screen_with_scores returns it."""
    if scored is None:
        scored = pd.DataFrame(columns=["symbol", "predicted_return"])
    return ScreenResult(candidates=list(candidates or []), scored=scored)


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


@pytest.fixture(autouse=True)
def _no_followups(monkeypatch):
    """The post-approval size confirmation must never reach a real phone from a test."""
    monkeypatch.setattr(trading_loop, "send_followup", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_hold_state_io(monkeypatch):
    """
    Hold-rule state lives in the DB; tests run without one. No prior
    misses by default — tests exercising the miss counter patch
    load_missed_cycles themselves.
    """
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {})
    monkeypatch.setattr(trading_loop, "store_missed_cycles", lambda engine, counts: None)


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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen(screen_called.append(1)))

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
        trading_loop, "run_screen_with_scores",
        lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]),
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([]))

    result = trading_loop.run_cycle("v3", ["AAPL"])

    assert result.status == "no_candidates"
    assert broker.submitted == []


def test_run_cycle_closes_positions_not_in_new_candidate_set(monkeypatch):
    """
    Positions that have been out of the shortlist HOLD_MAX_MISSED_CYCLES
    consecutive cycles get closed. Here OLD1/OLD2 already missed once, so
    this cycle's miss is their second — the exit condition fires.
    (A single miss is a hold — see the hold-rule tests further down.)
    """
    broker = _FakeBroker(positions={"OLD1": 10.0, "OLD2": -5.0, "TSLA": 3.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("TSLA", "long", 0.5)]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"TSLA": 100.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"OLD1": 1, "OLD2": 1})

    result = trading_loop.run_cycle("v3", ["TSLA", "OLD1", "OLD2"])

    submitted_symbols = {s for s, _ in broker.submitted}
    assert submitted_symbols == {"OLD1", "OLD2", "TSLA"}
    old1_order = next(shares for s, shares in broker.submitted if s == "OLD1")
    old2_order = next(shares for s, shares in broker.submitted if s == "OLD2")
    assert old1_order == 0.0
    assert old2_order == 0.0
    assert result.orders_placed == 3


def test_run_cycle_proposals_carry_reasoning_and_close_pnl(monkeypatch):
    """
    The human on the phone must see WHY: open proposals carry the
    screener's reasoning phases, close proposals carry a phase-4 story and
    the position's current P&L when the broker can report it.
    """
    broker = _FakeBroker(positions={"OLD1": 10.0})
    broker.get_positions_detailed = lambda: [
        {"symbol": "OLD1", "qty": 10.0, "unrealized_plpc": 0.042, "unrealized_pl": 421.5}
    ]
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")

    candidate = _candidate("TSLA", "long", 0.5)
    candidate.reasoning = [
        reasoning.phase_signals("trend", [{"feature_name": "mom_ret_5d", "value": 0.05, "contribution": 0.02}]),
        reasoning.phase_forecast(0.02, 0.9, 0.018),
    ]
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([candidate]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"TSLA": 100.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"OLD1": 1})  # this miss is its second -> close proposed

    seen = {}

    def capturing_gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["TSLA", "OLD1"], request_fn=capturing_gate)

    by_symbol = {p.symbol: p for p in seen["proposals"]}
    open_p = by_symbol["TSLA"]
    assert open_p.reasoning is candidate.reasoning
    close_p = by_symbol["OLD1"]
    assert close_p.current_pnl_pct == pytest.approx(0.042)
    assert close_p.current_pnl_usd == pytest.approx(421.5)
    assert close_p.reasoning and close_p.reasoning[0]["phase"] == 4


def test_run_cycle_proposals_survive_a_broker_without_pnl_support(monkeypatch):
    """No get_positions_detailed on the broker — proposals go out without P&L, the gate is not blocked."""
    broker = _FakeBroker(positions={"OLD1": 10.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"OLD1": 1})

    seen = {}

    def capturing_gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["OLD1"], request_fn=capturing_gate)

    (close_p,) = seen["proposals"]
    assert close_p.current_pnl_pct is None
    assert close_p.current_pnl_usd is None


def test_run_cycle_no_candidates_still_closes_positions_whose_exit_fired(monkeypatch):
    """Zero fresh candidates doesn't mute the exit rules: a position at its miss limit is still proposed for closing."""
    broker = _FakeBroker(positions={"OLD1": 10.0})
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"OLD1": 1})

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
        trading_loop, "run_screen_with_scores",
        lambda *a, **k: _screen([_candidate("BAD", "long", 0.1), _candidate("GOOD", "long", 0.05)]),
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]))
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([]))

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
        trading_loop, "run_screen_with_scores",
        lambda *a, **k: _screen([_candidate("YES", "long", 0.1), _candidate("NOPE", "long", 0.1)]),
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("TSLA", "long", 0.5)]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"TSLA": 100.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"KEEP": 1})  # exit fires -> close proposed

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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]))
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"AAPL": 100.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"OLD": 1})  # exit fires -> close proposed
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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]))

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
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([_candidate("AAPL", "long", 0.1)]))

    def fake_prices(symbols):
        order.append("prices")
        return {"AAPL": 100.0}

    def gate(proposals, *, context, **kwargs):
        order.append("gate")
        return _approve_all(proposals, context=context)

    monkeypatch.setattr(trading_loop, "_latest_prices", fake_prices)
    trading_loop.run_cycle("v3", ["AAPL"], request_fn=gate)

    assert order.index("gate") < order.index("prices")


# --- approve first, then size ----------------------------------------------


def _wire_basic_cycle(monkeypatch, broker, candidates, prices):
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen(candidates))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: prices)


def test_open_proposals_go_out_without_sizes(monkeypatch):
    """The human approves WHICH trades happen; sizes don't exist yet at that point."""
    broker = _FakeBroker()
    _wire_basic_cycle(monkeypatch, broker, [_candidate("AAPL", "long", 0.1)], {"AAPL": 100.0})

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["AAPL"], request_fn=gate)

    (open_p,) = seen["proposals"]
    assert open_p.target_position_pct is None


def test_partial_approval_deploys_full_capital_across_only_the_approved(monkeypatch):
    """
    The first-live-run bug: rejecting picks used to leave their capital in
    cash. Now the approved subset absorbs the full deployable capital,
    weighted by conviction (caps raised here so they don't bind).
    """
    broker = _FakeBroker(portfolio_value=100_000.0)
    monkeypatch.setattr(trading_loop.settings, "max_single_position_pct", 0.80)
    candidates = [
        _candidate("BIG", "long", 0.1, pred_return=0.04),   # conviction 0.036
        _candidate("SMALL", "long", 0.1, pred_return=0.02),  # conviction 0.018
        _candidate("NOPE", "long", 0.1, pred_return=0.02),
    ]
    _wire_basic_cycle(monkeypatch, broker, candidates, {"BIG": 100.0, "SMALL": 100.0, "NOPE": 100.0})

    trading_loop.run_cycle("v3", ["BIG", "SMALL", "NOPE"], request_fn=_gate_returning({"BIG", "SMALL"}))

    shares = dict(broker.submitted)
    assert "NOPE" not in shares  # rejected -> no allocation, no order
    # 2:1 conviction ratio over 100% of the book -> ~66.7% and ~33.3%.
    assert shares["BIG"] == pytest.approx(666.67, rel=1e-3)
    assert shares["SMALL"] == pytest.approx(333.33, rel=1e-3)


def test_caps_still_bind_after_approval_and_the_shortfall_is_reported(monkeypatch):
    """2 approved longs x 25% cap = 50% deployed, said out loud — never silently exceeded."""
    broker = _FakeBroker(portfolio_value=100_000.0)
    monkeypatch.setattr(trading_loop.settings, "max_single_position_pct", 0.25)  # pinned: a dev .env can override the default
    candidates = [_candidate("AAA", "long", 0.1), _candidate("BBB", "long", 0.1)]
    _wire_basic_cycle(monkeypatch, broker, candidates, {"AAA": 100.0, "BBB": 100.0})

    followups = []
    monkeypatch.setattr(trading_loop, "send_followup", lambda msg, **k: followups.append(msg))

    trading_loop.run_cycle("v3", ["AAA", "BBB"])

    shares = dict(broker.submitted)
    assert shares["AAA"] == pytest.approx(250.0)  # 25% cap at $100/share
    assert shares["BBB"] == pytest.approx(250.0)
    (msg,) = followups
    assert "50.0%" in msg  # deployed
    assert "not reached" in msg.lower() or "cap" in msg.lower()


def test_followup_confirmation_lists_each_approved_size(monkeypatch):
    broker = _FakeBroker(portfolio_value=100_000.0)
    monkeypatch.setattr(trading_loop.settings, "max_single_position_pct", 1.0)
    _wire_basic_cycle(monkeypatch, broker, [_candidate("AAPL", "long", 0.1)], {"AAPL": 100.0})

    followups = []
    monkeypatch.setattr(trading_loop, "send_followup", lambda msg, **k: followups.append(msg))

    trading_loop.run_cycle("v3", ["AAPL"])

    (msg,) = followups
    assert "AAPL (long): 100.0% of portfolio" in msg


def test_kept_positions_shrink_what_the_approved_picks_can_deploy(monkeypatch):
    """A rejected close stays open and its capital is NOT re-allocated on top."""
    broker = _FakeBroker(positions={"KEEP": 100.0}, portfolio_value=100_000.0)  # 100 sh * $400 = 40% of book
    monkeypatch.setattr(trading_loop.settings, "max_single_position_pct", 1.0)
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"KEEP": 1})  # exit fires -> close proposed (and rejected below)
    _wire_basic_cycle(
        monkeypatch, broker, [_candidate("AAPL", "long", 0.1)], {"AAPL": 100.0, "KEEP": 400.0}
    )

    trading_loop.run_cycle("v3", ["AAPL", "KEEP"], request_fn=_gate_returning({"AAPL"}))

    shares = dict(broker.submitted)
    # 60% deployable (40% stays in KEEP) -> 600 shares at $100.
    assert shares["AAPL"] == pytest.approx(600.0)
    assert "KEEP" not in shares


def test_approved_sizes_land_in_the_decisions_log(monkeypatch):
    """What gets logged is the post-approval allocation, not the pre-gate indication."""
    captured = {}
    broker = _FakeBroker(portfolio_value=100_000.0)
    monkeypatch.setattr(trading_loop.settings, "max_single_position_pct", 1.0)
    _wire_basic_cycle(monkeypatch, broker, [_candidate("AAPL", "long", 0.03)], {"AAPL": 100.0})
    monkeypatch.setattr(trading_loop, "_log_decisions", lambda *a, **k: captured.update(args=a, kwargs=k))

    trading_loop.run_cycle("v3", ["AAPL"])

    (logged_candidate,) = captured["args"][0]
    assert logged_candidate.target_position_pct == pytest.approx(1.0)  # full book, one approved pick


# --- multi-week hold rules (stop the weekly churn) --------------------------


def test_position_that_merely_slips_in_rank_is_held(monkeypatch):
    """The whole point of the hold rules: one missed Monday is not an exit condition."""
    broker = _FakeBroker(positions={"HELD": 10.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    _wire_basic_cycle(monkeypatch, broker, [_candidate("NEW", "long", 0.1)], {"NEW": 100.0, "HELD": 50.0})

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["NEW", "HELD"], request_fn=gate)

    proposed_closes = [p.symbol for p in seen["proposals"] if p.action == "close"]
    assert proposed_closes == []  # HELD slipped in rank but nothing fired
    assert ("HELD", 0.0) not in broker.submitted


def test_position_whose_prediction_flips_sign_is_proposed_for_closing(monkeypatch):
    """A confident prediction against the held side is a real exit condition, even on the first miss."""
    broker = _FakeBroker(positions={"FLIP": 10.0})  # held long
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 99)  # isolate the flip condition
    scored = pd.DataFrame({"symbol": ["FLIP"], "predicted_return": [-0.03]})  # model now says down 3%
    broker2 = broker
    monkeypatch.setattr(trading_loop, "get_broker", lambda: broker2)
    monkeypatch.setattr(trading_loop, "get_engine", lambda: None)
    monkeypatch.setattr(trading_loop, "_run_breaker_check", lambda b, e: [])
    monkeypatch.setattr(trading_loop, "_market_regime", lambda e: "trend")
    monkeypatch.setattr(trading_loop, "run_screen_with_scores", lambda *a, **k: _screen([], scored=scored))
    monkeypatch.setattr(trading_loop, "_latest_prices", lambda symbols: {"FLIP": 100.0})

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["FLIP"], request_fn=gate)

    (close_p,) = seen["proposals"]
    assert close_p.action == "close" and close_p.symbol == "FLIP"
    assert "against the long position" in close_p.reasoning[0]["summary"]
    assert ("FLIP", 0.0) in broker.submitted  # approved -> closed


def test_stop_loss_breach_is_proposed_for_closing(monkeypatch):
    broker = _FakeBroker(positions={"DOWN": 10.0})
    broker.get_positions_detailed = lambda: [
        {"symbol": "DOWN", "qty": 10.0, "unrealized_plpc": -0.12, "unrealized_pl": -1200.0}
    ]
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 99)
    monkeypatch.setattr(trading_loop.settings, "hold_stop_loss_pct", 0.08)
    _wire_basic_cycle(monkeypatch, broker, [], {"DOWN": 50.0})

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["DOWN"], request_fn=gate)

    (close_p,) = seen["proposals"]
    assert "stop loss" in close_p.reasoning[0]["summary"]


def test_take_profit_breach_is_proposed_for_closing(monkeypatch):
    broker = _FakeBroker(positions={"UP": 10.0})
    broker.get_positions_detailed = lambda: [
        {"symbol": "UP", "qty": 10.0, "unrealized_plpc": 0.11, "unrealized_pl": 1100.0}
    ]
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 99)
    monkeypatch.setattr(trading_loop.settings, "hold_take_profit_pct", 0.10)
    _wire_basic_cycle(monkeypatch, broker, [], {"UP": 50.0})

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["UP"], request_fn=gate)

    (close_p,) = seen["proposals"]
    assert "profit target" in close_p.reasoning[0]["summary"]


def test_position_already_closed_by_the_contradiction_monitor_is_not_double_closed(monkeypatch):
    """
    The contradiction monitor closed GONE mid-week, so the broker no longer
    holds it. The weekly cycle must not propose closing it again, and its
    stale miss counter must be dropped from the persisted hold state.
    """
    broker = _FakeBroker(positions={"KEPT": 10.0})  # GONE is absent: the monitor already closed it
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 2)
    _wire_basic_cycle(monkeypatch, broker, [_candidate("KEPT", "long", 0.1)], {"KEPT": 100.0})
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"GONE": 1, "KEPT": 0})

    stored = {}
    monkeypatch.setattr(trading_loop, "store_missed_cycles", lambda engine, counts: stored.update(counts=counts))

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    trading_loop.run_cycle("v3", ["KEPT", "GONE"], request_fn=gate)

    assert all(p.symbol != "GONE" for p in seen["proposals"])  # nothing proposed for a position that no longer exists
    assert "GONE" not in stored["counts"]  # stale state dropped
    assert "KEPT" in stored["counts"]


def test_miss_counter_advances_and_persists_for_held_positions(monkeypatch):
    broker = _FakeBroker(positions={"HELD": 10.0})
    monkeypatch.setattr(trading_loop.settings, "hold_max_missed_cycles", 3)
    _wire_basic_cycle(monkeypatch, broker, [_candidate("NEW", "long", 0.1)], {"NEW": 100.0, "HELD": 50.0})
    monkeypatch.setattr(trading_loop, "load_missed_cycles", lambda engine: {"HELD": 1})

    stored = {}
    monkeypatch.setattr(trading_loop, "store_missed_cycles", lambda engine, counts: stored.update(counts=counts))

    trading_loop.run_cycle("v3", ["NEW", "HELD"])

    assert stored["counts"]["HELD"] == 2  # second consecutive miss recorded, position still held
    assert stored["counts"]["NEW"] == 0  # fresh open starts clean
