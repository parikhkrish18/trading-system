import pandas as pd
import pytest

from execution import contradiction_monitor as cm
from execution.approval_gate import ApprovalOutcome, number_proposals


def _past_the_brake() -> float:
    """
    A move comfortably past the mid-week brake, whatever it is currently set
    to. Derived rather than hardcoded: these tests are about what happens
    once the brake trips, not about the size of the trip, and a literal here
    silently stops testing anything the next time the threshold is retuned.
    """
    return cm.settings.contradiction_momentum_pct + 0.05


def _approve_all(proposals, *, context, **kwargs):
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        list(ordered), [], status="auto", statuses={p.index: "auto" for p in ordered}
    )


def _reject_all(proposals, *, context, **kwargs):
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        [], list(ordered), status="replied", statuses={p.index: "rejected" for p in ordered}
    )


class _FakeClock:
    def __init__(self, is_open: bool):
        self.is_open = is_open


class _FakeClient:
    def __init__(self, is_open: bool = True):
        self._clock = _FakeClock(is_open)

    def get_clock(self):
        return self._clock


class _FakeBroker:
    def __init__(self, positions: dict[str, float], mode: str = "paper", is_open: bool = True, portfolio_value: float = 100_000.0):
        self._positions = dict(positions)
        self.mode = mode
        self.client = _FakeClient(is_open)
        self.closed: list[str] = []
        self._portfolio_value = portfolio_value

    def get_positions(self):
        return dict(self._positions)

    def get_portfolio_value(self):
        return self._portfolio_value

    def submit_target_position(self, symbol, target_shares):
        self.closed.append(symbol)
        self._positions[symbol] = target_shares
        return {"symbol": symbol, "qty": target_shares}


def _prices_df(symbol: str, closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2026-07-01", periods=len(closes))
    return pd.DataFrame({"ts": idx, "close": closes, "symbol": symbol})


@pytest.fixture(autouse=True)
def _no_slack(monkeypatch):
    monkeypatch.setattr(cm, "send_slack_alert", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _gate_wide_open(monkeypatch):
    """
    Pre-existing tests exercise detection and execution, not the gate — give
    them a gate that approves everything. Gate tests pass their own request_fn.
    """
    monkeypatch.setattr(cm, "request_approval", _approve_all)


@pytest.fixture(autouse=True)
def _no_followups(monkeypatch):
    """The post-approval size confirmation must never reach a real phone from a test."""
    monkeypatch.setattr(cm, "send_followup", lambda *a, **k: None)


def test_market_closed_is_a_clean_noop(monkeypatch):
    broker = _FakeBroker({"AAPL": 10}, is_open=False)
    monkeypatch.setattr(cm, "get_broker", lambda: broker)

    results = cm.run_contradiction_check()

    assert results == []
    assert broker.closed == []


def test_no_open_positions_is_a_clean_noop(monkeypatch):
    broker = _FakeBroker({})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())

    results = cm.run_contradiction_check()

    assert results == []


def test_negative_sentiment_closes_a_long_position(monkeypatch):
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (-0.8, 5))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: None)
    monkeypatch.setattr(cm, "_log_closure", lambda *a, **k: None)
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: None)

    results = cm.run_contradiction_check()

    assert broker.closed == ["AAPL"]
    assert results[0].closed
    assert results[0].reasons[0]["signal"] == "news_sentiment"


def test_reversed_momentum_closes_a_short_position(monkeypatch):
    broker = _FakeBroker({"TSLA": -20})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (None, 0))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: _past_the_brake())  # contradicts a short
    monkeypatch.setattr(cm, "_log_closure", lambda *a, **k: None)
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: None)

    results = cm.run_contradiction_check()

    assert broker.closed == ["TSLA"]
    assert results[0].reasons[0]["signal"] == "price_momentum"


def test_closure_triggers_reactivation_attempt(monkeypatch):
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (-0.8, 5))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: None)
    monkeypatch.setattr(cm, "_log_closure", lambda *a, **k: None)

    reactivation_calls = []
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: reactivation_calls.append(1))

    cm.run_contradiction_check()

    assert reactivation_calls == [1]


def test_no_closure_does_not_trigger_reactivation(monkeypatch):
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (0.5, 5))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: 0.02)

    reactivation_calls = []
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: reactivation_calls.append(1))

    cm.run_contradiction_check()

    assert reactivation_calls == []


def test_agreeing_signals_leave_the_position_open(monkeypatch):
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (0.5, 5))  # agrees with long
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: 0.02)  # agrees with long

    results = cm.run_contradiction_check()

    assert broker.closed == []
    assert not results[0].closed


def test_sparse_news_does_not_trigger_even_with_strong_sentiment(monkeypatch):
    """A single stray headline shouldn't be enough to close a position — need _MIN_NEWS_COUNT."""
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (-0.9, 1))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: None)

    cm.run_contradiction_check()

    assert broker.closed == []


def test_recent_momentum_computed_from_price_history(monkeypatch):
    prices = _prices_df("AAPL", [100, 101, 102, 103, 104, 90])  # sharp drop on the last bar
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: prices)

    momentum = cm._recent_momentum(engine=object(), symbol="AAPL")

    assert momentum < -0.09


def test_freed_capital_fraction_with_no_positions_is_fully_idle():
    broker = _FakeBroker({}, portfolio_value=100_000.0)
    fraction = cm._freed_capital_fraction(broker, engine=object())
    assert fraction == 1.0


def test_freed_capital_fraction_computed_from_held_value(monkeypatch):
    broker = _FakeBroker({"AAPL": 100}, portfolio_value=100_000.0)  # 100 sh * $200 = $20,000 held = 20%
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL"], "close": [200.0]}))

    fraction = cm._freed_capital_fraction(broker, engine=object())

    assert fraction == pytest.approx(0.80)


def test_attempt_reactivation_skips_when_freed_fraction_too_small(monkeypatch):
    broker = _FakeBroker({"AAPL": 495}, portfolio_value=100_000.0)  # ~99% held -> ~1% freed, below the 5% floor
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL"], "close": [200.0]}))
    run_screen_calls = []
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: run_screen_calls.append(1))

    cm._attempt_reactivation(broker, engine=object())

    assert run_screen_calls == []


def test_attempt_reactivation_opens_a_confident_candidate(monkeypatch):
    from models.screener import TradeCandidate

    broker = _FakeBroker({}, portfolio_value=100_000.0)  # nothing held -> 100% freed
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL"], "close": [50.0]}))

    candidate = TradeCandidate(
        symbol="AAPL", side="long", predicted_return=0.03, direction_agreement=0.9,
        conviction_score=0.027, target_position_pct=0.6,
        reasoning=[{"phase": 4, "title": "Candidate Selection & Sizing", "summary": "AAPL: long, 60.0% of capital.", "lines": ["some line"]}],
    )
    run_screen_calls = []
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: (run_screen_calls.append(k), [candidate])[1])
    monkeypatch.setattr(cm, "_log_reactivation", lambda *a, **k: None)

    cm._attempt_reactivation(broker, engine=object())

    assert broker.closed == ["AAPL"]  # submit_target_position was called for AAPL
    assert run_screen_calls[0]["total_deploy_pct"] == pytest.approx(1.0)


def test_attempt_reactivation_excludes_currently_held_symbols(monkeypatch):
    broker = _FakeBroker({"MSFT": 10}, portfolio_value=100_000.0)
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["MSFT"], "close": [10.0]}))  # tiny value -> big freed fraction

    captured_pool = {}

    def fake_run_screen(feature_set_id, symbols, **kwargs):
        captured_pool["symbols"] = symbols
        return []

    monkeypatch.setattr(cm, "run_screen", fake_run_screen)

    cm._attempt_reactivation(broker, engine=object())

    assert "MSFT" not in captured_pool["symbols"]
    assert "AAPL" in captured_pool["symbols"]


def test_attempt_reactivation_noop_when_no_confident_candidate(monkeypatch):
    broker = _FakeBroker({}, portfolio_value=100_000.0)
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": [], "close": []}))
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: [])

    cm._attempt_reactivation(broker, engine=object())

    assert broker.closed == []


def test_log_closure_builds_valid_phase_reasoning(monkeypatch):
    captured = {}

    class _FakeDF:
        def to_sql(self, *a, **k):
            captured["rows"] = a, k

    monkeypatch.setattr(cm.pd, "DataFrame", lambda rows: (captured.setdefault("raw_rows", rows), _FakeDF())[1])
    monkeypatch.setattr(cm, "get_engine", lambda: object())

    result = cm.ContradictionResult(
        symbol="AAPL",
        side="long",
        closed=True,
        reasons=[{"signal": "news_sentiment", "value": -0.8, "news_count": 5, "detail": "sentiment turned negative"}],
    )
    cm._log_closure(result, mode="paper", executed_position=0.0)

    row = captured["raw_rows"][0]
    import json

    phases = json.loads(row["reasoning"])
    assert [p["phase"] for p in phases] == [2, 4, 5, 7]
    assert "sentiment turned negative" in phases[0]["lines"]


# --- the human gate on mid-week closes and reactivations ------------------


def test_contradiction_closes_are_batched_into_one_gate_call(monkeypatch):
    broker = _FakeBroker({"AAPL": 10, "TSLA": -20})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (-0.8, 5) if symbol == "AAPL" else (None, 0))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: _past_the_brake() if symbol == "TSLA" else None)
    monkeypatch.setattr(cm, "_log_closure", lambda *a, **k: None)
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: None)

    gate_calls = []

    def gate(proposals, *, context, **kwargs):
        gate_calls.append({"context": context, "symbols": [p.symbol for p in proposals], "reasons": [p.reason for p in proposals]})
        return _approve_all(proposals, context=context)

    cm.run_contradiction_check(request_fn=gate)

    assert len(gate_calls) == 1  # ONE message for the whole batch, not one per position
    assert set(gate_calls[0]["symbols"]) == {"AAPL", "TSLA"}
    assert gate_calls[0]["reasons"] == ["contradiction", "contradiction"]
    assert set(broker.closed) == {"AAPL", "TSLA"}


def test_rejected_close_keeps_the_position_and_logs_the_flag(monkeypatch):
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())
    monkeypatch.setattr(cm, "ingest_news", lambda *a, **k: None)
    monkeypatch.setattr(cm, "backfill_unscored_news", lambda *a, **k: 0)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (-0.8, 5))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: None)

    logged = []
    monkeypatch.setattr(cm, "_log_rejected_closure", lambda result, mode, status: logged.append((result.symbol, status)))
    monkeypatch.setattr(cm, "_log_closure", lambda *a, **k: pytest.fail("a rejected close must not log as a closure"))

    alerts = []
    monkeypatch.setattr(cm, "send_slack_alert", lambda msg, severity="warning": alerts.append(msg))

    reactivations = []
    monkeypatch.setattr(cm, "_attempt_reactivation", lambda *a, **k: reactivations.append(1))

    results = cm.run_contradiction_check(request_fn=_reject_all)

    assert broker.closed == []  # nothing submitted
    assert logged == [("AAPL", "rejected")]
    assert any("flagged" in m and "NOT closed" in m for m in alerts)
    assert not results[0].closed  # the record reflects what actually happened
    assert reactivations == []  # nothing closed -> no capital freed -> no re-screen


def test_reactivation_opens_are_gated_with_their_own_proposal(monkeypatch):
    from models.screener import TradeCandidate

    broker = _FakeBroker({}, portfolio_value=100_000.0)
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL"], "close": [50.0]}))

    candidate = TradeCandidate(
        symbol="AAPL", side="long", predicted_return=0.03, direction_agreement=0.9,
        conviction_score=0.027, target_position_pct=0.6,
    )
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: [candidate])
    monkeypatch.setattr(cm, "_log_reactivation", lambda *a, **k: None)

    gate_calls = []

    def gate(proposals, *, context, **kwargs):
        gate_calls.append({"context": context, "reasons": [p.reason for p in proposals]})
        return _approve_all(proposals, context=context)

    cm._attempt_reactivation(broker, engine=object(), request_fn=gate)

    assert gate_calls[0]["reasons"] == ["reactivation"]
    assert "reactivation" in gate_calls[0]["context"]
    assert broker.closed == ["AAPL"]  # the open went through


def test_rejected_reactivation_stays_in_cash_and_is_logged(monkeypatch):
    from models.screener import TradeCandidate

    broker = _FakeBroker({}, portfolio_value=100_000.0)
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL"], "close": [50.0]}))

    candidate = TradeCandidate(
        symbol="AAPL", side="long", predicted_return=0.03, direction_agreement=0.9,
        conviction_score=0.027, target_position_pct=0.6,
    )
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: [candidate])

    logged = []
    monkeypatch.setattr(cm, "_log_rejected_reactivation", lambda c, mode, status: logged.append((c.symbol, status)))
    monkeypatch.setattr(cm, "_log_reactivation", lambda *a, **k: pytest.fail("a rejected reactivation must not log as opened"))

    cm._attempt_reactivation(broker, engine=object(), request_fn=_reject_all)

    assert broker.closed == []  # no order submitted
    assert logged == [("AAPL", "rejected")]


def test_reactivation_still_scopes_the_screen_to_freed_capital_only(monkeypatch):
    """Gating must not break the freed-capital-only redeployment contract."""
    broker = _FakeBroker({"MSFT": 100}, portfolio_value=100_000.0)  # $20k held -> 80% freed
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["MSFT"], "close": [200.0]}))

    captured = {}

    def fake_run_screen(feature_set_id, symbols, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cm, "run_screen", fake_run_screen)

    cm._attempt_reactivation(broker, engine=object(), request_fn=_approve_all)

    assert captured["total_deploy_pct"] == pytest.approx(0.80)


def test_news_refresh_failure_does_not_abort_the_check(monkeypatch):
    """If Polygon/Anthropic is down, still check against whatever sentiment is already in the DB."""
    broker = _FakeBroker({"AAPL": 10})
    monkeypatch.setattr(cm, "get_broker", lambda: broker)
    monkeypatch.setattr(cm, "get_engine", lambda: object())

    def _boom(*a, **k):
        raise RuntimeError("polygon is down")

    monkeypatch.setattr(cm, "ingest_news", _boom)
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (None, 0))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: None)

    results = cm.run_contradiction_check()

    assert not results[0].closed


def test_reactivation_proposals_carry_no_size_and_allocation_follows_approval(monkeypatch):
    """
    Approve-first on the reactivation path too: the proposal is size-less,
    and the freed capital is distributed across the approved subset by
    conviction after the human answers.
    """
    from models.screener import TradeCandidate

    broker = _FakeBroker({}, portfolio_value=100_000.0)  # nothing held -> 100% freed
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["AAPL", "MSFT"], "close": [50.0, 100.0]}))
    monkeypatch.setattr(cm.settings, "max_single_position_pct", 0.80)

    big = TradeCandidate(symbol="AAPL", side="long", predicted_return=0.04, direction_agreement=0.9,
                         conviction_score=0.036, target_position_pct=0.5)
    small = TradeCandidate(symbol="MSFT", side="long", predicted_return=0.02, direction_agreement=0.9,
                           conviction_score=0.018, target_position_pct=0.5)
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: [big, small])
    monkeypatch.setattr(cm, "_log_reactivation", lambda *a, **k: None)

    seen = {}

    def gate(proposals, *, context, **kwargs):
        seen["proposals"] = list(proposals)
        return _approve_all(proposals, context=context)

    cm._attempt_reactivation(broker, engine=object(), request_fn=gate)

    assert all(p.target_position_pct is None for p in seen["proposals"])
    # 2:1 conviction split of the full freed book: AAPL 66.7% (cap 80% doesn't bind), MSFT 33.3%.
    assert big.target_position_pct == pytest.approx(2 / 3, rel=1e-3)
    assert small.target_position_pct == pytest.approx(1 / 3, rel=1e-3)


def test_reactivation_allocates_only_the_freed_fraction(monkeypatch):
    """80% freed -> the approved pick gets 80% of the book (under a loose cap), not 100%."""
    from models.screener import TradeCandidate

    broker = _FakeBroker({"MSFT": 100}, portfolio_value=100_000.0)  # $20k held -> 80% freed
    monkeypatch.setattr(cm, "load_active_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cm.pd, "read_sql", lambda *a, **k: pd.DataFrame({"symbol": ["MSFT", "AAPL"], "close": [200.0, 100.0]}))
    monkeypatch.setattr(cm.settings, "max_single_position_pct", 1.0)

    candidate = TradeCandidate(symbol="AAPL", side="long", predicted_return=0.03, direction_agreement=0.9,
                               conviction_score=0.027, target_position_pct=0.6)
    monkeypatch.setattr(cm, "run_screen", lambda *a, **k: [candidate])
    monkeypatch.setattr(cm, "_log_reactivation", lambda *a, **k: None)

    cm._attempt_reactivation(broker, engine=object(), request_fn=_approve_all)

    assert candidate.target_position_pct == pytest.approx(0.80)
    # 80% of $100k at $100/share = 800 shares.
    submitted = {s: broker._positions[s] for s in broker.closed}
    assert submitted["AAPL"] == pytest.approx(800.0)


# --------------------------------------------------------------------------
# The mid-week brake is a brake, not a second opinion
# --------------------------------------------------------------------------


def _momentum_triggers(monkeypatch, move: float) -> bool:
    """Does a `move` against a long position trip the mid-week close?"""
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (None, 0))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: move)
    return cm._check_position(engine=object(), symbol="AAPL", qty=10).closed


def test_ordinary_weekly_movement_does_not_trip_the_brake(monkeypatch):
    """
    A typical S&P 500 name moves ~4% in a week just existing. The brake used
    to sit at exactly that, so it fired on noise — hourly — against a thesis
    that needs ~20 trading days to play out, reintroducing the churn the
    hold rules exist to prevent. Normal movement must now pass unremarked.
    """
    assert _momentum_triggers(monkeypatch, -0.04) is False
    assert _momentum_triggers(monkeypatch, -0.06) is False


def test_a_collapse_still_trips_the_brake(monkeypatch):
    """
    The brake's whole purpose: a position falling apart on a Tuesday must
    not wait until Monday to be noticed.
    """
    assert _momentum_triggers(monkeypatch, -0.15) is True


def test_the_brake_sits_well_clear_of_weekly_noise():
    """
    Pins the intent rather than the number. Weekly movement for a typical
    name is ~4%; the brake must stay far enough above that to mean
    something broke, not something wobbled. Retuning it below this is a
    decision that should require changing this test deliberately.
    """
    assert cm.settings.contradiction_momentum_pct >= 0.09


def test_the_brake_is_configurable_without_a_deploy(monkeypatch):
    """Tuning it should be an environment change, not a code change."""
    monkeypatch.setattr(cm.settings, "contradiction_momentum_pct", 0.20)

    assert _momentum_triggers(monkeypatch, -0.15) is False
    assert _momentum_triggers(monkeypatch, -0.25) is True


def test_the_brake_is_direction_aware_for_shorts(monkeypatch):
    """A short is hurt by a rise, so the sign that trips it is inverted."""
    monkeypatch.setattr(cm, "_recent_sentiment", lambda engine, symbol: (None, 0))
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: 0.15)

    assert cm._check_position(engine=object(), symbol="AAPL", qty=-10).closed is True
    # The same rise helps a long — it must not propose closing it.
    assert cm._check_position(engine=object(), symbol="AAPL", qty=10).closed is False
