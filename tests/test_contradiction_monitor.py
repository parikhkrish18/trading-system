import pandas as pd
import pytest

from execution import contradiction_monitor as cm


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
    monkeypatch.setattr(cm, "_recent_momentum", lambda engine, symbol: 0.06)  # up 6%, contradicts a short
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
