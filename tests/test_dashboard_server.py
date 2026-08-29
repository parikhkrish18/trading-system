import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from monitoring.dashboard import server


@pytest.fixture(autouse=True)
def _local_dev_bind(monkeypatch):
    """
    Every request is auth-gated now, so each test has to say which side of
    that gate it is on. The default is the local-development case —
    loopback bind, no password configured — which is the one nearly every
    test here cares nothing about. The gate's own tests override it.

    Pinned explicitly rather than inherited from the ambient settings so a
    developer's own .env cannot decide whether the suite passes.
    """
    monkeypatch.setattr(server.settings, "dashboard_host", "127.0.0.1")


@pytest.fixture
def client(monkeypatch):
    # Loopback base URL: with no DASHBOARD_PASSWORD the auth middleware only
    # lets requests through on a loopback bind (see the auth tests below).
    monkeypatch.setattr(server.settings, "dashboard_password", "")
    return TestClient(server.app, base_url="http://127.0.0.1")


class _FakeBroker:
    def __init__(self, positions):
        self._positions = positions

    def get_positions_detailed(self):
        return self._positions


def test_positions_empty_when_no_open_positions(monkeypatch, client):
    monkeypatch.setattr(server, "get_broker", lambda: _FakeBroker([]))
    resp = client.get("/api/positions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_positions_joins_latest_decision_and_parses_reasoning(monkeypatch, client):
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeBroker([{"symbol": "TSLA", "qty": 68.7, "side": "long", "avg_entry_price": 300.0,
                               "current_price": 310.0, "market_value": 21000.0, "cost_basis": 20600.0,
                               "unrealized_pl": 400.0, "unrealized_plpc": 0.019}]),
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    decisions_df = pd.DataFrame(
        [
            {
                "symbol": "TSLA", "ts": pd.Timestamp("2026-07-29T10:00:00Z"), "feature_set_id": "v3",
                "model_version": "ensemble_v1", "forecast": 0.026, "regime": "chop",
                "target_position": 0.21, "executed_position": 68.7, "mode": "paper",
                "reasoning": json.dumps([{"feature_name": "sentiment_mean_10d", "value": 0.4, "contribution": 0.01}]),
            }
        ]
    )
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: decisions_df)

    resp = client.get("/api/positions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "TSLA"
    assert body[0]["decision"]["regime"] == "chop"
    assert body[0]["decision"]["reasoning"] == [{"feature_name": "sentiment_mean_10d", "value": 0.4, "contribution": 0.01}]


def test_positions_include_exit_levels_when_recorded(monkeypatch, client):
    from execution.exit_levels import ExitLevels

    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeBroker([{"symbol": "TSLA", "qty": 68.7, "side": "long", "avg_entry_price": 300.0,
                               "current_price": 310.0, "market_value": 21000.0, "cost_basis": 20600.0,
                               "unrealized_pl": 400.0, "unrealized_plpc": 0.019}]),
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "reasoning"]))
    monkeypatch.setattr(
        server, "load_exit_levels",
        lambda engine: {"TSLA": ExitLevels(take_profit_pct=0.07, stop_loss_pct=0.05, derived=True)},
    )

    resp = client.get("/api/positions")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["exit_levels"] == {"take_profit_pct": 0.07, "stop_loss_pct": 0.05, "derived": True}


def test_positions_exit_levels_null_when_not_recorded(monkeypatch, client):
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeBroker([{"symbol": "ORPHAN", "qty": 1.0, "side": "long", "avg_entry_price": 10.0,
                               "current_price": 10.0, "market_value": 10.0, "cost_basis": 10.0,
                               "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]),
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "reasoning"]))
    monkeypatch.setattr(server, "load_exit_levels", lambda engine: {})

    resp = client.get("/api/positions")
    assert resp.json()[0]["exit_levels"] is None


def test_positions_exit_levels_load_failure_degrades_gracefully(monkeypatch, client):
    """A broken exit-levels lookup must not take down the whole positions panel."""
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeBroker([{"symbol": "TSLA", "qty": 1.0, "side": "long", "avg_entry_price": 10.0,
                               "current_price": 10.0, "market_value": 10.0, "cost_basis": 10.0,
                               "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]),
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "reasoning"]))

    def _boom(engine):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(server, "load_exit_levels", _boom)

    resp = client.get("/api/positions")
    assert resp.status_code == 200
    assert resp.json()[0]["exit_levels"] is None


def test_positions_no_matching_decision_is_null(monkeypatch, client):
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeBroker([{"symbol": "ORPHAN", "qty": 1.0, "side": "long", "avg_entry_price": 10.0,
                               "current_price": 10.0, "market_value": 10.0, "cost_basis": 10.0,
                               "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]),
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "reasoning"]))

    resp = client.get("/api/positions")
    assert resp.json()[0]["decision"] is None


def test_decisions_endpoint_parses_reasoning(monkeypatch, client):
    df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-07-29T10:00:00Z"), "symbol": "TSLA", "feature_set_id": "v3",
                "model_version": "ensemble_v1", "forecast": 0.026, "regime": "chop",
                "target_position": 0.21, "executed_position": 68.7, "mode": "paper",
                "reasoning": json.dumps([{"feature_name": "f1", "value": 1.0, "contribution": 0.02}]),
            }
        ]
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/decisions")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["reasoning"] == [{"feature_name": "f1", "value": 1.0, "contribution": 0.02}]


def test_equity_curve_endpoint(monkeypatch, client):
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-07-28", "2026-07-29"], utc=True), "mode": ["paper", "paper"], "equity_value": [100000.0, 99950.0]})
    monkeypatch.setattr(server, "load_equity_curve", lambda mode, limit: df)

    resp = client.get("/api/equity_curve")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[1]["equity_value"] == 99950.0


def test_circuit_breakers_endpoint(monkeypatch, client):
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2026-07-29"], utc=True), "breaker_name": ["max_drawdown"], "triggered": [False], "reason": [""]}
    )
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: df)

    resp = client.get("/api/circuit_breakers")
    assert resp.json()[0]["breaker_name"] == "max_drawdown"


def test_analysis_runs_endpoint_empty(monkeypatch, client):
    monkeypatch.setattr(server.mlflow, "search_runs", lambda experiment_names: pd.DataFrame())
    resp = client.get("/api/analysis/runs")
    assert resp.json() == []


def test_analysis_runs_endpoint_returns_records(monkeypatch, client):
    df = pd.DataFrame(
        {
            "start_time": pd.to_datetime(["2026-07-28"], utc=True),
            "params.fold_id": ["0"],
            "params.feature_set_id": ["v3"],
            "params.train_start": ["2024-01-01"],
            "params.train_end": ["2025-01-01"],
            "params.test_start": ["2025-01-01"],
            "params.test_end": ["2025-07-01"],
            "metrics.mae": [0.03],
            "metrics.rmse": [0.05],
            "metrics.directional_accuracy": [0.52],
            "metrics.directional_accuracy_when_confident": [0.53],
            "metrics.pct_rows_confident": [0.9],
        }
    )
    monkeypatch.setattr(server.mlflow, "search_runs", lambda experiment_names: df)

    resp = client.get("/api/analysis/runs")
    body = resp.json()
    assert body[0]["fold_id"] == "0"
    assert body[0]["directional_accuracy"] == pytest.approx(0.52)


def test_last_test_run_returns_none_when_no_cache(monkeypatch, client, tmp_path):
    monkeypatch.setattr(server, "LAST_TEST_RUN_PATH", tmp_path / "last_test_run.json")
    resp = client.get("/api/tests/last")
    assert resp.json() is None


def test_last_test_run_returns_cached_result(monkeypatch, client, tmp_path):
    cache_path = tmp_path / "last_test_run.json"
    cache_path.write_text(json.dumps({"ts": "2026-07-29T00:00:00Z", "passed": True, "summary": "5 passed", "output": "..."}))
    monkeypatch.setattr(server, "LAST_TEST_RUN_PATH", cache_path)

    resp = client.get("/api/tests/last")
    assert resp.json()["passed"] is True


def test_live_accuracy_no_decisions_returns_null_hit_rate(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "forecast", "mode"]))

    resp = client.get("/api/analysis/live_accuracy")
    body = resp.json()
    assert body["hit_rate"] is None
    assert body["n_matured"] == 0
    assert body["backfill"]["hit_rate"] is None
    assert body["backfill"]["n_matured"] == 0


def test_live_accuracy_computes_hit_rate_from_matured_decisions(monkeypatch, client):
    decisions = pd.DataFrame(
        {"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z"]), "forecast": [0.5], "mode": ["paper"]}
    )
    prices = pd.DataFrame(
        {"symbol": ["AAPL", "AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z", "2026-07-28T00:00:00Z"]), "close": [100.0, 105.0]}
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    resp = client.get("/api/analysis/live_accuracy")
    body = resp.json()
    assert body["hit_rate"] == pytest.approx(1.0)  # positive forecast, price went up
    assert body["n_matured"] == 1
    assert body["backfill"]["n_matured"] == 0


def test_live_accuracy_excludes_backfilled_history_from_the_live_number(monkeypatch, client):
    """
    The honesty split: 3 replayed rows (mode='backfill', all hits) + 1 real
    paper decision (a miss). Blended, that would read as a 75% hit rate; the
    live number must be the honest 0% and the replay reported separately.
    """
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "ts": pd.to_datetime(
                ["2026-07-20T12:00:00Z", "2026-07-21T12:00:00Z", "2026-07-22T12:00:00Z", "2026-07-27T12:00:00Z"]
            ),
            # Backfill rows predict up while price rises (hits); the one real
            # paper row predicts up right before the price falls (a miss).
            "forecast": [0.5, 0.5, 0.5, 0.5],
            "mode": ["backfill", "backfill", "backfill", "paper"],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "ts": pd.to_datetime(
                ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-27", "2026-07-28"]
            ).tz_localize("UTC"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 95.0],
        }
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    body = client.get("/api/analysis/live_accuracy").json()
    assert body["n_matured"] == 1
    assert body["hit_rate"] == pytest.approx(0.0)  # the real decision missed
    assert body["backfill"]["n_matured"] == 3
    assert body["backfill"]["hit_rate"] == pytest.approx(1.0)


def test_live_accuracy_with_only_backfill_rows_reports_no_live_number(monkeypatch, client):
    decisions = pd.DataFrame(
        {"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z"]), "forecast": [0.5], "mode": ["backfill"]}
    )
    prices = pd.DataFrame(
        {"symbol": ["AAPL", "AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z", "2026-07-28T00:00:00Z"]), "close": [100.0, 105.0]}
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    body = client.get("/api/analysis/live_accuracy").json()
    assert body["hit_rate"] is None  # nothing real has matured — say so, don't borrow history
    assert body["n_matured"] == 0
    assert body["backfill"]["n_matured"] == 1


def test_drift_no_decisions_returns_unavailable(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "forecast", "reasoning"]))

    body = client.get("/api/analysis/drift").json()
    assert body["available"] is False
    assert body["weekly"] == []
    assert body["feature_drag"] == []


def test_drift_no_matured_decisions_says_so(monkeypatch, client):
    decisions = pd.DataFrame(
        {"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z"]), "forecast": [0.5], "reasoning": [None]}
    )
    prices = pd.DataFrame(columns=["symbol", "ts", "close"])  # nothing to grade against yet
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    body = client.get("/api/analysis/drift").json()
    assert body["available"] is False
    assert "matured" in body["message"].lower()


def test_drift_flags_when_recent_weeks_undercut_the_walkforward_baseline(monkeypatch, client):
    """3 straight weeks of real misses against a strong baseline should flag."""
    phase2 = json.dumps([{"phase": 2, "title": "x", "summary": "x", "lines": [], "top_features": [{"feature_name": "mom_ret_5d", "value": 0.1, "contribution": 0.01}]}])
    weeks = [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-17")]
    rows, price_rows = [], []
    for w in weeks:
        for i in range(5):
            ts = w + pd.Timedelta(days=i)
            rows.append({"symbol": f"S{i}", "ts": ts.tz_localize("UTC"), "forecast": 0.5, "reasoning": phase2})
            # forecast is up; price goes DOWN the next day -> a miss every time
            price_rows.append({"symbol": f"S{i}", "ts": ts.tz_localize("UTC"), "close": 100.0})
            price_rows.append({"symbol": f"S{i}", "ts": (ts + pd.Timedelta(days=1)).tz_localize("UTC"), "close": 90.0})
    decisions = pd.DataFrame(rows)
    prices = pd.DataFrame(price_rows)
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))
    monkeypatch.setattr(
        server.report_card, "fetch_fold_runs",
        lambda tracking_uri: [{"run_name": "f0", "fold_id": "0", "metrics": {"directional_accuracy": 0.55}}],
    )

    body = client.get("/api/analysis/drift").json()
    assert body["available"] is True
    assert body["baseline_accuracy"] == pytest.approx(0.55)
    assert body["accuracy_flag"]["flagged"] is True
    assert len(body["weekly"]) == 3
    # every decision cited mom_ret_5d and every one missed -> hit_rate 0.0
    drag = {r["feature_name"]: r for r in body["feature_drag"]}
    assert drag["mom_ret_5d"]["hit_rate"] == pytest.approx(0.0)
    assert drag["mom_ret_5d"]["n"] == 15


def test_drift_baseline_unavailable_when_mlflow_is_down(monkeypatch, client):
    decisions = pd.DataFrame(
        {"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z"]), "forecast": [0.5], "reasoning": [None]}
    )
    prices = pd.DataFrame(
        {"symbol": ["AAPL", "AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z", "2026-07-28T00:00:00Z"]), "close": [100.0, 105.0]}
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    def _boom(tracking_uri):
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(server.report_card, "fetch_fold_runs", _boom)

    body = client.get("/api/analysis/drift").json()
    assert body["available"] is True  # live data is still shown even without a baseline to compare against
    assert body["baseline_accuracy"] is None
    assert body["accuracy_flag"]["flagged"] is False


def test_closed_trades_reconstructs_a_round_trip(monkeypatch, client):
    decisions = pd.DataFrame(
        [
            {"symbol": "AAPL", "ts": pd.Timestamp("2026-07-01T00:00:00Z"), "target_position": 0.5, "executed_position": 10.0, "mode": "paper"},
            {"symbol": "AAPL", "ts": pd.Timestamp("2026-07-08T00:00:00Z"), "target_position": 0.0, "executed_position": 0.0, "mode": "paper"},
        ]
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts": pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-08T00:00:00Z"]),
            "close": [100.0, 110.0],
        }
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    resp = client.get("/api/trades/closed")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    trade = body[0]
    assert trade["symbol"] == "AAPL"
    assert trade["side"] == "long"
    assert trade["shares"] == 10.0
    assert trade["realized_pnl"] == pytest.approx(100.0)  # (110-100) * 10 shares


def test_closed_trades_computes_short_pnl_correctly(monkeypatch, client):
    decisions = pd.DataFrame(
        [
            {"symbol": "IBM", "ts": pd.Timestamp("2026-07-01T00:00:00Z"), "target_position": -0.5, "executed_position": -10.0, "mode": "paper"},
            {"symbol": "IBM", "ts": pd.Timestamp("2026-07-08T00:00:00Z"), "target_position": 0.0, "executed_position": 0.0, "mode": "paper"},
        ]
    )
    prices = pd.DataFrame(
        {"symbol": ["IBM", "IBM"], "ts": pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-08T00:00:00Z"]), "close": [100.0, 90.0]}
    )
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    resp = client.get("/api/trades/closed")
    trade = resp.json()[0]
    assert trade["side"] == "short"
    assert trade["realized_pnl"] == pytest.approx(100.0)  # price dropped $10, short profits


def test_closed_trades_no_decisions_returns_empty(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame())

    resp = client.get("/api/trades/closed")
    assert resp.json() == []


def test_closed_trades_still_open_position_is_not_a_trade(monkeypatch, client):
    decisions = pd.DataFrame(
        [{"symbol": "AAPL", "ts": pd.Timestamp("2026-07-01T00:00:00Z"), "target_position": 0.5, "executed_position": 10.0, "mode": "paper"}]
    )
    prices = pd.DataFrame({"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-01T00:00:00Z"]), "close": [100.0]})
    calls = iter([decisions, prices])
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: next(calls))

    resp = client.get("/api/trades/closed")
    assert resp.json() == []


def test_positions_news_returns_empty_when_nothing_held(monkeypatch, client):
    class _NoPositionsBroker:
        def get_positions(self):
            return {}

    monkeypatch.setattr(server, "get_broker", lambda: _NoPositionsBroker())
    resp = client.get("/api/positions/news")
    assert resp.json() == {}


def test_positions_news_groups_headlines_by_symbol(monkeypatch, client):
    class _HeldBroker:
        def get_positions(self):
            return {"AAPL": 10.0, "MSFT": -5.0}

    monkeypatch.setattr(server, "get_broker", lambda: _HeldBroker())
    monkeypatch.setattr(server, "get_engine", lambda: None)
    news = pd.DataFrame(
        [
            {"symbol": "AAPL", "ts": pd.Timestamp("2026-07-29T00:00:00Z"), "headline": "AAPL beats earnings", "sentiment": 0.6, "source": "polygon"},
            {"symbol": "MSFT", "ts": pd.Timestamp("2026-07-28T00:00:00Z"), "headline": "MSFT antitrust probe", "sentiment": -0.4, "source": "polygon"},
        ]
    )
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: news)

    resp = client.get("/api/positions/news")
    body = resp.json()
    assert set(body.keys()) == {"AAPL", "MSFT"}
    assert body["AAPL"][0]["headline"] == "AAPL beats earnings"


def test_regime_history_returns_empty_with_insufficient_data(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["ts", "high", "low", "close"]))

    resp = client.get("/api/regime_history")
    assert resp.json() == []


def test_regime_history_classifies_trend_vs_chop(monkeypatch, client):
    idx = pd.bdate_range("2026-01-01", periods=40, tz="UTC")
    trend = pd.Series(range(40), dtype=float) + 100
    df = pd.DataFrame({"ts": idx, "high": trend + 1, "low": trend - 1, "close": trend})
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/regime_history")
    body = resp.json()
    assert len(body) > 0
    assert all(r["regime"] in ("trend", "chop") for r in body)


def test_feature_frequency_counts_across_decisions(monkeypatch, client):
    df = pd.DataFrame(
        {
            "reasoning": [
                json.dumps(
                    [{"phase": 2, "title": "x", "summary": "x", "lines": [], "top_features": [{"feature_name": "mom_ret_5d", "value": 0.1, "contribution": 0.02}]}]
                ),
                json.dumps(
                    [{"phase": 2, "title": "x", "summary": "x", "lines": [], "top_features": [{"feature_name": "mom_ret_5d", "value": 0.05, "contribution": 0.01}]}]
                ),
            ]
        }
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/analysis/feature_frequency")
    body = resp.json()
    assert body[0]["feature_name"] == "mom_ret_5d"
    assert body[0]["times_in_top5"] == 2
    assert body[0]["avg_abs_contribution"] == pytest.approx(0.015)


def test_feature_frequency_skips_old_decisions_without_top_features(monkeypatch, client):
    """Decisions logged before the 7-phase reasoning model don't have structured top_features -- skip, don't crash."""
    df = pd.DataFrame({"reasoning": [json.dumps([{"feature_name": "f1", "value": 1.0, "contribution": 0.02}])]})
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/analysis/feature_frequency")
    assert resp.json() == []


def test_run_tests_executes_subprocess_and_caches(monkeypatch, client, tmp_path):
    class _FakeResult:
        returncode = 0
        stdout = "..\n2 passed in 0.1s\n"
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(server, "LAST_TEST_RUN_PATH", tmp_path / "last_test_run.json")

    resp = client.post("/api/tests/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert "2 passed" in body["summary"]
    assert (tmp_path / "last_test_run.json").exists()


# --- model report card ------------------------------------------------------


def _fold_run(fold_id, accuracy, confident_accuracy, pct_confident):
    return {
        "run_name": f"fold_{fold_id}",
        "fold_id": str(fold_id),
        "metrics": {
            "directional_accuracy": accuracy,
            "directional_accuracy_when_confident": confident_accuracy,
            "pct_rows_confident": pct_confident,
            "mae": 0.01,
            "rmse": 0.02,
        },
    }


def test_report_card_endpoint_folds_metrics_into_headline_chart_and_callouts(monkeypatch, client):
    runs = [_fold_run(2, 0.60, 0.70, 0.5), _fold_run(1, 0.50, 0.60, 0.5)]
    monkeypatch.setattr(server.report_card, "fetch_fold_runs", lambda uri, experiment_name="forecast_lgbm": runs)

    resp = client.get("/api/analysis/report_card")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["headline"]["n_folds"] == 2
    assert body["headline"]["directional_accuracy"] == pytest.approx(0.55)
    # two folds x two series = four bars
    assert len(body["chart"]) == 4
    assert len(body["callouts"]) == 2
    assert any("models agreed" in c for c in body["callouts"])


def test_report_card_endpoint_is_empty_not_500_when_mlflow_is_down(monkeypatch, client):
    def _down(uri, experiment_name="forecast_lgbm"):
        raise ConnectionError("mlflow unreachable")

    monkeypatch.setattr(server.report_card, "fetch_fold_runs", _down)

    resp = client.get("/api/analysis/report_card")

    assert resp.status_code == 200
    assert resp.json() == {"available": False, "headline": None, "chart": [], "callouts": []}


def test_report_card_endpoint_unavailable_when_no_runs(monkeypatch, client):
    monkeypatch.setattr(server.report_card, "fetch_fold_runs", lambda uri, experiment_name="forecast_lgbm": [])

    resp = client.get("/api/analysis/report_card")

    assert resp.json()["available"] is False


# --- what-if thresholds -----------------------------------------------------


def _whatif_batch():
    ts = pd.Timestamp("2026-08-07T14:00:00Z")
    return pd.DataFrame(
        [
            {
                "ts": ts, "symbol": sym, "forecast": fc, "regime": "trend",
                "target_position": tp, "executed_position": None, "mode": "paper",
            }
            for sym, fc, tp in [
                ("AAPL", 0.05, 0.10),
                ("TSLA", -0.02, -0.05),
            ]
        ]
    )


def test_whatif_endpoint_filters_by_the_move_slider(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _whatif_batch())

    resp = client.get("/api/whatif?min_abs_move=0.03")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_before"] == 2
    assert body["n_after"] == 1  # TSLA's 2% move fails a 3% bar
    assert [r["Symbol"] for r in body["rows"]] == ["AAPL"]
    assert "1 pick" in body["summary"]


def test_whatif_endpoint_no_longer_takes_or_reports_agreement(monkeypatch, client):
    """
    The agreement slider went with the threshold it tuned. An unknown query
    parameter must be ignored rather than filtering on a number that
    predicts nothing.
    """
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _whatif_batch())

    body = client.get("/api/whatif?min_agreement=0.99").json()

    assert body["n_after"] == 2  # nothing was filtered
    assert "min_agreement" not in body
    assert not any("agreement" in str(k).lower() for k in body["rows"][0])


def test_whatif_endpoint_with_no_decisions_at_all(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame())

    resp = client.get("/api/whatif")

    body = resp.json()
    assert body["available"] is False
    assert body["rows"] == []


# --- /api/tests/run auth gate ------------------------------------------------


class _FakePytestResult:
    returncode = 0
    stdout = "..\n2 passed in 0.1s\n"
    stderr = ""


@pytest.fixture
def _runnable_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _FakePytestResult())
    monkeypatch.setattr(server, "LAST_TEST_RUN_PATH", tmp_path / "last_test_run.json")


def test_run_tests_allowed_on_loopback_with_no_password_set(client, _runnable_tests):
    """Local development needs no ceremony — the client fixture is already loopback, no password."""
    resp = client.post("/api/tests/run")

    assert resp.status_code == 200


def test_run_tests_forbidden_on_open_interface_without_a_password(client, _runnable_tests):
    """
    An open bind with no password must refuse — never run subprocesses for
    the whole network. `client` (loopback base_url) is only here for its
    fixture side effect of pinning dashboard_password=""; what makes this
    request "public" is the separate client's base_url below — the gate
    reads the actual connection's host (request.scope["server"]), not the
    dashboard_host setting.
    """
    public = TestClient(server.app, base_url="http://0.0.0.0")

    resp = public.post("/api/tests/run")

    assert resp.status_code == 503
    assert "DASHBOARD_PASSWORD" in resp.text


def test_run_tests_requires_the_session_cookie_when_a_password_is_configured(monkeypatch, client, _runnable_tests):
    monkeypatch.setattr(server.settings, "dashboard_password", "sekrit")

    no_cookie = client.post("/api/tests/run")

    client.cookies.set(server._SESSION_COOKIE, "wrong")
    wrong_cookie = client.post("/api/tests/run")

    client.cookies.set(server._SESSION_COOKIE, server._session_token("sekrit"))
    right_cookie = client.post("/api/tests/run")

    assert no_cookie.status_code == 401
    assert wrong_cookie.status_code == 401
    assert right_cookie.status_code == 200


def test_configured_password_is_checked_even_on_loopback(monkeypatch, client, _runnable_tests):
    """Configuring a password means wanting it enforced — the loopback exemption is only for the blank case."""
    monkeypatch.setattr(server.settings, "dashboard_password", "sekrit")

    resp = client.post("/api/tests/run")

    assert resp.status_code == 401


# --------------------------------------------------------------------------
# Where the server binds — the hosting contract
# --------------------------------------------------------------------------


def _captured_uvicorn_kwargs(monkeypatch) -> dict:
    """Run main() with uvicorn stubbed out, and report how it was called."""
    captured: dict = {}

    class _FakeUvicorn:
        @staticmethod
        def run(app, **kwargs):
            captured.update(app=app, **kwargs)

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)
    server.main()
    return captured


def test_main_binds_the_configured_host_and_port(monkeypatch):
    """
    A platform host (Railway, and PaaS generally) picks the port itself and
    injects it as $PORT, then routes the public domain there. Binding a
    hardcoded port instead means the health check hits nothing and the
    deploy is marked failed, so the port must come from settings.
    """
    monkeypatch.setattr(server.settings, "dashboard_host", "0.0.0.0")
    monkeypatch.setattr(server.settings, "dashboard_port", 3141)

    captured = _captured_uvicorn_kwargs(monkeypatch)

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 3141


def test_bind_defaults_stay_loopback_8501_when_nothing_is_injected(monkeypatch):
    """
    The unhosted default stays the deliberate loopback bind. Built from a
    clean environment rather than the ambient `settings`, so a developer's
    own .env cannot make this pass or fail.
    """
    for var in ("PORT", "DASHBOARD_HOST"):
        monkeypatch.delenv(var, raising=False)

    fresh = Settings(_env_file=None)

    assert fresh.dashboard_host == "127.0.0.1"
    assert fresh.dashboard_port == 8501


# --------------------------------------------------------------------------
# The gate covers reads, not just writes
# --------------------------------------------------------------------------


def test_reads_are_refused_without_a_session_cookie_once_a_password_is_configured(client, monkeypatch):
    """
    What these return is the book: open positions, their size, the model's
    reasoning, the equity curve. Configuring a password means every one of
    these needs the session cookie from /login — even on loopback, which is
    what `client` is bound to here; see test_configured_password_is_checked_
    even_on_loopback for that property in isolation.
    """
    monkeypatch.setattr(server.settings, "dashboard_password", "sekrit")

    for path in ("/api/positions", "/api/decisions", "/api/equity_curve", "/api/trades/closed"):
        assert client.get(path).status_code == 401, f"{path} leaked without a session cookie"


def test_reads_are_refused_on_a_public_bind_even_with_no_password_configured(client):
    """Fails closed: a blank DASHBOARD_PASSWORD must not mean 'open to all'."""
    public = TestClient(server.app, base_url="http://0.0.0.0")

    assert public.get("/api/positions").status_code == 503


def test_reads_still_work_on_loopback_without_a_password(monkeypatch, client):
    """Local development keeps its no-ceremony path."""
    monkeypatch.setattr(server, "get_broker", lambda: _FakeBroker([]))

    assert client.get("/api/positions").status_code == 200


# --------------------------------------------------------------------------
# Password login page + session cookie (replaces the old separate operator
# token + HTTP Basic Auth pair with one gate: enter the password once at
# /login, everything else — the static page, every /api route, the
# mutating job/test-run endpoints — is covered by the cookie it sets).
# --------------------------------------------------------------------------


def test_no_password_on_public_bind_is_refused(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "")
    public = TestClient(server.app, base_url="http://0.0.0.0")
    resp = public.get("/api/circuit_breakers")
    assert resp.status_code == 503
    assert "DASHBOARD_PASSWORD" in resp.text


def test_no_password_on_loopback_is_allowed(monkeypatch, client):
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    assert client.get("/api/circuit_breakers").status_code == 200


def test_login_page_is_reachable_without_a_session(monkeypatch):
    """The login flow itself must work before anyone is logged in."""
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0")

    resp = public.get("/login")

    assert resp.status_code == 200
    assert "<form" in resp.text and 'name="password"' in resp.text


def test_login_page_redirects_home_when_already_authenticated(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0", follow_redirects=False)
    public.cookies.set(server._SESSION_COOKIE, server._session_token("s3cret"))

    resp = public.get("/login")

    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/"


def test_wrong_password_is_401_and_sets_no_cookie(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0")

    resp = public.post("/login", data={"password": "nope"})

    assert resp.status_code == 401
    assert "Wrong password" in resp.text
    assert server._SESSION_COOKIE not in resp.cookies


def test_correct_password_sets_a_session_cookie_and_redirects_home(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0", follow_redirects=False)

    resp = public.post("/login", data={"password": "s3cret"})

    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    assert resp.cookies[server._SESSION_COOKIE] == server._session_token("s3cret")


def test_session_cookie_from_login_grants_access_to_the_page_and_the_api(monkeypatch):
    """
    The end-to-end flow: log in once, then both the static page and a gated
    /api route work. base_url is https here (unlike the other public-bind
    tests) because a real deploy always is — Railway serves HTTPS — and the
    session cookie is Secure-flagged (see login_submit), so a plain http
    round trip in the test client would drop it just like a browser would.
    """
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    public = TestClient(server.app, base_url="https://0.0.0.0", follow_redirects=False)

    login = public.post("/login", data={"password": "s3cret"})
    assert login.status_code == 303

    # TestClient carries cookies across requests within the same instance,
    # same as a browser would — the Set-Cookie from the login response above
    # is already in its jar for these two follow-up requests.
    assert public.get("/").status_code == 200
    assert public.get("/api/circuit_breakers").status_code == 200


def test_unauthenticated_api_request_gets_401_json_not_a_redirect(monkeypatch):
    """
    fetch() in app.js can't usefully follow a redirect into an HTML login
    page — it needs a machine-readable signal to act on (see fetchJSON in
    app.js, which sends the browser to /login on exactly this).
    """
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0")

    resp = public.get("/api/circuit_breakers")

    assert resp.status_code == 401
    assert resp.json()["detail"]


def test_unauthenticated_page_request_redirects_to_login(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0", follow_redirects=False)

    resp = public.get("/")

    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"


def test_static_page_is_gated_too(monkeypatch):
    """The mount isn't exempt: it needs the same session cookie as everything else."""
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0", follow_redirects=False)

    assert public.get("/").status_code in (302, 307)
    public.cookies.set(server._SESSION_COOKIE, server._session_token("s3cret"))
    assert public.get("/").status_code == 200


def test_logout_clears_the_session_cookie(monkeypatch):
    # The session is established through a real POST /login (rather than
    # poking the cookie jar directly) so the cookie is associated with the
    # right domain in the test client's jar — otherwise the deleting
    # Set-Cookie that /logout sends back doesn't line up with a manually
    # inserted one and this test would pass for the wrong reason. https
    # base_url for the same reason as the login test above: the cookie is
    # Secure-flagged.
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    public = TestClient(server.app, base_url="https://0.0.0.0", follow_redirects=False)
    public.post("/login", data={"password": "s3cret"})
    assert public.get("/api/circuit_breakers").status_code == 200

    logout = public.get("/logout")

    assert logout.status_code in (302, 307)
    assert public.get("/api/circuit_breakers").status_code == 401


def test_changing_the_password_invalidates_outstanding_sessions(monkeypatch):
    """No server-side session store to revoke — the token is derived from the password itself."""
    monkeypatch.setattr(server.settings, "dashboard_password", "old-password")
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    public = TestClient(server.app, base_url="https://0.0.0.0", follow_redirects=False)
    public.post("/login", data={"password": "old-password"})
    assert public.get("/api/circuit_breakers").status_code == 200

    monkeypatch.setattr(server.settings, "dashboard_password", "new-password")

    assert public.get("/api/circuit_breakers").status_code == 401


# --------------------------------------------------------------------------
# /api/news/live — universe-wide news feed for the dashboard's Live News tab
# --------------------------------------------------------------------------


def test_live_news_empty_when_no_articles(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "headline", "source", "sentiment"]))

    resp = client.get("/api/news/live")

    assert resp.status_code == 200
    assert resp.json() == []


def test_live_news_groups_multiple_symbols_under_the_same_story(monkeypatch, client):
    """One Fed headline tagged to two names must read as one story with two symbols, not two news items."""
    ts = pd.Timestamp("2026-08-28T14:30:00Z")
    df = pd.DataFrame(
        [
            {"symbol": "AAPL", "ts": ts, "headline": "Fed holds rates steady", "source": "alpaca_stream", "sentiment": 0.2},
            {"symbol": "MSFT", "ts": ts, "headline": "Fed holds rates steady", "source": "alpaca_stream", "sentiment": -0.1},
        ]
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/news/live")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["headline"] == "Fed holds rates steady"
    symbols = {s["symbol"]: s["sentiment"] for s in body[0]["symbols"]}
    assert symbols == {"AAPL": pytest.approx(0.2), "MSFT": pytest.approx(-0.1)}


def test_live_news_separate_headlines_stay_separate(monkeypatch, client):
    df = pd.DataFrame(
        [
            {"symbol": "AAPL", "ts": pd.Timestamp("2026-08-28T14:30:00Z"), "headline": "AAPL beats earnings", "source": "polygon", "sentiment": 0.6},
            {"symbol": "TSLA", "ts": pd.Timestamp("2026-08-28T15:00:00Z"), "headline": "TSLA recalls vehicles", "source": "polygon", "sentiment": -0.5},
        ]
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/news/live")

    body = resp.json()
    assert len(body) == 2
    assert body[0]["headline"] == "AAPL beats earnings"  # newest (ORDER BY ts DESC from the query) first


def test_live_news_leaves_sentiment_null_when_not_yet_scored(monkeypatch, client):
    """A freshly streamed article (see data/ingest/news_stream.py) has sentiment=NaN until the hourly scoring pass reaches it."""
    df = pd.DataFrame(
        [{"symbol": "NVDA", "ts": pd.Timestamp("2026-08-28T16:00:00Z"), "headline": "NVDA announces new chip", "source": "alpaca_stream", "sentiment": float("nan")}]
    )
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: df)

    resp = client.get("/api/news/live")

    assert resp.json()[0]["symbols"][0]["sentiment"] is None


# --------------------------------------------------------------------------
# /api/news/ingestion_status and /api/market_clock — the two status-strip
# indicators requested for Monday: is news ingestion actually alive right
# now (independent of market hours -- see data/ingest/news_stream.py), and
# are we inside NYSE regular trading hours.
# --------------------------------------------------------------------------


def _fake_read_sql_for_ingestion_status(latest_ts, count_last_hour):
    """
    get_news_ingestion_status runs two queries (MAX(ts), then COUNT(*)) --
    dispatch on which one a call is by sniffing the compiled SQL text, the
    same trick the endpoint's own SELECTs are named for.
    """

    def _read_sql(query, engine, params=None):
        sql = str(query)
        if "MAX(ts)" in sql:
            return pd.DataFrame([{"latest_ts": latest_ts}])
        if "COUNT(*)" in sql:
            return pd.DataFrame([{"n": count_last_hour}])
        raise AssertionError(f"Unexpected query in ingestion_status test: {sql}")

    return _read_sql


def test_news_ingestion_status_reports_seconds_since_latest_headline(monkeypatch, client):
    latest_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", _fake_read_sql_for_ingestion_status(latest_ts, 12))

    resp = client.get("/api/news/ingestion_status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count_last_hour"] == 12
    assert 250 < body["seconds_since_latest"] < 350  # ~5 minutes, allowing for test runtime


def test_news_ingestion_status_null_when_no_news_ever_ingested(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", _fake_read_sql_for_ingestion_status(pd.NaT, 0))

    resp = client.get("/api/news/ingestion_status")

    body = resp.json()
    assert body["latest_ts"] is None
    assert body["seconds_since_latest"] is None
    assert body["count_last_hour"] == 0


class _FakeClockBroker:
    def __init__(self, clock: dict):
        self._clock = clock

    def get_clock(self):
        return self._clock


def test_market_clock_passes_through_broker_response_when_open(monkeypatch, client):
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeClockBroker(
            {"is_open": True, "timestamp": "2026-08-25T14:00:00+00:00",
             "next_open": "2026-08-26T13:30:00+00:00", "next_close": "2026-08-25T20:00:00+00:00", "source": "alpaca"}
        ),
    )

    resp = client.get("/api/market_clock")

    assert resp.status_code == 200
    assert resp.json()["is_open"] is True


def test_market_clock_passes_through_broker_response_when_closed(monkeypatch, client):
    monkeypatch.setattr(
        server, "get_broker",
        lambda: _FakeClockBroker(
            {"is_open": False, "timestamp": "2026-08-29T14:00:00+00:00",
             "next_open": "2026-08-31T13:30:00+00:00", "next_close": "2026-08-31T20:00:00+00:00",
             "source": "computed_no_holiday_calendar"}
        ),
    )

    resp = client.get("/api/market_clock")

    body = resp.json()
    assert body["is_open"] is False
    assert body["next_open"] == "2026-08-31T13:30:00+00:00"
