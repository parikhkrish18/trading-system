import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from monitoring.dashboard import server


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
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["symbol", "ts", "forecast"]))

    resp = client.get("/api/analysis/live_accuracy")
    body = resp.json()
    assert body["hit_rate"] is None
    assert body["n_matured"] == 0


def test_live_accuracy_computes_hit_rate_from_matured_decisions(monkeypatch, client):
    decisions = pd.DataFrame(
        {"symbol": ["AAPL"], "ts": pd.to_datetime(["2026-07-27T00:00:00Z"]), "forecast": [0.5]}
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


# --- HTTP Basic Auth ---------------------------------------------------------


def _basic(user, password):
    import base64

    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_auth_no_password_on_public_bind_is_refused(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "")
    public = TestClient(server.app, base_url="http://0.0.0.0")
    resp = public.get("/api/circuit_breakers")
    assert resp.status_code == 503
    assert "DASHBOARD_PASSWORD" in resp.text


def test_auth_no_password_on_loopback_is_allowed(monkeypatch, client):
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    assert client.get("/api/circuit_breakers").status_code == 200


def test_auth_wrong_password_is_401_with_challenge(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0")
    for headers in ({}, _basic("admin", "nope"), _basic("someone", "s3cret")):
        resp = public.get("/api/circuit_breakers", headers=headers)
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"].startswith("Basic")


def test_auth_correct_password_is_200(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    monkeypatch.setattr(server, "load_latest_breaker_state", lambda limit: pd.DataFrame())
    public = TestClient(server.app, base_url="http://0.0.0.0")
    resp = public.get("/api/circuit_breakers", headers=_basic("admin", "s3cret"))
    assert resp.status_code == 200


def test_auth_static_page_is_gated_too(monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_password", "s3cret")
    public = TestClient(server.app, base_url="http://0.0.0.0")
    assert public.get("/").status_code == 401
    assert public.get("/", headers=_basic("admin", "s3cret")).status_code == 200
