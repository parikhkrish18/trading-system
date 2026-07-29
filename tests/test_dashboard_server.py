import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from monitoring.dashboard import server


@pytest.fixture
def client():
    return TestClient(server.app)


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
