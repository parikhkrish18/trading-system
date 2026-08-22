import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
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


def _whatif_batch(agreements):
    ts = pd.Timestamp("2026-08-07T14:00:00Z")
    return pd.DataFrame(
        [
            {
                "ts": ts, "symbol": sym, "forecast": fc, "regime": "trend",
                "target_position": tp, "executed_position": None, "mode": "paper",
                "direction_agreement": ag,
            }
            for sym, fc, tp, ag in [
                ("AAPL", 0.05, 0.10, agreements[0]),
                ("TSLA", -0.02, -0.05, agreements[1]),
            ]
        ]
    )


def test_whatif_endpoint_filters_by_slider_thresholds(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _whatif_batch([1.0, 0.6]))

    resp = client.get("/api/whatif?min_agreement=0.8&min_abs_move=0.0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_before"] == 2
    assert body["n_after"] == 1  # TSLA's 0.6 agreement fails the 0.8 bar
    assert [r["Symbol"] for r in body["rows"]] == ["AAPL"]
    assert body["rows"][0]["Model agreement"] == 1.0
    assert "1 pick" in body["summary"]


def test_whatif_endpoint_reports_no_scored_batch_for_pre_merge_rows(monkeypatch, client):
    """Rows logged before direction_agreement existed must say so, not silently filter on nothing."""
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: _whatif_batch([None, None]))

    resp = client.get("/api/whatif")

    body = resp.json()
    assert body["available"] is False
    assert "No scored batch" in body["message"]


def test_whatif_endpoint_with_no_decisions_at_all(monkeypatch, client):
    monkeypatch.setattr(server, "get_engine", lambda: None)
    monkeypatch.setattr(server.pd, "read_sql", lambda *a, **k: pd.DataFrame())

    resp = client.get("/api/whatif")

    body = resp.json()
    assert body["available"] is False
    assert body["rows"] == []


# --- /api/tests/run token gate ----------------------------------------------


class _FakePytestResult:
    returncode = 0
    stdout = "..\n2 passed in 0.1s\n"
    stderr = ""


@pytest.fixture
def _runnable_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _FakePytestResult())
    monkeypatch.setattr(server, "LAST_TEST_RUN_PATH", tmp_path / "last_test_run.json")


def test_run_tests_allowed_on_loopback_without_a_token(monkeypatch, client, _runnable_tests):
    monkeypatch.setattr(server.settings, "dashboard_api_token", "")
    monkeypatch.setattr(server.settings, "dashboard_host", "127.0.0.1")

    resp = client.post("/api/tests/run")

    assert resp.status_code == 200


def test_run_tests_forbidden_on_open_interface_without_a_token(monkeypatch, client, _runnable_tests):
    """An open bind with no token must refuse — never run subprocesses for the whole network."""
    monkeypatch.setattr(server.settings, "dashboard_api_token", "")
    monkeypatch.setattr(server.settings, "dashboard_host", "0.0.0.0")

    resp = client.post("/api/tests/run")

    assert resp.status_code == 403
    assert "DASHBOARD_API_TOKEN" in resp.json()["detail"]


def test_run_tests_requires_the_bearer_token_when_configured(monkeypatch, client, _runnable_tests):
    monkeypatch.setattr(server.settings, "dashboard_api_token", "sekrit")
    monkeypatch.setattr(server.settings, "dashboard_host", "0.0.0.0")

    no_token = client.post("/api/tests/run")
    wrong = client.post("/api/tests/run", headers={"Authorization": "Bearer wrong"})
    right = client.post("/api/tests/run", headers={"Authorization": "Bearer sekrit"})

    assert no_token.status_code == 403
    assert wrong.status_code == 403
    assert right.status_code == 200


def test_configured_token_is_checked_even_on_loopback(monkeypatch, client, _runnable_tests):
    """Configuring a token means wanting it enforced — the loopback exemption is only for the blank case."""
    monkeypatch.setattr(server.settings, "dashboard_api_token", "sekrit")
    monkeypatch.setattr(server.settings, "dashboard_host", "127.0.0.1")

    resp = client.post("/api/tests/run")

    assert resp.status_code == 403


# --- manual triggers (ingest / trading cycle) --------------------------------


@pytest.fixture
def _job_env(monkeypatch, tmp_path):
    """Fresh job registry, synchronous worker, fake subprocess, tmp cache file."""
    monkeypatch.setattr(server, "LAST_JOBS_PATH", tmp_path / "last_manual_jobs.json")
    monkeypatch.setattr(server, "_JOBS", {name: server._fresh_job_state() for name in server._JOB_COMMANDS})
    # Run the worker inline so tests see the final state deterministically.
    monkeypatch.setattr(server, "_spawn", lambda target, name: target(name))

    class _OkResult:
        returncode = 0
        stdout = "ingested 503 symbols\n"
        stderr = ""

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _OkResult())
    monkeypatch.setattr(server.settings, "dashboard_api_token", "")
    monkeypatch.setattr(server.settings, "dashboard_host", "127.0.0.1")


def test_run_job_starts_and_records_a_finished_run(client, _job_env):
    resp = client.post("/api/jobs/ingest/run")

    assert resp.status_code == 200
    status = client.get("/api/jobs").json()["ingest"]
    assert status["status"] == "finished"
    assert status["exit_code"] == 0
    assert "ingested 503 symbols" in status["output_tail"]
    assert status["started_at"] is not None and status["finished_at"] is not None


def test_run_job_records_a_failed_run(client, _job_env, monkeypatch):
    class _BadResult:
        returncode = 2
        stdout = ""
        stderr = "universe table is empty\n"

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _BadResult())

    client.post("/api/jobs/cycle/run")

    status = client.get("/api/jobs").json()["cycle"]
    assert status["status"] == "failed"
    assert status["exit_code"] == 2
    assert "universe table is empty" in status["output_tail"]


def test_run_job_refuses_a_second_copy_while_one_is_running(client, _job_env):
    server._JOBS["ingest"]["status"] = "running"
    server._JOBS["ingest"]["started_at"] = "2026-08-22T10:00:00+00:00"

    resp = client.post("/api/jobs/ingest/run")

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_run_job_unknown_name_is_404(client, _job_env):
    assert client.post("/api/jobs/nope/run").status_code == 404


def test_run_job_request_returns_before_completion_semantics(client, _job_env, monkeypatch):
    """The endpoint must hand work to _spawn (background), not run it in-request."""
    spawned = []
    monkeypatch.setattr(server, "_spawn", lambda target, name: spawned.append(name))

    resp = client.post("/api/jobs/ingest/run")

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"  # reported as started, not finished
    assert spawned == ["ingest"]


def test_run_job_is_token_gated_like_tests_run(client, _job_env, monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_api_token", "sekrit")
    monkeypatch.setattr(server.settings, "dashboard_host", "0.0.0.0")

    no_token = client.post("/api/jobs/ingest/run")
    right = client.post("/api/jobs/ingest/run", headers={"Authorization": "Bearer sekrit"})

    assert no_token.status_code == 403
    assert right.status_code == 200


def test_run_job_forbidden_on_open_interface_without_a_token(client, _job_env, monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_api_token", "")
    monkeypatch.setattr(server.settings, "dashboard_host", "0.0.0.0")

    assert client.post("/api/jobs/cycle/run").status_code == 403


def test_jobs_status_endpoint_is_readable_without_a_token(client, _job_env, monkeypatch):
    monkeypatch.setattr(server.settings, "dashboard_api_token", "sekrit")

    resp = client.get("/api/jobs")

    assert resp.status_code == 200
    assert set(resp.json()) == {"ingest", "cycle"}


def test_finished_jobs_survive_a_restart_but_running_does_not(client, _job_env, tmp_path):
    client.post("/api/jobs/ingest/run")  # writes the cache file

    reloaded = server._load_last_jobs()
    assert reloaded["ingest"]["status"] == "finished"

    # A "running" state from a dead process must not be resurrected.
    server.LAST_JOBS_PATH.write_text(json.dumps({"ingest": {"status": "running", "started_at": "x"}}))
    reloaded = server._load_last_jobs()
    assert reloaded["ingest"]["status"] == "never_run"


def test_cycle_job_command_still_goes_through_the_pipeline_entrypoint(client, _job_env):
    """
    The button starts scripts/run_weekly_cycle.py — the entrypoint whose
    cycle runs the Telegram approval gate. It must never call the broker
    or trading loop directly, or the button WOULD authorize trades.
    """
    command = server._JOB_COMMANDS["cycle"]["command"]
    assert "scripts.run_weekly_cycle" in command
    assert not any("broker" in part for part in command)


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
