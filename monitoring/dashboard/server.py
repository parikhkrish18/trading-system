"""
Custom monitoring dashboard — FastAPI JSON API + a static vanilla-JS
frontend (monitoring/dashboard/static/). Replaces the Streamlit version:
this is the full operational picture — every held position with the
model's actual reasoning for entering it (LightGBM per-prediction feature
contributions, see models/screener.py::_attach_reasoning), the walk-forward
analysis history from MLflow, equity/drawdown, circuit-breaker status, and
the test suite, runnable on demand.

Usage:
    python -m monitoring.dashboard.server
    # or: uvicorn monitoring.dashboard.server:app --port 8501
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine
from execution.broker import get_broker
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LAST_TEST_RUN_PATH = REPO_ROOT / "logs" / "last_test_run.json"

app = FastAPI(title="Trading System Monitor")


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT aren't valid JSON — swap for None before returning records."""
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records", date_format="iso"))


@app.get("/api/positions")
def get_positions() -> list[dict]:
    broker = get_broker()
    positions = broker.get_positions_detailed()
    if not positions:
        return []

    symbols = [p["symbol"] for p in positions]
    engine = get_engine()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    decisions = pd.read_sql(
        f"""SELECT DISTINCT ON (symbol) symbol, ts, feature_set_id, model_version,
                   forecast, regime, target_position, executed_position, mode, reasoning
            FROM decisions WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC""",
        engine,
    )
    decisions_by_symbol = {row["symbol"]: row.to_dict() for _, row in decisions.iterrows()}

    for p in positions:
        decision = decisions_by_symbol.get(p["symbol"])
        if decision is None:
            p["decision"] = None
            continue
        reasoning = decision.get("reasoning")
        if isinstance(reasoning, str):
            reasoning = json.loads(reasoning)
        ts = decision.get("ts")
        p["decision"] = {
            "ts": ts.isoformat() if isinstance(ts, (dt.datetime, pd.Timestamp)) else ts,
            "feature_set_id": decision.get("feature_set_id"),
            "model_version": decision.get("model_version"),
            "forecast": decision.get("forecast"),
            "regime": decision.get("regime"),
            "target_position": decision.get("target_position"),
            "executed_position": decision.get("executed_position"),
            "mode": decision.get("mode"),
            "reasoning": reasoning,
        }
    return positions


@app.get("/api/decisions")
def get_decisions(symbol: str | None = None, limit: int = Query(default=200, le=1000), offset: int = 0) -> list[dict]:
    engine = get_engine()
    where = "WHERE symbol = :symbol" if symbol else ""
    params: dict = {"limit": limit, "offset": offset}
    if symbol:
        params["symbol"] = symbol
    df = pd.read_sql(
        text(
            f"""SELECT ts, symbol, feature_set_id, model_version, forecast, regime,
                       target_position, executed_position, mode, reasoning
                FROM decisions {where} ORDER BY ts DESC LIMIT :limit OFFSET :offset"""
        ),
        engine,
        params=params,
    )
    records = _clean_records(df)
    for r in records:
        if isinstance(r.get("reasoning"), str):
            r["reasoning"] = json.loads(r["reasoning"])
    return records


@app.get("/api/equity_curve")
def get_equity_curve(mode: str = "paper", limit: int = 1000) -> list[dict]:
    df = load_equity_curve(mode=mode, limit=limit)
    return _clean_records(df)


@app.get("/api/circuit_breakers")
def get_circuit_breakers(limit: int = 200) -> list[dict]:
    df = load_latest_breaker_state(limit=limit)
    return _clean_records(df)


@app.get("/api/analysis/runs")
def get_analysis_runs(experiment: str = "forecast_lgbm") -> list[dict]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        df = mlflow.search_runs(experiment_names=[experiment])
    except Exception:
        return []
    if df.empty:
        return []

    wanted = {
        "start_time": "start_time",
        "params.fold_id": "fold_id",
        "params.feature_set_id": "feature_set_id",
        "params.train_start": "train_start",
        "params.train_end": "train_end",
        "params.test_start": "test_start",
        "params.test_end": "test_end",
        "metrics.mae": "mae",
        "metrics.rmse": "rmse",
        "metrics.directional_accuracy": "directional_accuracy",
        "metrics.directional_accuracy_when_confident": "directional_accuracy_when_confident",
        "metrics.pct_rows_confident": "pct_rows_confident",
        "metrics.mean_ensemble_std": "mean_ensemble_std",
    }
    present = {k: v for k, v in wanted.items() if k in df.columns}
    result = df[list(present.keys())].rename(columns=present)
    result = result.sort_values("start_time")
    return _clean_records(result)


@app.get("/api/analysis/live_accuracy")
def get_live_accuracy(limit: int = 500) -> dict:
    """
    How often real logged decisions' predicted direction matched what
    actually happened next — complements /api/analysis/runs (the
    walk-forward backtest) with a live, after-the-fact check.
    """
    engine = get_engine()
    decisions = pd.read_sql(
        text("SELECT symbol, ts, forecast FROM decisions WHERE forecast IS NOT NULL ORDER BY ts DESC LIMIT :limit"),
        engine,
        params={"limit": limit},
    )
    if decisions.empty:
        return {"hit_rate": None, "n_matured": 0, "rows": []}

    symbol_list = ", ".join(f"'{s}'" for s in decisions["symbol"].unique())
    prices = pd.read_sql(f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts", engine)

    result = compute_forecast_accuracy(decisions, prices)
    if result.empty:
        return {"hit_rate": None, "n_matured": 0, "rows": []}
    return {
        "hit_rate": float(result["hit"].mean()),
        "n_matured": len(result),
        "rows": _clean_records(result.sort_values("ts", ascending=False).head(50)),
    }


@app.get("/api/tests/last")
def get_last_test_run() -> dict | None:
    if not LAST_TEST_RUN_PATH.exists():
        return None
    return json.loads(LAST_TEST_RUN_PATH.read_text())


@app.post("/api/tests/run")
def run_tests() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    summary = output.strip().splitlines()[-1] if output.strip() else ""
    payload = {
        "ts": dt.datetime.now(tz=dt.UTC).isoformat(),
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "summary": summary,
        "output": output,
    }
    LAST_TEST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_TEST_RUN_PATH.write_text(json.dumps(payload))
    return payload


# Static frontend, mounted last so it doesn't shadow /api/* routes.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run("monitoring.dashboard.server:app", host="0.0.0.0", port=8501, reload=False)


if __name__ == "__main__":
    main()
