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

import base64
import datetime as dt
import hmac
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine
from execution.broker import get_broker
from features.quant.momentum import adx as compute_adx
from models.regime.trend_chop_classifier import RuleBasedRegime
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LAST_TEST_RUN_PATH = REPO_ROOT / "logs" / "last_test_run.json"

app = FastAPI(title="Trading System Monitor")

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _check_basic_auth(request: Request) -> PlainTextResponse | None:
    """HTTP Basic Auth for every request (API and static page alike).

    Returns a response to short-circuit with, or None to let the request
    through. With DASHBOARD_PASSWORD unset, only requests arriving on a
    loopback interface are served — any other interface fails closed instead of
    exposing positions and the test runner to the public internet.
    """
    password = settings.dashboard_password
    server_host = (request.scope.get("server") or ("", 0))[0]
    if not password:
        if server_host in _LOOPBACK_HOSTS:
            return None
        return PlainTextResponse(
            "DASHBOARD_PASSWORD is not set; refusing to serve on a non-loopback interface.",
            status_code=503,
        )
    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    ok = False
    if scheme.lower() == "basic":
        try:
            user, _, supplied = base64.b64decode(encoded).decode().partition(":")
        except (ValueError, UnicodeDecodeError):
            user = supplied = ""
        ok = hmac.compare_digest(user, settings.dashboard_user) & hmac.compare_digest(supplied, password)
    if ok:
        return None
    return PlainTextResponse(
        "Unauthorized", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Trading System Monitor"'}
    )


@app.middleware("http")
async def _basic_auth_middleware(request: Request, call_next):
    return _check_basic_auth(request) or await call_next(request)


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


def _price_at_or_before(sym_prices: pd.DataFrame, ts) -> float | None:
    """Nearest known close at or before `ts`; falls back to the earliest known close if none exists."""
    before = sym_prices[sym_prices["ts"] <= ts]
    if not before.empty:
        return float(before.iloc[-1]["close"])
    if not sym_prices.empty:
        return float(sym_prices.iloc[0]["close"])
    return None


@app.get("/api/trades/closed")
def get_closed_trades(limit: int = 100) -> list[dict]:
    """
    Reconstructs realized round-trip trades from the decisions log: an
    episode starts at the first decision that opens a nonzero position for a
    symbol and ends at the next decision that flattens it (executed_position
    == 0), using nearest-known prices at each end for entry/exit. This is an
    approximation, not a broker-verified fill record -- if a position was
    resized mid-episode (e.g. re-picked at a different weight the following
    week), the original entry size/price is used for the whole episode
    rather than tracking each resize individually.
    """
    engine = get_engine()
    decisions = pd.read_sql(
        text("SELECT symbol, ts, target_position, executed_position, mode FROM decisions WHERE mode = 'paper' ORDER BY symbol, ts"),
        engine,
    )
    if decisions.empty:
        return []

    symbols = decisions["symbol"].unique().tolist()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    prices = pd.read_sql(f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts", engine)

    trades: list[dict] = []
    for symbol, group in decisions.groupby("symbol"):
        group = group.sort_values("ts").reset_index(drop=True)
        sym_prices = prices[prices["symbol"] == symbol].sort_values("ts")
        entry = None
        for _, row in group.iterrows():
            executed = row["executed_position"]
            if executed is None or pd.isna(executed):
                continue
            if entry is None:
                if executed != 0:
                    entry = row
                continue
            if executed == 0:
                entry_price = _price_at_or_before(sym_prices, entry["ts"])
                exit_price = _price_at_or_before(sym_prices, row["ts"])
                shares = entry["executed_position"]
                if entry_price is not None and exit_price is not None and shares:
                    pnl = (exit_price - entry_price) * shares
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "long" if shares > 0 else "short",
                            "entry_ts": entry["ts"].isoformat(),
                            "exit_ts": row["ts"].isoformat(),
                            "shares": float(shares),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "realized_pnl": float(pnl),
                            "realized_pnl_pct": float((exit_price / entry_price - 1) * (1 if shares > 0 else -1)),
                        }
                    )
                entry = None
            # else: still open (possibly resized) -- keep the original entry, see docstring.

    trades.sort(key=lambda t: t["exit_ts"], reverse=True)
    return trades[:limit]


@app.get("/api/positions/news")
def get_positions_news(limit_per_symbol: int = 8) -> dict[str, list[dict]]:
    """Recent headlines feeding each held position's sentiment score, not just the aggregated number."""
    broker = get_broker()
    symbols = [s for s, q in broker.get_positions().items() if q != 0]
    if not symbols:
        return {}

    engine = get_engine()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    df = pd.read_sql(
        f"""SELECT symbol, ts, headline, sentiment, source FROM news_events
            WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC""",
        engine,
    )
    result: dict[str, list[dict]] = {s: [] for s in symbols}
    for symbol, group in df.groupby("symbol"):
        result[symbol] = _clean_records(group.head(limit_per_symbol))
    return result


@app.get("/api/regime_history")
def get_regime_history(market_proxy: str = "SPY") -> list[dict]:
    """Dense daily trend/chop classification (ADX on the market proxy) for overlaying on the equity chart."""
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT ts, high, low, close FROM prices WHERE symbol = :symbol ORDER BY ts"),
        engine,
        params={"symbol": market_proxy},
    )
    if len(df) < 20:
        return []
    df["adx"] = compute_adx(df["high"], df["low"], df["close"])
    df = df.dropna(subset=["adx"]).copy()
    if df.empty:
        return []
    df["regime"] = RuleBasedRegime().predict(df["adx"])
    return _clean_records(df[["ts", "regime", "adx"]])


@app.get("/api/analysis/feature_frequency")
def get_feature_frequency(limit: int = 200) -> list[dict]:
    """
    How often each feature has shown up in the top-5 SHAP drivers across
    recent decisions, and its average |contribution| when it does -- catches
    the model leaning on one signal (e.g. always the CPI countdown) rather
    than genuinely varying its reasoning. Only decisions logged after the
    7-phase reasoning model shipped carry the structured `top_features` data
    this needs; older rows are silently skipped.
    """
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT reasoning FROM decisions WHERE reasoning IS NOT NULL ORDER BY ts DESC LIMIT :limit"),
        engine,
        params={"limit": limit},
    )

    counts: dict[str, int] = {}
    total_abs_contribution: dict[str, float] = {}
    for reasoning in df["reasoning"]:
        if isinstance(reasoning, str):
            reasoning = json.loads(reasoning)
        if not isinstance(reasoning, list):
            continue
        phase2 = next((p for p in reasoning if p.get("phase") == 2), None)
        if not phase2:
            continue
        for f in phase2.get("top_features", []):
            name = f["feature_name"]
            counts[name] = counts.get(name, 0) + 1
            total_abs_contribution[name] = total_abs_contribution.get(name, 0.0) + abs(f["contribution"])

    result = [
        {"feature_name": name, "times_in_top5": count, "avg_abs_contribution": total_abs_contribution[name] / count}
        for name, count in counts.items()
    ]
    result.sort(key=lambda r: r["times_in_top5"], reverse=True)
    return result


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
