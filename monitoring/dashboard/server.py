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
import hmac
import json
import subprocess
import sys
import threading
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from execution.broker import get_broker
from features.quant.momentum import adx as compute_adx
from models.regime.trend_chop_classifier import RuleBasedRegime
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.dashboard import report_card, whatif
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LAST_TEST_RUN_PATH = REPO_ROOT / "logs" / "last_test_run.json"

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _require_api_token(request: Request) -> None:
    """
    The gate on every /api route — reads included. Three cases:

    - Token configured: every caller must present it as a bearer header,
      loopback or not — configuring a token means wanting it checked.
    - No token, loopback bind: allowed. The only people who can reach a
      127.0.0.1 dashboard are already on the machine, so local development
      needs no ceremony.
    - No token, non-loopback bind: 403 always. An open interface with an
      endpoint that runs subprocesses must not be one blank env var away
      from the network.

    Reads are gated as well as writes. What this dashboard returns is the
    book — every open position, its size, the model's reasoning for holding
    it, and the equity curve. On paper money that is merely embarrassing;
    the habit of publishing it is what must not survive to real money, and a
    URL that was public for months does not quietly become private later.
    """
    token = settings.dashboard_api_token
    if token:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {token}"):
            raise HTTPException(status_code=403, detail="Missing or invalid bearer token.")
        return
    if settings.dashboard_host not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="This endpoint is disabled: the dashboard is bound to a non-loopback "
                   "interface and DASHBOARD_API_TOKEN is not set.",
        )


# The gate is declared once, on the app, rather than per route: a new
# endpoint added later is then private by default instead of private only
# if its author remembered. tests/test_dashboard_server.py enforces that no
# /api route escapes it.
#
# The static frontend is mounted separately (see the bottom of this file)
# and is deliberately NOT behind the gate — the page has to load before
# anyone can type a token into it.
app = FastAPI(title="Trading System Monitor", dependencies=[Depends(_require_api_token)])


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
    symbol_list = symbol_in_clause(symbols)
    decisions = pd.read_sql(
        "SELECT DISTINCT ON (symbol) symbol, ts, feature_set_id, model_version, "  # noqa: S608 — symbols validated via symbol_in_clause
        "forecast, regime, target_position, executed_position, mode, reasoning "
        f"FROM decisions WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC",
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
            "SELECT ts, symbol, feature_set_id, model_version, forecast, regime, "  # noqa: S608 — WHERE fragment is a fixed literal; values are bind params
            "target_position, executed_position, mode, reasoning "
            f"FROM decisions {where} ORDER BY ts DESC LIMIT :limit OFFSET :offset"
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


# Decision modes that count as the system actually running. Anything else
# (mode='backfill' — replayed history loaded after the fact) is reported
# separately and must never be blended into what's presented as live results.
_LIVE_DECISION_MODES = ("paper", "live")


@app.get("/api/analysis/live_accuracy")
def get_live_accuracy(limit: int = 500) -> dict:
    """
    How often real logged decisions' predicted direction matched what
    actually happened next — complements /api/analysis/runs (the
    walk-forward backtest) with a live, after-the-fact check.

    The top-level hit_rate counts ONLY real decision modes (paper/live).
    Replayed history (mode='backfill') is scored the same way but returned
    under "backfill", clearly separated: blending 130 replayed rows into 23
    real ones would let history masquerade as live performance.
    """
    engine = get_engine()
    decisions = pd.read_sql(
        text("SELECT symbol, ts, forecast, mode FROM decisions WHERE forecast IS NOT NULL ORDER BY ts DESC LIMIT :limit"),
        engine,
        params={"limit": limit},
    )
    empty_bucket = {"hit_rate": None, "n_matured": 0, "rows": []}
    if decisions.empty:
        return {**empty_bucket, "backfill": dict(empty_bucket)}

    symbol_list = symbol_in_clause(decisions["symbol"].unique())
    prices = pd.read_sql(f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts", engine)  # noqa: S608 — symbols validated via symbol_in_clause

    def _scored(subset: pd.DataFrame) -> dict:
        if subset.empty:
            return dict(empty_bucket)
        result = compute_forecast_accuracy(subset, prices)
        if result.empty:
            return dict(empty_bucket)
        return {
            "hit_rate": float(result["hit"].mean()),
            "n_matured": len(result),
            "rows": _clean_records(result.sort_values("ts", ascending=False).head(50)),
        }

    is_live = decisions["mode"].isin(_LIVE_DECISION_MODES)
    return {**_scored(decisions[is_live]), "backfill": _scored(decisions[~is_live])}


@app.get("/api/analysis/report_card")
def get_report_card() -> dict:
    """
    The model's report card: fold-by-fold walk-forward metrics folded down to
    a headline, a grouped-bar chart, and two plain-English callouts about
    whether "the models agreed" is actually buying accuracy. Same
    fail-empty pattern as /api/analysis/runs — an unreachable MLflow means
    an unavailable panel, not a 500.
    """
    try:
        runs = report_card.fetch_fold_runs(settings.mlflow_tracking_uri)
    except Exception:
        return {"available": False, "headline": None, "chart": [], "callouts": []}

    folds = report_card.fold_metrics_frame(runs)
    if folds.empty:
        return {"available": False, "headline": None, "chart": [], "callouts": []}

    headline = report_card.headline_metrics(folds)
    return {
        "available": True,
        "headline": headline,
        "chart": _clean_records(report_card.accuracy_chart_frame(folds)),
        "callouts": [
            report_card.confidence_callout(headline["pct_rows_confident"]),
            report_card.agreement_edge_note(
                headline["directional_accuracy"], headline["directional_accuracy_when_confident"]
            ),
        ],
    }


@app.get("/api/whatif")
def get_whatif(min_agreement: float = 0.8, min_abs_move: float = 0.0) -> dict:
    """
    The what-if threshold playground: re-filter the latest logged screener
    batch at whatever bar the sliders ask for. Read-only — nothing is
    retrained or rescored; the question is only "which of these picks would
    still have made the cut".
    """
    engine = get_engine()
    batch = pd.read_sql(
        text(
            """SELECT ts, symbol, forecast, regime, target_position, executed_position,
                      mode, direction_agreement
               FROM decisions
               WHERE mode = 'paper' AND forecast IS NOT NULL
                 AND ts = (
                     SELECT MAX(ts) FROM decisions WHERE mode = 'paper' AND forecast IS NOT NULL
                 )"""
        ),
        engine,
    )
    if batch.empty:
        return {"available": False, "message": "No scored screener batch logged yet.", "rows": [], "summary": ""}
    if batch["direction_agreement"].isna().all():
        # Rows logged before the unified loop recorded agreement — the
        # sliders would filter on a number that was never written.
        return {
            "available": False,
            "message": "No scored batch yet — the latest picks were logged before model agreement was recorded. "
                       "The playground activates after the next screener run.",
            "rows": [],
            "summary": "",
        }

    filtered = whatif.filter_by_thresholds(batch, min_agreement=min_agreement, min_abs_move=min_abs_move)
    return {
        "available": True,
        "min_agreement": min_agreement,
        "min_abs_move": min_abs_move,
        "n_before": len(batch),
        "n_after": len(filtered),
        "summary": whatif.shortlist_summary(len(batch), len(filtered)),
        "rows": _clean_records(whatif.whatif_table(filtered)),
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
    symbol_list = symbol_in_clause(symbols)
    prices = pd.read_sql(f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts", engine)  # noqa: S608 — symbols validated via symbol_in_clause

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
    symbol_list = symbol_in_clause(symbols)
    df = pd.read_sql(
        "SELECT symbol, ts, headline, sentiment, source FROM news_events "  # noqa: S608 — symbols validated via symbol_in_clause
        f"WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC",
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


# --------------------------------------------------------------------------
# Manual triggers — data ingestion and a full trading cycle, from the UI.
#
# Both run as detached background subprocesses with a visible status; the
# request returns immediately. IMPORTANT SAFETY PROPERTY: the "Run trading
# cycle" button only STARTS the cycle — scripts/run_weekly_cycle.py still
# walks through the Telegram approval gate before any order exists, so the
# button authorizes computation, never trades. Like every /api route these
# sit behind the app-level _require_api_token gate; being state-changing,
# they are the ones that would hurt most if that ever came off.
# --------------------------------------------------------------------------

LAST_JOBS_PATH = REPO_ROOT / "logs" / "last_manual_jobs.json"
_JOB_OUTPUT_TAIL_CHARS = 8000

_JOB_COMMANDS: dict[str, dict] = {
    "init_db": {
        "label": "First-run setup (schema, universe, price history, features)",
        "command": [sys.executable, "-m", "scripts.init_database"],
        # Years of history for ~500 symbols from yfinance, then a full
        # feature build. Minutes rather than hours, but well past what the
        # daily ingest job budgets for.
        "timeout_s": 2 * 3600,
    },
    "ingest": {
        "label": "Data ingestion (prices, full universe)",
        "command": [sys.executable, "-m", "scripts.run_daily_ingest", "--universe", "--source", "yfinance"],
        "timeout_s": 3600,
    },
    "cycle": {
        "label": "Full trading cycle (ingest -> screen -> approval gate -> paper orders)",
        "command": [sys.executable, "-m", "scripts.run_weekly_cycle", "--feature-set-id", settings.feature_set_id],
        # Fundamentals + news for ~500 symbols on the free vendor tier take
        # ~2h alone (see run_weekly_cycle's docstring), plus the approval
        # wait — a short timeout would kill every legitimate run.
        "timeout_s": 4 * 3600,
    },
}


def _fresh_job_state() -> dict:
    return {"status": "never_run", "started_at": None, "finished_at": None, "exit_code": None, "output_tail": ""}


def _load_last_jobs() -> dict[str, dict]:
    """Completed runs survive a server restart, same idea as LAST_TEST_RUN_PATH."""
    states = {name: _fresh_job_state() for name in _JOB_COMMANDS}
    try:
        stored = json.loads(LAST_JOBS_PATH.read_text())
        for name, state in stored.items():
            # A job can't still be "running" in a previous process's lifetime.
            if name in states and state.get("status") in ("finished", "failed"):
                states[name] = {**_fresh_job_state(), **state}
    except (OSError, ValueError):
        pass
    return states


_JOBS: dict[str, dict] = _load_last_jobs()
_JOBS_LOCK = threading.Lock()


def _persist_jobs() -> None:
    try:
        LAST_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAST_JOBS_PATH.write_text(json.dumps(_JOBS))
    except OSError:
        pass  # a failed cache write must not fail the job itself


def _run_job_subprocess(name: str) -> None:
    """The background worker: run the command, record how it ended."""
    spec = _JOB_COMMANDS[name]
    try:
        result = subprocess.run(  # noqa: S603 — fixed command list, no user input
            spec["command"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=spec["timeout_s"]
        )
        output = (result.stdout or "") + (result.stderr or "")
        status = "finished" if result.returncode == 0 else "failed"
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        output = f"Timed out after {spec['timeout_s']}s and was killed."
        status, exit_code = "failed", None
    except Exception as exc:  # a crashed worker must still leave a readable status
        output = f"{type(exc).__name__}: {exc}"
        status, exit_code = "failed", None

    with _JOBS_LOCK:
        _JOBS[name].update(
            status=status,
            finished_at=dt.datetime.now(tz=dt.UTC).isoformat(),
            exit_code=exit_code,
            output_tail=output[-_JOB_OUTPUT_TAIL_CHARS:],
        )
        _persist_jobs()


def _spawn(target, name: str) -> None:
    """Thread launch, separated so tests can run the worker synchronously."""
    threading.Thread(target=target, args=(name,), daemon=True, name=f"manual-job-{name}").start()


@app.get("/api/jobs")
def get_jobs() -> dict:
    """Status of the manual-trigger jobs."""
    with _JOBS_LOCK:
        return {
            name: {"label": _JOB_COMMANDS[name]["label"], **state} for name, state in _JOBS.items()
        }


@app.post("/api/jobs/{job_name}/run")
def run_job(job_name: str) -> dict:
    if job_name not in _JOB_COMMANDS:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_name!r}. Available: {sorted(_JOB_COMMANDS)}.")
    with _JOBS_LOCK:
        if _JOBS[job_name]["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"{job_name} is already running (started {_JOBS[job_name]['started_at']}) — refusing a second copy.",
            )
        _JOBS[job_name] = {
            **_fresh_job_state(),
            "status": "running",
            "started_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        }
        snapshot = dict(_JOBS[job_name])
    _spawn(_run_job_subprocess, job_name)
    return {"job": job_name, **snapshot}


# Static frontend, mounted last so it doesn't shadow /api/* routes. A mount
# is not a route, so the app-level token dependency does not apply to it —
# which is what we want: the page, its JS and its CSS have to be reachable
# for anyone to type a token in. The page ships no data of its own; every
# number on it arrives through a gated /api call.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    # Loopback by default — exposing the dashboard to the network is an
    # explicit DASHBOARD_HOST=0.0.0.0 decision (the Docker image makes it).
    # The port comes from $PORT when a host injects one; see settings.
    uvicorn.run(
        "monitoring.dashboard.server:app",
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
