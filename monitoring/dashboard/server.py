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
import hashlib
import hmac
import html
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs

import mlflow
import pandas as pd
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from execution.broker import get_broker
from execution.broker_alpaca import AlpacaBroker
from execution.client_crypto import decrypt_credential, encrypt_credential, hash_password, verify_password
from execution.client_fanout import onboard_client
from execution.hold_rules import load_exit_levels
from features.quant.momentum import adx as compute_adx
from models.regime.trend_chop_classifier import RuleBasedRegime
from monitoring import drift
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.dashboard import report_card
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# One shared password gates the whole dashboard — the static page and every
# /api route (reads included) alike. There used to be a second, separate
# operator token for the mutating endpoints on top of HTTP Basic Auth for
# everything else; that split is gone. Log in once at /login and the
# session cookie it sets covers everything until it's cleared (see
# _session_token below for how logging out and changing the password both
# invalidate it).
app = FastAPI(title="Trading System Monitor")

_SESSION_COOKIE = "dashboard_session"
_SESSION_MAX_AGE_S = 30 * 24 * 3600  # 30 days — a shared operator password, not a per-user login


def _session_token(password: str) -> str:
    """
    Stateless session token: an HMAC of a fixed label under the current
    dashboard password. No server-side session store to manage — anyone who
    supplied the right password once is re-recognized by this token on
    later requests, and changing DASHBOARD_PASSWORD invalidates every
    outstanding cookie at once, since the recomputed token changes with it.
    """
    return hmac.new(password.encode(), b"trading-system-dashboard-session", hashlib.sha256).hexdigest()


def _is_authenticated(request: Request) -> bool:
    password = settings.dashboard_password
    if not password:
        return True  # the loopback-only fallback below is what actually gates this case
    cookie = request.cookies.get(_SESSION_COOKIE, "")
    return hmac.compare_digest(cookie, _session_token(password))


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Trading System Monitor — Log in</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: #0b0e14; color: #e6e9f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  form {{
    background: #161b26; border: 1px solid #262c3b; border-radius: 10px;
    padding: 32px 28px; width: 280px; text-align: center;
  }}
  h1 {{ font-size: 16px; margin: 0 0 20px; }}
  input {{
    width: 100%; padding: 9px 12px; margin-bottom: 12px; border-radius: 6px;
    border: 1px solid #262c3b; background: #131722; color: #e6e9f0; font-size: 14px; box-sizing: border-box;
  }}
  button {{
    width: 100%; padding: 9px 12px; border-radius: 6px; border: 1px solid #5b8cff;
    background: #5b8cff; color: white; font-size: 14px; font-weight: 600; cursor: pointer;
  }}
  .error {{ color: #e5484d; font-size: 12px; margin: -6px 0 14px; }}
</style>
</head>
<body>
  <form method="post" action="/login">
    <h1>Trading System Monitor</h1>
    {error_html}
    <input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password" />
    <button type="submit">Enter</button>
  </form>
</body>
</html>"""


@app.get("/login", include_in_schema=False)
def login_form(request: Request) -> Response:
    if _is_authenticated(request):
        return RedirectResponse("/")
    return HTMLResponse(_LOGIN_PAGE.format(error_html=""))


@app.post("/login", include_in_schema=False)
async def login_submit(request: Request) -> Response:
    # Parsed by hand (stdlib urllib) rather than FastAPI's Form(...), which
    # pulls in python-multipart as a new dependency for a single field on a
    # login form the app itself renders — not worth it for one text input.
    body = await request.body()
    password = (parse_qs(body.decode("utf-8")).get("password") or [""])[0]

    configured = settings.dashboard_password
    if not configured or not hmac.compare_digest(password, configured):
        return HTMLResponse(
            _LOGIN_PAGE.format(error_html='<div class="error">Wrong password.</div>'), status_code=401
        )
    resp = RedirectResponse("/", status_code=303)
    server_host = (request.scope.get("server") or ("", 0))[0]
    resp.set_cookie(
        _SESSION_COOKIE,
        _session_token(configured),
        max_age=_SESSION_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        # Secure unless we're plainly in loopback dev — a real deploy is
        # always HTTPS, and browsers treat 127.0.0.1/localhost as
        # "potentially trustworthy" so Secure cookies still round-trip there
        # over plain http during local testing.
        secure=server_host not in _LOOPBACK_HOSTS,
    )
    return resp


@app.get("/logout", include_in_schema=False)
def logout() -> Response:
    resp = RedirectResponse("/login")
    resp.delete_cookie(_SESSION_COOKIE)
    return resp


def _check_dashboard_auth(request: Request) -> PlainTextResponse | JSONResponse | RedirectResponse | None:
    """
    The one gate in front of EVERY request — the static page, every /api
    route (reads included), and /login itself. Returns a response to
    short-circuit with, or None to let the request through.

    - DASHBOARD_PASSWORD unset: served only on a loopback bind. The only
      people who can reach a 127.0.0.1 dashboard are already on the
      machine, so local development needs no ceremony; any other interface
      fails closed rather than exposing positions and the test runner to
      the public internet on a blank env var.
    - DASHBOARD_PASSWORD set: /login and /logout stay reachable logged out
      (the login flow has to work before anyone is logged in); everything
      else needs the session cookie /login sets. An unauthenticated /api
      call gets a 401 JSON body instead of a redirect — the frontend's own
      fetch wrapper sends the browser to /login on that, rather than a
      fetch() silently following a redirect into an HTML page.
    """
    # The client portal (/portal*, /api/portal/*) is gated by its own
    # per-client password, not the operator's DASHBOARD_PASSWORD — a client
    # has no reason to know that password, and shouldn't need it to see
    # their own account. See _require_client below for that gate.
    if request.url.path == "/portal" or request.url.path.startswith(("/portal/", "/api/portal/")):
        return None

    password = settings.dashboard_password
    server_host = (request.scope.get("server") or ("", 0))[0]
    if not password:
        if server_host in _LOOPBACK_HOSTS:
            return None
        return PlainTextResponse(
            "DASHBOARD_PASSWORD is not set; refusing to serve on a non-loopback interface.",
            status_code=503,
        )

    if request.url.path in ("/login", "/logout"):
        return None

    if _is_authenticated(request):
        return None

    if request.url.path.startswith("/api"):
        return JSONResponse({"detail": "Not authenticated. Log in at /login."}, status_code=401)
    return RedirectResponse("/login")


@app.middleware("http")
async def _dashboard_auth_middleware(request: Request, call_next):
    return _check_dashboard_auth(request) or await call_next(request)


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

    # The take-profit/stop-loss pair each open position was actually approved
    # with (execution/exit_levels.py, enforced weekly by execution/hold_rules.py).
    # Best-effort: a dashboard panel going blank on a DB hiccup beats the whole
    # positions endpoint 500ing over a field nothing above this depended on before.
    try:
        levels_by_symbol = load_exit_levels(engine)
    except Exception:
        logger.exception("Failed to load exit levels — positions will show without them.")
        levels_by_symbol = {}

    for p in positions:
        levels = levels_by_symbol.get(p["symbol"])
        p["exit_levels"] = (
            {
                "take_profit_pct": levels.take_profit_pct,
                "stop_loss_pct": levels.stop_loss_pct,
                "derived": levels.derived,
            }
            if levels
            else None
        )

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


@app.get("/api/analysis/drift")
def get_model_drift(consecutive_weeks: int = drift.DEFAULT_DRIFT_WEEKS) -> dict:
    """
    Read-only model-drift diagnostics (see monitoring/drift.py for the
    2026-08-28 decision behind this): whether live directional accuracy has
    fallen below the walk-forward baseline for several straight weeks, and
    which recent top-driver features have shown up in decisions that turned
    out wrong more often than right.

    Deliberately does nothing but report. Nothing here retrains, reweights,
    or otherwise changes the model — acting on what this shows (dropping a
    feature, running models/train.py early, tightening the confidence bar)
    stays a human call, same as every other change to what the model does.
    """
    empty = {"available": False, "message": "No live decisions logged yet.", "weekly": [], "accuracy_flag": None, "feature_drag": []}
    engine = get_engine()
    decisions = pd.read_sql(
        text(
            "SELECT symbol, ts, forecast, reasoning FROM decisions "
            "WHERE forecast IS NOT NULL AND mode IN ('paper', 'live') ORDER BY ts DESC LIMIT 1000"
        ),
        engine,
    )
    if decisions.empty:
        return empty

    symbol_list = symbol_in_clause(decisions["symbol"].unique())
    prices = pd.read_sql(f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts", engine)  # noqa: S608 — symbols validated via symbol_in_clause
    scored = compute_forecast_accuracy(decisions[["symbol", "ts", "forecast"]], prices)
    if scored.empty:
        return {**empty, "message": "No live decisions have matured yet (need a later price bar to grade against)."}

    weekly = drift.weekly_hit_rate(scored)

    baseline_accuracy = None
    try:
        runs = report_card.fetch_fold_runs(settings.mlflow_tracking_uri)
        folds = report_card.fold_metrics_frame(runs)
        baseline_accuracy = report_card.headline_metrics(folds)["directional_accuracy"]
    except Exception:
        logger.exception("Could not load the walk-forward baseline for the drift check — accuracy_flag will say why.")

    return {
        "available": True,
        "baseline_accuracy": baseline_accuracy,
        "weekly": _clean_records(weekly),
        "accuracy_flag": drift.accuracy_drift_flag(weekly, baseline_accuracy, consecutive_weeks=consecutive_weeks),
        "feature_drag": drift.feature_drag(decisions[["symbol", "ts", "reasoning"]], scored),
    }


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
        "SELECT symbol, ts, headline, sentiment, sentiment_reason, sentiment_relevant, source FROM news_events "  # noqa: S608 — symbols validated via symbol_in_clause
        f"WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC",
        engine,
    )
    # A headline a news vendor mistagged onto this symbol (see
    # data/schema/010_news_sentiment_relevance.sql) shouldn't show up as
    # "this symbol's news" at all -- IS NOT FALSE keeps NULL (not yet
    # scored / scored before this column existed) visible as before.
    if "sentiment_relevant" in df.columns:
        df = df[df["sentiment_relevant"] != False]  # noqa: E712 — NaN-safe: only an explicit False is dropped
    result: dict[str, list[dict]] = {s: [] for s in symbols}
    for symbol, group in df.groupby("symbol"):
        result[symbol] = _clean_records(group.head(limit_per_symbol))
    return result


@app.get("/api/news/ingestion_status")
def get_news_ingestion_status() -> dict:
    """
    Backs the dashboard's "news ingestion is live" indicator. Deliberately
    separate from market hours: data/ingest/news_stream.py's websocket runs
    continuously (see its module docstring) and Benzinga/Alpaca news
    publishes outside NYSE regular trading hours too, so this must never be
    read as "closed" just because /api/market_clock says the market isn't
    open.

    Reports both the most recent headline's timestamp AND how many
    headlines landed in the last hour, rather than a single row — one
    stray old row can't read as "live", and a quiet-but-connected stretch
    (overnight, weekend, a slow news day) doesn't read as "broken" off one
    data point alone. The dashboard is left to apply its own staleness
    thresholds to seconds_since_latest; this endpoint just reports facts.
    """
    engine = get_engine()
    latest = pd.read_sql(text("SELECT MAX(ts) AS latest_ts FROM news_events"), engine)
    latest_ts = latest["latest_ts"].iloc[0] if not latest.empty else None
    now = dt.datetime.now(tz=dt.UTC)

    count_last_hour = pd.read_sql(
        text("SELECT COUNT(*) AS n FROM news_events WHERE ts >= :cutoff"),
        engine,
        params={"cutoff": now - dt.timedelta(hours=1)},
    )["n"].iloc[0]

    seconds_since_latest = None
    latest_ts_iso = None
    if latest_ts is not None and not pd.isna(latest_ts):
        ts = latest_ts if latest_ts.tzinfo is not None else latest_ts.tz_localize("UTC")
        seconds_since_latest = (now - ts).total_seconds()
        latest_ts_iso = ts.isoformat()

    return {
        "latest_ts": latest_ts_iso,
        "seconds_since_latest": seconds_since_latest,
        "count_last_hour": int(count_last_hour),
        "checked_at": now.isoformat(),
    }


@app.get("/api/market_clock")
def get_market_clock() -> dict:
    """
    NYSE regular-trading-hours clock for the dashboard's market-hours
    label — separate from /api/news/ingestion_status on purpose, since
    news ingestion runs regardless of this. See AlpacaBroker.get_clock /
    IBKRBroker.get_clock for which source answers this depending on
    BROKER.
    """
    return get_broker().get_clock()


@app.get("/api/news/live")
def get_live_news(limit: int = Query(default=150, le=1000)) -> list[dict]:
    """
    The most recent news across the whole universe (not just held
    positions — see /api/positions/news for that narrower feed), grouped by
    story rather than listed one row per symbol.

    One article commonly fans out to several rows in news_events — a Fed
    headline tagged to five names writes five rows with the same headline
    and timestamp, one per symbol (see data/ingest/news_stream.py and
    data/ingest/news.py). Grouping by (headline, ts) here turns that back
    into "one story, these symbols, this sentiment each" instead of making
    that story read as five unrelated news items on the dashboard.

    Sentiment (and its accompanying reason) can be null for a little while:
    the real-time stream writes headlines to news_events immediately, but
    the LLM sentiment pass (features/qualitative/sentiment.py::
    backfill_unscored_news) only runs piggybacked on the hourly
    contradiction monitor and the weekly cycle — a very fresh headline can
    show "not yet scored" for up to about an hour during market hours.

    A symbol the vendor mistagged onto a story (sentiment_relevant is
    explicitly False — see data/schema/010_news_sentiment_relevance.sql,
    e.g. a Ballmer/Gates headline entirely about MSFT tagged with NYT) is
    dropped from that story's symbol list entirely, not just its
    sentiment blanked out; a story left with no relevant symbols is
    dropped from the response. NULL (not yet scored) stays visible as
    before.
    """
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT symbol, ts, headline, source, sentiment, sentiment_reason, sentiment_relevant FROM news_events "
            "WHERE headline IS NOT NULL AND headline != '' ORDER BY ts DESC LIMIT :limit"
        ),
        engine,
        params={"limit": limit},
    )
    if df.empty:
        return []

    grouped: dict[tuple, dict] = {}
    order: list[tuple] = []
    for _, row in df.iterrows():
        relevant = row["sentiment_relevant"]
        # pd.isna(relevant) is True for NULL (not yet scored / pre-migration)
        # -- those stay visible. `relevant is False` would miss a numpy.bool_
        # from the DB driver, so compare via bool() instead.
        if not pd.isna(relevant) and not bool(relevant):
            continue
        ts = row["ts"]
        ts_key = ts.isoformat() if hasattr(ts, "isoformat") else ts
        key = (row["headline"], ts_key)
        if key not in grouped:
            grouped[key] = {"headline": row["headline"], "ts": ts_key, "source": row["source"], "symbols": []}
            order.append(key)
        sentiment = row["sentiment"]
        reason = row["sentiment_reason"]
        grouped[key]["symbols"].append(
            {
                "symbol": row["symbol"],
                "sentiment": None if pd.isna(sentiment) else float(sentiment),
                "sentiment_reason": None if pd.isna(reason) else reason,
            }
        )
    return [grouped[k] for k in order if grouped[k]["symbols"]]


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


########################################################################
# Client accounts — operator-side management (admin-gated, above) and the
# client-facing read-only portal (its own password, see _require_client).
# See execution/client_fanout.py and execution/client_crypto.py for the
# trading and encryption logic this UI drives; CLIENT_TRADING_ENABLED
# (config/settings.py) is the separate kill switch that has to be on
# before any of this can place a real order.
########################################################################


class _NewClientRequest(BaseModel):
    name: str
    alpaca_api_key: str
    alpaca_api_secret: str
    margin_enabled: bool = False
    leverage_multiplier: int = 1
    password: str


class _ResetPasswordRequest(BaseModel):
    new_password: str


class _LeverageRequest(BaseModel):
    leverage_multiplier: int


# Hard backstop against a fat-fingered input (e.g. "20" meant as "2.0x")
# wiping out a client's account. Deliberately not settings-driven -- see
# data/schema/012_client_leverage.sql's comment on this same number for why
# raising it is meant to be a deliberate two-place code change (this
# constant AND that migration's CHECK constraint), not an env var someone
# bumps by accident. The DB constraint is the real backstop; this just gives
# a clean 400 with a helpful message instead of a raw constraint-violation
# error surfacing to the operator.
_MAX_CLIENT_LEVERAGE = 3


def _validate_leverage(leverage_multiplier: int, margin_enabled: bool) -> str | None:
    """Returns an error message if invalid, else None. Shared by create and update so the two paths can't drift."""
    if not (1 <= leverage_multiplier <= _MAX_CLIENT_LEVERAGE):
        return f"leverage_multiplier must be between 1 and {_MAX_CLIENT_LEVERAGE}."
    if leverage_multiplier > 1 and not margin_enabled:
        return "leverage_multiplier above 1x requires a margin-enabled account."
    return None


def _mask_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "•" * len(api_key)
    return f"{api_key[:4]}…{api_key[-4:]}"


@app.get("/api/clients/trading_status")
def get_client_trading_status() -> dict:
    """Whether CLIENT_TRADING_ENABLED is on — the dashboard's Clients tab shows this plainly,
    since adding a client is otherwise silent about whether it actually did anything yet."""
    return {"enabled": settings.client_trading_enabled}


@app.post("/api/clients")
def create_client(body: _NewClientRequest) -> dict:
    """
    Adds a client: verifies the Alpaca credentials actually work (a
    read-only get_account() call — this happens even if CLIENT_TRADING_ENABLED
    is still off, so a typo'd key is caught at add-time, not at the next
    fan-out), encrypts them at rest, hashes the portal password, and — if
    trading is enabled — immediately buys the client into whatever the
    master account currently holds (the "buy in right away" decision).
    """
    if not body.name.strip() or not body.password:
        return JSONResponse({"detail": "Name and password are required."}, status_code=400)

    leverage_error = _validate_leverage(body.leverage_multiplier, body.margin_enabled)
    if leverage_error:
        return JSONResponse({"detail": leverage_error}, status_code=400)

    try:
        verify_broker = AlpacaBroker(
            mode="live", confirm_live=True, api_key=body.alpaca_api_key, secret_key=body.alpaca_api_secret
        )
        verify_broker.get_account()
    except Exception as e:
        return JSONResponse({"detail": f"Could not connect to this Alpaca account: {e}"}, status_code=400)

    engine = get_engine()
    try:
        with engine.begin() as conn:
            result = conn.exec_driver_sql(
                "INSERT INTO clients "
                "(name, alpaca_api_key_encrypted, alpaca_api_secret_encrypted, margin_enabled, leverage_multiplier, password_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    body.name.strip(),
                    encrypt_credential(body.alpaca_api_key),
                    encrypt_credential(body.alpaca_api_secret),
                    body.margin_enabled,
                    body.leverage_multiplier,
                    hash_password(body.password),
                ),
            )
            client_id = result.scalar_one()
    except Exception as e:
        # Most likely the UNIQUE(name) constraint — surfaced as 400, not 500,
        # since it's a caller mistake (a duplicate name), not a server fault.
        return JSONResponse({"detail": f"Could not create client: {e}"}, status_code=400)

    buy_in_note = "Client trading is off (CLIENT_TRADING_ENABLED) — added but not yet buying in."
    if settings.client_trading_enabled:
        try:
            master_broker = get_broker()  # never passes confirm_live=True — paper-only by construction
            onboard_client(client_id, master_broker, engine)
            buy_in_note = "Buy-in submitted against the master account's current holdings."
        except Exception as e:
            logger.exception("Onboarding buy-in failed for new client %s", client_id)
            buy_in_note = f"Client added, but the immediate buy-in failed: {e}. Retry manually if needed."

    return {
        "id": client_id, "name": body.name.strip(), "margin_enabled": body.margin_enabled,
        "leverage_multiplier": body.leverage_multiplier, "active": True, "buy_in": buy_in_note,
    }


@app.get("/api/clients")
def list_clients() -> list[dict]:
    engine = get_engine()
    df = pd.read_sql(
        "SELECT id, name, alpaca_api_key_encrypted, margin_enabled, leverage_multiplier, active, created_at, "
        "trading_paused, pause_reason "
        "FROM clients ORDER BY created_at DESC",
        engine,
    )
    clients = []
    for _, row in df.iterrows():
        try:
            key_preview = _mask_key(decrypt_credential(row["alpaca_api_key_encrypted"]))
        except Exception:
            key_preview = "(could not decrypt)"
        clients.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "api_key_preview": key_preview,
                "margin_enabled": bool(row["margin_enabled"]),
                "leverage_multiplier": int(row["leverage_multiplier"]),
                "active": bool(row["active"]),
                # Client self-service pause (their own "Liquidate now" button,
                # or an auto-triggered max_drawdown/profit_target — see
                # execution/client_risk_controls.py) — surfaced here so the
                # operator can see a client paused themselves out without
                # having to check the portal or client_orders directly.
                "trading_paused": bool(row["trading_paused"]),
                "pause_reason": row["pause_reason"],
                "created_at": row["created_at"].isoformat() if pd.notna(row["created_at"]) else None,
            }
        )
    return clients


@app.post("/api/clients/{client_id}/leverage")
def set_client_leverage(client_id: int, body: _LeverageRequest) -> dict:
    """
    Operator-only leverage control — the client-facing portal has no
    equivalent endpoint, deliberately (see client_fanout.py's module
    docstring). Looks up the client's current margin_enabled to validate
    against, rather than trusting the caller, since a stale/wrong
    margin_enabled in the request body would otherwise let leverage > 1x
    through validation for an account that can't actually support it — the
    DB CHECK constraint (data/schema/012_client_leverage.sql) would still
    catch that at the UPDATE itself, but a fresh lookup gives the operator a
    clean, specific 400 instead of a raw constraint-violation error.
    """
    engine = get_engine()
    existing = pd.read_sql(
        text("SELECT margin_enabled FROM clients WHERE id = :id"), engine, params={"id": client_id}
    )
    if existing.empty:
        return JSONResponse({"detail": "No such client."}, status_code=404)

    leverage_error = _validate_leverage(body.leverage_multiplier, bool(existing["margin_enabled"].iloc[0]))
    if leverage_error:
        return JSONResponse({"detail": leverage_error}, status_code=400)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET leverage_multiplier = %s WHERE id = %s", (body.leverage_multiplier, client_id)
        )
    return {"id": client_id, "leverage_multiplier": body.leverage_multiplier}


@app.post("/api/clients/{client_id}/deactivate")
def deactivate_client(client_id: int) -> dict:
    """
    Takes a client out of every future fan-out (new trades, rebalances,
    closes) without touching whatever they currently hold — deactivating is
    not liquidating. Their existing positions stay exactly as they are
    until the client or operator acts on them directly.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE clients SET active = FALSE WHERE id = %s", (client_id,))
    return {"id": client_id, "active": False}


@app.post("/api/clients/{client_id}/reactivate")
def reactivate_client(client_id: int) -> dict:
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE clients SET active = TRUE WHERE id = %s", (client_id,))
    return {"id": client_id, "active": True}


@app.post("/api/clients/{client_id}/reset_password")
def reset_client_password(client_id: int, body: _ResetPasswordRequest) -> dict:
    if not body.new_password:
        return JSONResponse({"detail": "new_password is required."}, status_code=400)
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET password_hash = %s WHERE id = %s", (hash_password(body.new_password), client_id)
        )
    return {"id": client_id, "password_reset": True}


@app.get("/api/clients/{client_id}/orders")
def get_client_orders(client_id: int, limit: int = 100) -> list[dict]:
    """The operator's view of one client's fan-out history — same table the client's own portal reads."""
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT symbol, side, target_position_pct, target_shares, status, alpaca_order_id, error_message, ts "
            "FROM client_orders WHERE client_id = :client_id ORDER BY ts DESC LIMIT :limit"
        ),
        engine,
        params={"client_id": client_id, "limit": limit},
    )
    return _clean_records(df)


########################################################################
# Client portal — a separate login (client name + their own password, see
# execution/client_crypto.py) gating a read-only view of just that
# client's own positions, account, and trade history. Deliberately does
# NOT expose the model's reasoning, forecasts, or sentiment/news feed (the
# "results only" decision) — a client sees what happened to their money,
# not the strategy behind it.
########################################################################

_CLIENT_SESSION_COOKIE = "client_portal_session"
_CLIENT_SESSION_MAX_AGE_S = 30 * 24 * 3600


def _client_session_token(password_hash: str) -> str:
    """
    Self-invalidating exactly like the operator session (_session_token
    above): the token is an HMAC keyed by the client's OWN password hash,
    so resetting a client's password invalidates their outstanding cookie
    without a separate server-side session store.
    """
    return hmac.new(password_hash.encode(), b"client-portal-session", hashlib.sha256).hexdigest()


def _require_client(request: Request) -> dict | None:
    """Returns {id, name} for a valid client session cookie, or None."""
    cookie = request.cookies.get(_CLIENT_SESSION_COOKIE, "")
    if "." not in cookie:
        return None
    client_id_s, token = cookie.split(".", 1)
    try:
        client_id = int(client_id_s)
    except ValueError:
        return None
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT id, name, password_hash, active FROM clients WHERE id = :id"),
        engine,
        params={"id": client_id},
    )
    if df.empty or not bool(df.iloc[0]["active"]):
        return None
    row = df.iloc[0]
    if not hmac.compare_digest(token, _client_session_token(row["password_hash"])):
        return None
    return {"id": int(row["id"]), "name": row["name"]}


_CLIENT_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Client Portal — Log in</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: #0b0e14; color: #e6e9f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  form {{
    background: #161b26; border: 1px solid #262c3b; border-radius: 10px;
    padding: 32px 28px; width: 280px; text-align: center;
  }}
  h1 {{ font-size: 16px; margin: 0 0 20px; }}
  input {{
    width: 100%; padding: 9px 12px; margin-bottom: 12px; border-radius: 6px;
    border: 1px solid #262c3b; background: #131722; color: #e6e9f0; font-size: 14px; box-sizing: border-box;
  }}
  button {{
    width: 100%; padding: 9px 12px; border-radius: 6px; border: 1px solid #5b8cff;
    background: #5b8cff; color: white; font-size: 14px; font-weight: 600; cursor: pointer;
  }}
  .error {{ color: #e5484d; font-size: 12px; margin: -6px 0 14px; }}
</style>
</head>
<body>
  <form method="post" action="/portal/login">
    <h1>Client Portal</h1>
    {error_html}
    <input type="text" name="name" placeholder="Name" autofocus autocomplete="username" />
    <input type="password" name="password" placeholder="Password" autocomplete="current-password" />
    <button type="submit">Log in</button>
  </form>
</body>
</html>"""


# Self-contained (inline CSS + JS), same as _CLIENT_LOGIN_PAGE above, rather
# than a separate static/portal.js file — a portal.js served through the
# root StaticFiles mount would hit the OPERATOR'S auth gate (it doesn't
# start with /portal/), locking every client out of their own page. Built
# with str.replace() rather than .format(): the JS below is full of literal
# { } that .format() would otherwise treat as fields.
_CLIENT_PORTAL_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Client Portal</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0b0e14; color: #e6e9f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid #262c3b; background: #12151f;
  }
  header h1 { font-size: 16px; margin: 0; }
  header a { color: #9aa4b8; font-size: 13px; text-decoration: none; }
  main { max-width: 900px; margin: 0 auto; padding: 20px; }
  section { margin-bottom: 28px; }
  h2 { font-size: 14px; color: #9aa4b8; text-transform: uppercase; letter-spacing: 0.03em; margin: 0 0 10px; }
  .summary-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .stat-card {
    background: #161b26; border: 1px solid #262c3b; border-radius: 10px;
    padding: 12px 16px; min-width: 140px;
  }
  .stat-card .label { font-size: 11px; color: #9aa4b8; text-transform: uppercase; }
  .stat-card .value { font-size: 20px; font-weight: 600; margin-top: 4px; }
  .table-wrap { background: #161b26; border: 1px solid #262c3b; border-radius: 10px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #262c3b; white-space: nowrap; }
  th { color: #9aa4b8; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  tbody tr:last-child td { border-bottom: none; }
  .empty-state { color: #9aa4b8; font-size: 13px; padding: 16px 0; }
  .long { color: #3ecf8e; } .short { color: #e5484d; }
  .status-submitted { color: #3ecf8e; } .status-failed { color: #e5484d; }
  .status-skipped_no_margin, .status-no_price, .status-no_change { color: #9aa4b8; }
  button.btn {
    padding: 8px 14px; border-radius: 6px; border: 1px solid #5b8cff;
    background: #5b8cff; color: white; font-size: 13px; font-weight: 600; cursor: pointer;
  }
  button.btn.danger { border-color: #e5484d; background: #e5484d; }
  button.btn.secondary { background: transparent; color: #9aa4b8; border-color: #262c3b; }
  .risk-form-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; margin-top: 10px; }
  .risk-form-row label { display: flex; flex-direction: column; font-size: 11px; color: #9aa4b8; gap: 4px; }
  .risk-form-row input {
    padding: 7px 10px; border-radius: 6px; border: 1px solid #262c3b; background: #131722;
    color: #e6e9f0; font-size: 13px; width: 100px; box-sizing: border-box;
  }
  .risk-note { color: #9aa4b8; font-size: 12px; margin: 8px 0; }
</style>
</head>
<body>
  <header>
    <h1>__CLIENT_NAME__'s Portfolio</h1>
    <a href="/portal/logout">Log out</a>
  </header>
  <main>
    <section>
      <h2>Account</h2>
      <div id="account-summary" class="summary-row"></div>
    </section>
    <section>
      <h2>Positions</h2>
      <div class="table-wrap">
        <table id="positions-table">
          <thead><tr><th>Symbol</th><th>Side</th><th>Shares</th><th>Value</th><th>Unrealized P/L</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>Trade History</h2>
      <div class="table-wrap">
        <table id="trades-table">
          <thead><tr><th>When</th><th>Symbol</th><th>Side</th><th>Target</th><th>Shares</th><th>Status</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>Risk Controls</h2>
      <div id="risk-status" class="empty-state">Loading…</div>
      <div class="risk-form-row">
        <button id="liquidate-btn" class="btn danger">Liquidate now</button>
        <button id="resume-btn" class="btn secondary" hidden>Resume trading</button>
      </div>
      <div id="liquidate-confirm" hidden>
        <p class="risk-note">This closes every open position on your account right now and pauses trading until you resume. Are you sure?</p>
        <button id="liquidate-yes" class="btn danger">Yes, liquidate everything</button>
        <button id="liquidate-no" class="btn secondary">Cancel</button>
      </div>
      <p class="risk-note">Optional: automatically liquidate and pause if your account drops from its peak, or automatically secure gains and pause for the rest of the window once you hit a profit target. Leave a field blank to turn it off.</p>
      <form id="risk-settings-form" class="risk-form-row">
        <label>Max drawdown %<input id="risk-max-drawdown" type="number" min="1" max="50" step="0.5" placeholder="off" /></label>
        <label>Profit target %<input id="risk-profit-target" type="number" min="1" max="100" step="0.5" placeholder="off" /></label>
        <label>Window (days)<input id="risk-profit-window" type="number" min="1" max="365" step="1" placeholder="days" /></label>
        <button type="submit" class="btn">Save</button>
      </form>
      <div id="risk-settings-status" class="risk-note"></div>
    </section>
  </main>
  <script>
    function esc(s) {
      const div = document.createElement("div");
      div.textContent = s ?? "";
      return div.innerHTML;
    }
    function fmtMoney(n) {
      return n == null ? "—" : n.toLocaleString(undefined, { style: "currency", currency: "USD" });
    }
    async function fetchJSON(url) {
      const res = await fetch(url);
      if (res.status === 401) {
        window.location.href = "/portal";
        throw new Error("not authenticated");
      }
      if (!res.ok) throw new Error(url + " -> " + res.status);
      return res.json();
    }
    async function loadAccount() {
      const el = document.getElementById("account-summary");
      try {
        const a = await fetchJSON("/api/portal/account");
        el.innerHTML =
          '<div class="stat-card"><div class="label">Equity</div><div class="value">' + fmtMoney(a.equity) + '</div></div>' +
          '<div class="stat-card"><div class="label">Cash</div><div class="value">' + fmtMoney(a.cash) + '</div></div>' +
          '<div class="stat-card"><div class="label">Buying power</div><div class="value">' + fmtMoney(a.buying_power) + '</div></div>';
      } catch {
        el.innerHTML = '<div class="empty-state">Could not load your account right now.</div>';
      }
    }
    async function loadPositions() {
      const tbody = document.querySelector("#positions-table tbody");
      try {
        const rows = await fetchJSON("/api/portal/positions");
        tbody.innerHTML = rows.length ? rows.map((p) => {
          const sideClass = p.side === "short" ? "short" : "long";
          return '<tr><td>' + esc(p.symbol) + '</td><td class="' + sideClass + '">' + esc(p.side) + '</td>' +
            '<td>' + Number(p.qty).toLocaleString() + '</td><td>' + fmtMoney(p.market_value) + '</td>' +
            '<td class="' + (p.unrealized_pl >= 0 ? "long" : "short") + '">' + fmtMoney(p.unrealized_pl) + '</td></tr>';
        }).join("") : '<tr><td colspan="5" class="empty-state">No open positions.</td></tr>';
      } catch {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Could not load positions right now.</td></tr>';
      }
    }
    async function loadTrades() {
      const tbody = document.querySelector("#trades-table tbody");
      try {
        const rows = await fetchJSON("/api/portal/trades");
        tbody.innerHTML = rows.length ? rows.map((t) => {
          const pct = t.target_position_pct == null ? "—" : (t.target_position_pct * 100).toFixed(1) + "%";
          const shares = t.target_shares == null ? "—" : Number(t.target_shares).toLocaleString();
          return '<tr><td>' + new Date(t.ts).toLocaleString() + '</td><td>' + esc(t.symbol) + '</td>' +
            '<td>' + esc(t.side || "") + '</td><td>' + pct + '</td><td>' + shares + '</td>' +
            '<td class="status-' + esc(t.status) + '">' + esc(t.status) + '</td></tr>';
        }).join("") : '<tr><td colspan="6" class="empty-state">No trades yet.</td></tr>';
      } catch {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Could not load trade history right now.</td></tr>';
      }
    }
    async function postJSON(url, body) {
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
      if (res.status === 401) {
        window.location.href = "/portal";
        throw new Error("not authenticated");
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || (url + " -> " + res.status));
      return data;
    }
    const _pauseReasonText = {
      client_liquidate: "you liquidated your account",
      max_drawdown: "your max drawdown limit was hit",
      profit_target: "your profit target was hit",
    };
    async function loadRiskSettings() {
      const statusEl = document.getElementById("risk-status");
      try {
        const s = await fetchJSON("/api/portal/risk_settings");
        document.getElementById("liquidate-btn").hidden = s.trading_paused;
        document.getElementById("resume-btn").hidden = !s.trading_paused;
        statusEl.textContent = s.trading_paused
          ? "Trading is paused — " + (_pauseReasonText[s.pause_reason] || "trading is paused") + "."
          : "Trading is active.";
        document.getElementById("risk-max-drawdown").value = s.max_drawdown_pct != null ? (s.max_drawdown_pct * 100) : "";
        document.getElementById("risk-profit-target").value = s.profit_target_pct != null ? (s.profit_target_pct * 100) : "";
        document.getElementById("risk-profit-window").value = s.profit_target_window_days != null ? s.profit_target_window_days : "";
      } catch {
        statusEl.textContent = "Could not load risk settings right now.";
      }
    }
    document.getElementById("liquidate-btn").addEventListener("click", () => {
      document.getElementById("liquidate-confirm").hidden = false;
    });
    document.getElementById("liquidate-no").addEventListener("click", () => {
      document.getElementById("liquidate-confirm").hidden = true;
    });
    document.getElementById("liquidate-yes").addEventListener("click", async () => {
      document.getElementById("liquidate-confirm").hidden = true;
      try {
        await postJSON("/api/portal/liquidate");
        await Promise.all([loadAccount(), loadPositions(), loadTrades(), loadRiskSettings()]);
      } catch (e) {
        alert("Could not liquidate: " + e.message);
      }
    });
    document.getElementById("resume-btn").addEventListener("click", async () => {
      try {
        await postJSON("/api/portal/resume");
        await loadRiskSettings();
      } catch (e) {
        alert("Could not resume: " + e.message);
      }
    });
    document.getElementById("risk-settings-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const statusEl = document.getElementById("risk-settings-status");
      const dd = document.getElementById("risk-max-drawdown").value;
      const pt = document.getElementById("risk-profit-target").value;
      const pw = document.getElementById("risk-profit-window").value;
      const body = {
        max_drawdown_pct: dd === "" ? null : Number(dd) / 100,
        profit_target_pct: pt === "" ? null : Number(pt) / 100,
        profit_target_window_days: pw === "" ? null : Number(pw),
      };
      try {
        await postJSON("/api/portal/risk_settings", body);
        statusEl.textContent = "Saved.";
        await loadRiskSettings();
      } catch (e) {
        statusEl.textContent = e.message;
      }
    });
    loadAccount();
    loadPositions();
    loadTrades();
    loadRiskSettings();
    setInterval(() => { loadAccount(); loadPositions(); loadRiskSettings(); }, 60000);
  </script>
</body>
</html>"""


@app.get("/portal", include_in_schema=False)
def portal_page(request: Request) -> Response:
    client = _require_client(request)
    if not client:
        return HTMLResponse(_CLIENT_LOGIN_PAGE.format(error_html=""))
    # html.escape: the client name is operator-entered (not the client's own
    # self-registration), but it's still external text landing in markup —
    # same discipline as escapeHTML() on the JS side for headlines/reasons.
    safe_name = html.escape(client["name"])
    page = _CLIENT_PORTAL_PAGE.replace("__CLIENT_NAME__", safe_name)
    return HTMLResponse(page)


@app.post("/portal/login", include_in_schema=False)
async def portal_login(request: Request) -> Response:
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8"))
    name = (parsed.get("name") or [""])[0]
    password = (parsed.get("password") or [""])[0]

    engine = get_engine()
    df = pd.read_sql(text("SELECT id, password_hash, active FROM clients WHERE name = :name"), engine, params={"name": name})
    if df.empty or not bool(df.iloc[0]["active"]) or not verify_password(password, df.iloc[0]["password_hash"]):
        return HTMLResponse(_CLIENT_LOGIN_PAGE.format(error_html='<div class="error">Wrong name or password.</div>'), status_code=401)

    row = df.iloc[0]
    resp = RedirectResponse("/portal", status_code=303)
    server_host = (request.scope.get("server") or ("", 0))[0]
    resp.set_cookie(
        _CLIENT_SESSION_COOKIE,
        f"{int(row['id'])}.{_client_session_token(row['password_hash'])}",
        max_age=_CLIENT_SESSION_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=server_host not in _LOOPBACK_HOSTS,
    )
    return resp


@app.get("/portal/logout", include_in_schema=False)
def portal_logout() -> Response:
    resp = RedirectResponse("/portal")
    resp.delete_cookie(_CLIENT_SESSION_COOKIE)
    return resp


def _client_broker(client_id: int) -> AlpacaBroker:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT alpaca_api_key_encrypted, alpaca_api_secret_encrypted FROM clients WHERE id = :id"),
        engine,
        params={"id": client_id},
    )
    row = df.iloc[0]
    return AlpacaBroker(
        mode="live",
        confirm_live=True,
        api_key=decrypt_credential(row["alpaca_api_key_encrypted"]),
        secret_key=decrypt_credential(row["alpaca_api_secret_encrypted"]),
    )


@app.get("/api/portal/positions", response_model=None)
def portal_positions(request: Request) -> list[dict] | JSONResponse:
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    try:
        return _client_broker(client["id"]).get_positions_detailed()
    except Exception as e:
        return JSONResponse({"detail": f"Could not reach your Alpaca account: {e}"}, status_code=502)


@app.get("/api/portal/account", response_model=None)
def portal_account(request: Request) -> dict | JSONResponse:
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    try:
        account = _client_broker(client["id"]).get_account()
    except Exception as e:
        return JSONResponse({"detail": f"Could not reach your Alpaca account: {e}"}, status_code=502)
    return {
        "equity": float(account.get("equity", 0) or 0),
        "cash": float(account.get("cash", 0) or 0),
        "buying_power": float(account.get("buying_power", 0) or 0),
    }


@app.get("/api/portal/trades", response_model=None)
def portal_trades(request: Request, limit: int = 100) -> list[dict] | JSONResponse:
    """
    This client's own fan-out history — deliberately just symbol/side/size/
    status/timestamp, nothing about why the model picked it (no reasoning,
    no sentiment, no forecast) per the results-only decision.
    """
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT symbol, side, target_position_pct, target_shares, status, ts "
            "FROM client_orders WHERE client_id = :client_id ORDER BY ts DESC LIMIT :limit"
        ),
        engine,
        params={"client_id": client["id"], "limit": limit},
    )
    return _clean_records(df)


class _RiskSettingsRequest(BaseModel):
    max_drawdown_pct: float | None = None
    profit_target_pct: float | None = None
    profit_target_window_days: int | None = None


# Bounds mirror the DB CHECK constraints in
# data/schema/013_client_risk_controls.sql exactly -- kept as separate
# constants (not shared with the master account's settings.max_drawdown_pct)
# since a client's own stop is a completely independent decision from the
# master account's own circuit breaker.
_MIN_CLIENT_DRAWDOWN_PCT = 0.01
_MAX_CLIENT_DRAWDOWN_PCT = 0.5
_MIN_CLIENT_PROFIT_TARGET_PCT = 0.01
_MAX_CLIENT_PROFIT_TARGET_PCT = 1.0
_MIN_PROFIT_TARGET_WINDOW_DAYS = 1
_MAX_PROFIT_TARGET_WINDOW_DAYS = 365


def _validate_risk_settings(body: _RiskSettingsRequest) -> str | None:
    """
    Returns an error message if invalid, else None. Mirrors the DB CHECK
    constraints so a bad input gets a clean 400 here instead of a raw
    constraint-violation error surfacing to the client. profit_target_pct
    and profit_target_window_days must be set together or not at all -- a
    target % with no window (or a window with no target) is meaningless.
    """
    if body.max_drawdown_pct is not None and not (
        _MIN_CLIENT_DRAWDOWN_PCT <= body.max_drawdown_pct <= _MAX_CLIENT_DRAWDOWN_PCT
    ):
        return f"max_drawdown_pct must be between {_MIN_CLIENT_DRAWDOWN_PCT:.0%} and {_MAX_CLIENT_DRAWDOWN_PCT:.0%}, or omitted to turn it off."
    if (body.profit_target_pct is None) != (body.profit_target_window_days is None):
        return "profit_target_pct and profit_target_window_days must be set together (or both left off)."
    if body.profit_target_pct is not None and not (
        _MIN_CLIENT_PROFIT_TARGET_PCT <= body.profit_target_pct <= _MAX_CLIENT_PROFIT_TARGET_PCT
    ):
        return f"profit_target_pct must be between {_MIN_CLIENT_PROFIT_TARGET_PCT:.0%} and {_MAX_CLIENT_PROFIT_TARGET_PCT:.0%}."
    if body.profit_target_window_days is not None and not (
        _MIN_PROFIT_TARGET_WINDOW_DAYS <= body.profit_target_window_days <= _MAX_PROFIT_TARGET_WINDOW_DAYS
    ):
        return f"profit_target_window_days must be between {_MIN_PROFIT_TARGET_WINDOW_DAYS} and {_MAX_PROFIT_TARGET_WINDOW_DAYS}."
    return None


@app.post("/api/portal/liquidate", response_model=None)
def portal_liquidate(request: Request) -> dict | JSONResponse:
    """
    The client's own "exit everything, right now" button -- closes every
    open position on THEIR account only (never the master account, never
    another client) and pauses their trading with pause_reason=
    'client_liquidate', the same way an auto-triggered max_drawdown pauses
    -- so a signal an hour later doesn't immediately buy them right back
    into what they just chose to exit. They resume whenever they're ready
    via POST /api/portal/resume.
    """
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    engine = get_engine()
    try:
        _client_broker(client["id"]).flatten_all()
    except Exception as e:
        return JSONResponse({"detail": f"Could not reach your Alpaca account: {e}"}, status_code=502)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET trading_paused = TRUE, pause_reason = %s WHERE id = %s",
            ("client_liquidate", client["id"]),
        )
        conn.exec_driver_sql(
            "INSERT INTO client_orders (client_id, symbol, status) VALUES (%s, %s, %s)",
            (client["id"], "ALL", "client_liquidate"),
        )
    return {"status": "liquidated", "trading_paused": True}


@app.post("/api/portal/resume", response_model=None)
def portal_resume(request: Request) -> dict | JSONResponse:
    """
    Un-pauses this client's trading, whatever the reason it was paused for
    (their own liquidate click, or an auto-triggered max_drawdown /
    profit_target) -- and resets the drawdown peak / profit-target baseline
    to right now, so resuming doesn't immediately re-trigger off a stale
    peak or an already-elapsed window.
    """
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    engine = get_engine()
    now = dt.datetime.now(tz=dt.UTC)
    try:
        equity = float(_client_broker(client["id"]).get_account().get("equity", 0) or 0)
    except Exception as e:
        return JSONResponse({"detail": f"Could not reach your Alpaca account: {e}"}, status_code=502)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET trading_paused = FALSE, pause_reason = NULL, equity_peak = %s, "
            "profit_target_period_start_equity = %s, profit_target_period_start_ts = %s WHERE id = %s",
            (equity or None, equity or None, now, client["id"]),
        )
    return {"status": "resumed", "trading_paused": False}


@app.get("/api/portal/risk_settings", response_model=None)
def get_portal_risk_settings(request: Request) -> dict | JSONResponse:
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT trading_paused, pause_reason, max_drawdown_pct, profit_target_pct, profit_target_window_days "
            "FROM clients WHERE id = :id"
        ),
        engine,
        params={"id": client["id"]},
    )
    row = df.iloc[0]
    return {
        "trading_paused": bool(row["trading_paused"]),
        "pause_reason": row["pause_reason"],
        "max_drawdown_pct": float(row["max_drawdown_pct"]) if pd.notna(row["max_drawdown_pct"]) else None,
        "profit_target_pct": float(row["profit_target_pct"]) if pd.notna(row["profit_target_pct"]) else None,
        "profit_target_window_days": int(row["profit_target_window_days"]) if pd.notna(row["profit_target_window_days"]) else None,
    }


@app.post("/api/portal/risk_settings", response_model=None)
def set_portal_risk_settings(request: Request, body: _RiskSettingsRequest) -> dict | JSONResponse:
    """
    Self-service, unlike leverage -- the client sets their OWN thresholds
    directly, no operator involved. Omitting a field turns that threshold
    off; setting one (or changing its value) reseeds that threshold's
    baseline (equity_peak for drawdown, the period start for profit-target)
    to right now, so a new or changed limit is measured going forward, not
    against a peak/baseline from before this call.
    """
    client = _require_client(request)
    if not client:
        return JSONResponse({"detail": "Not authenticated."}, status_code=401)
    error = _validate_risk_settings(body)
    if error:
        return JSONResponse({"detail": error}, status_code=400)

    engine = get_engine()
    now = dt.datetime.now(tz=dt.UTC)
    equity = None
    if body.max_drawdown_pct is not None or body.profit_target_pct is not None:
        try:
            equity = float(_client_broker(client["id"]).get_account().get("equity", 0) or 0)
        except Exception as e:
            return JSONResponse({"detail": f"Could not reach your Alpaca account: {e}"}, status_code=502)

    equity_peak = equity if (body.max_drawdown_pct is not None and equity) else None
    period_equity = equity if (body.profit_target_pct is not None and equity) else None
    period_ts = now if body.profit_target_pct is not None else None

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET max_drawdown_pct = %s, equity_peak = %s, "
            "profit_target_pct = %s, profit_target_window_days = %s, "
            "profit_target_period_start_equity = %s, profit_target_period_start_ts = %s "
            "WHERE id = %s",
            (
                body.max_drawdown_pct, equity_peak,
                body.profit_target_pct, body.profit_target_window_days,
                period_equity, period_ts,
                client["id"],
            ),
        )
    return {
        "max_drawdown_pct": body.max_drawdown_pct,
        "profit_target_pct": body.profit_target_pct,
        "profit_target_window_days": body.profit_target_window_days,
    }


# Static frontend, mounted last so it doesn't shadow /api/* routes. A mount
# is not a route-level dependency, but the middleware above gates it anyway
# (it runs in front of every request, mount or not) — the page, its JS and
# its CSS are only reachable once the session cookie from /login is
# present. The page ships no data of its own; every number on it arrives
# through a gated /api call.
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
