"""
Demo data + monkeypatches so monitoring/dashboard/server.py can run and be
screenshotted without a live Postgres, MLflow, or broker connection.

Every number here is synthetic, but it flows through the REAL application
code: reasoning.py builds the actual 7-phase explanations, exit_levels.py
computes the actual take-profit/stop-loss formula, picks.py/whatif.py/
report_card.py/forecast_accuracy.py all run unmodified on this fake data.
Only the I/O boundary (DB reads, the broker, MLflow) is faked.

Usage: uvicorn demo_seed:app --host 127.0.0.1 --port 8501
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from execution.exit_levels import exit_levels_for
from monitoring import reasoning

np.random.seed(7)
NOW = dt.datetime(2026, 8, 28, 15, 30, tzinfo=dt.UTC)  # Friday afternoon, market open


# --------------------------------------------------------------------------
# Synthetic price history — deterministic random walk per symbol, daily bars
# going back far enough for the regime chart (needs >=20 rows) and for the
# closed-trade reconstruction below.
# --------------------------------------------------------------------------
def _price_series(symbol: str, start_price: float, days: int, drift: float, vol: float) -> pd.DataFrame:
    dates = pd.bdate_range(end=NOW.date(), periods=days, tz="UTC")
    rets = np.random.normal(drift, vol, size=days)
    closes = start_price * np.cumprod(1 + rets)
    highs = closes * (1 + np.abs(np.random.normal(0, vol / 2, size=days)))
    lows = closes * (1 - np.abs(np.random.normal(0, vol / 2, size=days)))
    return pd.DataFrame({"symbol": symbol, "ts": dates, "close": closes, "high": highs, "low": lows})


PRICES = pd.concat(
    [
        _price_series("SPY", 560.0, 90, 0.0007, 0.008),
        _price_series("NVDA", 185.0, 90, 0.0018, 0.028),
        _price_series("JPM", 245.0, 90, 0.0009, 0.013),
        _price_series("XOM", 118.0, 90, 0.0006, 0.015),
        _price_series("DIS", 108.0, 40, -0.0035, 0.022),  # the one the watcher closed
    ],
    ignore_index=True,
)


def _last_price(symbol: str) -> float:
    return float(PRICES[PRICES["symbol"] == symbol].sort_values("ts")["close"].iloc[-1])


def _price_on(symbol: str, ts: dt.datetime) -> float:
    sub = PRICES[(PRICES["symbol"] == symbol) & (PRICES["ts"] <= ts)].sort_values("ts")
    return float(sub["close"].iloc[-1]) if not sub.empty else float(PRICES[PRICES["symbol"] == symbol]["close"].iloc[0])


# --------------------------------------------------------------------------
# News headlines (feeds the "Recent news" panel + gave the contradiction
# monitor its DIS sentiment signal below)
# --------------------------------------------------------------------------
NEWS = pd.DataFrame(
    [
        {"symbol": "NVDA", "ts": NOW - dt.timedelta(hours=6), "headline": "NVIDIA data-center orders reported ahead of schedule into Q3", "sentiment": 0.52, "source": "polygon"},
        {"symbol": "NVDA", "ts": NOW - dt.timedelta(hours=30), "headline": "Analysts raise price targets after supply-chain checks", "sentiment": 0.41, "source": "polygon"},
        {"symbol": "JPM", "ts": NOW - dt.timedelta(hours=10), "headline": "JPMorgan trading desk revenue tracking above guidance", "sentiment": 0.22, "source": "polygon"},
        {"symbol": "JPM", "ts": NOW - dt.timedelta(hours=40), "headline": "Regional bank credit spreads narrow on rate-cut bets", "sentiment": 0.11, "source": "polygon"},
        {"symbol": "XOM", "ts": NOW - dt.timedelta(hours=14), "headline": "Crude holds steady as OPEC+ maintains current output", "sentiment": 0.05, "source": "polygon"},
        {"symbol": "XOM", "ts": NOW - dt.timedelta(hours=52), "headline": "Refining margins soften into month-end", "sentiment": -0.08, "source": "polygon"},
        {"symbol": "DIS", "ts": NOW - dt.timedelta(days=5, hours=2), "headline": "Streaming subscriber growth slows sharply, guidance cut", "sentiment": -0.61, "source": "polygon"},
        {"symbol": "DIS", "ts": NOW - dt.timedelta(days=5, hours=9), "headline": "Park attendance softens as consumers pull back on travel", "sentiment": -0.48, "source": "polygon"},
        {"symbol": "DIS", "ts": NOW - dt.timedelta(days=4, hours=20), "headline": "Two analysts cut price targets after guidance miss", "sentiment": -0.55, "source": "polygon"},
    ]
)


# --------------------------------------------------------------------------
# Build the decisions log using the REAL reasoning + exit_levels modules.
# --------------------------------------------------------------------------
def _make_open_decision(symbol, ts, forecast, regime, weight_pct, top_features, shares, vol):
    levels = exit_levels_for(predicted_return=forecast, daily_volatility=vol, horizon_days=20)
    phases = reasoning.combine_phases(
        reasoning.phase_pretrade_risk([]),
        reasoning.phase_signals(regime, top_features),
        reasoning.phase_forecast(forecast, abs(forecast)),
        reasoning.phase_selection_diversified(symbol, "long" if forecast >= 0 else "short", weight_pct, n_confident=14, n_selected=3, top_k=10),
        reasoning.phase_execution(symbol, "opened", shares, "market"),
        reasoning.phase_reconciliation(symbol, shares, shares, flagged=False),
        reasoning.phase_ongoing_monitoring(closed=False),
    )
    return {
        "ts": ts, "symbol": symbol, "feature_set_id": "v4", "model_version": "ensemble_v1",
        "forecast": forecast, "regime": regime, "target_position": weight_pct,
        "executed_position": shares, "mode": "paper", "reasoning": json.dumps(phases),
    }, levels


OPEN_TS = NOW - dt.timedelta(days=4, hours=6)  # this week's Monday cycle

nvda_row, nvda_levels = _make_open_decision(
    "NVDA", OPEN_TS, 0.041, "trend", 0.132,
    [
        {"feature_name": "mom_ret_20d", "value": 0.187, "contribution": 0.021},
        {"feature_name": "sentiment_mean_10d", "value": 0.38, "contribution": 0.011},
        {"feature_name": "meanrev_rsi_14", "value": 64.0, "contribution": 0.006},
    ],
    shares=round(150000 * 0.132 / _price_on("NVDA", OPEN_TS), 2), vol=0.028,
)
jpm_row, jpm_levels = _make_open_decision(
    "JPM", OPEN_TS, 0.024, "trend", 0.101,
    [
        {"feature_name": "sentiment_mean_10d", "value": 0.19, "contribution": 0.009},
        {"feature_name": "mom_ret_5d", "value": 0.021, "contribution": 0.005},
        {"feature_name": "adx_14", "value": 27.0, "contribution": 0.004},
    ],
    shares=round(150000 * 0.101 / _price_on("JPM", OPEN_TS), 2), vol=0.013,
)
xom_row, xom_levels = _make_open_decision(
    "XOM", OPEN_TS, 0.019, "chop", 0.083,
    [
        {"feature_name": "meanrev_zscore_20d", "value": -1.4, "contribution": 0.007},
        {"feature_name": "vol_realized_20d", "value": 0.21, "contribution": -0.002},
        {"feature_name": "days_to_next_fomc", "value": 11.0, "contribution": 0.001},
    ],
    shares=round(150000 * 0.083 / _price_on("XOM", OPEN_TS), 2), vol=0.015,
)

# --- DIS: opened 12 days ago, closed 5 days ago by the contradiction monitor ---
DIS_OPEN_TS = NOW - dt.timedelta(days=12)
dis_shares = round(150000 * 0.09 / _price_on("DIS", DIS_OPEN_TS), 2)
dis_open_row, dis_levels = _make_open_decision(
    "DIS", DIS_OPEN_TS, 0.027, "trend", 0.09,
    [
        {"feature_name": "mom_ret_20d", "value": 0.061, "contribution": 0.012},
        {"feature_name": "sentiment_mean_10d", "value": 0.14, "contribution": 0.004},
    ],
    shares=dis_shares, vol=0.019,
)

DIS_CLOSE_TS = NOW - dt.timedelta(days=5, hours=1)
dis_contradiction_reasons = [
    {"signal": "news_sentiment", "value": -0.55, "news_count": 3,
     "detail": "mean sentiment -0.55 over last 24h contradicts long position"},
    {"signal": "price_momentum", "value": -0.137,
     "detail": "5d return -13.7% contradicts long position"},
]
dis_close_phases = reasoning.combine_phases(
    reasoning.phase_contradiction(dis_contradiction_reasons),
    {
        "phase": 4, "title": "Candidate Selection & Sizing",
        "summary": "DIS closed outside the weekly screen — no new position opened.",
        "lines": [
            "This wasn't a weekly screen decision — the hourly contradiction check triggered mid-week.",
            "Re-entry (if any) is left to the next weekly screen, not decided here.",
        ],
    },
    reasoning.phase_execution("DIS", "closed", None, "market"),
    reasoning.phase_ongoing_monitoring(closed=True),
)
dis_close_row = {
    "ts": DIS_CLOSE_TS, "symbol": "DIS", "feature_set_id": "contradiction_monitor",
    "model_version": "rule_based_v1", "forecast": None, "regime": None, "target_position": 0.0,
    "executed_position": 0.0, "mode": "paper", "reasoning": json.dumps(dis_close_phases),
}

# --- a couple of older, already-closed round trips so Decision History / feature
# frequency / live accuracy have more than one week of data ---
def _flat_decision(symbol, ts, forecast, regime, weight, shares, top_features):
    if forecast is None:
        # A hold-rules close: missed the shortlist for enough consecutive
        # cycles that the exit condition fired — see execution/hold_rules.py.
        phases = reasoning.combine_phases(
            reasoning.phase_pretrade_risk([]),
            reasoning.phase_signals(regime, top_features),
            reasoning.phase_hold_exit(symbol, ["out of the shortlist 2 consecutive cycle(s) (limit 2)"], missed_cycles=2),
            reasoning.phase_execution(symbol, "closed", None, "market"),
            reasoning.phase_ongoing_monitoring(closed=True),
        )
    else:
        phases = reasoning.combine_phases(
            reasoning.phase_pretrade_risk([]),
            reasoning.phase_signals(regime, top_features),
            reasoning.phase_forecast(forecast, abs(forecast)),
            reasoning.phase_selection_diversified(symbol, "long" if forecast >= 0 else "short", weight, 12, 3, 10),
            reasoning.phase_execution(symbol, "opened", shares, "market"),
            reasoning.phase_reconciliation(symbol, shares, shares, flagged=False),
            reasoning.phase_ongoing_monitoring(closed=False),
        )
    return {
        "ts": ts, "symbol": symbol, "feature_set_id": "v4", "model_version": "ensemble_v1",
        "forecast": forecast, "regime": regime, "target_position": weight,
        "executed_position": shares, "mode": "paper", "reasoning": json.dumps(phases),
    }


older_rows = [
    _flat_decision("MSFT", NOW - dt.timedelta(days=19), 0.031, "trend", 0.11,
                    round(150000 * 0.11 / _price_on("SPY", NOW - dt.timedelta(days=19)) * 1.0, 2),
                    [{"feature_name": "mom_ret_20d", "value": 0.09, "contribution": 0.014}]),
    _flat_decision("MSFT", NOW - dt.timedelta(days=12), None, None, 0.0, 0.0,
                    [{"feature_name": "mom_ret_20d", "value": -0.02, "contribution": -0.003}]),
    _flat_decision("AAPL", NOW - dt.timedelta(days=26), -0.014, "chop", 0.07,
                    round(150000 * 0.07 / 210.0, 2),
                    [{"feature_name": "meanrev_rsi_14", "value": 71.0, "contribution": -0.006}]),
    _flat_decision("AAPL", NOW - dt.timedelta(days=19), None, None, 0.0, 0.0,
                    [{"feature_name": "meanrev_rsi_14", "value": 55.0, "contribution": 0.001}]),
]

DECISIONS = pd.DataFrame(
    [nvda_row, jpm_row, xom_row, dis_open_row, dis_close_row, *older_rows]
).sort_values("ts").reset_index(drop=True)

LEVELS_BY_SYMBOL = {"NVDA": nvda_levels, "JPM": jpm_levels, "XOM": xom_levels}


# --------------------------------------------------------------------------
# Fake broker — mirrors execution/broker.py's public shape used by the
# dashboard (get_positions_detailed, get_positions).
# --------------------------------------------------------------------------
class DemoBroker:
    mode = "paper"

    def __init__(self):
        self._rows = {}
        for symbol, row in (("NVDA", nvda_row), ("JPM", jpm_row), ("XOM", xom_row)):
            entry_price = _price_on(symbol, row["ts"])
            current_price = _last_price(symbol)
            qty = row["executed_position"]
            market_value = qty * current_price
            cost_basis = qty * entry_price
            self._rows[symbol] = {
                "symbol": symbol, "qty": qty, "side": "long" if qty >= 0 else "short",
                "avg_entry_price": entry_price, "current_price": current_price,
                "market_value": market_value, "cost_basis": cost_basis,
                "unrealized_pl": market_value - cost_basis,
                "unrealized_plpc": (market_value - cost_basis) / cost_basis if cost_basis else 0.0,
            }

    def get_positions_detailed(self):
        return list(self._rows.values())

    def get_positions(self):
        return {s: r["qty"] for s, r in self._rows.items()}


def get_broker_fake():
    return DemoBroker()


# --------------------------------------------------------------------------
# Equity curve — a slightly bumpy uptrend with one real drawdown, so the
# chart isn't a straight line.
# --------------------------------------------------------------------------
def _equity_curve() -> pd.DataFrame:
    dates = pd.bdate_range(end=NOW.date(), periods=70, tz="UTC")
    base = 150000.0
    rets = np.random.normal(0.0009, 0.006, size=len(dates))
    rets[40:47] -= 0.008  # a visible drawdown mid-history
    curve = base * np.cumprod(1 + rets)
    return pd.DataFrame({"ts": dates, "mode": "paper", "equity_value": curve})


EQUITY = _equity_curve()

BREAKERS = pd.DataFrame(
    [
        {"ts": NOW, "breaker_name": "max_drawdown", "triggered": False, "reason": ""},
        {"ts": NOW, "breaker_name": "max_correlated_exposure", "triggered": False, "reason": ""},
        {"ts": NOW, "breaker_name": "max_single_position:NVDA", "triggered": False, "reason": ""},
        {"ts": NOW, "breaker_name": "max_single_position:JPM", "triggered": False, "reason": ""},
        {"ts": NOW, "breaker_name": "max_single_position:XOM", "triggered": False, "reason": ""},
    ]
)


# --------------------------------------------------------------------------
# Walk-forward fold metrics, shaped like BENCHMARK_RESULTS.md's real numbers
# (this repo's actual measured result: ~coin-flip accuracy, no demonstrated
# edge) — used by both /api/analysis/runs and /api/analysis/report_card.
# --------------------------------------------------------------------------
FOLD_METRICS = [
    {"fold_id": i, "directional_accuracy": acc, "directional_accuracy_when_confident": acc_c,
     "pct_rows_confident": conf, "mae": mae, "rmse": mae * 1.6}
    for i, (acc, acc_c, conf, mae) in enumerate(
        [
            (0.512, 0.524, 0.91, 0.031), (0.498, 0.507, 0.89, 0.033), (0.505, 0.519, 0.93, 0.030),
            (0.489, 0.481, 0.88, 0.034), (0.517, 0.522, 0.90, 0.029), (0.502, 0.511, 0.92, 0.031),
            (0.494, 0.489, 0.87, 0.035), (0.509, 0.516, 0.94, 0.028), (0.500, 0.503, 0.90, 0.032),
            (0.521, 0.531, 0.91, 0.030),
        ]
    )
]


def fetch_fold_runs_fake(tracking_uri, experiment_name="forecast_lgbm"):
    return [
        {"run_name": f"fold-{m['fold_id']}", "fold_id": str(m["fold_id"]), "metrics": {k: v for k, v in m.items() if k != "fold_id"}}
        for m in FOLD_METRICS
    ]


def _mlflow_runs_df() -> pd.DataFrame:
    rows = []
    start = NOW - dt.timedelta(days=200)
    for m in FOLD_METRICS:
        fold_start = start + dt.timedelta(days=18 * m["fold_id"])
        rows.append(
            {
                "start_time": fold_start,
                "params.fold_id": str(m["fold_id"]),
                "params.feature_set_id": "v4",
                "params.train_start": (fold_start - dt.timedelta(days=365)).date().isoformat(),
                "params.train_end": fold_start.date().isoformat(),
                "params.test_start": fold_start.date().isoformat(),
                "params.test_end": (fold_start + dt.timedelta(days=18)).date().isoformat(),
                "metrics.mae": m["mae"], "metrics.rmse": m["rmse"],
                "metrics.directional_accuracy": m["directional_accuracy"],
                "metrics.directional_accuracy_when_confident": m["directional_accuracy_when_confident"],
                "metrics.pct_rows_confident": m["pct_rows_confident"],
                "metrics.mean_ensemble_std": 0.012,
            }
        )
    return pd.DataFrame(rows)


MLFLOW_RUNS_DF = _mlflow_runs_df()


# --------------------------------------------------------------------------
# pd.read_sql dispatcher — routes every query the real endpoint code issues
# to the right in-memory table, so the REAL server/report_card/whatif/
# picks/forecast_accuracy code runs unmodified on top of this fake data.
# --------------------------------------------------------------------------
def fake_read_sql(query, con=None, params=None, **kwargs):
    q = str(query)
    if "FROM news_events" in q:
        return NEWS.copy()
    if "FROM prices" in q:
        if params and "symbol" in params:
            return PRICES[PRICES["symbol"] == params["symbol"]][["ts", "high", "low", "close"]].sort_values("ts").reset_index(drop=True)
        return PRICES.sort_values(["symbol", "ts"]).reset_index(drop=True)
    if "FROM decisions" in q:
        df = DECISIONS.copy()
        if params and params.get("symbol"):
            df = df[df["symbol"] == params["symbol"]]
        if "mode = 'paper'" in q:
            df = df[df["mode"] == "paper"]
        if "forecast IS NOT NULL" in q:
            df = df[df["forecast"].notna()]
        if "reasoning IS NOT NULL" in q:
            df = df[df["reasoning"].notna()]
        return df.sort_values("ts", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def install(server_module):
    server_module.get_broker = get_broker_fake
    server_module.get_engine = lambda: "demo-engine"
    server_module.pd.read_sql = fake_read_sql
    server_module.load_equity_curve = lambda mode="paper", limit=1000: EQUITY.copy()
    server_module.load_latest_breaker_state = lambda limit=200: BREAKERS.copy()
    server_module.load_exit_levels = lambda engine: dict(LEVELS_BY_SYMBOL)
    server_module.mlflow.search_runs = lambda experiment_names: MLFLOW_RUNS_DF.copy()
    server_module.report_card.fetch_fold_runs = fetch_fold_runs_fake


app = None  # set below


def _build_app():
    global app
    from monitoring.dashboard import server as server_module

    install(server_module)
    return server_module.app


app = _build_app()
