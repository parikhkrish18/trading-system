"""
Phase 8 monitoring dashboard. Streamlit is enough for a 3-person team per
the plan — swap for Grafana later if you want more polish, the underlying
queries stay the same either way.

Run locally with: streamlit run monitoring/dashboard/app.py
Or via docker-compose (see top-level docker-compose.yml, `dashboard` service).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.ingest.db import get_engine
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

st.set_page_config(page_title="Trading System Monitor", layout="wide")
st.title("Trading System — Monitoring")

engine = get_engine()


@st.cache_data(ttl=60)
def load_recent_decisions(limit: int = 500) -> pd.DataFrame:
    return pd.read_sql(
        f"SELECT * FROM decisions ORDER BY ts DESC LIMIT {limit}", engine
    )


@st.cache_data(ttl=60)
def load_recent_prices(symbols: list[str], limit_days: int = 120) -> pd.DataFrame:
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    return pd.read_sql(
        f"""SELECT symbol, ts, close FROM prices
            WHERE symbol IN ({symbol_list})
            ORDER BY ts DESC LIMIT {limit_days * max(len(symbols), 1)}""",
        engine,
    )


decisions = load_recent_decisions()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Open positions (from latest decisions)", int((decisions.groupby("symbol")["executed_position"].last() != 0).sum()) if not decisions.empty else 0)
with col2:
    st.metric("Mode", decisions["mode"].iloc[0] if not decisions.empty else "n/a")
with col3:
    st.metric("Decisions logged", len(decisions))

st.subheader("Recent decisions")
if decisions.empty:
    st.info("No decisions logged yet — this fills in once Phase 6 paper trading is running.")
else:
    st.dataframe(decisions, use_container_width=True)

    st.subheader("Forecast vs. executed position, by symbol")
    for symbol, sub in decisions.groupby("symbol"):
        st.write(f"**{symbol}**")
        st.line_chart(sub.set_index("ts")[["forecast", "executed_position"]])

st.subheader("Price history")
symbols_input = st.text_input("Symbols (comma-separated)", value="SPY,QQQ,TQQQ,SQQQ")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
if symbols:
    prices = load_recent_prices(symbols)
    if prices.empty:
        st.info("No price data found — run data.ingest.prices first.")
    else:
        pivot = prices.pivot(index="ts", columns="symbol", values="close").sort_index()
        st.line_chart(pivot)

st.subheader("Model forecast-accuracy trend")
if decisions.empty or not symbols:
    st.info("Needs both logged decisions and price history above to compute.")
else:
    forecasted = decisions.dropna(subset=["forecast"])[["symbol", "ts", "forecast"]]
    accuracy = compute_forecast_accuracy(forecasted, load_recent_prices(symbols, limit_days=400))
    if accuracy.empty:
        st.info("No decision has a matured forward return yet (needs at least one price bar after the decision timestamp).")
    else:
        st.metric("Directional hit rate (sign of forecast vs. sign of next-bar return)", f"{accuracy['hit'].mean():.1%}")
        rolling_hit_rate = accuracy.sort_values("ts").set_index("ts")["hit"].astype(float).rolling(20, min_periods=5).mean()
        st.line_chart(rolling_hit_rate.rename("rolling_20_hit_rate"))

st.subheader("Equity curve & drawdown")
equity = load_equity_curve()
if equity.empty:
    st.info(
        "No equity snapshots recorded yet — call "
        "monitoring.equity.record_equity_snapshot(equity_value, mode) from the "
        "execution loop once paper trading is running."
    )
else:
    equity = equity.sort_values("ts")
    running_peak = equity["equity_value"].cummax()
    drawdown = (equity["equity_value"] / running_peak - 1).rename("drawdown")
    st.line_chart(equity.set_index("ts")["equity_value"])
    st.line_chart(pd.DataFrame({"drawdown": drawdown.values}, index=equity["ts"]))

st.subheader("Circuit breaker status")
breaker_state = load_latest_breaker_state()
if breaker_state.empty:
    st.info(
        "No breaker checks recorded yet — call "
        "monitoring.breaker_state.check_and_record_breakers(...) from the "
        "execution loop in place of risk.circuit_breakers.run_all_breakers(...)."
    )
else:
    latest_per_breaker = breaker_state.sort_values("ts").groupby("breaker_name", as_index=False).tail(1)
    for _, row in latest_per_breaker.sort_values("breaker_name").iterrows():
        label = f"{row['breaker_name']} — last checked {row['ts']}"
        if row["triggered"]:
            st.error(f"TRIGGERED — {row['reason']} ({label})")
        else:
            st.success(f"OK ({label})")

    with st.expander("Recent breaker trip history"):
        tripped = breaker_state[breaker_state["triggered"]].sort_values("ts", ascending=False)
        if tripped.empty:
            st.write("No trips in recorded history.")
        else:
            st.dataframe(tripped, use_container_width=True)
