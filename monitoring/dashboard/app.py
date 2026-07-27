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

st.caption(
    "TODO once live: model forecast-accuracy trend, drawdown chart from equity "
    "curve, and a circuit-breaker status panel wired to risk/circuit_breakers.py."
)
