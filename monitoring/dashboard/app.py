"""
Phase 8 monitoring dashboard. Streamlit is enough for a 3-person team per
the plan — swap for Grafana later if you want more polish, the underlying
queries stay the same either way.

Read-only: every query here is a SELECT. Nothing on this page places, cancels
or sizes a trade — it reports what the screener and execution loop already did.

Run locally with: streamlit run monitoring/dashboard/app.py
Or via docker-compose (see top-level docker-compose.yml, `dashboard` service).
"""
from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import bindparam, text

from data.ingest.db import get_engine
from models.regime.trend_chop_classifier import CHOP, TREND
from monitoring.breaker_state import load_latest_breaker_state
from monitoring.dashboard.evidence import (
    confidence_note,
    evidence_table,
    news_sentiment_note,
    regime_note,
)
from monitoring.dashboard.picks import (
    COL_FORECAST,
    COL_REGIME,
    COL_SIZE,
    batch_summary,
    latest_batch,
    latest_picks_table,
    regime_counts,
)
from monitoring.equity import load_equity_curve
from monitoring.forecast_accuracy import compute_forecast_accuracy

st.set_page_config(page_title="Trading System Monitor", layout="wide")

engine = get_engine()

# --- Chart palette -----------------------------------------------------------
# Both modes are chosen for their own surface rather than flipped automatically.
# Charts here are single-series, so identity never rests on colour alone.
_LIGHT = {
    "series": "#2a78d6",
    "surface": "#fcfcfb",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "muted": "#898781",
    "text": "#52514e",
}
_DARK = {
    "series": "#3987e5",
    "surface": "#1a1a19",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "muted": "#898781",
    "text": "#c3c2b7",
}

# Regime badges pair a fill with the regime's own word in the cell, so a
# red/green-confusable pair never has to carry the meaning on hue alone.
_REGIME_BADGE = {
    TREND: "background-color: #dff2df; color: #0b5f0b;",
    CHOP: "background-color: #fdeccd; color: #8a5a00;",
}


def _palette() -> dict[str, str]:
    try:
        base = st.context.theme.type
    except Exception:  # bare mode / older runtime without st.context
        base = None
    return _DARK if base == "dark" else _LIGHT


COLORS = _palette()


# --- Queries (all read-only) -------------------------------------------------
@st.cache_data(ttl=60)
def load_recent_decisions(limit: int = 500) -> pd.DataFrame:
    query = text(
        "SELECT ts, symbol, forecast, direction_agreement, regime, target_position, "
        "executed_position, mode, feature_set_id, model_version "
        "FROM decisions ORDER BY ts DESC LIMIT :limit"
    )
    return pd.read_sql(query, engine, params={"limit": limit})


@st.cache_data(ttl=60)
def load_batch_evidence(batch_ts) -> pd.DataFrame:
    """
    Every "why this pick" row belonging to one screener run (see
    data/schema/004_decision_evidence.sql). Loaded per batch rather than per
    symbol so the panel opens without a round trip per expander.
    """
    if batch_ts is None or pd.isna(batch_ts):
        return pd.DataFrame(
            columns=["symbol", "feature_name", "feature_value", "contribution", "contribution_rank"]
        )
    query = text(
        "SELECT symbol, feature_name, feature_value, contribution, contribution_rank "
        "FROM decision_evidence WHERE ts = :ts ORDER BY symbol, contribution_rank"
    )
    return pd.read_sql(query, engine, params={"ts": pd.Timestamp(batch_ts).to_pydatetime()})


@st.cache_data(ttl=300)
def load_price_symbols() -> list[str]:
    df = pd.read_sql(text("SELECT DISTINCT symbol FROM prices ORDER BY symbol"), engine)
    return df["symbol"].tolist()


@st.cache_data(ttl=300)
def load_universe_size() -> int:
    df = pd.read_sql(text("SELECT count(*) AS n FROM universe WHERE is_active"), engine)
    return int(df["n"].iloc[0]) if not df.empty else 0


@st.cache_data(ttl=60)
def load_prices(symbols: list[str], lookback_days: int = 365) -> pd.DataFrame:
    """
    Daily closes for `symbols` over the trailing window. Bound parameters
    throughout — no string-interpolated symbol lists.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "ts", "close"])
    cutoff = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=lookback_days)
    query = text(
        "SELECT symbol, ts, close FROM prices "
        "WHERE symbol IN :symbols AND ts >= :cutoff ORDER BY symbol, ts"
    ).bindparams(bindparam("symbols", expanding=True))
    return pd.read_sql(query, engine, params={"symbols": list(symbols), "cutoff": cutoff})


# --- Chart helpers -----------------------------------------------------------
def _timeseries_chart(
    df: pd.DataFrame,
    value_col: str,
    y_title: str,
    value_format: str,
    value_label: str,
    zero: bool = False,
    area: bool = False,
    baseline: float | None = None,
) -> alt.LayerChart:
    """
    One-series line over time: 2px round-capped line, hairline grid, and a
    crosshair + tooltip on hover. No legend — a single series is named by the
    panel heading above it.

    `area` adds a 10%-opacity wash, and is only for series measured against a
    meaningful zero (drawdown). A wash forces the y-axis to include zero, which
    on a price or account-value chart squashes the whole line into the top
    fraction of the plot — those stay line-only with a fitted scale.
    """
    base = alt.Chart(df).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
    )
    y = alt.Y(
        f"{value_col}:Q",
        title=y_title,
        scale=alt.Scale(zero=zero),
        axis=alt.Axis(format=value_format),
    )

    layers = []
    if baseline is not None:
        # Reference line only — no inline label. A horizontal rule spans the full
        # width, so any text on it lands on top of the series somewhere; the
        # panel caption names what the line means instead.
        layers.append(
            alt.Chart(pd.DataFrame({"baseline": [baseline]}))
            .mark_rule(color=COLORS["axis"], strokeWidth=1)
            .encode(y=alt.Y("baseline:Q"))
        )
    if area:
        layers.append(base.mark_area(color=COLORS["series"], opacity=0.10).encode(y=y))
    layers.append(
        base.mark_line(color=COLORS["series"], strokeWidth=2, strokeCap="round", strokeJoin="round").encode(y=y)
    )

    hover = alt.selection_point(nearest=True, on="pointermove", fields=["ts"], empty=False)
    tooltip = [
        alt.Tooltip("ts:T", title="Date", format="%d %b %Y"),
        alt.Tooltip(f"{value_col}:Q", title=value_label, format=value_format),
    ]
    layers.append(
        base.mark_rule(color=COLORS["muted"], strokeWidth=1)
        .encode(opacity=alt.condition(hover, alt.value(0.5), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    layers.append(
        base.mark_point(
            size=80, filled=True, color=COLORS["series"], stroke=COLORS["surface"], strokeWidth=2
        ).encode(y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip)
    )

    return (
        alt.layer(*layers)
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=COLORS["grid"],
            domainColor=COLORS["axis"],
            tickColor=COLORS["axis"],
            labelColor=COLORS["muted"],
            titleColor=COLORS["text"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
    )


def _show(chart: alt.LayerChart) -> None:
    # theme=None so the palette configured above wins over Streamlit's own.
    st.altair_chart(chart, width="stretch", theme=None)


# --- Page --------------------------------------------------------------------
st.title("Trading System — Monitoring")
st.caption(
    "A read-only view of what the stock screener has been doing. Nothing on this "
    "page buys or sells anything — it only reports what was already decided."
)

decisions = load_recent_decisions()
batch = latest_batch(decisions)
summary = batch_summary(batch)
universe_size = load_universe_size()

# --- Stat tiles --------------------------------------------------------------
tile1, tile2, tile3, tile4 = st.columns(4)
with tile1:
    st.metric("Picks in latest run", summary["n_picks"])
    st.caption("Stocks the screener shortlisted the last time it ran.")
with tile2:
    st.metric("Long vs short", f"{summary['n_long']} / {summary['n_short']}")
    st.caption("Long = betting the price rises. Short = betting it falls.")
with tile3:
    last_run = summary["last_run"]
    if last_run is None:
        st.metric("Last run", "never")
        st.caption("The screener has not logged a run yet.")
    else:
        last_run_ts = pd.Timestamp(last_run)
        age_days = (pd.Timestamp.now(tz=last_run_ts.tz) - last_run_ts).days
        st.metric("Last run", last_run_ts.strftime("%d %b %Y %H:%M"))
        st.caption(f"{age_days} day(s) ago" + (f" · mode: {summary['mode']}" if summary["mode"] else ""))
with tile4:
    st.metric("Stocks watched", f"{universe_size:,}")
    st.caption("Size of the universe the screener chooses from.")

st.divider()

# --- Latest picks ------------------------------------------------------------
st.subheader("Latest picks")
st.caption(
    "The shortlist from the most recent screener run. Each row is one stock the "
    "model was confident enough about to propose."
)

table = latest_picks_table(batch)

if batch.empty:
    st.info(
        "No picks logged yet — this fills in once the screener runs with logging on "
        "(`python -m models.screener --feature-set-id <id> --universe --log`)."
    )
else:
    counts = regime_counts(batch)
    styled = (
        table.style.format({COL_FORECAST: "{:+.2%}", COL_SIZE: "{:.2%}"}, na_rep="—")
        .map(lambda v: _REGIME_BADGE.get(v, ""), subset=[COL_REGIME])
    )
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption(
        f"**Predicted move** is how much the model expects the price to change. "
        f"**Target size** is how much of the portfolio the pick would take up. "
        f"**Market regime** describes the stock's recent behaviour — "
        f"*trend* (green, moving steadily one way, {counts[TREND]} here) or "
        f"*chop* (orange, drifting sideways, {counts[CHOP]} here); "
        f"picks in chop are deliberately sized smaller. "
        f"**Placed?** says whether the trade was actually sent to the broker — "
        f"the screener only proposes."
    )

    # --- Why this pick? ------------------------------------------------------
    st.markdown("##### Why this pick?")
    st.caption(
        "Open a stock to see what the model was actually reacting to. Each line is "
        "one thing it measured, what it saw, and which way that pushed the "
        "prediction. The factors are ranked — the top one is the biggest single "
        "reason the stock made the shortlist."
    )

    batch_evidence = load_batch_evidence(summary["last_run"])
    batch_by_symbol = batch.set_index("symbol")

    for symbol in table["Symbol"]:
        pick = batch_by_symbol.loc[symbol]
        # A symbol can in principle appear twice in one batch; take the first.
        if isinstance(pick, pd.DataFrame):
            pick = pick.iloc[0]

        direction = "Long" if float(pick["target_position"]) > 0 else "Short"
        with st.expander(f"{symbol} — {direction}, why?"):
            symbol_evidence = batch_evidence[batch_evidence["symbol"] == symbol]

            st.write(regime_note(pick.get("regime")))
            st.write(confidence_note(pick.get("direction_agreement")))

            rows = evidence_table(symbol_evidence)
            if rows.empty:
                st.info(
                    "No per-factor evidence stored for this pick. Evidence is recorded "
                    "from the screener run that logged it — picks logged before this "
                    "panel existed won't have any."
                )
            else:
                st.dataframe(rows, width="stretch", hide_index=True)

            st.caption(news_sentiment_note(symbol_evidence))

    with st.expander("Full decision history (all logged runs)"):
        st.dataframe(decisions, width="stretch", hide_index=True)

st.divider()

# --- Price history -----------------------------------------------------------
st.subheader("Price history")
st.caption("The daily closing price of one stock, to sanity-check what the screener saw.")

price_symbols = load_price_symbols()
if not price_symbols:
    st.info("No price data stored yet — run `python -m data.ingest.prices` first.")
else:
    _RANGES = {"3 months": 90, "6 months": 180, "1 year": 365, "3 years": 1095}
    pick_col, range_col, _ = st.columns([1, 1, 2])
    with pick_col:
        default_symbol = table.iloc[0]["Symbol"] if not table.empty else price_symbols[0]
        selected_symbol = st.selectbox(
            "Stock",
            price_symbols,
            index=price_symbols.index(default_symbol) if default_symbol in price_symbols else 0,
        )
    with range_col:
        selected_range = st.selectbox("Time range", list(_RANGES), index=2)

    prices = load_prices([selected_symbol], lookback_days=_RANGES[selected_range])
    if prices.empty:
        st.info(f"No stored prices for {selected_symbol} in the last {selected_range.lower()}.")
    else:
        prices = prices.sort_values("ts")
        first_close, last_close = float(prices["close"].iloc[0]), float(prices["close"].iloc[-1])
        change = last_close / first_close - 1 if first_close else 0.0
        st.metric(
            f"{selected_symbol} — latest close",
            f"${last_close:,.2f}",
            delta=f"{change:+.1%} over {selected_range.lower()}",
        )
        _show(
            _timeseries_chart(
                prices[["ts", "close"]],
                value_col="close",
                y_title="Closing price ($)",
                value_format="$,.2f",
                value_label="Close",
            )
        )

st.divider()

# --- Forecast accuracy -------------------------------------------------------
st.subheader("Was the model right?")
st.caption(
    "Compares each past prediction against what the price actually did next. "
    "A hit rate above 50% means the model called the direction correctly more "
    "often than a coin flip — the grey line on the chart marks that 50% mark."
)

forecasted = decisions.dropna(subset=["forecast"])[["symbol", "ts", "forecast"]] if not decisions.empty else decisions
if decisions.empty or forecasted.empty:
    st.info("Needs logged picks with forecasts before accuracy can be measured.")
else:
    accuracy = compute_forecast_accuracy(
        forecasted, load_prices(sorted(forecasted["symbol"].unique()), lookback_days=1095)
    )
    if accuracy.empty:
        st.info(
            "No prediction has had time to play out yet (needs at least one trading "
            "day of price history after the prediction was made)."
        )
    else:
        st.metric("Directional hit rate", f"{accuracy['hit'].mean():.1%}")
        rolling = (
            accuracy.sort_values("ts")
            .assign(hit_rate=lambda d: d["hit"].astype(float).rolling(20, min_periods=5).mean())
            .dropna(subset=["hit_rate"])[["ts", "hit_rate"]]
        )
        if rolling.empty:
            st.caption("Not enough predictions yet to plot a trend (needs at least 5).")
        else:
            _show(
                _timeseries_chart(
                    rolling,
                    value_col="hit_rate",
                    y_title="Hit rate (rolling 20 predictions)",
                    value_format=".0%",
                    value_label="Hit rate",
                    baseline=0.5,
                )
            )

st.divider()

# --- Equity curve ------------------------------------------------------------
st.subheader("Account value over time")
st.caption(
    "How much the portfolio is worth, and how far it has fallen below its own "
    "previous peak (the 'drawdown' — the worst dip an investor would have sat through)."
)

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
    equity = equity.assign(drawdown=equity["equity_value"] / running_peak - 1)
    _show(
        _timeseries_chart(
            equity[["ts", "equity_value"]],
            value_col="equity_value",
            y_title="Account value ($)",
            value_format="$,.0f",
            value_label="Value",
        )
    )
    _show(
        _timeseries_chart(
            equity[["ts", "drawdown"]],
            value_col="drawdown",
            y_title="Drawdown (% below peak)",
            value_format=".1%",
            value_label="Drawdown",
            zero=True,
            area=True,
        )
    )

st.divider()

# --- Circuit breakers --------------------------------------------------------
st.subheader("Safety switches")
st.caption(
    "Automatic checks that halt trading if something goes wrong — for example "
    "losing too much, or putting too much money into one stock."
)

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
            st.dataframe(tripped, width="stretch", hide_index=True)
