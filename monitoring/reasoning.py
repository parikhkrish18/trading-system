"""
Plain-English trade reasoning, phase by phase.

Every automated decision this system makes — trade or no trade, open or
close — goes through the same 7 stages, in this order:

    1. Pre-Trade Risk Check     — did a circuit breaker block trading at all?
    2. Market Regime & Signals  — what the data actually looked like
    3. Forecast & Confidence    — what the model predicted, and how sure it was
    4. Candidate Selection & Sizing — why this symbol, why this much capital
    5. Execution                — what order actually got placed and how
    6. Reconciliation & Post-Trade Check — did the fill match the intent
    7. Ongoing Monitoring       — what happens to this position next

This module turns the raw numbers behind each stage into short, plain-
English lines — one function per phase, called at the point in
execution/trading_loop.py or execution/contradiction_monitor.py where that
phase's facts become available. The result of each is a small dict with a
`summary` (one line, meant for Slack) and `lines` (the full explanation,
meant for the dashboard). Pure formatting only — no DB/API calls here.
"""
from __future__ import annotations

FEATURE_LABELS = {
    "mom_ret_5d": "5-day price momentum",
    "mom_ret_20d": "20-day price momentum",
    "adx_14": "trend strength (ADX)",
    "vol_realized_20d": "20-day realized volatility",
    "vol_atr_14": "average daily trading range (ATR)",
    "vol_of_vol": "volatility-of-volatility",
    "meanrev_zscore_20d": "distance from the 20-day average price (z-score)",
    "meanrev_bollinger_pctb": "position within the Bollinger band",
    "meanrev_rsi_14": "RSI (overbought/oversold pressure)",
    "sentiment_mean_10d": "10-day average news sentiment",
    "sentiment_mean_3d": "3-day average news sentiment",
    "sentiment_momentum_3v10": "news sentiment trend (3-day vs 10-day)",
    "news_volume_3d": "3-day news volume",
    "days_to_next_fomc": "days until the next Fed meeting",
    "days_to_next_cpi": "days until the next CPI report",
    "days_to_next_jobs": "days until the next jobs report",
    "fund_eps_actual_latest": "latest reported EPS",
    "fund_revenue_actual_latest": "latest reported revenue",
    "fund_net_income_latest": "latest reported net income",
    "fund_gross_profit_latest": "latest reported gross profit",
    "fund_total_assets_latest": "latest reported total assets",
    "fund_total_liabilities_latest": "latest reported total liabilities",
}

_PCT_PREFIXES = ("mom_ret", "vol_realized", "sentiment_mean", "sentiment_momentum")


def _fmt_feature_value(feature_name: str, value: float | None) -> str:
    if value is None:
        return "unavailable"
    if feature_name.startswith(_PCT_PREFIXES):
        return f"{value:+.1%}"
    if feature_name.startswith("days_to_next"):
        return f"{value:.0f} day(s) away"
    if feature_name == "news_volume_3d":
        return f"{value:.0f} article(s)"
    return f"{value:,.2f}"


def explain_feature(feature_name: str, value: float | None, contribution: float) -> str:
    """One plain-English sentence for a single SHAP feature contribution."""
    label = FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))
    label = label[0].upper() + label[1:]
    direction = "pushed the forecast up" if contribution >= 0 else "pushed the forecast down"
    return f"{label} was {_fmt_feature_value(feature_name, value)} — {direction}."


def phase_pretrade_risk(triggers: list) -> dict:
    """Phase 1. `triggers`: circuit_breakers.BreakerResult list (empty if none tripped)."""
    if not triggers:
        return {
            "phase": 1,
            "title": "Pre-Trade Risk Check",
            "summary": "All circuit breakers clear — trading proceeded.",
            "lines": [
                "Checked max drawdown, max single-position size, and max correlated exposure.",
                "None were breached, so the cycle was allowed to trade.",
            ],
        }
    reasons = "; ".join(t.reason for t in triggers)
    return {
        "phase": 1,
        "title": "Pre-Trade Risk Check",
        "summary": f"Circuit breaker tripped — all positions flattened. {reasons}",
        "lines": [f"Breached: {reasons}", "Every open position was closed instead of trading this cycle."],
    }


def phase_signals(regime: str | None, top_features: list[dict]) -> dict:
    """Phase 2. `top_features`: [{feature_name, value, contribution}, ...], strongest first."""
    lines = []
    if regime:
        lines.append(f"Market regime read as {regime.upper()} (based on trend strength on SPY).")
    lines += [explain_feature(f["feature_name"], f.get("value"), f["contribution"]) for f in top_features]
    top_label = FEATURE_LABELS.get(top_features[0]["feature_name"], top_features[0]["feature_name"]) if top_features else "n/a"
    return {
        "phase": 2,
        "title": "Market Regime & Signals",
        "summary": f"Regime: {regime or 'n/a'}. Strongest driver: {top_label}.",
        "lines": lines or ["No feature contributions available for this decision."],
    }


def phase_forecast(
    predicted_return: float,
    direction_agreement: float,
    conviction_score: float,
    min_direction_agreement: float = 0.8,
) -> dict:
    """Phase 3. What the 5-model ensemble predicted and how confident it was."""
    pct = f"{predicted_return:+.2%}"
    agree_pct = f"{direction_agreement:.0%}"
    lines = [
        f"The 5-model ensemble forecasts a {pct} move over the next 5 trading days.",
        f"{agree_pct} of the 5 models agree on that direction "
        f"(needed at least {min_direction_agreement:.0%} to qualify for a trade).",
        f"Conviction score (agreement × size of the move): {conviction_score:.4f} — this is what ranks candidates against each other.",
    ]
    return {
        "phase": 3,
        "title": "Forecast & Confidence",
        "summary": f"{pct} forecast, {agree_pct} model agreement.",
        "lines": lines,
    }


def phase_selection_closed(symbol: str) -> dict:
    """Phase 4, for a position that fell out of this cycle's top picks and is being closed."""
    return {
        "phase": 4,
        "title": "Candidate Selection & Sizing",
        "summary": f"{symbol} was not one of this cycle's top picks.",
        "lines": [
            "Only the top 2 candidates by conviction score are traded each cycle.",
            f"{symbol} did not rank in the top 2 this time, so its position was closed rather than left open on stale conviction.",
        ],
    }


def phase_selection(
    symbol: str,
    side: str,
    weight_pct: float,
    n_confident: int,
    n_selected: int,
    max_leg_pct: float,
    min_leg_pct: float,
) -> dict:
    """Phase 4. Why this symbol specifically, and how much capital it got."""
    lines = [f"{n_confident} symbol(s) cleared the confidence bar this cycle; the top {n_selected} by conviction score were picked."]
    lines.append(f"{symbol} was sized as a {side} position for {abs(weight_pct):.1%} of total portfolio capital.")
    if n_selected == 1:
        lines.append("Only one candidate cleared the bar, so it received the full deployable capital alone rather than splitting with a weaker second pick.")
    else:
        at_bound = abs(abs(weight_pct) - max_leg_pct) < 1e-6 or abs(abs(weight_pct) - min_leg_pct) < 1e-6
        if at_bound:
            lines.append(
                f"The split between the two picks is capped to [{min_leg_pct:.0%}, {max_leg_pct:.0%}] of capital per leg "
                "so one pick can't swallow the whole book even at an extreme confidence gap — this leg hit that cap."
            )
        else:
            lines.append("The split between the two picks is weighted by their relative conviction scores.")
    return {
        "phase": 4,
        "title": "Candidate Selection & Sizing",
        "summary": f"{symbol}: {side}, {abs(weight_pct):.1%} of capital.",
        "lines": lines,
    }


def phase_execution(
    symbol: str,
    action: str,  # "opened" | "closed" | "adjusted"
    shares: float | None,
    order_type: str,  # "market" | "limit (extended hours)"
) -> dict:
    """Phase 5. What order actually got sent to the broker."""
    if action == "closed":
        lines = [f"{symbol} was closed out — the target position is 0 shares.", f"Order type: {order_type}."]
        summary = f"{symbol} closed."
    else:
        share_str = f"{abs(shares):.4g} shares" if shares is not None else "an unknown number of shares"
        lines = [f"Target: {share_str} of {symbol} ({'long' if (shares or 0) >= 0 else 'short'}).", f"Order type: {order_type}."]
        summary = f"{symbol}: {share_str} via {order_type}."
    return {"phase": 5, "title": "Execution", "summary": summary, "lines": lines}


def phase_reconciliation(symbol: str, intended_shares: float, actual_shares: float, flagged: bool) -> dict:
    """Phase 6. Did the actual fill match what was intended."""
    diff = actual_shares - intended_shares
    if flagged:
        lines = [
            f"Intended {intended_shares:.4g} shares, actually filled {actual_shares:.4g} — a {diff:+.4g} share divergence beyond the normal 2% tolerance.",
            "This was flagged and alerted — worth a manual look.",
        ]
        summary = f"{symbol}: reconciliation FLAGGED (diff {diff:+.4g})."
    else:
        lines = [f"Intended {intended_shares:.4g} shares, filled {actual_shares:.4g} — within the normal 2% tolerance."]
        summary = f"{symbol}: reconciled cleanly."
    return {"phase": 6, "title": "Reconciliation & Post-Trade Check", "summary": summary, "lines": lines}


def phase_ongoing_monitoring(closed: bool = False) -> dict:
    """Phase 7. What happens between now and the next weekly screen."""
    if closed:
        lines = ["This position is closed — no further monitoring needed until it (or something else) is picked again."]
        summary = "No position held — nothing to monitor."
    else:
        lines = [
            "This position is checked hourly during market hours against fresh news sentiment and 5-day price momentum.",
            "If either signal turns strongly against this position before the next weekly screen, it will be closed automatically (see execution/contradiction_monitor.py).",
        ]
        summary = "Hourly contradiction checks active until the next weekly screen."
    return {"phase": 7, "title": "Ongoing Monitoring", "summary": summary, "lines": lines}


def phase_contradiction(reasons: list[dict]) -> dict:
    """Phase 2 substitute used by the hourly contradiction monitor instead of phase_signals."""
    lines = [r["detail"] for r in reasons]
    signals = ", ".join(r["signal"] for r in reasons)
    return {
        "phase": 2,
        "title": "Market Regime & Signals",
        "summary": f"Contradiction detected via: {signals}.",
        "lines": lines,
    }


def combine_phases(*phases: dict) -> list[dict]:
    """Sorts and packages phase dicts into the final list stored in decisions.reasoning."""
    return sorted(phases, key=lambda p: p["phase"])
