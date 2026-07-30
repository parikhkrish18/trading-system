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


def _bucket(value: float, edges: list[tuple[float, str]], default: str) -> str:
    """Returns the label for the first edge `value` doesn't exceed, walking low to high."""
    for threshold, label in edges:
        if value <= threshold:
            return label
    return default


def _momentum_narrative(window_label: str) :
    def fn(value: float | None) -> str:
        if value is None:
            return "Price momentum data was unavailable."
        pct = f"{value:+.1%}"
        mood = _bucket(
            value,
            [(-0.15, "sold off sharply"), (-0.05, "drifted lower"), (0.05, "traded roughly flat"), (0.15, "climbed steadily")],
            default="rallied hard",
        )
        colour = {
            "sold off sharply": "a decline steep enough to reflect either real deterioration in sentiment or a stretched, oversold setup",
            "drifted lower": "a mild downtrend rather than a sharp break",
            "traded roughly flat": "no strong directional push either way",
            "climbed steadily": "a healthy uptrend without looking overextended",
            "rallied hard": "the kind of move that can mark genuine strength or a chase-prone, overbought rally",
        }[mood]
        return f"The stock has {mood} over the past {window_label} ({pct}) — {colour}."

    return fn


def _adx_narrative(value: float | None) -> str:
    if value is None:
        return "Trend-strength (ADX) data was unavailable."
    level = _bucket(
        value,
        [(20, "weak"), (25, "borderline"), (40, "solid")],
        default="very strong",
    )
    colour = {
        "weak": "price action looks choppy and range-bound, which tends to make directional bets less reliable",
        "borderline": "the market isn't clearly trending or chopping either way",
        "solid": "the stock is in a real directional trend, not just noise",
        "very strong": "a strong, possibly stretched trend that can either persist or snap back hard",
    }[level]
    return f"Trend strength (ADX {value:.0f}) is {level} — {colour}."


def _volatility_narrative(value: float | None) -> str:
    if value is None:
        return "20-day volatility data was unavailable."
    pct = f"{value:.0%}"
    level = _bucket(value, [(0.20, "low"), (0.40, "moderate"), (0.70, "elevated")], default="very high")
    colour = {
        "low": "the stock has been trading calmly, which usually supports more confident directional calls",
        "moderate": "normal day-to-day noise, nothing unusual",
        "elevated": "the stock has been swinging hard, adding real uncertainty to any forecast",
        "very high": "sharp, unstable price action that makes the near-term direction much harder to call",
    }[level]
    return f"20-day volatility ({pct} annualized) is {level} — {colour}."


def _atr_narrative(value: float | None) -> str:
    if value is None:
        return "Average daily trading range (ATR) data was unavailable."
    return (
        f"The stock's average daily trading range (ATR) is {value:,.2f} points — "
        "a measure of typical day-to-day movement, used here alongside volatility to gauge how much noise to expect."
    )


def _vol_of_vol_narrative(value: float | None) -> str:
    if value is None:
        return "Volatility-of-volatility data was unavailable."
    return (
        f"Volatility-of-volatility is reading {value:.3f} — how unstable the stock's own volatility has been. "
        "A higher reading points to choppier, less predictable risk conditions around this stock."
    )


def _zscore_narrative(value: float | None) -> str:
    if value is None:
        return "Distance from the 20-day average price was unavailable."
    side = "above" if value >= 0 else "below"
    level = _bucket(abs(value), [(1.0, "near its recent average"), (2.0, "moderately stretched")], default="extremely stretched")
    colour = {
        "near its recent average": "nothing unusual about where price sits right now",
        "moderately stretched": "a setup that can either keep extending or start mean-reverting",
        "extremely stretched": "an unusually large deviation, the kind that often precedes either a sharp reversal or a genuine breakout",
    }[level]
    return f"Price is {abs(value):.1f} standard deviations {side} its 20-day average ({level}) — {colour}."


def _bollinger_narrative(value: float | None) -> str:
    if value is None:
        return "Bollinger Band position data was unavailable."
    zone = _bucket(value, [(0.0, "below the lower band"), (0.2, "near the lower band"), (0.8, "mid-band")], default="near the upper band")
    if value > 1.0:
        zone = "above the upper band"
    colour = {
        "below the lower band": "deep oversold territory relative to its own recent volatility",
        "near the lower band": "leaning toward oversold",
        "mid-band": "sitting in a normal, unremarkable range",
        "near the upper band": "leaning toward overbought",
        "above the upper band": "deep overbought territory relative to its own recent volatility",
    }[zone]
    return f"Price is {zone} of its recent volatility range (%B = {value:.2f}) — {colour}."


def _rsi_narrative(value: float | None) -> str:
    if value is None:
        return "RSI data was unavailable."
    level = _bucket(value, [(30, "oversold"), (45, "leaning weak"), (55, "neutral"), (70, "leaning strong")], default="overbought")
    colour = {
        "oversold": "selling may be overdone, a condition that often precedes a bounce even in a downtrend",
        "leaning weak": "momentum is soft but not extreme",
        "neutral": "no strong overbought or oversold pressure either way",
        "leaning strong": "momentum is firm but not extreme",
        "overbought": "buying may be overdone, a condition that often precedes a pause or pullback even in an uptrend",
    }[level]
    return f"RSI is at {value:.0f} ({level}) — {colour}."


def _sentiment_narrative(window_label: str) :
    def fn(value: float | None) -> str:
        if value is None:
            return "News sentiment data was unavailable."
        level = _bucket(
            value,
            [(-0.3, "clearly negative"), (-0.1, "mildly negative"), (0.1, "mixed/neutral"), (0.3, "mildly positive")],
            default="clearly positive",
        )
        return f"News coverage over the last {window_label} reads {level} (sentiment {value:+.2f})."

    return fn


def _sentiment_momentum_narrative(value: float | None) -> str:
    if value is None:
        return "News sentiment trend data was unavailable."
    trend = "improving" if value >= 0 else "deteriorating"
    return f"News sentiment has been {trend} recently versus its 10-day baseline ({value:+.2f})."


def _news_volume_narrative(value: float | None) -> str:
    if value is None:
        return "News volume data was unavailable."
    level = _bucket(value, [(2, "low"), (8, "typical")], default="elevated")
    colour = {
        "low": "little coverage, so this signal carries less weight",
        "typical": "an ordinary amount of coverage",
        "elevated": "a burst of attention, the kind that can precede or amplify a larger move",
    }[level]
    return f"There were {value:.0f} news article(s) in the last 3 days — {colour}."


def _event_narrative(label: str) :
    def fn(value: float | None) -> str:
        if value is None:
            return f"Days until {label} was unavailable."
        urgency = _bucket(value, [(5, "imminent"), (15, "approaching")], default="not yet a near-term factor")
        colour = {
            "imminent": "expect elevated pre-event positioning and volatility risk into it",
            "approaching": "starting to factor into near-term positioning",
            "not yet a near-term factor": "still far enough out that it isn't driving near-term risk yet",
        }[urgency]
        return f"{label.capitalize()} is {value:.0f} day(s) away ({urgency}) — {colour}."

    return fn


def _fundamentals_narrative(label: str) :
    def fn(value: float | None) -> str:
        if value is None:
            return f"{label} data was unavailable."
        return f"Latest reported {label.lower()} is {value:,.2f} (from the most recent filed report)."

    return fn


_NARRATIVE_FNS = {
    "mom_ret_5d": _momentum_narrative("5 days"),
    "mom_ret_20d": _momentum_narrative("20 days"),
    "adx_14": _adx_narrative,
    "vol_realized_20d": _volatility_narrative,
    "vol_atr_14": _atr_narrative,
    "vol_of_vol": _vol_of_vol_narrative,
    "meanrev_zscore_20d": _zscore_narrative,
    "meanrev_bollinger_pctb": _bollinger_narrative,
    "meanrev_rsi_14": _rsi_narrative,
    "sentiment_mean_10d": _sentiment_narrative("10 days"),
    "sentiment_mean_3d": _sentiment_narrative("3 days"),
    "sentiment_momentum_3v10": _sentiment_momentum_narrative,
    "news_volume_3d": _news_volume_narrative,
    "days_to_next_fomc": _event_narrative("the next Fed meeting"),
    "days_to_next_cpi": _event_narrative("the next CPI report"),
    "days_to_next_jobs": _event_narrative("the next jobs report"),
    "fund_eps_actual_latest": _fundamentals_narrative("EPS"),
    "fund_revenue_actual_latest": _fundamentals_narrative("Revenue"),
    "fund_net_income_latest": _fundamentals_narrative("Net income"),
    "fund_gross_profit_latest": _fundamentals_narrative("Gross profit"),
    "fund_total_assets_latest": _fundamentals_narrative("Total assets"),
    "fund_total_liabilities_latest": _fundamentals_narrative("Total liabilities"),
}


def explain_feature(feature_name: str, value: float | None, contribution: float) -> str:
    """
    One plain-English, value-aware sentence for a single SHAP feature
    contribution — describes what the raw value actually implies about the
    stock (e.g. "sold off sharply", "RSI is overbought"), then states which
    way the model weighted it, so the reasoning reads like something a
    person with basic investing knowledge would say, not a raw number dump.
    """
    narrative_fn = _NARRATIVE_FNS.get(feature_name)
    if narrative_fn is not None:
        narrative = narrative_fn(value)
    else:
        label = feature_name.replace("_", " ")
        label = label[0].upper() + label[1:]
        narrative = f"{label} was {_fmt_feature_value(feature_name, value)}."

    if value is None:
        direction = "still factored in, pushing the forecast up" if contribution >= 0 else "still factored in, pushing the forecast down"
        return f"{narrative} It's {direction}."

    direction = "This pushed the forecast up" if contribution >= 0 else "This pushed the forecast down"
    return f"{narrative} {direction}."


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
    side = "rise" if predicted_return >= 0 else "fall"
    agreement_colour = (
        "every model independently reached the same conclusion, which is a much stronger signal than one model alone"
        if direction_agreement >= 0.999
        else "most, but not all, of the models agree — a real signal, but with some internal disagreement about direction"
    )
    lines = [
        f"The 5-model ensemble expects this stock to {side} about {pct} over the next 5 trading days.",
        f"{agree_pct} of the 5 independently-trained models agree on that direction "
        f"(needed at least {min_direction_agreement:.0%} to qualify for a trade) — {agreement_colour}.",
        f"Conviction score (agreement × size of the move): {conviction_score:.4f} — the higher this is, the more capital "
        "this pick gets relative to the other one selected this cycle.",
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
