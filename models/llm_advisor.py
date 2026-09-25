"""
Claude Sonnet 5 advisory pass over the screener's confident-candidate pool.

Downstream of the quantitative model, never upstream of it: this only ever
sees candidates models/screener.py has already put through score_universe's
cost/ATR-floor hurdle and apply_macro_sector_block's hard block -- it can
rank and pick among them, but can never introduce a symbol that failed a
hard gate (see _parse_advice's valid_symbols check).

Its confidence score RE-WEIGHTS capital within select_concentrated_trades's
existing _bounded_conviction_weights floor/cap, and its suggested TP/SL is
clamped to execution/exit_levels.py's existing volatility-derived bounds
(exit_levels_advised) -- it can never set a position size or exit level
outside what the system's existing risk settings already allow.

Its predicted_return_pct DOES replace the quant ensemble's own forecast
for a picked symbol (see models/screener.py's _apply_llm_advice_to_scored)
-- magnitude only, never direction: the candidate's side (long/short) was
already decided by the quant model and can't be flipped here, only how big
the move is expected to be, informed by qualitative context (news,
Donchian support/resistance, sector sentiment) the quant model never sees.
That replaces conviction_score too, and everything downstream that reads
it -- position sizing weight, take-profit target sizing. A missing or
invalid predicted_return_pct falls back to the quant forecast unchanged,
same fail-open convention as everything else here.

Fails closed: an unset ANTHROPIC_API_KEY, an API error, or a response that
never parses after retries all return None. The caller
(models/screener.py) falls back to the exact pre-existing quant-only
selection and template-based reasoning in every one of those cases -- an
LLM outage or a bad response must never block or corrupt a live trading
cycle, only lose the extra judgment layer for that one cycle.
"""
from __future__ import annotations

import json
import logging
import math

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
# A malformed/empty response is usually sampling variance (a stray unescaped
# quote, an omitted field), not a persistent problem with this pool's
# content -- re-asking the identical question often just works. Retried
# immediately, no backoff, same convention as
# features/qualitative/sentiment.py -- this is a malformed-response retry,
# not a rate-limit/network one (the anthropic client already retries those
# itself before an exception ever reaches this module).
_MAX_ATTEMPTS = 3

def _build_system_prompt() -> str:
    """
    Interpolates the live horizon/ATR-multiple settings rather than
    hardcoding them, so the prompt can never drift out of sync with
    execution/exit_levels.py's actual bounds (settings.target_horizon_days,
    exit_take_profit_min_atr_mult/max_atr_mult) if those are ever retuned.
    """
    horizon_days = settings.target_horizon_days
    tp_min_mult = settings.exit_take_profit_min_atr_mult
    tp_max_mult = settings.exit_take_profit_max_atr_mult
    donchian_window = settings.donchian_exit_window
    return (
        "You are a trading analyst reviewing a shortlist of stocks that have "
        "ALREADY cleared a quantitative screen (a cost/ATR-floor hurdle and "
        "a market/sector sentiment check) -- your job is "
        "not to decide whether these are tradeable at all, but to synthesize "
        "everything given about each one (the model's forecast, its top "
        "feature drivers, market regime, sector sentiment, recent news, and "
        "volatility) into genuine judgment: how confident are you in each "
        "pick, what take-profit/stop-loss makes sense for it, and which of "
        "them are the strongest trades to actually take right now. "
        "signal_to_noise on each candidate (|predicted move| / how much the "
        "ensemble's members disagree on its size) is informational, not a "
        "pass/fail bar -- it hasn't been shown to predict being right on its "
        "own, so weigh it alongside everything else rather than gating on "
        "it. Any fund_*_latest line in top_features (EPS, revenue, net "
        "income, etc.) is only a fresh, near-term catalyst on the day it "
        "was filed or the day after -- the market has typically priced a "
        "financial statement in by then. The narrative itself will tell "
        "you when a filing is stale (\"long-term speculation support, not "
        "a fresh catalyst\"); treat those as background context on the "
        "business, never as a reason for near-term conviction on their "
        "own.\n\n"
        f"THE STRATEGY: this book holds swing trades for about {horizon_days} "
        "trading days (roughly a week), not day trades and not multi-month "
        "positions. Every pick should be a setup you genuinely expect to "
        f"resolve -- hit its target or invalidate its thesis -- within that "
        "window, not a slow-grind long-term story. Take-profit and "
        f"stop-loss are sized as a multiple of the stock's own ATR (atr_pct "
        f"on each candidate: its 14-day Average True Range as a fraction of "
        f"price) scaled to this horizon -- normally {tp_min_mult:g}x-{tp_max_mult:g}x "
        "of that horizon-scaled ATR for the target -- and then SOLIDIFIED "
        "against this stock's own recent trading range across several "
        f"timeframes at once, from a {donchian_window}-day channel up to a "
        "200-day one (the wider the timeframe, the more that level has "
        "actually been tested and held, so a genuinely major level always "
        "wins over a minor one that just happens to sit closer): "
        "donchian_support/donchian_resistance on each candidate are the "
        "actual rolling low/high of the single widest one currently in "
        "play, and both the target and the stop get pulled in toward "
        "whichever side is closer, so a target never sits past a "
        "resistance this stock has actually failed to clear recently and a "
        "stop never sits past a support it has actually held. "
        "donchian_levels lists EVERY one of those timeframes the stock has "
        "enough history for (200/100/50/20-day), each as its own "
        "window_days/support/resistance, largest first -- donchian_support/"
        "donchian_resistance is just the single widest one still in play; "
        "the full list lets you see, for example, that a 20-day resistance "
        "sits right on top of entry while the 100- and 200-day ones are "
        "much further out, meaning there's real room for this move even "
        "though the nearest timeframe alone would look capped, or the "
        "reverse -- several timeframes clustering at the same price is "
        "stronger evidence of a real ceiling/floor than any one of them "
        "alone.\n\n"
        "A Donchian level only tells you price REACHED that number, not "
        "that anyone traded size there -- a single thin, undefended spike "
        "prints the exact same rolling high as a level the market fought "
        "over all week. volume_profile_poc/value_area_low/value_area_high "
        "on each candidate are built from real intraday trading (not the "
        "daily bars everything else here comes from): poc is where the "
        "most volume actually traded over the recent window, and the value "
        "area (value_area_low to value_area_high) is the band holding "
        "about 70% of it -- the market's real zone of agreement on fair "
        "value. donchian_support_volume_confirmed/"
        "donchian_resistance_volume_confirmed tell you directly whether "
        "each Donchian level sits inside that value area (true), was an "
        "untested move outside it (false), or there wasn't enough intraday "
        "data to check (null -- treat the Donchian level on its own "
        "merits, same as always). An unconfirmed resistance is weaker "
        "evidence the target has real room, and an unconfirmed support is "
        "weaker evidence the stop will actually hold -- weigh confidence "
        "down accordingly, the same way you'd discount a technical level "
        "you knew nobody had actually traded near. A stock with "
        f"no realistic path to a {tp_min_mult:g}x-ATR move in about a week, "
        "or with real resistance/support sitting right on top of entry, is "
        "a weak pick here even if the raw forecast number looks fine; a "
        "stock whose setup (momentum, breakout, catalyst, news) genuinely "
        f"supports reaching the top of that band ({tp_max_mult:g}x ATR) with "
        "clear room before the next real level is a stronger one. Weigh "
        "confidence accordingly -- this is exactly the kind of judgment the "
        "quant model alone can't make.\n\n"
        "For EACH candidate, write three short sections a retail trader could "
        "read at a glance, matching this exact framework:\n"
        "  - signals: what the market regime and the top feature drivers say "
        "about this stock\n"
        "  - forecast: what's predicted, and how confident you are it plays "
        "out within the week given everything else you were told (not just "
        "restating the model's own forecast number back)\n"
        "  - selection: why this stock specifically, in plain English -- "
        "including whether its setup realistically supports a move in the "
        f"{tp_min_mult:g}x-{tp_max_mult:g}x ATR range this week, and whether "
        "its own support/resistance leaves room for that move or sits in "
        "the way of it\n"
        "Each section needs a one-sentence summary and 2-4 short prose lines "
        "(no jargon a retail trader wouldn't know).\n\n"
        "Assign each candidate a confidence score from 0.0 to 1.0 -- your own "
        "independent judgment, which may differ from the model's own "
        "conviction_score if the qualitative context (news, sentiment, "
        "regime) argues for more or less conviction than that raw number "
        "alone suggests, or if the setup doesn't realistically support "
        "resolving within the week.\n\n"
        "Also give your own predicted_return_pct: how large a move you "
        "genuinely expect over the week, as a POSITIVE fraction (e.g. 0.04 "
        "for 4%) -- magnitude only, in the direction the candidate's side "
        "already establishes (long or short was decided upstream of you "
        "and can't change here). predicted_return on each candidate is the "
        "quant model's own forecast, informational, not a floor or ceiling "
        "-- weigh it against everything else you were given (top_features, "
        "recent_headlines, donchian_support/donchian_resistance, "
        "macro_sector_sentiment, atr_pct, signal_to_noise) and give your "
        "own honest estimate. A story with strong confirming news and a "
        "clean breakout past resistance can genuinely support a bigger "
        "move than the quant number alone suggests; a stock with the same "
        "quant forecast but stale/contradicting news or real resistance "
        "sitting right on top of entry should get a smaller one. This "
        "number replaces the quant forecast for sizing and the "
        "take-profit target on any candidate you pick, so it should be a "
        "real, considered estimate, not a rubber stamp of the input.\n\n"
        "Suggest take_profit_pct and stop_loss_pct (positive fractions of "
        "entry price, e.g. 0.07 for 7%) for an entry at current_price -- "
        "quant_take_profit_pct/quant_stop_loss_pct on each candidate show "
        f"what the existing formula ({tp_min_mult:g}x-{tp_max_mult:g}x this "
        "stock's own horizon-scaled ATR, solidified against its "
        "donchian_support/donchian_resistance) would set for that stock as "
        "a reference point; you may suggest something different if you "
        "have good reason, but a similar magnitude is usually right -- a "
        "target far outside that band on a normal setup usually means the "
        "horizon is wrong for this stock, not that the target should be.\n\n"
        'Finally, recommend "picks": an ordered list of AT MOST max_picks '
        "symbols (fewer is fine if you're not genuinely confident in that "
        "many) -- the strongest trades to actually take this cycle, most "
        "confident first. Only recommend symbols from the given candidate "
        "list.\n\n"
        "Respond with ONLY a JSON object of this exact shape, no other text:\n"
        '{"candidates": [{"symbol": <string>, "confidence": <float 0-1>, '
        '"predicted_return_pct": <float, positive>, '
        '"take_profit_pct": <float>, "stop_loss_pct": <float>, '
        '"signals_summary": <string>, "signals_lines": [<string>, ...], '
        '"forecast_summary": <string>, "forecast_lines": [<string>, ...], '
        '"selection_summary": <string>, "selection_lines": [<string>, ...]}, '
        '...], "picks": [<string>, ...]}'
    )


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_advice(text: str, valid_symbols: set[str]) -> dict:
    """
    Raises on anything that doesn't validate -- the caller retries or falls
    back. A symbol in the response that isn't in valid_symbols is silently
    dropped rather than trusted: it would mean advising on, or worse
    picking, a stock that never cleared score_universe/
    apply_macro_sector_block's gates this pool was built from.
    """
    data = json.loads(text)
    by_symbol: dict[str, dict] = {}
    for c in data["candidates"]:
        symbol = str(c["symbol"])
        if symbol not in valid_symbols:
            continue
        predicted_return_pct = c.get("predicted_return_pct")
        if predicted_return_pct is not None:
            predicted_return_pct = abs(float(predicted_return_pct))
            if not math.isfinite(predicted_return_pct):
                predicted_return_pct = None
        by_symbol[symbol] = {
            "confidence": max(0.0, min(1.0, float(c["confidence"]))),
            "predicted_return_pct": predicted_return_pct,
            "take_profit_pct": float(c["take_profit_pct"]) if c.get("take_profit_pct") is not None else None,
            "stop_loss_pct": float(c["stop_loss_pct"]) if c.get("stop_loss_pct") is not None else None,
            "reasoning": [
                {
                    "phase": 2,
                    "title": "Market Regime & Signals",
                    "summary": str(c["signals_summary"]),
                    "lines": [str(x) for x in c["signals_lines"]],
                },
                {
                    "phase": 3,
                    "title": "Forecast & Confidence",
                    "summary": str(c["forecast_summary"]),
                    "lines": [str(x) for x in c["forecast_lines"]],
                },
                {
                    "phase": 4,
                    "title": "Candidate Selection & Sizing",
                    "summary": str(c["selection_summary"]),
                    "lines": [str(x) for x in c["selection_lines"]],
                },
            ],
        }
    picks = [str(s) for s in data["picks"] if str(s) in by_symbol]
    return {"by_symbol": by_symbol, "picks": picks}


def _call_claude(client: Anthropic, payload: dict) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=[{"type": "text", "text": _build_system_prompt(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    # Sonnet 5 runs adaptive extended thinking by default (no `thinking`
    # param needed to turn it on), so resp.content[0] is very often a
    # ThinkingBlock ahead of the TextBlock -- content[0].text blew up with
    # "'ThinkingBlock' object has no attribute 'text'" on every single
    # call in production, silently falling back to quant-only selection
    # every cycle. Find the actual text block instead of assuming position 0.
    for block in resp.content:
        if block.type == "text":
            return block.text
    raise ValueError("Claude's response contained no text block (thinking/other content only).")


def get_llm_trade_advice(candidates: list[dict], market_context: dict, max_picks: int) -> dict | None:
    """
    `candidates`: plain dicts, one per pool member -- symbol, side,
    predicted_return, conviction_score, macro_sector_sentiment,
    current_price, daily_volatility_pct, top_features, recent_headlines,
    exit_bounds (see models/screener.py's _build_llm_candidate_pool for the
    exact assembly).

    Returns {"by_symbol": {symbol: {confidence, predicted_return_pct,
    take_profit_pct, stop_loss_pct, reasoning}}, "picks": [symbol, ...]} on
    success, or None (never raises) when the API key is unset, the pool is
    empty, the call fails, or the response never validates after retries.
    predicted_return_pct is a positive magnitude (direction already fixed
    by the candidate's own side) or None when Claude didn't give one --
    see models/screener.py's _apply_llm_advice_to_scored for how it
    replaces the quant forecast.
    """
    if not settings.anthropic_api_key or not candidates:
        return None

    valid_symbols = {c["symbol"] for c in candidates}
    payload = {"market_context": market_context, "candidates": candidates, "max_picks": max_picks}
    client = Anthropic(api_key=settings.anthropic_api_key)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            text = _strip_code_fence(_call_claude(client, payload))
            advice = _parse_advice(text, valid_symbols)
            if not advice["picks"]:
                raise ValueError("Claude returned no valid picks from the given candidate pool")
            return advice
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
            last_exc = exc
            logger.warning(
                "Could not parse Claude's trade-advisory response, attempt %d/%d.",
                attempt, _MAX_ATTEMPTS, exc_info=True,
            )
        except Exception:
            # A transient network/rate-limit error is already retried
            # inside the Anthropic client itself -- anything that still
            # reaches here is a harder failure (auth, an outage outlasting
            # the SDK's own retries) that re-asking the identical request
            # won't fix.
            logger.exception("Claude trade-advisory call failed -- falling back to quant-only selection.")
            return None

    logger.warning(
        "Falling back to quant-only candidate selection after %d failed LLM-advisory attempts: %s",
        _MAX_ATTEMPTS, last_exc,
    )
    return None
