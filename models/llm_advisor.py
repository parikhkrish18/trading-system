"""
Claude Sonnet 5 advisory pass over the screener's confident-candidate pool.

Downstream of the quantitative model, never upstream of it: this only ever
sees candidates models/screener.py has already put through score_universe's
cost hurdle + Donchian breakout gate and apply_macro_sector_block's hard
block -- it can rank and pick among them, but can never introduce a symbol
that failed a hard gate (see _parse_advice's valid_symbols check).

Advisory, not authoritative, on money: its confidence score only
RE-WEIGHTS capital within select_concentrated_trades's existing
_bounded_conviction_weights floor/cap, and its suggested TP/SL is clamped
to execution/exit_levels.py's existing volatility-derived bounds
(exit_levels_advised) -- it can never set a position size or exit level
outside what the system's existing risk settings already allow.

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

_SYSTEM_PROMPT = (
    "You are a trading analyst reviewing a shortlist of stocks that have "
    "ALREADY cleared a quantitative screen (a cost hurdle, a breakout-"
    "direction gate, and a market/sector sentiment check) -- your job is "
    "not to decide whether these are tradeable at all, but to synthesize "
    "everything given about each one (the model's forecast, its top "
    "feature drivers, market regime, sector sentiment, recent news, and "
    "volatility) into genuine judgment: how confident are you in each "
    "pick, what take-profit/stop-loss makes sense for it, and which of "
    "them are the strongest trades to actually take right now.\n\n"
    "For EACH candidate, write three short sections a retail trader could "
    "read at a glance, matching this exact framework:\n"
    "  - signals: what the market regime and the top feature drivers say "
    "about this stock\n"
    "  - forecast: what's predicted, and how confident you are in it "
    "given everything else you were told (not just restating the model's "
    "own forecast number back)\n"
    "  - selection: why this stock specifically, in plain English\n"
    "Each section needs a one-sentence summary and 2-4 short prose lines "
    "(no jargon a retail trader wouldn't know).\n\n"
    "Assign each candidate a confidence score from 0.0 to 1.0 -- your own "
    "independent judgment, which may differ from the model's own "
    "conviction_score if the qualitative context (news, sentiment, "
    "regime) argues for more or less conviction than that raw number "
    "alone suggests.\n\n"
    "Suggest take_profit_pct and stop_loss_pct (positive fractions of "
    "entry price, e.g. 0.07 for 7%) for an entry at current_price -- "
    "quant_take_profit_pct/quant_stop_loss_pct on each candidate show "
    "what the existing volatility-derived formula would set for that "
    "stock as a reference point; you may suggest something different if "
    "you have good reason, but a similar magnitude is usually right.\n\n"
    'Finally, recommend "picks": an ordered list of AT MOST max_picks '
    "symbols (fewer is fine if you're not genuinely confident in that "
    "many) -- the strongest trades to actually take this cycle, most "
    "confident first. Only recommend symbols from the given candidate "
    "list.\n\n"
    "Respond with ONLY a JSON object of this exact shape, no other text:\n"
    '{"candidates": [{"symbol": <string>, "confidence": <float 0-1>, '
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
        by_symbol[symbol] = {
            "confidence": max(0.0, min(1.0, float(c["confidence"]))),
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
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    return resp.content[0].text


def get_llm_trade_advice(candidates: list[dict], market_context: dict, max_picks: int) -> dict | None:
    """
    `candidates`: plain dicts, one per pool member -- symbol, side,
    predicted_return, conviction_score, macro_sector_sentiment,
    current_price, daily_volatility_pct, top_features, recent_headlines,
    exit_bounds (see models/screener.py's _build_llm_candidate_pool for the
    exact assembly).

    Returns {"by_symbol": {symbol: {confidence, take_profit_pct,
    stop_loss_pct, reasoning}}, "picks": [symbol, ...]} on success, or None
    (never raises) when the API key is unset, the pool is empty, the call
    fails, or the response never validates after retries.
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
