"""
Claude second-opinion pass over positions the hourly contradiction check
(execution/contradiction_monitor.py) already flagged on a SOFT signal --
news sentiment, macro/sector alignment, or price momentum contradicting the
held side. Never consulted on a stop-loss/take-profit hit (a mechanical
exit sized to the position's own volatility, not a judgment call) and never
consulted on a position nothing flagged -- this can only ever confirm or
reverse a trip the deterministic signals already found, never generate a
contradiction claim of its own.

Fails closed the OPPOSITE way from models/llm_advisor.py's weekly-screen
pass: an unset key, an API error, or a response that never parses all fall
back to CLOSING -- the rule-based signal's own original call -- never to
silently keeping a flagged position open because Claude was unreachable.
"""
from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
# Same convention as models/llm_advisor.py: a malformed/empty response is
# usually sampling variance, not a persistent problem with this pool's
# content -- re-asking the identical question often just works.
_MAX_ATTEMPTS = 3


def _build_system_prompt() -> str:
    return (
        "You are reviewing positions an automated rule-based check has "
        "already flagged as contradicting recent news sentiment, "
        "sector-wide sentiment, or price momentum -- your job is to give a "
        "second, more qualitative opinion before the position is actually "
        "closed. The rule-based signal already fired; you are deciding "
        "whether the broader context genuinely supports closing now, or "
        "whether this is a false alarm the trade should ride through (a "
        "single noisy or thinly-sourced headline, a technical bounce that "
        "doesn't change the underlying thesis, a momentum reading that's "
        "really just short-term overextension). This is NOT a stop-loss or "
        "take-profit hit -- those close automatically on their own and are "
        "never sent here for review.\n\n"
        "For each position, weigh the specific signal(s) that tripped "
        "(reasons), the recent headlines, and the current unrealized P&L "
        "(pnl_pct). Default to trusting the rule-based signal unless you "
        "have a genuinely good reason not to -- sentiment or momentum "
        "contradicting the held side is real evidence most of the time, "
        "not noise, and this check exists to catch the minority of cases "
        "where it plainly isn't.\n\n"
        "Respond with ONLY a JSON object of this exact shape, no other "
        "text:\n"
        '{"positions": [{"symbol": <string>, "close": <bool>, '
        '"reasoning": <string, one or two sentences>}, ...]}'
    )


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_response(text: str, valid_symbols: set[str]) -> dict[str, dict]:
    """
    Raises on anything that doesn't validate -- the caller retries or falls
    back. A symbol in the response that isn't in valid_symbols is silently
    dropped rather than trusted, same convention as models/llm_advisor.py.
    """
    data = json.loads(text)
    by_symbol: dict[str, dict] = {}
    for p in data["positions"]:
        symbol = str(p["symbol"])
        if symbol not in valid_symbols:
            continue
        by_symbol[symbol] = {"close": bool(p["close"]), "reasoning": str(p["reasoning"])}
    return by_symbol


def _call_claude(client: Anthropic, payload: dict) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=[{"type": "text", "text": _build_system_prompt(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    # Sonnet 5 runs adaptive extended thinking by default, so
    # resp.content[0] is very often a ThinkingBlock (no .text attribute)
    # ahead of the TextBlock -- see models/llm_advisor.py's identical fix
    # for why this must not assume position 0.
    for block in resp.content:
        if block.type == "text":
            return block.text
    raise ValueError("Claude's response contained no text block (thinking/other content only).")


def get_second_opinions(positions: list[dict]) -> dict[str, dict] | None:
    """
    `positions`: one dict per flagged position -- symbol, side, reasons
    (list of the tripped signals' detail strings), pnl_pct, recent_headlines.

    Returns {symbol: {"close": bool, "reasoning": str}} on success, or None
    (never raises) when the API key is unset, `positions` is empty, the
    call fails, or the response never validates after retries. Callers
    MUST treat None as "close every one of these" -- the rule-based
    signal's own call -- never as "keep them all open": an LLM outage must
    never silently suppress a real risk signal that already tripped.
    """
    if not settings.anthropic_api_key or not positions:
        return None

    valid_symbols = {p["symbol"] for p in positions}
    payload = {"positions": positions}
    client = Anthropic(api_key=settings.anthropic_api_key)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            text = _strip_code_fence(_call_claude(client, payload))
            parsed = _parse_response(text, valid_symbols)
            if not parsed:
                raise ValueError("Claude returned no valid positions from the given set")
            return parsed
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
            last_exc = exc
            logger.warning(
                "Could not parse Claude's contradiction second-opinion response, attempt %d/%d.",
                attempt, _MAX_ATTEMPTS, exc_info=True,
            )
        except Exception:
            # A transient network/rate-limit error is already retried
            # inside the Anthropic client itself -- anything that still
            # reaches here is a harder failure (auth, an outage outlasting
            # the SDK's own retries) that re-asking the identical request
            # won't fix.
            logger.exception(
                "Claude contradiction second-opinion call failed -- falling back to closing every flagged position."
            )
            return None

    logger.warning(
        "Falling back to closing every flagged position after %d failed second-opinion attempts: %s",
        _MAX_ATTEMPTS, last_exc,
    )
    return None
