"""
Between weekly screen-and-trade cycles, a held position can go stale: fresh
news can break against the direction we entered, or short-term price action
can reverse hard enough to contradict the original thesis, days before the
next scheduled screen would otherwise notice. This module checks every
currently held position against both signals and closes out any position
that's now fighting the evidence it was opened on.

Close-only for the contradiction check itself -- reversing requires a fresh
conviction call, which is exactly what a screen does. But after a close
frees up capital, this module immediately re-screens (same 80% confidence
bar, same selection logic as the weekly cycle) to redeploy it right away
rather than leaving it in cash until next week -- see _attempt_reactivation.
The strategy doesn't change, it just doesn't have to wait for Monday.

Safety boundary, same as trading_loop.py: only ever calls get_broker()
without confirm_live=True.

Usage:
    python -m execution.contradiction_monitor
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from data.ingest.db import get_engine, symbol_in_clause
from data.ingest.news import ingest_news
from data.ingest.universe import load_active_universe
from execution.approval_gate import ProposedTrade, request_approval
from execution.broker import get_broker
from execution.trading_loop import current_pnl_by_symbol
from features.qualitative.sentiment import backfill_unscored_news
from features.quant.momentum import rolling_return
from models.screener import run_screen
from monitoring import reasoning
from monitoring.alerts import configure_file_logging, send_slack_alert

logger = logging.getLogger(__name__)

_FEATURE_SET_ID = "contradiction_monitor"
_MODEL_VERSION = "rule_based_v1"
_REACTIVATION_MODEL_VERSION = "ensemble_v1"

# News lookback for the sentiment read — wider than the check interval so a
# position isn't judged on a single stale article between checks.
_SENTIMENT_LOOKBACK_HOURS = 24
_MIN_NEWS_COUNT = 2
_SENTIMENT_CONTRADICTION_THRESHOLD = 0.4  # mean sentiment must be this strongly opposite to trigger

_MOMENTUM_WINDOW_DAYS = 5
_MOMENTUM_CONTRADICTION_THRESHOLD = 0.04  # 4% move against the position over the window

# Don't bother re-screening for a sliver of freed capital too small to matter.
_MIN_REACTIVATION_FRACTION = 0.05


@dataclasses.dataclass
class ContradictionResult:
    symbol: str
    side: str
    closed: bool
    reasons: list[dict]


def _recent_sentiment(engine, symbol: str) -> tuple[float | None, int]:
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=_SENTIMENT_LOOKBACK_HOURS)
    df = pd.read_sql(
        "SELECT sentiment FROM news_events WHERE symbol = %(symbol)s AND ts >= %(since)s AND sentiment IS NOT NULL",
        engine,
        params={"symbol": symbol, "since": since},
    )
    if df.empty:
        return None, 0
    return float(df["sentiment"].mean()), len(df)


def _recent_momentum(engine, symbol: str) -> float | None:
    df = pd.read_sql(
        "SELECT ts, close FROM prices WHERE symbol = %(symbol)s ORDER BY ts DESC LIMIT %(limit)s",
        engine,
        params={"symbol": symbol, "limit": _MOMENTUM_WINDOW_DAYS + 1},
    )
    if len(df) < _MOMENTUM_WINDOW_DAYS + 1:
        return None
    df = df.sort_values("ts")
    ret = rolling_return(df["close"], _MOMENTUM_WINDOW_DAYS).iloc[-1]
    return None if pd.isna(ret) else float(ret)


def _check_position(engine, symbol: str, qty: float) -> ContradictionResult:
    side = "long" if qty > 0 else "short"
    sign = 1.0 if qty > 0 else -1.0
    reasons: list[dict] = []

    sentiment, news_count = _recent_sentiment(engine, symbol)
    if sentiment is not None and news_count >= _MIN_NEWS_COUNT:
        # Contradiction: sentiment points opposite the held side, strongly enough.
        if sign * sentiment <= -_SENTIMENT_CONTRADICTION_THRESHOLD:
            reasons.append(
                {
                    "signal": "news_sentiment",
                    "value": sentiment,
                    "news_count": news_count,
                    "detail": f"mean sentiment {sentiment:.2f} over last {_SENTIMENT_LOOKBACK_HOURS}h contradicts {side} position",
                }
            )

    momentum = _recent_momentum(engine, symbol)
    if momentum is not None and sign * momentum <= -_MOMENTUM_CONTRADICTION_THRESHOLD:
        reasons.append(
            {
                "signal": "price_momentum",
                "value": momentum,
                "detail": f"{_MOMENTUM_WINDOW_DAYS}d return {momentum:.2%} contradicts {side} position",
            }
        )

    return ContradictionResult(symbol=symbol, side=side, closed=bool(reasons), reasons=reasons)


def _log_closure(
    result: ContradictionResult, mode: str, executed_position: float | None, approval_status: str | None = "approved"
) -> None:
    # Phases 1/3/6 don't apply here (this isn't the weekly screen — no fresh
    # risk gate, forecast, or fill reconciliation to report); phases 2/4/5/7
    # cover what actually happened: what contradicted, why it wasn't part of
    # normal selection, what order got sent, and what happens next.
    phase2 = reasoning.phase_contradiction(result.reasons)
    phase4 = {
        "phase": 4,
        "title": "Candidate Selection & Sizing",
        "summary": f"{result.symbol} closed outside the weekly screen — no new position opened.",
        "lines": [
            "This wasn't a weekly screen decision — the hourly contradiction check triggered mid-week.",
            "Re-entry (if any) is left to the next weekly screen, not decided here.",
        ],
    }
    phase5 = reasoning.phase_execution(result.symbol, "closed", None, "market")
    phase7 = reasoning.phase_ongoing_monitoring(closed=True)
    full_reasoning = reasoning.combine_phases(phase2, phase4, phase5, phase7)

    row = {
        "ts": dt.datetime.now(tz=dt.UTC),
        "symbol": result.symbol,
        "feature_set_id": _FEATURE_SET_ID,
        "model_version": _MODEL_VERSION,
        "forecast": None,
        "regime": None,
        "target_position": 0.0,
        "executed_position": executed_position,
        "mode": mode,
        "reasoning": json.dumps(full_reasoning),
        "approval_status": approval_status,
    }
    pd.DataFrame([row]).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def _log_rejected_closure(result: ContradictionResult, mode: str, approval_status: str) -> None:
    """
    The variant row for a contradiction the human declined (or ignored) —
    the record must show the system FLAGGED this position even though the
    position is still open.
    """
    phase2 = reasoning.phase_contradiction(result.reasons)
    phase4 = {
        "phase": 4,
        "title": "Candidate Selection & Sizing",
        "summary": f"{result.symbol} flagged mid-week, but the close was not approved — position kept.",
        "lines": [
            "The hourly contradiction check proposed closing this position mid-week.",
            "The close was not approved, so the position stays open on the human's call.",
        ],
    }
    phase5 = reasoning.phase_execution_rejected(result.symbol, approval_status)
    # closed=False: the position is still open and still under hourly watch.
    phase7 = reasoning.phase_ongoing_monitoring(closed=False)
    full_reasoning = reasoning.combine_phases(phase2, phase4, phase5, phase7)

    row = {
        "ts": dt.datetime.now(tz=dt.UTC),
        "symbol": result.symbol,
        "feature_set_id": _FEATURE_SET_ID,
        "model_version": _MODEL_VERSION,
        "forecast": None,
        "regime": None,
        "target_position": 0.0,
        "executed_position": 0.0,
        "mode": mode,
        "reasoning": json.dumps(full_reasoning),
        "approval_status": approval_status,
    }
    pd.DataFrame([row]).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def _latest_prices(engine, symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    symbol_list = symbol_in_clause(symbols)
    df = pd.read_sql(
        "SELECT DISTINCT ON (symbol) symbol, close FROM prices "  # noqa: S608 — symbols validated via symbol_in_clause
        f"WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC",
        engine,
    )
    return dict(zip(df["symbol"], df["close"], strict=False))


def _freed_capital_fraction(broker, engine) -> float:
    """Fraction of portfolio_value currently sitting idle (not allocated to any open position)."""
    positions = {s: q for s, q in broker.get_positions().items() if q != 0}
    portfolio_value = broker.get_portfolio_value()
    if portfolio_value <= 0:
        return 0.0
    if not positions:
        return 1.0
    prices = _latest_prices(engine, list(positions.keys()))
    held_value = sum(abs(qty) * prices[sym] for sym, qty in positions.items() if sym in prices)
    return max(0.0, 1.0 - held_value / portfolio_value)


def _log_rejected_reactivation(candidate, mode: str, approval_status: str) -> None:
    """A reactivation pick the human turned down — recorded, not opened."""
    phases = list(candidate.reasoning or [])
    phase5 = reasoning.phase_execution_rejected(candidate.symbol, approval_status)
    phase7 = reasoning.phase_ongoing_monitoring(closed=True)
    full_reasoning = reasoning.combine_phases(*phases, phase5, phase7)

    row = {
        "ts": dt.datetime.now(tz=dt.UTC),
        "symbol": candidate.symbol,
        "feature_set_id": _FEATURE_SET_ID,
        "model_version": _REACTIVATION_MODEL_VERSION,
        "forecast": candidate.predicted_return,
        "regime": None,
        "target_position": candidate.target_position_pct,
        "executed_position": 0.0,
        "mode": mode,
        "reasoning": json.dumps(full_reasoning),
        "direction_agreement": candidate.direction_agreement,
        "approval_status": approval_status,
    }
    pd.DataFrame([row]).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def _log_reactivation(
    candidate, executed_position: float | None, mode: str, target_shares: float, approval_status: str | None = "approved"
) -> None:
    # Phases 2/3/4 already come fully built from run_screen (same selection
    # logic as the weekly cycle) -- just note on phase 4 that this fired
    # mid-week rather than at the normal weekly screen, then add phases 5/7.
    phases = list(candidate.reasoning or [])
    phases = [
        {**p, "lines": [*p["lines"], "Picked mid-week to redeploy capital freed by a contradiction close, using the same confidence bar as the weekly screen."]}
        if p["phase"] == 4
        else p
        for p in phases
    ]
    phase5 = reasoning.phase_execution(candidate.symbol, "opened", target_shares, "market")
    phase7 = reasoning.phase_ongoing_monitoring(closed=False)
    full_reasoning = reasoning.combine_phases(*phases, phase5, phase7)

    row = {
        "ts": dt.datetime.now(tz=dt.UTC),
        "symbol": candidate.symbol,
        "feature_set_id": _FEATURE_SET_ID,
        "model_version": _REACTIVATION_MODEL_VERSION,
        "forecast": candidate.predicted_return,
        "regime": None,
        "target_position": candidate.target_position_pct,
        "executed_position": executed_position,
        "mode": mode,
        "reasoning": json.dumps(full_reasoning),
        "direction_agreement": candidate.direction_agreement,
        "approval_status": approval_status,
    }
    pd.DataFrame([row]).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def _attempt_reactivation(broker, engine, request_fn=None) -> None:
    """
    After a contradiction close, checks whether meaningful capital is now
    sitting idle and, if so, immediately re-screens for a new candidate to
    redeploy it -- same 80% confidence bar and top-2-concentrated selection
    logic as the weekly cycle (models.screener.run_screen), just scoped to
    the freed fraction of capital instead of the whole book, and restricted
    to symbols not currently held. No-ops if the freed slice is too small to
    bother with, or nothing confident turns up.
    """
    freed_fraction = _freed_capital_fraction(broker, engine)
    if freed_fraction < _MIN_REACTIVATION_FRACTION:
        return

    held_symbols = {s for s, q in broker.get_positions().items() if q != 0}
    candidate_pool = [s for s in load_active_universe() if s not in held_symbols]
    if not candidate_pool:
        return

    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    try:
        candidates = run_screen("v3", candidate_pool, is_shortable_fn=is_shortable_fn, total_deploy_pct=freed_fraction)
    except Exception:
        logger.exception("Reactivation screen failed — leaving freed capital in cash until the next check.")
        return

    if not candidates:
        logger.info("No confident candidate to redeploy %.1f%% freed capital — staying in cash for now.", freed_fraction * 100)
        return

    # Reactivation opens go through the same human gate as everything else —
    # a separate message from the closes, since it is a separate question
    # ("re-deploy the freed capital into X?").
    gate = request_fn if request_fn is not None else request_approval
    proposals = [
        ProposedTrade(
            index=0, symbol=c.symbol, action="open", side=c.side,
            target_position_pct=c.target_position_pct,
            predicted_return=c.predicted_return, reason="reactivation",
            reasoning=c.reasoning,
        )
        for c in candidates
    ]
    outcome = gate(proposals, context="contradiction monitor — reactivation")
    status_by_symbol = {p.symbol: outcome.statuses.get(p.index) for p in proposals}
    approved_symbols = {p.symbol for p in outcome.approved}

    for c in candidates:
        if c.symbol not in approved_symbols:
            status = status_by_symbol.get(c.symbol) or "rejected"
            logger.warning("Reactivation of %s not approved (%s) — freed capital stays in cash.", c.symbol, status)
            _log_rejected_reactivation(c, broker.mode, status)
            send_slack_alert(
                f"Reactivation proposal for {c.symbol} was not approved ({status}) — capital stays in cash.",
                severity="info",
            )
    candidates = [c for c in candidates if c.symbol in approved_symbols]
    if not candidates:
        return

    # Prices read after the gate, so shares are sized off post-wait quotes.
    portfolio_value = broker.get_portfolio_value()
    prices = _latest_prices(engine, [c.symbol for c in candidates])

    for c in candidates:
        price = prices.get(c.symbol)
        if not price:
            logger.warning("No price for reactivation candidate %s — skipping.", c.symbol)
            continue
        target_shares = (c.target_position_pct * portfolio_value) / price
        logger.warning("Reactivating freed capital: %s %s, %.1f%% of portfolio.", c.symbol, c.side, abs(c.target_position_pct) * 100)
        send_slack_alert(
            f"Phase 4 — Candidate Selection & Sizing: Mid-week reactivation — {c.symbol} ({c.side}), "
            f"{abs(c.target_position_pct):.1%} of capital.",
            severity="info",
        )
        try:
            broker.submit_target_position(c.symbol, target_shares)
        except Exception:
            logger.exception("Failed to open reactivation position %s.", c.symbol)
            continue

        executed = broker.get_positions().get(c.symbol, 0.0)
        _log_reactivation(c, executed, broker.mode, target_shares, approval_status=status_by_symbol.get(c.symbol) or "approved")
        send_slack_alert(f"Phase 5 — Execution: {c.symbol} opened mid-week (reactivation) — {target_shares:+.4g} shares.", severity="info")


def run_contradiction_check(request_fn=None) -> list[ContradictionResult]:
    """
    Checks every currently held position for news/momentum contradicting the
    side it's held on, then asks a human (one batched proposal message)
    before closing anything that tripped a threshold. Approved closes are
    submitted and logged; rejected ones are logged as flagged-but-kept and
    alerted. If anything actually closed, immediately attempts to redeploy
    the freed capital (_attempt_reactivation, itself gated the same way)
    rather than leaving it idle until next week. No-ops cleanly if nothing
    is held or nothing contradicts.
    """
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()

    if hasattr(broker, "client") and not broker.client.get_clock().is_open:
        logger.info("Market is closed — skipping this check (runs hourly during market hours).")
        return []

    positions = {s: q for s, q in broker.get_positions().items() if q != 0}
    if not positions:
        logger.info("No open positions — nothing to check.")
        return []

    symbols = list(positions.keys())
    try:
        ingest_news(symbols, since_hours=_SENTIMENT_LOOKBACK_HOURS)
        backfill_unscored_news()
    except Exception:
        logger.exception("News refresh failed — checking against whatever sentiment is already in the DB.")

    # Detect first, act later: every position is checked, and everything
    # that tripped goes to the human as ONE batch instead of a message per
    # position.
    results: list[ContradictionResult] = []
    flagged: list[ContradictionResult] = []
    for symbol, qty in positions.items():
        result = _check_position(engine, symbol, qty)
        results.append(result)
        if result.closed:
            flagged.append(result)
            detail = "; ".join(r["detail"] for r in result.reasons)
            logger.warning("Contradiction detected for %s (%s). %s", symbol, result.side, detail)
            send_slack_alert(f"Phase 2 — Market Regime & Signals: {symbol} ({result.side}) — {detail}", severity="warning")

    if not flagged:
        return results

    gate = request_fn if request_fn is not None else request_approval
    # Each close proposal carries the contradiction evidence and the
    # position's current P&L, so the phone message says WHY the system
    # wants out and what the position stands at — not just "close X".
    pnl_by_symbol = current_pnl_by_symbol(broker)
    proposals = [
        ProposedTrade(
            index=0, symbol=r.symbol, action="close", side=r.side,
            target_position_pct=0.0, reason="contradiction",
            reasoning=[reasoning.phase_contradiction(r.reasons)],
            current_pnl_pct=pnl_by_symbol.get(r.symbol, (None, None))[0],
            current_pnl_usd=pnl_by_symbol.get(r.symbol, (None, None))[1],
        )
        for r in flagged
    ]
    outcome = gate(proposals, context="contradiction monitor")
    status_by_symbol = {p.symbol: outcome.statuses.get(p.index) for p in proposals}
    approved_symbols = {p.symbol for p in outcome.approved}

    closed_any = False
    for result in flagged:
        status = status_by_symbol.get(result.symbol)
        if result.symbol not in approved_symbols:
            result.closed = False  # the record must not claim a close that didn't happen
            logger.warning("Close of %s not approved (%s) — position stays open.", result.symbol, status or "rejected")
            _log_rejected_closure(result, broker.mode, status or "rejected")
            send_slack_alert(
                f"{result.symbol} was flagged as contradicted but NOT closed ({status or 'rejected'}) — "
                "position stays open on the human's call.",
                severity="warning",
            )
            continue

        try:
            broker.submit_target_position(result.symbol, 0.0)
        except Exception:
            logger.exception("Failed to close contradicted position %s.", result.symbol)
            continue

        closed_any = True
        executed = broker.get_positions().get(result.symbol, 0.0)
        _log_closure(result, broker.mode, executed, approval_status=status or "approved")
        send_slack_alert(f"Phase 5 — Execution: {result.symbol} closed mid-week (contradiction).", severity="warning")

    if closed_any:
        _attempt_reactivation(broker, engine, request_fn=request_fn)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_file_logging()  # logs survive the console closing
    results = run_contradiction_check()
    closed = [r for r in results if r.closed]
    print(f"Checked {len(results)} position(s), closed {len(closed)}.")


if __name__ == "__main__":
    main()
