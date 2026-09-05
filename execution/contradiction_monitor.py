"""
Between weekly screen-and-trade cycles, a held position can go stale: fresh
news can break against the direction we entered, or short-term price action
can reverse hard enough to contradict the original thesis, days before the
next scheduled screen would otherwise notice. This module checks every
currently held position against both signals — plus its own take-profit/
stop-loss (see check_stop_or_target below) — and closes out any position
that's now fighting the evidence it was opened on, or that has simply
finished: the swing trade the position was opened for is sized to that
stock's own volatility (execution/exit_levels.py) and can resolve in a few
days for a volatile name or take a week or more for a calm one — checking
it hourly, same clock as the two contradiction signals, means a position
closes when it resolves rather than sitting past its own target or stop
until the next weekly checkpoint just because the calendar hadn't come
around yet.

Close-only for the contradiction check itself -- reversing requires a fresh
conviction call, which is exactly what a screen does. But after a close
frees up capital, this module immediately re-screens (same 80% confidence
bar, same selection logic as the weekly cycle) to redeploy it right away
rather than leaving it in cash until next week -- see _attempt_reactivation.
The strategy doesn't change, it just doesn't have to wait for Monday.

Safety boundary, same as trading_loop.py: only ever calls get_broker()
without confirm_live=True, so it can never fire a live order on the MASTER
account no matter what TRADING_MODE is set to.

That guarantee is scoped to the master account only. This file also calls
check_all_clients_risk(...) and replicate_to_clients(...) below, both of
which submit real orders on each client's OWN broker account whenever
CLIENT_TRADING_ENABLED is True -- a separate switch from TRADING_MODE, and
real money regardless of whether the master account above is trading paper
or live. See execution/client_risk_controls.py and execution/client_fanout.py.

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

from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from data.ingest.news import ingest_news
from data.ingest.universe import load_active_universe
from execution import hold_rules
from execution.approval_gate import ProposedTrade, advisory_lock, request_approval, send_followup
from execution.broker import get_broker
from execution.client_fanout import replicate_to_clients
from execution.client_risk_controls import check_all_clients_risk
from execution.exit_levels import ExitLevels
from execution.trading_loop import (
    _allocation_confirmation,
    _apply_allocation,
    _correlation_matrix,
    _flatten_and_alert,
    _run_breaker_check,
    current_pnl_by_symbol,
)
from features.qualitative.sentiment import backfill_unscored_news
from features.quant.momentum import rolling_return
from models.screener import run_screen
from monitoring import reasoning
from monitoring.alerts import configure_file_logging, send_slack_alert
from risk.sizing import allocate_by_conviction

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
# The trigger size is config, not a constant — see
# settings.contradiction_momentum_pct for why it is where it is (an
# emergency brake, deliberately quiet) and what would justify moving it.

# Don't bother re-screening for a sliver of freed capital too small to matter.
_MIN_REACTIVATION_FRACTION = 0.05

# One fixed key for "an hourly contradiction-check pass is running" — a slow
# news backfill call can push one run past the next hourly trigger, and two
# overlapping passes would double-submit closes/reactivations against the
# same positions. Distinct from approval_gate.APPROVAL_LOCK_KEY (that one
# guards Telegram's single-consumer getUpdates poll, a different resource).
_CONTRADICTION_LOCK_KEY = 903218


@dataclasses.dataclass
class ContradictionResult:
    symbol: str
    side: str
    closed: bool
    reasons: list[dict]


def _recent_sentiment(engine, symbol: str) -> tuple[float | None, int]:
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=_SENTIMENT_LOOKBACK_HOURS)
    df = pd.read_sql(
        "SELECT sentiment FROM news_events WHERE symbol = %(symbol)s AND ts >= %(since)s AND sentiment IS NOT NULL "
        # A headline a news vendor mistagged onto this symbol (see
        # data/schema/010_news_sentiment_relevance.sql) must not be able to
        # trigger closing a real position -- IS NOT FALSE is the NULL-safe
        # form, so unscored-for-relevance and pre-migration rows still count.
        "AND sentiment_relevant IS NOT FALSE",
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


def _check_position(
    engine,
    symbol: str,
    qty: float,
    pnl_pct: float | None = None,
    levels: ExitLevels | None = None,
) -> ContradictionResult:
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
    if momentum is not None and sign * momentum <= -settings.contradiction_momentum_pct:
        reasons.append(
            {
                "signal": "price_momentum",
                "value": momentum,
                "detail": f"{_MOMENTUM_WINDOW_DAYS}d return {momentum:.2%} contradicts {side} position",
            }
        )

    # This is not a second contradiction signal, it's the swing trade
    # actually finishing: a stock's own target/stop, sized to its own
    # volatility (execution/exit_levels.py), can resolve in a few days for a
    # volatile name or take longer for a calm one. Checking it on the same
    # hourly clock as the two signals above means a trade closes when IT
    # resolves rather than sitting past its own target or stop until next
    # Monday's weekly cycle just because the calendar hadn't come around.
    hit = hold_rules.check_stop_or_target(pnl_pct, levels, settings.hold_stop_loss_pct, settings.hold_take_profit_pct)
    if hit:
        reasons.append({"signal": hit.kind, "value": pnl_pct, "detail": hit.message})

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


def _attempt_reactivation(broker, engine, request_fn=None, excluded_symbols=None) -> None:
    """
    After a confirmed mid-cycle exit, production calls this with the symbols
    that just closed. That path delegates to the whole-book optimizer so
    surviving positions can grow, shrink, or be displaced rather than an
    empty slot merely being refilled.

    Calls without `excluded_symbols` retain the historical slice-only path.
    That keeps direct tooling/backward-compatible callers stable while the
    real post-exit path gets the stronger whole-book semantics.
    """
    if excluded_symbols is not None:
        from execution.full_book_rebalance import rebalance_after_exit

        def _log_displaced_close(symbol: str, approval_status: str | None) -> None:
            result = ContradictionResult(
                symbol=symbol,
                side="long",
                closed=True,
                reasons=[
                    {
                        "signal": "portfolio_rebalance",
                        "value": None,
                        "detail": "position displaced by a higher-conviction full-book rebalance after another exit",
                    }
                ],
            )
            _log_closure(result, broker.mode, 0.0, approval_status=approval_status or "approved")

        rebalance_after_exit(
            broker,
            engine,
            excluded_symbols=excluded_symbols,
            request_fn=request_fn,
            log_candidate=_log_reactivation,
            log_displaced_close=_log_displaced_close,
        )
        return

    freed_fraction = _freed_capital_fraction(broker, engine)
    if freed_fraction < _MIN_REACTIVATION_FRACTION:
        return

    held_symbols = {s for s, q in broker.get_positions().items() if q != 0}

    max_positions_override = None
    if settings.strategy_mode == "concentrated":
        open_slots = max(0, settings.max_concentrated_positions - len(held_symbols))
        if open_slots <= 0:
            logger.info(
                "Book already holds %d of %d target position(s) — freed capital stays in cash rather than "
                "adding a name beyond the concentrated cap.",
                len(held_symbols), settings.max_concentrated_positions,
            )
            return
        max_positions_override = open_slots

    candidate_pool = [s for s in load_active_universe() if s not in held_symbols]
    if not candidate_pool:
        return

    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    try:
        candidates = run_screen(
            "v3", candidate_pool, is_shortable_fn=is_shortable_fn,
            total_deploy_pct=freed_fraction, max_positions_override=max_positions_override,
        )
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
    # Approve-first, same as the weekly cycle: proposals carry the "why"
    # but no size — the freed capital is allocated across whatever subset
    # the human approves, and the sizes are confirmed in a follow-up.
    proposals = [
        ProposedTrade(
            index=0, symbol=c.symbol, action="open", side=c.side,
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
    candidates = [c for c in candidates if c.symbol in approved_symbols]
    if not candidates:
        return

    # Approve first, THEN size: the freed capital goes to the approved
    # subset only, weighted by conviction, under the same caps as the
    # weekly cycle. A rejected pick's share is redistributed to the
    # approved ones (up to the caps) instead of idling in cash.
    allocation = allocate_by_conviction(
        {c.symbol: (c.conviction_score if c.side == "long" else -c.conviction_score) for c in candidates},
        max_position_pct=settings.max_single_position_pct,
        max_short_position_pct=settings.max_short_position_pct,
        max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
        correlation_matrix=_correlation_matrix(engine, [c.symbol for c in candidates]),
        target_allocation=freed_fraction,
    )
    _apply_allocation(allocation, candidates)
    if not allocation.reached_target and allocation.reason:
        logger.warning("%s", allocation.reason)
    send_followup(_allocation_confirmation(allocation, candidates))

    # Prices read after the gate, so shares are sized off post-wait quotes.
    portfolio_value = broker.get_portfolio_value()
    prices = _latest_prices(engine, [c.symbol for c in candidates])

    reopened: list[str] = []
    reopened_target_pct: dict[str, float] = {}
    for c in candidates:
        price = prices.get(c.symbol)
        if not price:
            logger.warning("No price for reactivation candidate %s — skipping.", c.symbol)
            continue
        if abs(c.target_position_pct or 0.0) < 1e-9:
            logger.warning("%s was approved but the caps left it no allocation — no order.", c.symbol)
            continue
        target_shares = (c.target_position_pct * portfolio_value) / price
        logger.warning("Reactivating freed capital: %s %s, %.1f%% of portfolio.", c.symbol, c.side, abs(c.target_position_pct) * 100)
        try:
            broker.submit_target_position(c.symbol, target_shares)
        except Exception:
            logger.exception("Failed to open reactivation position %s.", c.symbol)
            continue

        executed = broker.get_positions().get(c.symbol, 0.0)
        _log_reactivation(c, executed, broker.mode, target_shares, approval_status=status_by_symbol.get(c.symbol) or "approved")
        reopened.append(f"{c.symbol} {target_shares:+,.2f} sh")
        reopened_target_pct[c.symbol] = c.target_position_pct or 0.0

    if reopened_target_pct:
        # Same reactivation onto every client's own account, sized to
        # their own capital (execution/client_fanout.py) — never allowed
        # to affect the master account's own outcome above.
        try:
            replicate_to_clients(reopened_target_pct, prices, engine)
        except Exception:
            logger.exception("Client fan-out failed for this reactivation — the master account's own trades above are unaffected.")

    if reopened:
        message = f"♻️ Freed capital redeployed: {', '.join(reopened)}"
        send_slack_alert(message, severity="info")
        send_followup(message)  # Telegram's post-trade update — see approval_gate module docstring


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

    Runs under advisory_lock(_CONTRADICTION_LOCK_KEY): this fires hourly, and
    a slow pass (e.g. the news backfill call below) can run long enough to
    still be going when the next hourly trigger fires. A second overlapping
    call finds the lock held and no-ops (logs and returns []) rather than
    double-processing the same positions.
    """
    with advisory_lock(_CONTRADICTION_LOCK_KEY) as got_lock:
        if not got_lock:
            logger.warning(
                "Another contradiction-check pass is already running (advisory lock held) — "
                "skipping this run rather than double-processing. It will run again next hour."
            )
            return []
        return _run_contradiction_check(request_fn)


def _run_contradiction_check(request_fn=None) -> list[ContradictionResult]:
    """The actual check, run under run_contradiction_check's advisory lock — see its docstring."""
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()

    if hasattr(broker, "client") and not broker.client.get_clock().is_open:
        logger.info("Market is closed — skipping this check (runs hourly during market hours).")
        return []

    # Master-account circuit breakers (risk/circuit_breakers.py) — the same
    # checks the weekly cycle runs before/after trading (trading_loop.py's
    # _run_breaker_check). This hourly monitor is documented everywhere as
    # the emergency brake between weekly cycles, but until now it only ever
    # checked the two contradiction signals below, never the master
    # account's own risk limits — a breach between Mondays went unnoticed
    # until the next weekly cycle. A trip here flattens the master account
    # exactly like a weekly-cycle trip does (same alert, same flatten call,
    # see _flatten_and_alert) and skips the rest of this pass.
    breaker_triggers = _run_breaker_check(broker, engine)
    if breaker_triggers:
        reasons = "; ".join(r.reason for r in breaker_triggers)
        _flatten_and_alert(broker, reasons)
        return []

    # Client self-service risk controls (max-drawdown auto-close,
    # profit-target auto-secure — see execution/client_risk_controls.py)
    # run on this same hourly, market-hours clock, independent of whether
    # the MASTER account itself holds anything below. One client's failed
    # check never raises past this call.
    try:
        check_all_clients_risk(engine)
    except Exception:
        logger.exception("Client risk-control check failed this pass — will retry next hour.")

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

    # Computed once, up front, so every position's stop/target check (inside
    # _check_position below) reads the same P&L and the same levels it was
    # actually approved with, rather than each position pulling its own
    # separately. Also reused below for the approval message's P&L display.
    pnl_by_symbol = current_pnl_by_symbol(broker)
    try:
        levels_by_symbol = hold_rules.load_exit_levels(engine)
    except Exception:
        # Same degrade-gracefully posture as the dashboard's positions
        # endpoint: a position with no recorded levels still gets checked,
        # just against the global HOLD_STOP_LOSS_PCT/HOLD_TAKE_PROFIT_PCT
        # fallback (see check_stop_or_target) instead of skipping the
        # stop/target check for everyone because one table read failed.
        logger.exception("Could not load per-position exit levels — falling back to the global stop/target settings.")
        levels_by_symbol = {}

    # Detect first, act later: every position is checked, and everything
    # that tripped goes to the human as ONE batch instead of a message per
    # position.
    results: list[ContradictionResult] = []
    flagged: list[ContradictionResult] = []
    for symbol, qty in positions.items():
        pnl_pct = pnl_by_symbol.get(symbol, (None, None))[0]
        result = _check_position(engine, symbol, qty, pnl_pct=pnl_pct, levels=levels_by_symbol.get(symbol))
        results.append(result)
        if result.closed:
            flagged.append(result)
            detail = "; ".join(r["detail"] for r in result.reasons)
            logger.warning("Contradiction detected for %s (%s). %s", symbol, result.side, detail)

    if not flagged:
        return results

    gate = request_fn if request_fn is not None else request_approval
    # Each close proposal carries the contradiction/exit evidence and the
    # position's current P&L, so the phone message says WHY the system
    # wants out and what the position stands at — not just "close X".
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

    # Collected rather than announced one at a time: the human already saw
    # the proposal message naming each of these, so what they need
    # afterwards is one line saying how it ended.
    closed: list[str] = []
    kept: list[str] = []
    closed_any = False
    for result in flagged:
        status = status_by_symbol.get(result.symbol)
        if result.symbol not in approved_symbols:
            result.closed = False  # the record must not claim a close that didn't happen
            logger.warning("Close of %s not approved (%s) — position stays open.", result.symbol, status or "rejected")
            _log_rejected_closure(result, broker.mode, status or "rejected")
            kept.append(result.symbol)
            continue

        try:
            broker.submit_target_position(result.symbol, 0.0)
        except Exception:
            logger.exception("Failed to close contradicted position %s.", result.symbol)
            continue

        closed_any = True
        executed = broker.get_positions().get(result.symbol, 0.0)
        _log_closure(result, broker.mode, executed, approval_status=status or "approved")
        closed.append(result.symbol)

        # Every client holding this symbol exits it too, on their own
        # account (execution/client_fanout.py) — never allowed to affect
        # the master account's own close above, which already happened.
        try:
            replicate_to_clients({result.symbol: 0.0}, {}, engine)
        except Exception:
            logger.exception("Client fan-out failed closing %s — the master account's own close above is unaffected.", result.symbol)

    parts = []
    if closed:
        parts.append(f"🔻 Closed mid-week: {', '.join(closed)}")
    if kept:
        parts.append(f"🤝 Flagged but kept open on your call: {', '.join(kept)}")
    if parts:
        outcome_message = "Contradiction check done.\n" + "\n".join(parts)
        send_slack_alert(outcome_message, severity="warning")
        send_followup(outcome_message)  # Telegram's post-trade update — see approval_gate module docstring

    if closed_any:
        _attempt_reactivation(
            broker,
            engine,
            request_fn=request_fn,
            excluded_symbols=set(closed),
        )

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_file_logging()  # logs survive the console closing
    results = run_contradiction_check()
    closed = [r for r in results if r.closed]
    print(f"Checked {len(results)} position(s), closed {len(closed)}.")


if __name__ == "__main__":
    main()