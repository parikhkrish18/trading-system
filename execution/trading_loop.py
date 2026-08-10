"""
The step models/screener.py explicitly doesn't do: take its shortlist and
actually place (paper) orders, then reconcile, record equity, run circuit
breakers, and alert. This is the file that turns "the pipeline produces
ranked candidates" into "positions actually move at a broker."

Safety boundary, enforced by construction, not by config: this only ever
calls get_broker() without confirm_live=True, so it can never fire a live
order no matter what TRADING_MODE is set to. Going live requires a human to
deliberately change this file, not flip an environment variable.

Usage:
    python -m execution.trading_loop --feature-set-id v3 --universe --dry-run
    python -m execution.trading_loop --feature-set-id v3 --symbols AAPL,MSFT,TSLA
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import time

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause
from data.ingest.universe import resolve_symbols
from execution.approval_gate import ProposedTrade, request_approval
from execution.broker import get_broker
from execution.reconciliation import reconcile_positions, summarize
from features.quant.momentum import adx
from models.regime.trend_chop_classifier import CHOP, RuleBasedRegime
from models.screener import build_correlation_matrix, run_screen
from monitoring import reasoning
from monitoring.alerts import alert_circuit_breaker, configure_file_logging, send_slack_alert
from monitoring.breaker_state import check_and_record_breakers
from monitoring.equity import load_equity_curve, record_equity_snapshot

logger = logging.getLogger(__name__)

_MODEL_VERSION = "ensemble_v1"


@dataclasses.dataclass
class CycleResult:
    status: str  # "flattened_pre_trade" | "flattened_post_trade" | "no_candidates" | "dry_run" | "traded"
    candidates_screened: int
    orders_placed: int
    reconciliation_summary: str | None
    portfolio_value: float


def _latest_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    symbol_list = symbol_in_clause(symbols)
    engine = get_engine()
    df = pd.read_sql(
        "SELECT DISTINCT ON (symbol) symbol, close FROM prices "  # noqa: S608 — symbols validated via symbol_in_clause
        f"WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC",
        engine,
    )
    return dict(zip(df["symbol"], df["close"], strict=False))


def _positions_value(positions: dict[str, float], prices: dict[str, float]) -> dict[str, float]:
    """shares -> dollar value, dropping any held symbol we don't have a price for."""
    return {sym: shares * prices[sym] for sym, shares in positions.items() if sym in prices and shares != 0}


def _market_regime(engine, market_proxy: str = "SPY") -> str:
    """
    One regime value for the whole cycle (matches models.screener.run_screen's
    single `regime` argument) — ADX on a broad market proxy, not per-symbol.
    """
    df = pd.read_sql(
        "SELECT ts, high, low, close FROM prices WHERE symbol = %(symbol)s ORDER BY ts",
        engine,
        params={"symbol": market_proxy},
    )
    if len(df) < 20:
        logger.warning("Not enough %s price history for a regime read — defaulting to CHOP (conservative).", market_proxy)
        return CHOP
    latest_adx = adx(df["high"], df["low"], df["close"]).iloc[-1]
    if pd.isna(latest_adx):
        return CHOP
    return RuleBasedRegime().predict(pd.Series([latest_adx])).item()


def _run_breaker_check(broker, engine) -> list:
    positions = broker.get_positions()
    prices = _latest_prices(list(positions.keys()))
    positions_by_value = _positions_value(positions, prices)
    portfolio_value = broker.get_portfolio_value()

    price_history = pd.read_sql(
        "SELECT symbol, ts, close FROM prices "  # noqa: S608 — symbols validated via symbol_in_clause
        f"WHERE symbol IN ({symbol_in_clause(positions_by_value)}) ORDER BY ts",
        engine,
    )
    correlation_matrix = build_correlation_matrix(price_history) if not price_history.empty else pd.DataFrame()
    equity_curve = load_equity_curve(mode="paper")["equity_value"].tolist()

    return check_and_record_breakers(
        equity_curve=equity_curve,
        positions_by_symbol=positions_by_value,
        portfolio_value=portfolio_value,
        correlation_matrix=correlation_matrix,
        max_drawdown_pct=settings.max_drawdown_pct,
        max_single_position_pct=settings.max_single_position_pct,
        max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
    )


def current_pnl_by_symbol(broker) -> dict[str, tuple[float | None, float | None]]:
    """
    {symbol: (unrealized P&L %, unrealized P&L $)} for every open position,
    for the approval message's close proposals. Best-effort: a broker
    without get_positions_detailed (or a failing call) yields {} and the
    proposals simply omit the P&L rather than blocking the gate.
    """
    if not hasattr(broker, "get_positions_detailed"):
        return {}
    try:
        return {
            p["symbol"]: (p.get("unrealized_plpc"), p.get("unrealized_pl"))
            for p in broker.get_positions_detailed()
        }
    except Exception:
        logger.warning("Could not fetch position P&L for the approval message — proposing without it.")
        return {}


def _flatten_and_alert(broker, reason: str) -> None:
    logger.critical("Circuit breaker triggered: %s — flattening all positions.", reason)
    broker.flatten_all()
    alert_circuit_breaker(reason)
    record_equity_snapshot(broker.get_portfolio_value(), mode=broker.mode)


def _send_phase_alert(phase: dict) -> None:
    send_slack_alert(f"Phase {phase['phase']} — {phase['title']}: {phase['summary']}", severity="info")


def _order_type(broker) -> str:
    if hasattr(broker, "client") and not broker.client.get_clock().is_open:
        return "limit (extended hours)"
    return "market"


def _log_decisions(
    candidates,
    closing_symbols: list[str],
    executed: dict[str, float],
    intended_shares: dict[str, float],
    feature_set_id: str,
    mode: str,
    regime: str,
    phase1: dict,
    phase6_by_symbol: dict[str, dict],
    order_type: str,
    rejected_candidates=(),
    rejected_close_symbols=(),
    approval_status_by_symbol: dict[str, str] | None = None,
) -> None:
    """
    Logs one decisions row per symbol touched this cycle — new/adjusted
    candidates, anything closed for falling out of the shortlist, AND
    anything the human approval gate turned down (executed_position 0.0,
    approval_status rejected/timeout), so the record shows what the system
    wanted and didn't get, not just what happened. Every row carries the
    full 7-phase reasoning: phases 1/5/6/7 are cycle-level facts merged in
    here, phases 2/3/4 come from run_screen for real candidates (see
    monitoring/reasoning.py).
    """
    now = dt.datetime.now(tz=dt.UTC)
    statuses = approval_status_by_symbol or {}
    rows = []

    def _row(symbol, forecast, target_position, executed_position, agreement, full_reasoning):
        return {
            "ts": now,
            "symbol": symbol,
            "feature_set_id": feature_set_id,
            "model_version": _MODEL_VERSION,
            "forecast": forecast,
            "regime": regime,
            "target_position": target_position,
            "executed_position": executed_position,
            "mode": mode,
            "reasoning": json.dumps(full_reasoning),
            "direction_agreement": agreement,
            "approval_status": statuses.get(symbol),
        }

    for c in candidates:
        shares = intended_shares.get(c.symbol, 0.0)
        phase5 = reasoning.phase_execution(c.symbol, "opened", shares, order_type)
        phase6 = phase6_by_symbol.get(c.symbol) or reasoning.phase_reconciliation(c.symbol, shares, executed.get(c.symbol, 0.0), False)
        phase7 = reasoning.phase_ongoing_monitoring(closed=False)
        full_reasoning = reasoning.combine_phases(phase1, *(c.reasoning or []), phase5, phase6, phase7)
        rows.append(_row(c.symbol, c.predicted_return, c.target_position_pct, executed.get(c.symbol), c.direction_agreement, full_reasoning))

    for symbol in closing_symbols:
        phase4 = reasoning.phase_selection_closed(symbol)
        phase5 = reasoning.phase_execution(symbol, "closed", None, order_type)
        phase6 = phase6_by_symbol.get(symbol) or reasoning.phase_reconciliation(symbol, 0.0, executed.get(symbol, 0.0), False)
        phase7 = reasoning.phase_ongoing_monitoring(closed=True)
        full_reasoning = reasoning.combine_phases(phase1, phase4, phase5, phase6, phase7)
        rows.append(_row(symbol, None, 0.0, executed.get(symbol), None, full_reasoning))

    for c in rejected_candidates:
        status = statuses.get(c.symbol, "rejected")
        phase5 = reasoning.phase_execution_rejected(c.symbol, status)
        phase7 = reasoning.phase_ongoing_monitoring(closed=True)
        full_reasoning = reasoning.combine_phases(phase1, *(c.reasoning or []), phase5, phase7)
        rows.append(_row(c.symbol, c.predicted_return, c.target_position_pct, 0.0, c.direction_agreement, full_reasoning))

    for symbol in rejected_close_symbols:
        status = statuses.get(symbol, "rejected")
        phase4 = reasoning.phase_selection_closed(symbol)
        phase5 = reasoning.phase_execution_rejected(symbol, status)
        # closed=False: the close was refused, so the position is still open
        # and still under hourly contradiction watch.
        phase7 = reasoning.phase_ongoing_monitoring(closed=False)
        full_reasoning = reasoning.combine_phases(phase1, phase4, phase5, phase7)
        rows.append(_row(symbol, None, 0.0, 0.0, None, full_reasoning))

    if not rows:
        return
    pd.DataFrame(rows).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def run_cycle(
    feature_set_id: str,
    symbols: list[str],
    dry_run: bool = False,
    request_fn=None,
) -> CycleResult:
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()
    logger.info("Starting trading cycle in %s mode — %s real money involved.", broker.mode, "NO" if broker.mode == "paper" else "REAL")
    send_slack_alert(f"Trading cycle starting (mode={broker.mode}, {len(symbols)} symbols).", severity="info")

    pre_trade_triggers = _run_breaker_check(broker, engine)
    phase1 = reasoning.phase_pretrade_risk(pre_trade_triggers)
    _send_phase_alert(phase1)
    if pre_trade_triggers:
        reasons = "; ".join(r.reason for r in pre_trade_triggers)
        _flatten_and_alert(broker, reasons)
        return CycleResult("flattened_pre_trade", 0, 0, None, broker.get_portfolio_value())

    regime = _market_regime(engine)
    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    candidates = run_screen(feature_set_id, symbols, regime=regime, is_shortable_fn=is_shortable_fn)

    # Phases 2-4 (signals, forecast, selection/sizing) were built per-candidate
    # inside run_screen — surface a one-line summary per phase here so Slack
    # gets the same 7-phase breakdown the dashboard shows per position.
    for phase_num in (2, 3, 4):
        # next(..., None): a candidate missing a phase must cost a summary
        # line, not crash the whole trading cycle with StopIteration.
        with_phase = [
            (c.symbol, next((p for p in c.reasoning if p["phase"] == phase_num), None))
            for c in candidates
            if c.reasoning
        ]
        with_phase = [(symbol, p) for symbol, p in with_phase if p is not None]
        if with_phase:
            title = with_phase[0][1]["title"]
            summaries = [f"{symbol}: {p['summary']}" for symbol, p in with_phase]
            send_slack_alert(f"Phase {phase_num} — {title}: " + " | ".join(summaries), severity="info")

    if dry_run:
        logger.info("Dry run — %s candidate(s) screened, broker untouched.", len(candidates))
        for c in candidates:
            logger.info("  %s %s target=%.4f pred_return=%.4f agreement=%.2f", c.symbol, c.side, c.target_position_pct, c.predicted_return, c.direction_agreement)
        return CycleResult("dry_run", len(candidates), 0, None, broker.get_portfolio_value())

    # Full rebalance to the new target book, not just adjustments layered on
    # top of whatever was already open: anything currently held that isn't
    # one of this cycle's candidates gets closed. Without this, "top 2"
    # would only ever mean "2 new positions added," while last cycle's
    # picks — no longer confident, by definition, since they didn't make
    # this cycle's shortlist — would sit open indefinitely.
    portfolio_value = broker.get_portfolio_value()
    current_positions = broker.get_positions()
    candidate_symbols = {c.symbol for c in candidates}
    closing_symbols = [s for s, qty in current_positions.items() if s not in candidate_symbols and qty != 0]

    if not candidates and not closing_symbols:
        logger.info("No candidates cleared the confidence bar, and nothing open to close — staying in cash.")
        send_slack_alert("No confident candidates this cycle — nothing traded.", severity="info")
        return CycleResult("no_candidates", 0, 0, None, portfolio_value)

    # --- The human gate. Everything below this point only acts on what a
    # human approved: one numbered Telegram message (closes first, then
    # opens), replies polled until answered or timeout, silence = rejected.
    # request_fn is injectable for tests; the default asks a real phone.
    # Every proposal carries its "why" (the screener's reasoning phases for
    # opens, the selection story for closes) plus current P&L for closes —
    # the human on the phone gets the same explanation Slack and the
    # dashboard do, not just a ticker and a size.
    pnl_by_symbol = current_pnl_by_symbol(broker)
    proposals = [
        ProposedTrade(
            index=0, symbol=s, action="close",
            side="long" if current_positions.get(s, 0) >= 0 else "short",
            target_position_pct=0.0, reason="out_of_book",
            reasoning=[reasoning.phase_selection_closed(s)],
            current_pnl_pct=pnl_by_symbol.get(s, (None, None))[0],
            current_pnl_usd=pnl_by_symbol.get(s, (None, None))[1],
        )
        for s in closing_symbols
    ] + [
        ProposedTrade(
            index=0, symbol=c.symbol, action="open", side=c.side,
            target_position_pct=c.target_position_pct,
            predicted_return=c.predicted_return, reason="screen",
            reasoning=c.reasoning,
        )
        for c in candidates
    ]
    gate = request_fn if request_fn is not None else request_approval
    outcome = gate(proposals, context="weekly cycle")

    approved_close_symbols = [p.symbol for p in outcome.approved_closes()]
    approved_open_symbols = {p.symbol for p in outcome.approved_opens()}
    approved_candidates = [c for c in candidates if c.symbol in approved_open_symbols]
    rejected_candidates = [c for c in candidates if c.symbol not in approved_open_symbols]
    rejected_close_symbols = [s for s in closing_symbols if s not in set(approved_close_symbols)]
    approval_status_by_symbol = {p.symbol: outcome.statuses.get(p.index) for p in proposals}

    if outcome.rejected:
        logger.info(
            "Approval gate rejected %d of %d proposal(s) (status=%s) — acting only on the approved subset.",
            len(outcome.rejected), len(proposals), outcome.status,
        )

    # Prices fetched AFTER the gate, not before: a human reply can take up
    # to the full approval timeout, and shares must be sized off quotes
    # from after that wait, not before it.
    prices = _latest_prices([c.symbol for c in approved_candidates] + approved_close_symbols)

    intended_shares: dict[str, float] = {}
    orders_placed = 0

    for symbol in approved_close_symbols:
        intended_shares[symbol] = 0.0
        try:
            order = broker.submit_target_position(symbol, 0.0)
            if order is not None:
                orders_placed += 1
        except Exception:
            logger.exception("Failed to close out-of-book position %s — continuing with the rest of the cycle.", symbol)

    # A close the human refused keeps its current share count on purpose —
    # recording that as the intent keeps reconciliation honest (the position
    # is *supposed* to still be there now).
    for symbol in rejected_close_symbols:
        intended_shares[symbol] = current_positions.get(symbol, 0.0)

    for c in approved_candidates:
        price = prices.get(c.symbol)
        if not price:
            logger.warning("No price for %s — skipping this candidate.", c.symbol)
            continue
        target_shares = (c.target_position_pct * portfolio_value) / price
        intended_shares[c.symbol] = target_shares
        try:
            order = broker.submit_target_position(c.symbol, target_shares)
            if order is not None:
                orders_placed += 1
        except Exception:
            logger.exception("Order failed for %s — continuing with the rest of the cycle.", c.symbol)

    order_type = _order_type(broker)
    execution_summaries = [f"{s}: closed" for s in approved_close_symbols]
    execution_summaries += [f"{c.symbol}: {intended_shares.get(c.symbol, 0.0):+.4g} sh via {order_type}" for c in approved_candidates]
    execution_summaries += [
        f"{p.symbol}: {p.action} NOT executed ({approval_status_by_symbol.get(p.symbol) or 'rejected'})"
        for p in outcome.rejected
    ]
    send_slack_alert("Phase 5 — Execution: " + " | ".join(execution_summaries), severity="info")

    time.sleep(5)  # let paper fills settle before reading positions back
    actual_positions = broker.get_positions()

    reconciliation = reconcile_positions(intended_shares, actual_positions)
    reconciliation_summary = summarize(reconciliation)
    phase6_by_symbol = {
        r.symbol: reasoning.phase_reconciliation(r.symbol, r.intended_shares, r.actual_shares, r.flagged) for r in reconciliation
    }
    phase6_summaries = [p["summary"] for p in phase6_by_symbol.values()]
    send_slack_alert(
        "Phase 6 — Reconciliation & Post-Trade Check: " + " | ".join(phase6_summaries),
        severity="warning" if any(r.flagged for r in reconciliation) else "info",
    )

    _log_decisions(
        approved_candidates, approved_close_symbols, actual_positions, intended_shares, feature_set_id, broker.mode, regime,
        phase1, phase6_by_symbol, order_type,
        rejected_candidates=rejected_candidates,
        rejected_close_symbols=rejected_close_symbols,
        approval_status_by_symbol=approval_status_by_symbol,
    )

    phase7_open = len(approved_candidates)
    phase7_closed = len(approved_close_symbols)
    send_slack_alert(
        f"Phase 7 — Ongoing Monitoring: {phase7_open} position(s) now under hourly contradiction watch, "
        f"{phase7_closed} closed position(s) need no further monitoring.",
        severity="info",
    )

    new_portfolio_value = broker.get_portfolio_value()
    record_equity_snapshot(new_portfolio_value, mode=broker.mode)

    total_attempted = len(approved_candidates) + len(approved_close_symbols)

    post_trade_triggers = _run_breaker_check(broker, engine)
    if post_trade_triggers:
        reasons = "; ".join(r.reason for r in post_trade_triggers)
        _flatten_and_alert(broker, reasons)
        return CycleResult("flattened_post_trade", len(candidates), orders_placed, reconciliation_summary, broker.get_portfolio_value())

    send_slack_alert(
        f"Trading cycle complete: {orders_placed}/{total_attempted} order(s) placed "
        f"({len(approved_candidates)} approved candidate(s), {len(approved_close_symbols)} position(s) closed, "
        f"{len(outcome.rejected)} proposal(s) rejected at the gate). "
        f"Portfolio value ${new_portfolio_value:,.2f}. {reconciliation_summary}",
        severity="info",
    )
    return CycleResult("traded", len(candidates), orders_placed, reconciliation_summary, new_portfolio_value)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_file_logging()  # logs survive the console closing
    parser = argparse.ArgumentParser(description="Run one screen-and-trade cycle (paper only).")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--dry-run", action="store_true", help="Screen only — never touches the broker.")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    result = run_cycle(args.feature_set_id, symbols, dry_run=args.dry_run)
    print(result)


if __name__ == "__main__":
    main()
