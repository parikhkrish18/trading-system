"""Whole-book reallocation after a mid-cycle exit.

A freed slice of capital is not a slot. Once a stop/target/contradiction
closes a position, this module re-screens the whole eligible universe and
sets targets for the resulting book. Existing holdings may grow, shrink, or
be displaced by stronger candidates. Symbols that just exited are excluded
from the immediate pass so a take-profit does not close and reopen the same
trade in one cycle.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from config.settings import settings
from data.ingest.universe import load_active_universe
from execution.approval_gate import ProposedTrade, request_approval, send_followup
from execution.client_fanout import replicate_to_clients
from execution.trading_loop import (
    _allocation_confirmation,
    _apply_allocation,
    _correlation_matrix,
    _deployable_fraction,
)
from models.screener import TradeCandidate, run_screen
from risk.sizing import allocate_by_conviction

logger = logging.getLogger(__name__)

_MIN_REBALANCE_FRACTION = 0.05


def _latest_prices(engine, symbols: list[str]) -> dict[str, float]:
    from execution.contradiction_monitor import _latest_prices as latest_prices

    return latest_prices(engine, symbols)


def _freed_capital_fraction(broker, engine) -> float:
    from execution.contradiction_monitor import _freed_capital_fraction as freed_capital_fraction

    return freed_capital_fraction(broker, engine)


def rebalance_after_exit(
    broker,
    engine,
    *,
    excluded_symbols: Iterable[str] = (),
    request_fn=None,
    log_candidate: Callable[[TradeCandidate, float | None, str, float, str | None], None] | None = None,
    log_displaced_close: Callable[[str, str | None], None] | None = None,
) -> None:
    """Re-optimize the whole master book after confirmed capital is freed."""
    excluded = set(excluded_symbols)
    current_positions = {s: q for s, q in broker.get_positions().items() if q != 0}

    # The close order may be queued or partially filled. Never size a fresh
    # book against capital that has not actually been released, and never
    # submit a second close through the rebalance path for the same symbol.
    still_exiting = excluded.intersection(current_positions)
    if still_exiting:
        logger.info(
            "Post-exit rebalance deferred: %s still present at the broker; waiting for the close to settle.",
            ", ".join(sorted(still_exiting)),
        )
        return

    freed_fraction = _freed_capital_fraction(broker, engine)
    if freed_fraction < _MIN_REBALANCE_FRACTION:
        return

    universe = [s for s in load_active_universe() if s not in excluded]
    if not universe:
        return

    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    max_positions_override = (
        settings.max_concentrated_positions if settings.strategy_mode == "concentrated" else None
    )
    try:
        candidates = run_screen(
            settings.feature_set_id,
            universe,
            is_shortable_fn=is_shortable_fn,
            total_deploy_pct=1.0,
            max_positions_override=max_positions_override,
        )
    except Exception:
        logger.exception("Full-book reactivation screen failed — leaving the current post-exit book unchanged.")
        return

    if not candidates:
        logger.info("No confident candidates after the exit — keeping surviving positions and cash as-is.")
        return

    candidate_symbols = {c.symbol for c in candidates}
    displaced = sorted(set(current_positions) - candidate_symbols)

    gate = request_fn if request_fn is not None else request_approval
    proposals = [
        ProposedTrade(
            index=0,
            symbol=symbol,
            action="close",
            side="long" if current_positions[symbol] > 0 else "short",
            target_position_pct=0.0,
            reason="portfolio_rebalance",
        )
        for symbol in displaced
    ] + [
        ProposedTrade(
            index=0,
            symbol=c.symbol,
            action="open",
            side=c.side,
            predicted_return=c.predicted_return,
            reason="reactivation",
            reasoning=c.reasoning,
            exit_levels=c.exit_levels,
        )
        for c in candidates
    ]
    outcome = gate(proposals, context="mid-cycle full-book rebalance")
    status_by_symbol = {p.symbol: outcome.statuses.get(p.index) for p in proposals}
    approved_close_symbols = {p.symbol for p in outcome.approved_closes()}
    approved_candidate_symbols = {p.symbol for p in outcome.approved_opens()}

    approved_candidates = [c for c in candidates if c.symbol in approved_candidate_symbols]
    if not approved_candidates and not approved_close_symbols:
        return

    rejected_close_symbols = [s for s in displaced if s not in approved_close_symbols]
    rejected_held_candidates = [
        c.symbol
        for c in candidates
        if c.symbol not in approved_candidate_symbols and current_positions.get(c.symbol)
    ]
    kept_symbols = rejected_close_symbols + rejected_held_candidates

    portfolio_value = broker.get_portfolio_value()
    price_symbols = sorted(
        set(current_positions)
        | {c.symbol for c in approved_candidates}
        | set(approved_close_symbols)
        | set(kept_symbols)
    )
    prices = _latest_prices(engine, price_symbols)
    deployable = _deployable_fraction(
        portfolio_value,
        current_positions,
        prices,
        kept_symbols,
    )

    if approved_candidates:
        allocation = allocate_by_conviction(
            {
                c.symbol: (c.conviction_score if c.side == "long" else -c.conviction_score)
                for c in approved_candidates
            },
            max_position_pct=settings.max_single_position_pct,
            max_short_position_pct=settings.max_short_position_pct,
            max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
            correlation_matrix=_correlation_matrix(engine, [c.symbol for c in approved_candidates]),
            target_allocation=deployable,
        )
        _apply_allocation(allocation, approved_candidates)
        if not allocation.reached_target and allocation.reason:
            logger.warning("%s", allocation.reason)
        send_followup(_allocation_confirmation(allocation, approved_candidates))

    target_pct_by_symbol: dict[str, float] = dict.fromkeys(approved_close_symbols, 0.0)
    target_pct_by_symbol.update(
        {c.symbol: c.target_position_pct or 0.0 for c in approved_candidates}
    )

    changed: list[str] = []
    for symbol in approved_close_symbols:
        try:
            broker.submit_target_position(symbol, 0.0)
        except Exception:
            logger.exception("Failed to close %s during the full-book rebalance.", symbol)
            continue
        changed.append(f"{symbol} → 0%")
        if log_displaced_close is not None:
            log_displaced_close(symbol, status_by_symbol.get(symbol) or "approved")

    for candidate in approved_candidates:
        price = prices.get(candidate.symbol)
        if not price:
            logger.warning("No current price for %s during full-book rebalance — leaving it unchanged.", candidate.symbol)
            continue
        target_pct = candidate.target_position_pct or 0.0
        target_shares = (target_pct * portfolio_value) / price
        try:
            broker.submit_target_position(candidate.symbol, target_shares)
        except Exception:
            logger.exception("Failed to retarget %s during the full-book rebalance.", candidate.symbol)
            continue

        executed = broker.get_positions().get(candidate.symbol, 0.0)
        if log_candidate is not None:
            log_candidate(
                candidate,
                executed,
                broker.mode,
                target_shares,
                status_by_symbol.get(candidate.symbol) or "approved",
            )
        changed.append(f"{candidate.symbol} → {abs(target_pct):.1%}")

    if target_pct_by_symbol:
        try:
            replicate_to_clients(target_pct_by_symbol, prices, engine)
        except Exception:
            logger.exception("Client fan-out failed during full-book rebalance; master targets are unaffected.")

    if changed:
        message = "♻️ Full-book rebalance after exit: " + ", ".join(changed)
        from monitoring.alerts import send_slack_alert

        send_slack_alert(message, severity="info")
        send_followup(message)
