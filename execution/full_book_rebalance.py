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
import time
from collections.abc import Callable, Iterable

import pandas as pd

from config.settings import settings
from data.ingest.universe import load_active_universe
from execution import hold_rules
from execution.approval_gate import ProposedTrade, request_approval, send_followup
from execution.client_fanout import replicate_to_clients
from execution.exit_levels import ExitLevels
from execution.trading_loop import (
    _allocation_confirmation,
    _apply_allocation,
    _correlation_matrix,
    _deployable_fraction,
)
from models.candidate_pool_store import load_recent_pool
from models.screener import TradeCandidate, run_screen, select_concentrated_trades
from risk.sizing import allocate_by_conviction

logger = logging.getLogger(__name__)

_MIN_REBALANCE_FRACTION = 0.05
# concentrated-mode-only fallback when settings.max_concentrated_positions
# isn't the active cap (diversified mode never reaches this function's fast
# path at all — see _candidates_from_recent_pool).
_DEFAULT_MAX_POSITIONS = 2

# A market close doesn't always settle at the broker instantly -- a
# get_positions() read taken right after submit_target_position(0.0) can
# still show the pre-fill quantity for a few minutes. 15 minutes is a
# generous margin for a paper fill to post.
_SETTLEMENT_WAIT_SECONDS = 15 * 60


def _latest_prices(engine, symbols: list[str]) -> dict[str, float]:
    from execution.contradiction_monitor import _latest_prices as latest_prices

    return latest_prices(engine, symbols)


def _freed_capital_fraction(broker, engine) -> float:
    from execution.contradiction_monitor import _freed_capital_fraction as freed_capital_fraction

    return freed_capital_fraction(broker, engine)


def _drop_immediately_contradicted(engine, candidates: list[TradeCandidate]) -> list[TradeCandidate]:
    """
    Filters out any candidate that would immediately trip the exact same
    sentiment/sector/momentum contradiction signals
    execution/contradiction_monitor.py already applies hourly to HELD
    positions (see its _contradiction_reasons) -- without this, a
    reactivation can reopen the very symbol/side a contradiction close just
    exited. Real gap, not hypothetical: this candidate can come from
    _candidates_from_recent_pool's persisted pool (models/
    candidate_pool_store.py), which is up to MAX_POOL_AGE old and has no way
    to know about news/momentum that arrived after it was saved -- and even
    a same-cycle fresh run_screen doesn't apply this specific check, only
    score_universe's own gates. excluded_symbols above only blocks a
    same-cycle reopen of a symbol closed THIS pass; this blocks any
    candidate, from either source, that contradicts current signals right
    now, whatever cycle closed it.
    """
    if not candidates:
        return candidates
    from execution.contradiction_monitor import _contradiction_reasons, _sector_by_symbol

    sector_by_symbol = _sector_by_symbol(engine, [c.symbol for c in candidates])
    kept = []
    for c in candidates:
        reasons = _contradiction_reasons(engine, c.symbol, c.side, sector_by_symbol.get(c.symbol))
        if reasons:
            detail = "; ".join(r["detail"] for r in reasons)
            logger.warning(
                "Reactivation candidate %s (%s) dropped — would immediately contradict current signals: %s",
                c.symbol, c.side, detail,
            )
            continue
        kept.append(c)
    return kept


def _candidates_from_recent_pool(
    excluded: set[str], is_shortable_fn, max_positions: int
) -> list[TradeCandidate] | None:
    """
    Tries to build this cycle's candidates from the persisted Claude-advised
    pool (models/candidate_pool_store.py) — this week's already-computed
    analysis — instead of a fresh ensemble retrain + rescreen, which costs
    several minutes on top of what a full-book reactivation otherwise needs.
    Concentrated-mode only, the shape that pool was written in.

    Returns None (never []) when there's nothing usable to redeploy from —
    no persisted pool, everything in it excluded, or nothing clears
    select_concentrated_trades's own gates — so the caller can tell "use
    this" apart from "fall back to a fresh screen" and never silently skip
    a redeploy just because the fast path came up empty.
    """
    if settings.strategy_mode != "concentrated":
        return None
    pool = [c for c in load_recent_pool(settings.feature_set_id) if c["symbol"] not in excluded]
    if not pool:
        return None

    df = pd.DataFrame(pool)
    df["confident"] = True
    df["llm_confidence"] = df["confidence"]

    candidates = select_concentrated_trades(
        df,
        max_leg_pct=settings.max_concentrated_position_pct,
        min_leg_floor_fraction=settings.min_concentrated_leg_floor_fraction,
        max_positions=max_positions,
        total_deploy_pct=1.0,
        is_shortable_fn=is_shortable_fn,
        allow_shorts=settings.allow_shorts,
        rank_score_col="llm_confidence",
        weight_score_col="llm_confidence",
    )
    if not candidates:
        return None

    info_by_symbol = {c["symbol"]: c for c in pool}
    for candidate in candidates:
        info = info_by_symbol.get(candidate.symbol)
        if info is None:
            continue
        candidate.reasoning = info["reasoning"]
        if info.get("take_profit_pct") is not None and info.get("stop_loss_pct") is not None:
            candidate.exit_levels = ExitLevels(take_profit_pct=info["take_profit_pct"], stop_loss_pct=info["stop_loss_pct"])
    return candidates


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

    # Best-effort, same fallback-to-empty pattern execution/trading_loop.py
    # uses for the identical reads: a state read failing must never block a
    # rebalance, only cost it the carried-forward levels/miss counts below.
    try:
        prior_levels = hold_rules.load_exit_levels(engine)
    except Exception:
        logger.warning("Could not load per-position exit levels — kept positions may lose their recorded levels.")
        prior_levels = {}
    try:
        prior_missed = hold_rules.load_missed_cycles(engine)
    except Exception:
        logger.warning("Could not load hold-state miss counters — kept positions restart their counter at 0.")
        prior_missed = {}

    # The close order may be queued or partially filled. Never size a fresh
    # book against capital that has not actually been released, and never
    # submit a second close through the rebalance path for the same symbol.
    still_exiting = excluded.intersection(current_positions)
    if still_exiting:
        # Checking again immediately would very likely see the same
        # still-open read the close order itself just produced. Give the
        # fill real time to post, then make the settlement call for real,
        # rather than deferring on a read taken the instant the close was
        # submitted.
        logger.info(
            "%s still shows open right after the close order — waiting %d minutes for the fill to settle before deciding.",
            ", ".join(sorted(still_exiting)), _SETTLEMENT_WAIT_SECONDS // 60,
        )
        time.sleep(_SETTLEMENT_WAIT_SECONDS)
        current_positions = {s: q for s, q in broker.get_positions().items() if q != 0}
        still_exiting = excluded.intersection(current_positions)
        if still_exiting:
            logger.info(
                "Post-exit rebalance deferred: %s still present at the broker %d minutes after the close; "
                "waiting for the close to settle.",
                ", ".join(sorted(still_exiting)), _SETTLEMENT_WAIT_SECONDS // 60,
            )
            return

    freed_fraction = _freed_capital_fraction(broker, engine)
    if freed_fraction < _MIN_REBALANCE_FRACTION:
        return

    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    max_positions_override = (
        settings.max_concentrated_positions if settings.strategy_mode == "concentrated" else None
    )

    candidates = _candidates_from_recent_pool(excluded, is_shortable_fn, max_positions_override or _DEFAULT_MAX_POSITIONS)
    if candidates is not None:
        logger.info("Reactivating from this week's already-computed candidate pool — skipping a fresh retrain.")
    else:
        universe = [s for s in load_active_universe() if s not in excluded]
        if not universe:
            return
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

    candidates = _drop_immediately_contradicted(engine, candidates)
    if not candidates:
        logger.info(
            "No confident candidates after the exit — every candidate would immediately contradict current "
            "news/momentum signals."
        )
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
    closed_ok: set[str] = set()
    opened_ok: set[str] = set()
    for symbol in approved_close_symbols:
        try:
            broker.submit_target_position(symbol, 0.0)
        except Exception:
            logger.exception("Failed to close %s during the full-book rebalance.", symbol)
            continue
        closed_ok.add(symbol)
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
        opened_ok.add(candidate.symbol)

        # target_shares, not a live re-query: broker.get_positions() read
        # right after submission can still show the pre-fill (unsettled)
        # quantity -- see the settlement-wait comment above and
        # contradiction_monitor.py's _log_closure/_log_reactivation, which
        # have the identical issue on the same root cause.
        if log_candidate is not None:
            log_candidate(
                candidate,
                target_shares,
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

    # Persist exit levels and miss counters for the book as it now stands --
    # same intended (not re-queried) state used above, for the same
    # unsettled-read reason. Without this, a candidate opened here (e.g. the
    # hourly contradiction monitor's mid-week reactivation) never gets its
    # ATR/Donchian-derived take-profit/stop-loss recorded anywhere: the
    # dashboard shows it as unrecorded, and hold_rules.evaluate_holds /
    # execution/contradiction_monitor.py's own stop/target check falls back
    # to the generic global thresholds instead of this position's own levels.
    final_open_symbols = (set(current_positions) - closed_ok) | opened_ok
    levels_by_symbol = {
        **prior_levels,
        **{c.symbol: c.exit_levels for c in approved_candidates if c.exit_levels},
    }
    missed_by_symbol = {s: (0 if s in opened_ok else prior_missed.get(s, 0)) for s in final_open_symbols}
    try:
        hold_rules.store_missed_cycles(
            engine, missed_by_symbol, {s: lv for s, lv in levels_by_symbol.items() if s in final_open_symbols}
        )
    except Exception:
        logger.exception("Could not persist hold state after the full-book rebalance.")

    if changed:
        message = "♻️ Full-book rebalance after exit: " + ", ".join(changed)
        from monitoring.alerts import send_slack_alert

        send_slack_alert(message, severity="info")
        send_followup(message)
