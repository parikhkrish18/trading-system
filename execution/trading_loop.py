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
import logging
import time

import pandas as pd

from config.settings import settings
from data.ingest.db import get_engine
from data.ingest.universe import resolve_symbols
from execution.broker import get_broker
from execution.reconciliation import reconcile_positions, summarize
from features.quant.momentum import adx
from models.regime.trend_chop_classifier import CHOP, RuleBasedRegime
from models.screener import build_correlation_matrix, run_screen
from monitoring.alerts import alert_circuit_breaker, send_slack_alert
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
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    engine = get_engine()
    df = pd.read_sql(
        f"""SELECT DISTINCT ON (symbol) symbol, close FROM prices
            WHERE symbol IN ({symbol_list}) ORDER BY symbol, ts DESC""",
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
        f"SELECT ts, high, low, close FROM prices WHERE symbol = '{market_proxy}' ORDER BY ts",
        engine,
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
        f"""SELECT symbol, ts, close FROM prices
            WHERE symbol IN ({", ".join(f"'{s}'" for s in positions_by_value) or "''"})
            ORDER BY ts""",
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


def _flatten_and_alert(broker, reason: str) -> None:
    logger.critical("Circuit breaker triggered: %s — flattening all positions.", reason)
    broker.flatten_all()
    alert_circuit_breaker(reason)
    record_equity_snapshot(broker.get_portfolio_value(), mode=broker.mode)


def _log_decisions(candidates, executed: dict[str, float], feature_set_id: str, mode: str) -> None:
    if not candidates:
        return
    now = dt.datetime.now(tz=dt.UTC)
    rows = [
        {
            "ts": now,
            "symbol": c.symbol,
            "feature_set_id": feature_set_id,
            "model_version": _MODEL_VERSION,
            "forecast": c.predicted_return,
            "regime": None,
            "target_position": c.target_position_pct,
            "executed_position": executed.get(c.symbol),
            "mode": mode,
        }
        for c in candidates
    ]
    pd.DataFrame(rows).to_sql("decisions", get_engine(), if_exists="append", index=False)


def run_cycle(
    feature_set_id: str,
    symbols: list[str],
    dry_run: bool = False,
) -> CycleResult:
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()
    logger.info("Starting trading cycle in %s mode — %s real money involved.", broker.mode, "NO" if broker.mode == "paper" else "REAL")
    send_slack_alert(f"Trading cycle starting (mode={broker.mode}, {len(symbols)} symbols).", severity="info")

    pre_trade_triggers = _run_breaker_check(broker, engine)
    if pre_trade_triggers:
        reasons = "; ".join(r.reason for r in pre_trade_triggers)
        _flatten_and_alert(broker, reasons)
        return CycleResult("flattened_pre_trade", 0, 0, None, broker.get_portfolio_value())

    regime = _market_regime(engine)
    is_shortable_fn = broker.is_shortable if hasattr(broker, "is_shortable") else None
    candidates = run_screen(feature_set_id, symbols, regime=regime, is_shortable_fn=is_shortable_fn)

    if not candidates:
        logger.info("No candidates cleared the confidence bar this cycle.")
        send_slack_alert("No confident candidates this cycle — nothing traded.", severity="info")
        return CycleResult("no_candidates", 0, 0, None, broker.get_portfolio_value())

    if dry_run:
        logger.info("Dry run — %s candidate(s) screened, broker untouched.", len(candidates))
        for c in candidates:
            logger.info("  %s %s target=%.4f pred_return=%.4f agreement=%.2f", c.symbol, c.side, c.target_position_pct, c.predicted_return, c.direction_agreement)
        return CycleResult("dry_run", len(candidates), 0, None, broker.get_portfolio_value())

    portfolio_value = broker.get_portfolio_value()
    prices = _latest_prices([c.symbol for c in candidates])

    intended_shares: dict[str, float] = {}
    orders_placed = 0
    for c in candidates:
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

    time.sleep(5)  # let paper fills settle before reading positions back
    actual_positions = broker.get_positions()
    _log_decisions(candidates, actual_positions, feature_set_id, broker.mode)

    reconciliation = reconcile_positions(intended_shares, actual_positions)
    reconciliation_summary = summarize(reconciliation)
    if any(r.flagged for r in reconciliation):
        send_slack_alert(f"Reconciliation flagged divergence:\n{reconciliation_summary}", severity="warning")

    new_portfolio_value = broker.get_portfolio_value()
    record_equity_snapshot(new_portfolio_value, mode=broker.mode)

    post_trade_triggers = _run_breaker_check(broker, engine)
    if post_trade_triggers:
        reasons = "; ".join(r.reason for r in post_trade_triggers)
        _flatten_and_alert(broker, reasons)
        return CycleResult("flattened_post_trade", len(candidates), orders_placed, reconciliation_summary, broker.get_portfolio_value())

    send_slack_alert(
        f"Trading cycle complete: {orders_placed}/{len(candidates)} order(s) placed. "
        f"Portfolio value ${new_portfolio_value:,.2f}. {reconciliation_summary}",
        severity="info",
    )
    return CycleResult("traded", len(candidates), orders_placed, reconciliation_summary, new_portfolio_value)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
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
