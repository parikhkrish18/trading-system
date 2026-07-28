"""
Alpaca broker integration. Same API for paper and live — per the plan,
build and prove out against paper first (Phase 5, point 3), and treat this
file as one of the two in the repo (with circuit_breakers.py) where a bug
does the most damage (Security section).

Safety choices baked in deliberately:
  - Paper and live use fully separate credentials (config/settings.py),
    never the same key pair.
  - Constructing an AlpacaBroker in "live" mode requires an explicit,
    separate `confirm_live=True` flag in addition to settings.trading_mode
    being "live" — a config typo alone can't fire a live order.
  - Every order placement is logged before submission, not just on success.
"""
from __future__ import annotations

import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config.settings import settings

logger = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, mode: str | None = None, confirm_live: bool = False):
        mode = mode or settings.trading_mode
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")

        if mode == "live" and not confirm_live:
            raise RuntimeError(
                "Refusing to construct a live broker without confirm_live=True. "
                "This is deliberate friction — pass it explicitly at the call site "
                "that's actually meant to trade live, not from a config default."
            )

        self.mode = mode
        if mode == "paper":
            api_key, secret_key = settings.alpaca_paper_api_key, settings.alpaca_paper_secret_key
        else:
            api_key, secret_key = settings.alpaca_live_api_key, settings.alpaca_live_secret_key

        if not api_key or not secret_key:
            raise RuntimeError(f"Alpaca {mode} API credentials are not set in the environment.")

        self.client = TradingClient(api_key, secret_key, paper=(mode == "paper"))

    def get_account(self) -> dict:
        account = self.client.get_account()
        return account.model_dump()

    def get_positions(self) -> dict[str, float]:
        positions = self.client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def get_portfolio_value(self) -> float:
        return float(self.client.get_account().equity)

    def is_shortable(self, symbol: str) -> bool:
        """
        Whether Alpaca will currently let this symbol be sold short — checks
        both `shortable` (the asset class allows it at all) and
        `easy_to_borrow` (shares are actually available right now, not just
        theoretically shortable). Callers (the screener) should skip any
        short candidate this returns False for rather than submitting an
        order that's just going to get rejected.
        """
        asset = self.client.get_asset(symbol)
        return bool(getattr(asset, "shortable", False) and getattr(asset, "easy_to_borrow", False))

    def submit_target_position(self, symbol: str, target_shares: float) -> dict | None:
        """
        Submits a single market order to move from the current position in
        `symbol` to `target_shares`. Returns the order dict, or None if no
        trade was needed. Fractional shares by default (Alpaca supports it
        for most symbols) — except orders that open or increase a short
        position, which Alpaca rejects outright if fractional ("fractional
        orders cannot be sold short"); those get rounded to whole shares.
        """
        current_positions = self.get_positions()
        current_shares = current_positions.get(symbol, 0.0)
        delta = target_shares - current_shares

        if abs(delta) < 1e-9:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = abs(round(delta, 4))

        if side == OrderSide.SELL and target_shares < 0:
            qty = float(round(qty))
            if qty < 1:
                logger.info("Skipping %s: rounds to 0 whole shares for a short order.", symbol)
                return None

        logger.info(
            "Submitting %s order: %s %s shares (mode=%s, current=%s, target=%s)",
            side, qty, symbol, self.mode, current_shares, target_shares,
        )

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_request)
        return order.model_dump()

    def flatten_all(self) -> None:
        """Closes every open position — the action a triggered circuit breaker should call."""
        logger.warning("Flattening all positions (mode=%s)", self.mode)
        self.client.close_all_positions(cancel_orders=True)
