"""
Interactive Brokers (TWS / IB Gateway) integration via ib_insync.

Connection model matches the Blue Chip Trading Bot setup: the bot connects to
a locally running TWS or IB Gateway socket — there are no REST API keys.
Log in to TWS/Gateway and enable API access before calling this module.

Ports (same as Blue_Chip_Trading_Bot/tws_bot.py):
  TWS paper  7497 | TWS live  7496
  IBG paper  4002 | IBG live  4001

Safety choices mirror broker_alpaca.py:
  - Paper and live use separate ports, selected by TRADING_MODE.
  - Constructing IBKRBroker in "live" mode requires confirm_live=True.
  - Every order placement is logged before submission.

Short-selling limitation: unlike broker_alpaca.py, this module has no
is_shortable() pre-check. ib_insync doesn't expose IBKR's short-locate
inventory through the base API without an extra market-data subscription,
so there's no reliable way to know in advance whether a short will be
filled. An unshortable order will be rejected by IBKR itself at submission
time — that rejection surfaces via the returned trade's orderStatus, and
callers (the screener/trading loop) must check and log it rather than
assume the order went through just because placeOrder() didn't raise.

Extended-hours limitation: unlike broker_alpaca.py, this module always
submits regular-hours market orders — no outsideRth handling yet. IBKR
does support extended-hours execution (Order.outsideRth=True, generally
still wants a limit order for the same thin-liquidity reasons), it's just
not implemented here. An order submitted outside RTH will queue at IBKR
until the next open, same as broker_alpaca.py's behavior when
ALLOW_EXTENDED_HOURS_TRADING is off.
"""
from __future__ import annotations

import logging
import time

from ib_insync import IB, MarketOrder, Stock

from config.settings import settings

logger = logging.getLogger(__name__)


class IBKRBroker:
    def __init__(self, mode: str | None = None, confirm_live: bool = False):
        mode = mode or settings.trading_mode
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")

        if mode == "live" and not confirm_live:
            raise RuntimeError(
                "Refusing to construct a live IBKR broker without confirm_live=True. "
                "This is deliberate friction — pass it explicitly at the call site "
                "that's actually meant to trade live, not from a config default."
            )

        self.mode = mode
        self.host = settings.ibkr_host
        self.client_id = settings.ibkr_client_id if mode == "paper" else settings.ibkr_live_client_id
        self.port = settings.ibkr_paper_port if mode == "paper" else settings.ibkr_live_port

        self.ib = IB()
        self._connect()

    def _connect(self, retries: int = 5, delay: int = 10) -> None:
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    "Connecting to IBKR %s:%s clientId=%s (attempt %s/%s, mode=%s)",
                    self.host,
                    self.port,
                    self.client_id,
                    attempt,
                    retries,
                    self.mode,
                )
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                logger.info("IBKR connected. Server version: %s", self.ib.client.serverVersion())
                return
            except Exception as e:
                logger.warning("IBKR connection failed: %s", e)
                if attempt < retries:
                    time.sleep(delay)

        raise RuntimeError(
            f"Could not connect to IBKR at {self.host}:{self.port}. "
            "Ensure TWS or IB Gateway is running, logged in, and API access is enabled "
            "(Edit → Global Configuration → API → Settings → Enable ActiveX and Socket Clients)."
        )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def get_account(self) -> dict:
        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = item.value
        summary["mode"] = self.mode
        summary["host"] = self.host
        summary["port"] = self.port
        return summary

    def get_positions(self) -> dict[str, float]:
        self.ib.reqPositions()
        self.ib.sleep(0.5)
        return {pos.contract.symbol: float(pos.position) for pos in self.ib.positions()}

    def get_order(self, order_id: str) -> dict | None:
        """
        Current state of a previously submitted order, matching
        AlpacaBroker.get_order so reconciliation can ask either broker the
        same question.

        ib_insync tracks orders as Trade objects on the session rather than
        by a queryable id, so this looks through the open trades for a
        matching order id and reports its status. Anything not found is
        None, which reconciliation reads as "still pending" — the
        deliberately quiet fallback, since a status we cannot read is not
        evidence that something failed.
        """
        for trade in self.ib.trades():
            if str(getattr(trade.order, "orderId", "")) == str(order_id):
                return {"status": str(trade.orderStatus.status).lower(), "filled": float(trade.orderStatus.filled)}
        return None

    def get_portfolio_value(self) -> float:
        for item in self.ib.accountSummary():
            if item.tag == "NetLiquidation" and item.currency == "USD":
                return float(item.value)
        for item in self.ib.accountSummary():
            if item.tag == "NetLiquidation":
                return float(item.value)
        raise RuntimeError("Could not read NetLiquidation from IBKR account summary.")

    def _qualify_stock(self, symbol: str) -> Stock:
        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Could not qualify IBKR contract for {symbol}")
        return qualified[0]

    def submit_target_position(self, symbol: str, target_shares: float) -> dict | None:
        """
        Submits a single market order to move from the current position in
        `symbol` to `target_shares`. Returns order metadata, or None if no
        trade was needed. Uses whole shares.
        """
        current_positions = self.get_positions()
        current_shares = current_positions.get(symbol, 0.0)
        delta = target_shares - current_shares

        if abs(delta) < 1e-9:
            return None

        side = "BUY" if delta > 0 else "SELL"
        qty = abs(round(delta))
        if qty < 1:
            return None

        logger.info(
            "Submitting %s order: %s %s shares (mode=%s, current=%s, target=%s)",
            side,
            qty,
            symbol,
            self.mode,
            current_shares,
            target_shares,
        )

        contract = self._qualify_stock(symbol)
        order = MarketOrder(side, qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
        }

    def flatten_all(self) -> None:
        """Closes every open position — the action a triggered circuit breaker should call."""
        logger.warning("Flattening all positions via IBKR (mode=%s)", self.mode)

        for trade in self.ib.openTrades():
            self.ib.cancelOrder(trade.order)
        self.ib.sleep(1)

        self.ib.reqPositions()
        self.ib.sleep(0.5)
        for pos in list(self.ib.positions()):
            if pos.position == 0:
                continue
            action = "SELL" if pos.position > 0 else "BUY"
            qty = abs(int(pos.position))
            symbol = pos.contract.symbol
            logger.info("Closing %s position: %s %s shares", self.mode, symbol, qty)
            contract = self._qualify_stock(symbol)
            order = MarketOrder(action, qty)
            order.tif = "DAY"
            self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
