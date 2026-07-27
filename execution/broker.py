"""
Broker factory — returns the configured execution backend.

Set BROKER=ibkr or BROKER=alpaca in .env (default: ibkr).
"""
from __future__ import annotations

from config.settings import settings
from execution.broker_alpaca import AlpacaBroker
from execution.broker_ibkr import IBKRBroker


def get_broker(mode: str | None = None, confirm_live: bool = False):
    broker = settings.broker.lower()
    if broker == "ibkr":
        return IBKRBroker(mode=mode, confirm_live=confirm_live)
    if broker == "alpaca":
        return AlpacaBroker(mode=mode, confirm_live=confirm_live)
    raise ValueError(f"BROKER must be 'ibkr' or 'alpaca', got {settings.broker!r}")
