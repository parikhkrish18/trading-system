"""
Broker factory — returns the configured execution backend.

Set BROKER=ibkr or BROKER=alpaca in .env (default: ibkr).
"""
from __future__ import annotations

import logging
import socket

from config.settings import settings
from execution.broker_alpaca import AlpacaBroker
from execution.broker_ibkr import IBKRBroker

logger = logging.getLogger(__name__)

# Fast sanity-check timeout for the IBKR reachability pre-check below.
# Deliberately much shorter than IBKRBroker's own connection loop (5 retries
# x 10s delay, execution/broker_ibkr.py) — this only answers "is anything
# listening at all", not "did TWS log in and enable API access". The point
# is catching a misconfigured deployment (BROKER left at its 'ibkr' default
# on a host with no TWS/IB Gateway, e.g. Railway) in ~1-2s instead of
# burning the full ~40s retry sequence before that cycle silently no-ops.
# A real, briefly-unreachable IBKR instance still gets the full retry loop
# below once this pre-check passes (or for any case that isn't the
# fast-fail path) — this does not change that behavior.
_IBKR_PRECHECK_TIMEOUT_S = 1.5


def _ibkr_port_for(mode: str) -> int:
    return settings.ibkr_paper_port if mode == "paper" else settings.ibkr_live_port


def _ibkr_reachable(host: str, port: int, timeout: float = _IBKR_PRECHECK_TIMEOUT_S) -> bool:
    """Raw TCP connect check — no IB protocol handshake, just "is a socket listening"."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_broker(mode: str | None = None, confirm_live: bool = False):
    broker = settings.broker.lower()
    if broker == "ibkr":
        effective_mode = mode or settings.trading_mode
        host = settings.ibkr_host
        port = _ibkr_port_for(effective_mode)
        if not _ibkr_reachable(host, port):
            raise RuntimeError(
                f"BROKER=ibkr but no TWS/IB Gateway reachable at {host}:{port} "
                f"(fast {_IBKR_PRECHECK_TIMEOUT_S}s reachability check — not the "
                "slower connection retry that follows a successful check). "
                "If this is a hosted/Railway deployment, set BROKER=alpaca instead: "
                "IBKR needs a locally running TWS/IB Gateway socket that no "
                "container has. If TWS/IB Gateway is supposed to be running here, "
                "start it, log in, and enable API access (Edit -> Global "
                "Configuration -> API -> Settings -> Enable ActiveX and Socket "
                "Clients) before retrying."
            )
        return IBKRBroker(mode=mode, confirm_live=confirm_live)
    if broker == "alpaca":
        return AlpacaBroker(mode=mode, confirm_live=confirm_live)
    raise ValueError(f"BROKER must be 'ibkr' or 'alpaca', got {settings.broker!r}")
