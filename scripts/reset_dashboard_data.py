"""
DESTRUCTIVE, one-off reset: flattens every open paper position and wipes
every market-data/trading-history table, for a completely clean dashboard.

Deliberately narrow about what it touches:
  - `universe` (the active S&P 500 list, data/ingest/universe.py) is left
    alone -- the system can't screen or trade at all without knowing what
    it's allowed to hold.
  - `clients` / `client_orders` are left alone entirely -- those are real,
    separately-owned client accounts, unrelated to the master book's own
    market-data/decision history.

No FK constraints exist between any of the wiped tables (see
data/schema/*.sql), so truncation order doesn't matter.

Usage:
    python -m scripts.reset_dashboard_data --yes
"""
from __future__ import annotations

import argparse
import logging

from data.ingest.db import get_engine
from execution.broker import get_broker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TABLES_TO_WIPE = [
    "prices",
    "news_events",
    "fundamentals",
    "features",
    "decisions",
    "equity_curve",
    "circuit_breaker_state",
    "position_hold_state",
    "universe_snapshot",
]


def reset_dashboard_data() -> None:
    broker = get_broker()  # never confirm_live=True -- paper-only by construction
    open_symbols = sorted(s for s, q in broker.get_positions().items() if q != 0)
    if open_symbols:
        logger.warning("Flattening %d open position(s): %s", len(open_symbols), ", ".join(open_symbols))
        broker.flatten_all()
    else:
        logger.info("No open positions to flatten.")

    engine = get_engine()
    with engine.begin() as conn:
        for table in _TABLES_TO_WIPE:
            logger.warning("Truncating %s", table)
            # Table names come from the fixed literal list above, not user input.
            conn.exec_driver_sql(f"TRUNCATE TABLE {table} RESTART IDENTITY")

    logger.warning("Reset complete: %d table(s) truncated, positions flattened.", len(_TABLES_TO_WIPE))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DESTRUCTIVE: flattens every open paper position and truncates "
            f"{', '.join(_TABLES_TO_WIPE)}. Irreversible -- no backup is taken."
        )
    )
    parser.add_argument("--yes", action="store_true", required=True, help="Required acknowledgement.")
    parser.parse_args()
    reset_dashboard_data()


if __name__ == "__main__":
    main()
