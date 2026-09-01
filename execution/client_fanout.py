"""
Replicates the master account's trades onto every client's own Alpaca
account, sized as the same percentage of THEIR capital rather than a fixed
dollar amount — a client with $10,000 and a client with $50,000 both end
up at the same target weight per symbol, just different share counts.

A client can additionally carry a leverage_multiplier (1-3x, operator-set
only — see data/schema/012_client_leverage.sql), which scales that target
weight further against the client's own equity before sizing, capped so
leverage can't push a single position's exposure past
settings.max_single_position_pct. Requires margin_enabled (enforced by a DB
constraint) since leverage on a cash account isn't meaningful.

A client can also be trading_paused (see data/schema/013_client_risk_controls.sql)
-- via their own portal "Liquidate now" button, or automatically by
execution/client_risk_controls.py's drawdown/profit-target checks.
load_active_clients() excludes a paused client from every fan-out pass the
same way active=FALSE already does; nothing below this point needs to know
why a client didn't show up in the list.

Every attempt (submitted, skipped, failed) is logged to `client_orders` —
that table is both the operator's audit trail and the source for each
client's own "what happened to my capital" trade history on the portal.

Deliberately isolated per client: one client's bad/revoked API key,
insufficient buying power, or a rejected order must never stop the other
clients' trades from going through. Nothing here touches the master
account's own broker/order flow (execution/trading_loop.py calls this
*after* its own orders are placed, passing the same symbols/prices it just
used) — a bug in this module can misallocate client capital, but it can't
by itself misfire the operator's own trades.
"""
from __future__ import annotations

import logging

import pandas as pd
from alpaca.data.requests import StockLatestQuoteRequest
from sqlalchemy.engine import Engine

from config.settings import settings
from execution.broker_alpaca import AlpacaBroker
from execution.client_crypto import decrypt_credential

logger = logging.getLogger(__name__)


def _log_order(
    engine: Engine,
    client_id: int,
    symbol: str,
    *,
    side: str | None = None,
    target_position_pct: float | None = None,
    target_shares: float | None = None,
    status: str,
    alpaca_order_id: str | None = None,
    error_message: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO client_orders "
            "(client_id, symbol, side, target_position_pct, target_shares, status, alpaca_order_id, error_message) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (client_id, symbol, side, target_position_pct, target_shares, status, alpaca_order_id, error_message),
        )


def load_active_clients(engine: Engine, client_ids: list[int] | None = None) -> list[dict]:
    """
    Decrypted client records ready to trade: id, name, api_key, api_secret,
    margin_enabled, leverage_multiplier. Decryption happens here, just-in-
    time, rather than anywhere credentials might linger longer than one
    fan-out pass.

    client_ids restricts to specific clients (used to buy in a single
    newly-onboarded client) — None means every active client.
    """
    query = (
        "SELECT id, name, alpaca_api_key_encrypted, alpaca_api_secret_encrypted, margin_enabled, leverage_multiplier "
        # trading_paused excludes a client from every fan-out pass the same
        # way active=FALSE already does -- set by the client's own
        # "Liquidate now" button or automatically by
        # execution/client_risk_controls.py (drawdown/profit-target
        # triggers). See data/schema/013_client_risk_controls.sql.
        "FROM clients WHERE active = TRUE AND trading_paused = FALSE"
    )
    params: dict = {}
    if client_ids is not None:
        if not client_ids:
            return []
        query += " AND id = ANY(%(client_ids)s)"
        params["client_ids"] = list(client_ids)
    rows = pd.read_sql(query, engine, params=params or None)

    clients = []
    for _, row in rows.iterrows():
        try:
            api_key = decrypt_credential(row["alpaca_api_key_encrypted"])
            api_secret = decrypt_credential(row["alpaca_api_secret_encrypted"])
        except Exception:
            logger.exception("Could not decrypt credentials for client %s (id=%s) — skipping.", row["name"], row["id"])
            continue
        clients.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "api_key": api_key,
                "api_secret": api_secret,
                "margin_enabled": bool(row["margin_enabled"]),
                # .get with a default rather than row["leverage_multiplier"] directly: the column is
                # NOT NULL DEFAULT 1 at the DB level (see data/schema/012_client_leverage.sql), but this
                # keeps load_active_clients from hard-crashing for a caller/test still on the pre-migration
                # SELECT shape rather than silently misdirecting client capital if it ever did.
                "leverage_multiplier": int(row["leverage_multiplier"]) if "leverage_multiplier" in row else 1,
            }
        )
    return clients


def current_master_weights(broker) -> dict[str, float]:
    """
    The master account's current holdings expressed as signed
    percent-of-portfolio weights (negative = short) — used both to fan out
    a freshly-decided trade (trading_loop.py passes the just-computed
    target weights directly) and to buy a newly-onboarded client into
    whatever the book already holds, mirroring the master's actual current
    allocation rather than replaying history.
    """
    portfolio_value = broker.get_portfolio_value()
    if portfolio_value <= 0:
        return {}
    positions = broker.get_positions_detailed()
    return {p["symbol"]: p["market_value"] / portfolio_value for p in positions}


def replicate_to_clients(
    target_positions: dict[str, float],
    prices: dict[str, float],
    engine: Engine,
    client_ids: list[int] | None = None,
) -> None:
    """
    For every active client (or just `client_ids`, if given): size each
    symbol in target_positions as that percentage of the CLIENT's own
    account equity, and submit the order through the client's own Alpaca
    credentials. A short (target_position_pct < 0) is dropped for a client
    without margin_enabled rather than submitted and rejected by Alpaca --
    a cash account cannot hold a short position at all.

    Every outcome is written to client_orders. Nothing here raises past
    this function: a client-level or symbol-level failure is logged and the
    loop continues, so one bad client can never block the rest.
    """
    if not settings.client_trading_enabled:
        logger.info("CLIENT_TRADING_ENABLED is off — skipping client fan-out for %d symbol(s).", len(target_positions))
        return

    clients = load_active_clients(engine, client_ids=client_ids)
    if not clients:
        return

    for client in clients:
        try:
            broker = AlpacaBroker(mode="live", confirm_live=True, api_key=client["api_key"], secret_key=client["api_secret"])
            portfolio_value = broker.get_portfolio_value()
        except Exception as e:
            logger.exception("Could not connect to %s's Alpaca account — skipping this client entirely this cycle.", client["name"])
            for symbol, target_pct in target_positions.items():
                _log_order(
                    engine, client["id"], symbol,
                    target_position_pct=target_pct, status="failed", error_message=f"account connection failed: {e}",
                )
            continue

        for symbol, target_pct in target_positions.items():
            side = "long" if (target_pct or 0.0) >= 0 else "short"

            if target_pct < 0 and not client["margin_enabled"]:
                logger.info("Skipping %s short for %s — account is not margin-enabled.", symbol, client["name"])
                _log_order(
                    engine, client["id"], symbol, side=side, target_position_pct=target_pct,
                    status="skipped_no_margin",
                )
                continue

            # A full close (target_pct == 0) needs no price -- submitting
            # target_shares=0 works the same regardless, and a client
            # holding a position in a symbol the master no longer prices
            # (delisted, or just not in this cycle's quote batch) must
            # still be able to exit it.
            if target_pct == 0.0:
                target_shares = 0.0
                effective_pct = 0.0
            else:
                price = prices.get(symbol)
                if not price:
                    _log_order(
                        engine, client["id"], symbol, side=side, target_position_pct=target_pct,
                        status="no_price",
                    )
                    continue
                # Leverage multiplies the master's target weight against this
                # client's own equity -- a leverage=1 client (every client
                # before this feature, and the default for every new one)
                # computes byte-for-byte the same target_shares as before.
                # The cap only engages once leverage > 1: it deliberately
                # does NOT retroactively clamp an unlevered client whose
                # target_pct alone might already exceed max_single_position_pct
                # (e.g. one candidate getting the full deployable book) --
                # that's an existing, intentional master-account sizing
                # decision this module has never second-guessed. What's new
                # here is specifically bounding how much EXTRA exposure
                # leverage can add, so a client's leveraged dollar risk in
                # any one symbol never ends up higher than what an unlevered
                # client's own cap already allows.
                effective_pct = target_pct * client["leverage_multiplier"]
                if client["leverage_multiplier"] > 1:
                    cap = settings.max_single_position_pct
                    if abs(effective_pct) > cap:
                        effective_pct = cap if effective_pct > 0 else -cap
                target_shares = (effective_pct * portfolio_value) / price
            try:
                order = broker.submit_target_position(symbol, target_shares)
            except Exception as e:
                logger.exception("Order failed for %s on %s's account — continuing with the rest of this client's book.", symbol, client["name"])
                _log_order(
                    engine, client["id"], symbol, side=side, target_position_pct=effective_pct,
                    target_shares=target_shares, status="failed", error_message=str(e),
                )
                continue

            if order is None:
                _log_order(
                    engine, client["id"], symbol, side=side, target_position_pct=effective_pct,
                    target_shares=target_shares, status="no_change",
                )
            else:
                _log_order(
                    engine, client["id"], symbol, side=side, target_position_pct=effective_pct,
                    target_shares=target_shares, status="submitted", alpaca_order_id=order.get("id"),
                )


def onboard_client(client_id: int, master_broker, engine: Engine) -> None:
    """
    Buys a newly-added client into the master account's CURRENT holdings
    immediately (the "buy in right away" decision), rather than waiting for
    the next scheduled signal — mirrors current_master_weights() using
    live quotes for whatever's currently held.

    Client fan-out is Alpaca-only (see execution/broker_alpaca.py's
    docstring on why IBKR doesn't have a workable per-client credential
    model) regardless of which broker the operator's own master account
    runs on — this needs master_broker's Alpaca market-data client
    specifically to price the buy-in, so it's a no-op with a clear log
    line rather than a crash when the master account isn't Alpaca.
    """
    if not isinstance(master_broker, AlpacaBroker):
        logger.warning(
            "Client onboarding needs an Alpaca master broker to price the buy-in (got %s) — "
            "client %s was added but not bought in. Run it manually once the master account is on Alpaca.",
            type(master_broker).__name__, client_id,
        )
        return

    weights = current_master_weights(master_broker)
    if not weights:
        logger.info("Master account currently holds nothing — new client %s starts in cash, nothing to buy in.", client_id)
        return

    prices = {}
    for symbol in weights:
        try:
            quote = master_broker.data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
            prices[symbol] = quote.ask_price or quote.bid_price
        except Exception:
            logger.exception("Could not price %s for new client %s's buy-in.", symbol, client_id)

    replicate_to_clients(weights, prices, engine, client_ids=[client_id])
