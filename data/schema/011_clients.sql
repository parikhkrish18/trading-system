-- Client accounts for the "manage via their own Alpaca API keys" model.
--
-- Each client funds and holds their own live Alpaca account; this table
-- only stores the credentials needed to trade it on their behalf (Fernet-
-- encrypted at rest, see execution/client_crypto.py -- the master
-- encryption key lives in CLIENT_KEY_ENCRYPTION_KEY, an env var, never in
-- this database) and the password gating their read-only portal login.
--
-- margin_enabled is set by the operator when the client is added (Alpaca's
-- API doesn't expose a clean single "can this account short" flag) and
-- drives execution/client_fanout.py: a client without margin has the short
-- leg of a trade skipped for them rather than rejected by Alpaca at
-- submission time, since a cash account can't hold a short position at all.
--
-- active=false takes a client out of every future fan-out (new trades,
-- rebalances, closes) without touching whatever they're currently holding
-- -- deactivating is not the same as liquidating, and this table never
-- fires an order on its own.
CREATE TABLE IF NOT EXISTS clients (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    alpaca_api_key_encrypted BYTEA NOT NULL,
    alpaca_api_secret_encrypted BYTEA NOT NULL,
    margin_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    password_hash TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per (client, symbol) fan-out attempt -- the audit trail behind
-- both the operator's Clients tab and each client's own "what happened to
-- my capital" trade history. status is one of:
--   submitted          -- an order was placed at Alpaca (alpaca_order_id set)
--   no_change           -- target already matched the held position, nothing to do
--   skipped_no_margin   -- a short leg was dropped because the client isn't margin-enabled
--   no_price             -- couldn't price the symbol, nothing was attempted
--   failed              -- Alpaca rejected the order or the call raised (error_message set)
-- Deliberately append-only and never joined against `decisions` (the
-- master account's own table) -- a client's target_position_pct can differ
-- from the master's on any given symbol once per-client margin skips start
-- happening, so each client's history has to stand on its own.
CREATE TABLE IF NOT EXISTS client_orders (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id),
    symbol TEXT NOT NULL,
    side TEXT,
    target_position_pct DOUBLE PRECISION,
    target_shares DOUBLE PRECISION,
    status TEXT NOT NULL,
    alpaca_order_id TEXT,
    error_message TEXT,
    ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_client_orders_client_ts ON client_orders (client_id, ts DESC);
