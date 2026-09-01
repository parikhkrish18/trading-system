"""
Client self-service risk controls: independent of the master account's own
circuit breakers (risk/circuit_breakers.py) and independent of every OTHER
client, each client can optionally set a max-drawdown auto-close and/or a
profit-target auto-secure threshold on their OWN account (see
data/schema/013_client_risk_controls.sql), plus flatten it immediately with
the portal's "Liquidate now" button (monitoring/dashboard/server.py's
POST /api/portal/liquidate, which calls flatten_all() directly rather than
going through this module -- that path is instant, not on this hourly
clock). A client with neither threshold configured (the default, and the
only behavior for every client that existed before this feature) is never
touched by check_all_clients_risk.

Two triggers here, two different postures once tripped -- a deliberate
distinction, not an oversight:

  - max_drawdown_pct: "my equity dropped this much from its peak -- get me
    out." This is the client flagging that something is wrong. Tripping it
    flattens the account AND pauses it (trading_paused=TRUE,
    pause_reason='max_drawdown') until the client explicitly clicks Resume
    on the portal (POST /api/portal/resume) -- a drawdown big enough to
    configure a stop for deserves a human look before capital goes back to
    work, not a silent auto-restart next hour.

  - profit_target_pct / profit_target_window_days: "if I'm up this much in
    this many days, lock it in and sit out the rest of the window." This is
    routine profit-taking, not a red flag, so it auto-resumes on its own at
    the start of the next window (handled in _roll_profit_target_windows
    below, same hourly pass) rather than waiting on the client to notice
    and click Resume.

Called from execution/contradiction_monitor.py's hourly, market-hours pass
-- clients aren't re-priced faster than that anywhere else in the system,
so a hidden faster clock here would just mean this module reacts to a price
the rest of the system hasn't seen yet.

Deliberately isolated per client, same posture as execution/client_fanout.py:
one client's bad API key, a stale quote, or a failed flatten must never
stop the rest of the book from being checked.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging

import pandas as pd
from sqlalchemy.engine import Engine

from execution.broker_alpaca import AlpacaBroker
from execution.client_crypto import decrypt_credential

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ClientRiskAction:
    client_id: int
    name: str
    action: str  # "max_drawdown_flatten" | "profit_target_flatten" | "profit_window_rolled" | "error"
    detail: str = ""


def _load_risk_rows(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT id, name, alpaca_api_key_encrypted, alpaca_api_secret_encrypted, "
        "trading_paused, pause_reason, max_drawdown_pct, equity_peak, "
        "profit_target_pct, profit_target_window_days, "
        "profit_target_period_start_equity, profit_target_period_start_ts "
        "FROM clients WHERE active = TRUE",
        engine,
    )


def _broker_for_row(row) -> AlpacaBroker:
    return AlpacaBroker(
        mode="live",
        confirm_live=True,
        api_key=decrypt_credential(row["alpaca_api_key_encrypted"]),
        secret_key=decrypt_credential(row["alpaca_api_secret_encrypted"]),
    )


def _set_profit_period(engine: Engine, client_id: int, equity: float, ts: dt.datetime) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET profit_target_period_start_equity = %s, profit_target_period_start_ts = %s WHERE id = %s",
            (equity, ts, client_id),
        )


def _set_equity_peak(engine: Engine, client_id: int, peak: float) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE clients SET equity_peak = %s WHERE id = %s", (peak, client_id))


def _unpause(engine: Engine, client_id: int) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET trading_paused = FALSE, pause_reason = NULL WHERE id = %s", (client_id,)
        )


def _flatten_and_pause(broker, engine: Engine, client_id: int, name: str, reason: str) -> None:
    try:
        broker.flatten_all()
    except Exception:
        # Still pause even if the flatten call itself failed -- leaving the
        # client tradeable after a tripped threshold, just because Alpaca's
        # call errored, would defeat the whole point of the feature. The
        # error is logged loudly so the operator can flatten manually.
        logger.exception("%s: %s triggered but flatten_all() failed -- pausing anyway, flatten needs a manual retry.", name, reason)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE clients SET trading_paused = TRUE, pause_reason = %s WHERE id = %s", (reason, client_id)
        )
        # symbol='ALL': this is an account-wide event, not a per-symbol fill
        # -- client_orders is reused as the append-only audit trail rather
        # than adding a second table for one more event type.
        conn.exec_driver_sql(
            "INSERT INTO client_orders (client_id, symbol, status) VALUES (%s, %s, %s)",
            (client_id, "ALL", reason),
        )
    logger.warning("%s: %s triggered -- account flattened and paused.", name, reason)


def _roll_profit_target_windows(
    engine: Engine, rows: pd.DataFrame, now: dt.datetime
) -> tuple[pd.DataFrame, list[ClientRiskAction]]:
    """
    First pass, over every client with a profit_target configured,
    regardless of trading_paused: advances a finished window (or
    initializes a never-set one) and auto-resumes a client that was paused
    specifically for having hit last window's target. Returns `rows` with
    the rolled columns updated in place so the second pass (trigger checks,
    below) sees this pass's results without a second DB round-trip, plus one
    ClientRiskAction per client actually auto-resumed here -- the same
    audit-trail treatment every other trigger in this module gets.
    """
    rows = rows.copy()
    actions: list[ClientRiskAction] = []
    for idx, row in rows.iterrows():
        if pd.isna(row["profit_target_pct"]):
            continue
        client_id = int(row["id"])
        window_days = int(row["profit_target_window_days"])
        start_ts = row["profit_target_period_start_ts"]

        if pd.isna(start_ts) or pd.isna(row["profit_target_period_start_equity"]):
            # Never initialized (feature just turned on) -- seed the window
            # with wherever equity is right now rather than guessing a
            # historical baseline. No price is known yet at this point in
            # the pass, so this is finished with a real equity read in the
            # trigger pass below on the FIRST run only; subsequent rows
            # already have a baseline and skip straight to the elapsed check.
            continue

        start_ts = start_ts if start_ts.tzinfo else start_ts.replace(tzinfo=dt.UTC)
        if now - start_ts < dt.timedelta(days=window_days):
            continue  # window still running, nothing to roll yet

        if bool(row["trading_paused"]) and row["pause_reason"] == "profit_target":
            _unpause(engine, client_id)
            rows.at[idx, "trading_paused"] = False
            rows.at[idx, "pause_reason"] = None
            actions.append(ClientRiskAction(client_id, row["name"], "profit_window_rolled", "auto-resumed for the new window"))

        # Window elapsed either way (paused or not) -- the NEXT window's
        # baseline is set once a fresh equity read happens in the trigger
        # pass; mark it here as "needs reseed" by clearing the start fields
        # so that pass's isna() check catches it.
        rows.at[idx, "profit_target_period_start_ts"] = pd.NaT
        rows.at[idx, "profit_target_period_start_equity"] = None

    return rows, actions


def check_all_clients_risk(engine: Engine) -> list[ClientRiskAction]:
    """
    Two passes over every active client: roll/auto-resume profit-target
    windows first, then check both thresholds against a fresh equity read.
    A client that's trading_paused for max_drawdown (or any reason other
    than an elapsed profit_target window) is skipped by the trigger pass
    entirely -- only the client's own Resume click clears that.
    """
    rows = _load_risk_rows(engine)
    if rows.empty:
        return []

    now = dt.datetime.now(tz=dt.UTC)
    rows, actions = _roll_profit_target_windows(engine, rows, now)

    for _, row in rows.iterrows():
        client_id = int(row["id"])
        name = row["name"]
        has_drawdown = pd.notna(row["max_drawdown_pct"])
        has_profit_target = pd.notna(row["profit_target_pct"])
        if not has_drawdown and not has_profit_target:
            continue

        already_paused = bool(row["trading_paused"])
        if already_paused:
            # Still paused after the roll above (max_drawdown, or a
            # profit_target window that hasn't elapsed yet) -- nothing to
            # check until the client resumes or the window elapses.
            continue

        # One broker built per client per pass, reused for both the equity
        # read below and a flatten_all() if a threshold trips -- rather than
        # a fresh connection per call.
        broker = _broker_for_row(row)
        try:
            equity = float(broker.get_account().get("equity", 0) or 0)
        except Exception:
            logger.exception("Could not read %s's account for risk checks this pass -- skipping.", name)
            actions.append(ClientRiskAction(client_id, name, "error", "could not read account"))
            continue
        if equity <= 0:
            continue

        if has_profit_target and pd.isna(row["profit_target_period_start_ts"]):
            # Freshly seeded (either brand new, or just rolled over above).
            _set_profit_period(engine, client_id, equity, now)
        elif has_profit_target:
            baseline = float(row["profit_target_period_start_equity"])
            gain = (equity / baseline) - 1.0 if baseline > 0 else 0.0
            if gain >= float(row["profit_target_pct"]):
                _flatten_and_pause(broker, engine, client_id, name, "profit_target")
                actions.append(ClientRiskAction(client_id, name, "profit_target_flatten", f"+{gain:.2%} >= target"))
                continue  # flattened+paused this pass -- skip the drawdown check below

        if has_drawdown:
            prior_peak = row["equity_peak"]
            peak = equity if pd.isna(prior_peak) else max(float(prior_peak), equity)
            if pd.isna(prior_peak) or peak > float(prior_peak):
                _set_equity_peak(engine, client_id, peak)
            drawdown = (equity / peak) - 1.0 if peak > 0 else 0.0
            if drawdown < -abs(float(row["max_drawdown_pct"])):
                _flatten_and_pause(broker, engine, client_id, name, "max_drawdown")
                actions.append(ClientRiskAction(client_id, name, "max_drawdown_flatten", f"{drawdown:.2%} <= -limit"))

    return actions
