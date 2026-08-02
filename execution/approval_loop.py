"""
Human-in-the-loop approval bridge between the screener and the broker.

models/screener.py stops at "here are today's candidates" — it writes rows
into `decisions` with executed_position left NULL and never touches
execution/broker.py. This module is the missing link: read the latest
un-executed batch, show it to a human on Telegram, and only act on what
that human explicitly approves. Nothing here decides *what* to trade; the
screener already did that. This decides *whether* an already-made proposal
becomes an order, and records the answer.

STATUS: REVIEWABLE SKELETON — EXECUTION IS INTENTIONALLY STUBBED.
All the structure is real, working code: the DB read, the message
formatting, the fraction-of-portfolio to whole-shares conversion, the
executed_position write-back, and the post-batch monitoring calls. The two
functions that would cause a real side effect outside this process are
deliberately `raise NotImplementedError`:

  1. request_approvals()   — would talk to the Telegram API (send + poll).
  2. submit_paper_order()  — the ONLY place broker.submit_target_position
                             is ever called.

Every path routes through those two chokepoints, so turning this from a
skeleton into a working loop is a two-function job with no restructuring.
Nothing in this module is runnable end to end today.

The one runnable entrypoint is `python -m execution.approval_loop --dry-run`:
a read-only walkthrough that shows the pending batch as the messages the real
loop would send and asks y/n on the console. It exists to demonstrate the
shape of the loop without any of its side effects, so it routes *around* the
two stubs rather than un-stubbing them — it never contacts Telegram, never
constructs a broker, and never writes to the database. There is no non-dry-run
CLI mode, by design: the flag is required and there is nothing else to pass.

PAPER MODE ONLY, BY DESIGN. `MODE` is hardcoded to "paper" and the broker
is constructed via get_broker(mode=MODE) — confirm_live is never passed,
so the live-broker guard in broker_alpaca.py / broker_ibkr.py can never be
satisfied from this module. Live trading must stay impossible here even
after the two stubs are filled in; if live is ever wanted it belongs in a
separate, separately-reviewed call site.

Rejections are recorded too: a rejected candidate gets executed_position
written as 0.0 rather than left NULL, so it drops out of the "pending"
query instead of being re-proposed to the human every single run.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import sys
from collections.abc import Callable
from typing import Protocol

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from config.settings import settings
from data.ingest.db import get_engine
from execution.broker import get_broker
from monitoring.breaker_state import check_and_record_breakers
from monitoring.equity import load_equity_curve, record_equity_snapshot

logger = logging.getLogger(__name__)

# Hardcoded on purpose — see the module docstring. Do not make this a
# parameter, a setting, or anything else a typo can flip to "live".
MODE = "paper"


class Broker(Protocol):
    """The slice of execution/broker_*.py this module actually depends on."""

    def get_portfolio_value(self) -> float: ...

    def get_positions(self) -> dict[str, float]: ...

    def submit_target_position(self, symbol: str, target_shares: float) -> dict | None: ...


@dataclasses.dataclass
class PendingDecision:
    """One un-executed row from the `decisions` table, awaiting a human."""

    decision_id: int
    ts: dt.datetime
    symbol: str
    forecast: float
    regime: str
    target_position: float  # fraction of portfolio, signed (negative = short)

    @property
    def side(self) -> str:
        return "long" if self.target_position >= 0 else "short"


@dataclasses.dataclass
class ExecutionResult:
    """What happened to one candidate after the human answered."""

    decision: PendingDecision
    approved: bool
    target_shares: float = 0.0
    executed_position: float | None = None  # written back to the decisions row
    note: str = ""


# --------------------------------------------------------------------------
# 1. Read the latest un-executed batch
# --------------------------------------------------------------------------

_PENDING_SQL = text(
    """
    SELECT id, ts, symbol, forecast, regime, target_position
    FROM decisions
    WHERE executed_position IS NULL
      AND mode = :mode
      AND ts = (
          SELECT MAX(ts) FROM decisions
          WHERE executed_position IS NULL AND mode = :mode
      )
    ORDER BY ABS(target_position) DESC, symbol
    """
)


def load_pending_decisions(engine: Engine | None = None) -> list[PendingDecision]:
    """
    The most recent batch of proposals the screener logged and nobody has
    acted on yet. Deliberately one batch only: if yesterday's run was never
    approved, those picks are stale and should not be silently mixed into
    today's proposal — they'd be sized off today's portfolio value from a
    forecast made against different prices.
    """
    engine = engine or get_engine()
    df = pd.read_sql(_PENDING_SQL, engine, params={"mode": MODE})
    return [
        PendingDecision(
            decision_id=int(row["id"]),
            ts=row["ts"],
            symbol=str(row["symbol"]),
            forecast=float(row["forecast"]),
            regime=str(row["regime"]),
            target_position=float(row["target_position"]),
        )
        for _, row in df.iterrows()
    ]


# --------------------------------------------------------------------------
# 2. Format the proposal a human has to read on a phone
# --------------------------------------------------------------------------


def format_candidate(index: int, decision: PendingDecision) -> str:
    """One candidate, two lines. Numbered so a reply can say "approve 2"."""
    return (
        f"{index}. {decision.symbol} {decision.side.upper()} "
        f"target {decision.target_position:+.2%} of portfolio\n"
        f"   forecast {decision.forecast:+.2%} | regime {decision.regime} | id {decision.decision_id}"
    )


def format_proposal(decisions: list[PendingDecision]) -> str:
    """
    The whole batch as one plain-text message. Kept plain (no Telegram
    markdown) so the same string is readable in a log file, a test failure,
    or a chat window without escaping rules leaking into this module.
    """
    if not decisions:
        return "No pending trade proposals."

    batch_ts = decisions[0].ts
    header = (
        f"Trade proposals — batch {_format_ts(batch_ts)} | {MODE} mode | "
        f"{len(decisions)} candidate(s)"
    )
    body = "\n".join(format_candidate(i, d) for i, d in enumerate(decisions, start=1))
    footer = (
        'Reply "approve <id>" or "reject <id>" for each candidate '
        '("approve all" / "reject all" also accepted).\n'
        "Anything not explicitly approved is treated as rejected. "
        f"{MODE.capitalize()} mode only — this loop cannot reach a live account."
    )
    return f"{header}\n\n{body}\n\n{footer}"


def _format_ts(ts) -> str:
    """Batch timestamps are tz-aware UTC coming out of the DB; keep it obvious."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------
# 3. Ask the human — CHOKEPOINT 1 (stub)
# --------------------------------------------------------------------------


def request_approvals(message: str, decisions: list[PendingDecision]) -> dict[int, bool]:
    """
    Send `message` to the Telegram group and block until every candidate has
    an answer, returning {decision_id: approved}.

    STUB — the only function in this repo that would touch the Telegram API.
    A real implementation would: POST the message via sendMessage with
    TELEGRAM_BOT_TOKEN to TELEGRAM_CHAT_ID, poll getUpdates for replies,
    parse "approve <id>" / "reject <id>" / "approve all" / "reject all",
    ignore replies from unknown chats, and time out into all-rejected rather
    than hanging or defaulting to approval.

    Callers must treat a missing decision_id in the returned mapping as
    rejected — never as approved.
    """
    raise NotImplementedError(
        "Telegram approval transport is not implemented — this module is a "
        "reviewable skeleton. Implement request_approvals() (send + poll via "
        "settings.telegram_bot_token / settings.telegram_chat_id) to enable it. "
        "It must fail closed: any timeout, parse failure, or unknown sender "
        "means rejected."
    )


# --------------------------------------------------------------------------
# 4. Approved -> shares, and CHOKEPOINT 2 (stub)
# --------------------------------------------------------------------------


def target_shares_for(target_position: float, portfolio_value: float, price: float) -> float:
    """
    Fraction of portfolio -> whole shares. Rounds toward zero so rounding can
    only ever make a position smaller than the human approved, never larger.
    Both broker backends round to whole shares at submission anyway; doing it
    here means the number written to executed_position matches what was
    actually asked for.
    """
    if price <= 0 or portfolio_value <= 0:
        return 0.0
    return float(int(target_position * portfolio_value / price))


def executed_fraction(target_shares: float, price: float, portfolio_value: float) -> float:
    """
    The share count expressed back as a signed fraction of the portfolio, so
    executed_position is directly comparable with target_position in the same
    row (they differ by the whole-share rounding above).
    """
    if portfolio_value <= 0:
        return 0.0
    return target_shares * price / portfolio_value


def submit_paper_order(broker: Broker, symbol: str, target_shares: float) -> dict | None:
    """
    Move `symbol` to `target_shares` at the paper broker.

    STUB — the single, only place broker.submit_target_position is invoked
    from this module. Everything upstream (approval, sizing, rounding)
    funnels here, so this one function is the whole blast radius: while it
    raises, no order can leave the process no matter what the rest of the
    loop does.

    A real implementation is one line — `return
    broker.submit_target_position(symbol, target_shares)` — plus whatever
    logging and order-status checking you want (note IBKR rejects
    unshortable orders at submission time, see broker_ibkr.py's docstring).
    The broker handed in here is always paper; see MODE.
    """
    raise NotImplementedError(
        "Order submission is not implemented — this module is a reviewable "
        f"skeleton. Would have submitted: {symbol} -> {target_shares} shares "
        f"(mode={MODE}). Implement submit_paper_order() to enable execution."
    )


# --------------------------------------------------------------------------
# 5. Write the outcome back to the decision row
# --------------------------------------------------------------------------

_WRITE_BACK_SQL = text(
    "UPDATE decisions SET executed_position = :executed_position "
    "WHERE id = :decision_id AND mode = :mode"
)


def record_execution(decision_id: int, executed_position: float, engine: Engine | None = None) -> None:
    """
    Stamp what actually happened onto the decision row. Also called with 0.0
    for rejected and for rounds-to-nothing candidates — a decided-and-declined
    row must not look identical to a never-reviewed one, or the pending query
    keeps re-proposing it.
    """
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            _WRITE_BACK_SQL,
            {"executed_position": float(executed_position), "decision_id": int(decision_id), "mode": MODE},
        )


# --------------------------------------------------------------------------
# Prices and correlations (real reads, used for sizing and the breakers)
# --------------------------------------------------------------------------

_LATEST_PRICE_SQL = text(
    """
    SELECT DISTINCT ON (symbol) symbol, close
    FROM prices
    WHERE symbol IN :symbols
    ORDER BY symbol, ts DESC
    """
).bindparams(bindparam("symbols", expanding=True))


def latest_close_prices(symbols: list[str], engine: Engine | None = None) -> dict[str, float]:
    """Most recent stored close per symbol — what fraction-of-portfolio gets divided by."""
    if not symbols:
        return {}
    engine = engine or get_engine()
    df = pd.read_sql(_LATEST_PRICE_SQL, engine, params={"symbols": list(symbols)})
    return {str(row["symbol"]): float(row["close"]) for _, row in df.iterrows()}


_RECENT_CLOSES_SQL = text(
    "SELECT symbol, ts, close FROM prices WHERE symbol IN :symbols ORDER BY ts"
).bindparams(bindparam("symbols", expanding=True))


def recent_correlations(
    symbols: list[str], lookback_days: int = 60, engine: Engine | None = None
) -> pd.DataFrame:
    """
    Trailing daily-return correlation matrix for the symbols currently held —
    what check_and_record_breakers needs to spot a cluster of positions that
    all move together. Same construction as
    models.screener.build_correlation_matrix, read straight from `prices` here
    so this module doesn't drag the whole model stack in as an import.
    """
    if not symbols:
        return pd.DataFrame()
    engine = engine or get_engine()
    df = pd.read_sql(_RECENT_CLOSES_SQL, engine, params={"symbols": list(symbols)})
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    return pivot.pct_change().tail(lookback_days).corr()


# --------------------------------------------------------------------------
# 6. Post-batch monitoring — the calls that finally feed the dashboard
# --------------------------------------------------------------------------


def record_batch_monitoring(
    broker: Broker,
    price_fn: Callable[[list[str]], dict[str, float]] = latest_close_prices,
    correlation_fn: Callable[[list[str]], pd.DataFrame] = recent_correlations,
) -> list:
    """
    Snapshot portfolio value and run the circuit breakers once per batch.
    monitoring/equity.py and monitoring/breaker_state.py have had zero
    callers until now — this is the caller. Returns the triggered breakers
    (empty list = all clear); acting on a trigger is the caller's business,
    this module only ever records.
    """
    portfolio_value = broker.get_portfolio_value()
    record_equity_snapshot(portfolio_value, mode=MODE)

    positions = broker.get_positions()
    prices = price_fn(list(positions))
    position_values = {s: shares * prices.get(s, 0.0) for s, shares in positions.items()}

    equity_curve = load_equity_curve(mode=MODE)["equity_value"].tolist()

    return check_and_record_breakers(
        equity_curve=equity_curve,
        positions_by_symbol=position_values,
        portfolio_value=portfolio_value,
        correlation_matrix=correlation_fn(list(positions)),
        max_drawdown_pct=settings.max_drawdown_pct,
        max_single_position_pct=settings.max_single_position_pct,
        max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def paper_broker() -> Broker:
    """
    The only broker constructor in this module. mode is pinned to MODE and
    confirm_live is never passed, so the live guard in the broker classes
    cannot be satisfied from here.
    """
    return get_broker(mode=MODE)


def run_approval_loop(
    broker: Broker | None = None,
    request_fn: Callable[[str, list[PendingDecision]], dict[int, bool]] = request_approvals,
    submit_fn: Callable[[Broker, str, float], dict | None] = submit_paper_order,
    price_fn: Callable[[list[str]], dict[str, float]] = latest_close_prices,
    engine: Engine | None = None,
) -> list[ExecutionResult]:
    """
    One full pass: pending batch -> proposal -> human answer -> sized orders
    -> write-back -> monitoring snapshot.

    The two side-effecting steps are injected (defaulting to the stubs) so
    this orchestration can be exercised in tests without either stub being
    reachable by accident in production — the defaults still raise.

    Fails closed at every step: a candidate with no price, no answer, or a
    size that rounds to zero shares is not traded.
    """
    decisions = load_pending_decisions(engine=engine)
    if not decisions:
        logger.info("No pending decisions awaiting approval (mode=%s).", MODE)
        return []

    approvals = request_fn(format_proposal(decisions), decisions)

    broker = broker if broker is not None else paper_broker()
    portfolio_value = broker.get_portfolio_value()
    prices = price_fn([d.symbol for d in decisions])

    results: list[ExecutionResult] = [
        _handle_decision(
            decision,
            # .get default False: a candidate the human never answered for is
            # rejected, never approved. Fail closed.
            approved=bool(approvals.get(decision.decision_id, False)),
            broker=broker,
            portfolio_value=portfolio_value,
            price=prices.get(decision.symbol),
            submit_fn=submit_fn,
            engine=engine,
        )
        for decision in decisions
    ]

    record_batch_monitoring(broker, price_fn=price_fn)
    return results


def _handle_decision(
    decision: PendingDecision,
    approved: bool,
    broker: Broker,
    portfolio_value: float,
    price: float | None,
    submit_fn: Callable[[Broker, str, float], dict | None],
    engine: Engine | None,
) -> ExecutionResult:
    """One candidate's fate, including its write-back. Split out to keep run_approval_loop readable."""
    if not approved:
        logger.info("Decision %s (%s) rejected by human.", decision.decision_id, decision.symbol)
        record_execution(decision.decision_id, 0.0, engine=engine)
        return ExecutionResult(decision, approved=False, executed_position=0.0, note="rejected")

    if price is None or price <= 0:
        # No usable price means no defensible share count — leave the row
        # pending rather than guessing, so it can be re-proposed once the
        # price data catches up.
        logger.warning("No price for %s — skipping decision %s.", decision.symbol, decision.decision_id)
        return ExecutionResult(decision, approved=True, note="skipped: no price available")

    shares = target_shares_for(decision.target_position, portfolio_value, price)
    if shares == 0:
        logger.info(
            "Decision %s (%s) approved but rounds to 0 shares at %.2f.",
            decision.decision_id, decision.symbol, price,
        )
        record_execution(decision.decision_id, 0.0, engine=engine)
        return ExecutionResult(decision, approved=True, executed_position=0.0, note="rounds to zero shares")

    submit_fn(broker, decision.symbol, shares)

    executed = executed_fraction(shares, price, portfolio_value)
    record_execution(decision.decision_id, executed, engine=engine)
    return ExecutionResult(
        decision, approved=True, target_shares=shares, executed_position=executed, note="submitted"
    )


# --------------------------------------------------------------------------
# Dry run — the loop's shape, none of its side effects
# --------------------------------------------------------------------------
#
# Everything below is deliberately isolated from run_approval_loop. It shares
# only load_pending_decisions (a SELECT) and the PendingDecision dataclass.
# It does not call request_approvals, submit_paper_order, paper_broker,
# record_execution, or record_batch_monitoring — so it stays runnable while
# those remain stubs, and no amount of answering "y" can move money or mutate
# a row. The stubs are routed around, never un-stubbed.


@dataclasses.dataclass
class DryRunSummary:
    """Tally of one dry-run pass. Counts only — nothing was acted on."""

    total: int = 0
    approved: int = 0
    skipped: int = 0


def format_dry_run_proposal(decision: PendingDecision) -> str:
    """
    One candidate as the single-line message the real loop would push to a
    phone. Direction lives in the LONG/SHORT word, so the size is shown
    unsigned — "SHORT TSLA — -8.0%" reads like a double negative on a small
    screen. The expected return keeps its sign, since that is the number the
    human is being asked to disagree with.
    """
    return (
        f"🤖 Pulse proposes: {decision.side.upper()} {decision.symbol} — "
        f"{abs(decision.target_position):.1%} of portfolio | "
        f"regime: {decision.regime} | "
        f"expected: {decision.forecast:+.1%} — Approve? [y/n] "
    )


def _read_approval(prompt: str, input_fn: Callable[[str], str]) -> bool:
    """
    y/yes approves; everything else — including a blank line, a typo, or the
    console closing out from under us — is a skip. Same fail-closed rule the
    real Telegram path is specified to follow: silence is never consent.
    """
    try:
        answer = input_fn(prompt)
    except EOFError:
        return False
    # lstrip the BOM: PowerShell prepends one when answers are piped in rather
    # than typed, which would otherwise turn the first "y" into a silent skip.
    return str(answer).lstrip("﻿").strip().lower() in ("y", "yes")


def run_dry_run(
    decisions: list[PendingDecision] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    engine: Engine | None = None,
) -> DryRunSummary:
    """
    Walk the latest un-executed batch on the console: one proposal message per
    candidate, a y/n answer, and a tally at the end. Read-only from start to
    finish — the DB read is the only thing that leaves this process.

    input_fn/print_fn are injected so tests can drive it without a terminal.
    """
    decisions = load_pending_decisions(engine=engine) if decisions is None else decisions

    print_fn(f"DRY RUN — {MODE} mode, nothing will be executed and nothing will be written.\n")

    if not decisions:
        print_fn("No pending trade proposals.")
        return DryRunSummary()

    summary = DryRunSummary(total=len(decisions))
    for decision in decisions:
        if _read_approval(format_dry_run_proposal(decision), input_fn):
            summary.approved += 1
            print_fn(f"✅ approved — would execute in {MODE.upper()} mode (broker not connected)")
        else:
            summary.skipped += 1
            print_fn("⏭️ skipped")

    print_fn(
        f"\n{summary.approved} approved, {summary.skipped} skipped, {summary.total} total. "
        "Nothing was executed and no rows were changed — approvals here are a preview only."
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint. --dry-run is required and is the only mode: there is no
    flag, env var, or argument here that reaches a broker, and adding one
    would mean adding a call site for submit_paper_order, which is the whole
    thing this module is arranged to make hard.
    """
    parser = argparse.ArgumentParser(
        prog="python -m execution.approval_loop",
        description=(
            "Preview the pending trade proposals on the console. Dry run only — "
            "this command cannot place an order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required. Print each pending proposal and ask y/n, without executing or writing anything.",
    )
    parser.parse_args(argv)

    run_dry_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
