"""
Human-in-the-loop approval bridge between the screener and the broker.

models/screener.py stops at "here are today's candidates" — it writes rows
into `decisions` with executed_position left NULL and never touches
execution/broker.py. This module is the missing link: read the latest
un-executed batch, show it to a human on Telegram, and only act on what
that human explicitly approves. Nothing here decides *what* to trade; the
screener already did that. This decides *whether* an already-made proposal
becomes an order, and records the answer.

STATUS: EXECUTION IS WIRED TO THE ALPACA PAPER ACCOUNT; TELEGRAM CAN SEND BUT
NOT LISTEN. There were two chokepoints where a side effect could leave this
process. One of them is now real:

  1. request_approvals()   — STILL A STUB. Sending to Telegram now works
                             (execution/telegram.py, --notify), but nothing
                             reads replies, so a chat message cannot approve.
  2. submit_paper_order()  — LIVE (paper account). The ONLY place
                             broker.submit_target_position is called.

So the loop cannot yet run unattended end to end: with nothing injected it
stops at the human step, exactly as before. What changed is that once an
approval arrives — from an injected request_fn, or from Telegram when that
stub is filled in — an order really is placed against the Alpaca paper
account, executed_position really is written back, and the equity/breaker
snapshots really are recorded.

Telegram is a one-way notification channel here, deliberately. `--notify`
pushes the pending batch to a phone so a human knows there is something to
look at; the answer still has to be given at the console. Nothing in the
Telegram path parses a reply, so nobody who can write into that chat can cause
a trade.

The CLI has three modes and not one of them can place an order:

  --dry-run        a read-only walkthrough that shows the pending batch as the
                   messages the real loop would send and asks y/n on the
                   console. It never contacts Telegram, never constructs a
                   broker, and never writes to the database, so answering "y"
                   there cannot move anything.
  --telegram-setup prints the chat id of whoever has messaged the bot, for .env.
  --notify         sends the pending batch to that chat as one message.

There is deliberately no executing CLI mode: adding one would mean adding an
unattended call site for real orders.

PAPER MODE ONLY, BY DESIGN, AND MORE SO NOW THAT ORDERS ARE REAL. `MODE` is
hardcoded to "paper" and paper_broker() constructs AlpacaBroker(mode=MODE)
directly — not via get_broker(), which reads settings.broker and so would
let an .env edit redirect where these orders land. confirm_live is never
passed and never read here, so the live guard in broker_alpaca.py cannot be
satisfied from this module no matter what the config says. If live is ever
wanted it belongs in a separate, separately-reviewed call site.

Alpaca paper keys are optional: with none set, the loop says so and returns
without asking anyone to approve orders it could not place.

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
from execution import telegram
from execution.broker_alpaca import AlpacaBroker
from monitoring.breaker_state import check_and_record_breakers
from monitoring.equity import load_equity_curve, record_equity_snapshot

logger = logging.getLogger(__name__)

# Hardcoded on purpose — see the module docstring. Do not make this a
# parameter, a setting, or anything else a typo can flip to "live".
MODE = "paper"

MISSING_PAPER_KEYS_MESSAGE = (
    "Alpaca paper keys not set — add ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY "
    "to .env to enable paper execution.\n"
    "Everything up to the order is already wired; there is just no account to send "
    "it to yet. A paper account is free at https://alpaca.markets (no funding, no "
    "card). Nothing was proposed, approved, or written."
)


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


def format_pick_line(decision: PendingDecision) -> str:
    """
    One candidate as the single line a human reads on a phone.

    Direction lives in the LONG/SHORT word, so the size is shown unsigned —
    "SHORT TSLA — -8.0%" reads like a double negative on a small screen. The
    expected return keeps its sign, since that is the number the human is being
    asked to disagree with.

    Shared by the console dry run and the Telegram notification so the two can
    never drift into describing the same pick differently: whichever one you
    read, you are reading this string.
    """
    return (
        f"🤖 Pulse proposes: {decision.side.upper()} {decision.symbol} — "
        f"{abs(decision.target_position):.1%} of portfolio | "
        f"regime: {decision.regime} | "
        f"expected: {decision.forecast:+.1%}"
    )


# --------------------------------------------------------------------------
# 3. Ask the human — CHOKEPOINT 1 (stub)
# --------------------------------------------------------------------------


def request_approvals(message: str, decisions: list[PendingDecision]) -> dict[int, bool]:
    """
    Send `message` to the Telegram group and block until every candidate has
    an answer, returning {decision_id: approved}.

    STILL A STUB, and now a stub for a narrower reason. The send half exists:
    execution/telegram.py can put this message on a phone, and --notify does.
    What is missing is the half that turns a chat reply into consent — poll
    getUpdates, parse "approve <id>" / "reject <id>" / "approve all" /
    "reject all", ignore replies from any chat but the configured one, and time
    out into all-rejected rather than hanging or defaulting to approval.

    That half is left unbuilt on purpose. Filling it in makes a message from a
    phone sufficient to place an order, which is a security question (who can
    write into that chat?) that deserves its own review rather than arriving as
    a side effect of wiring up notifications. Until then approval happens at
    the console.

    Callers must treat a missing decision_id in the returned mapping as
    rejected — never as approved.
    """
    raise NotImplementedError(
        "Telegram approval read-back is not implemented. Sending works — see "
        "execution.telegram.send_message and --notify — but nothing parses "
        "replies, so no chat message can approve a trade. Approve at the "
        "console with --dry-run. Any implementation here must fail closed: any "
        "timeout, parse failure, or unknown sender means rejected."
    )


# --------------------------------------------------------------------------
# 4. Approved -> shares, and CHOKEPOINT 2 (live against the paper account)
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

    The single, only place broker.submit_target_position is invoked from this
    module. Everything upstream (approval, sizing, rounding) funnels here, so
    this one function is the whole blast radius — which is exactly why it is
    worth keeping as its own function now that it really does place orders.

    The broker handed in here is always paper: run_approval_loop either gets
    one from paper_broker() (mode pinned to MODE) or from a caller that
    injected its own, and no path in this module constructs anything else.

    Logged before the call as well as after, so an order that is submitted
    but never returns still leaves a record that it went out.
    """
    logger.info("Submitting %s -> %s shares (mode=%s).", symbol, target_shares, MODE)
    order = broker.submit_target_position(symbol, target_shares)
    if order is None:
        logger.info("%s already at %s shares — no order needed.", symbol, target_shares)
    return order


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


def paper_keys_present() -> bool:
    """
    Whether there is an Alpaca paper account to talk to at all. Checked
    before anyone is asked to approve anything — proposing trades that
    provably cannot be placed just trains a human to rubber-stamp.
    """
    return bool(settings.alpaca_paper_api_key and settings.alpaca_paper_secret_key)


def paper_broker() -> Broker:
    """
    The only broker constructor in this module. AlpacaBroker directly rather
    than get_broker(): the factory picks its class from settings.broker, which
    would make where these orders land an .env-editable question. mode is
    pinned to MODE and confirm_live is never passed, so the live guard in
    broker_alpaca.py cannot be satisfied from here.
    """
    return AlpacaBroker(mode=MODE)


def run_approval_loop(
    broker: Broker | None = None,
    request_fn: Callable[[str, list[PendingDecision]], dict[int, bool]] = request_approvals,
    submit_fn: Callable[[Broker, str, float], dict | None] = submit_paper_order,
    price_fn: Callable[[list[str]], dict[str, float]] = latest_close_prices,
    correlation_fn: Callable[[list[str]], pd.DataFrame] = recent_correlations,
    engine: Engine | None = None,
    print_fn: Callable[[str], None] = print,
) -> list[ExecutionResult]:
    """
    One full pass: pending batch -> proposal -> human answer -> sized orders
    -> write-back -> monitoring snapshot.

    Both side-effecting steps are injected so this orchestration can be
    exercised in tests. The default request_fn is still the Telegram stub and
    still raises, so calling this with nothing injected stops at the human
    step rather than trading unattended; the default submit_fn now really
    places the order once an approval does arrive.

    Fails closed at every step: a candidate with no price, no answer, or a
    size that rounds to zero shares is not traded. With no Alpaca paper keys
    configured it says so and returns without asking anyone anything.
    """
    decisions = load_pending_decisions(engine=engine)
    if not decisions:
        logger.info("No pending decisions awaiting approval (mode=%s).", MODE)
        return []

    # Checked before the human is asked, not after: an approval this loop
    # cannot act on is worse than no approval at all. Only when the caller
    # didn't bring its own broker — an injected one needs no keys of ours.
    if broker is None and not paper_keys_present():
        logger.warning("Alpaca paper credentials are not configured; skipping the batch.")
        print_fn(MISSING_PAPER_KEYS_MESSAGE)
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

    record_batch_monitoring(broker, price_fn=price_fn, correlation_fn=correlation_fn)
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
# those stay unreachable from here, and no amount of answering "y" can move
# money or mutate a row. That mattered when submit_paper_order was a stub and
# it matters more now that it isn't: this path's inertness comes from what it
# does not call, not from the other functions being harmless.


@dataclasses.dataclass
class DryRunSummary:
    """Tally of one dry-run pass. Counts only — nothing was acted on."""

    total: int = 0
    approved: int = 0
    skipped: int = 0


def format_dry_run_proposal(decision: PendingDecision) -> str:
    """
    The shared pick line plus the console's y/n prompt. The line itself is
    format_pick_line — the same text --notify puts on the phone — so the
    console preview and the real message stay one string, not two.
    """
    return f"{format_pick_line(decision)} — Approve? [y/n] "


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


# --------------------------------------------------------------------------
# Notify — put the pending batch on a phone. SEND ONLY.
# --------------------------------------------------------------------------
#
# This section has the same inertness as the dry run above, and gets it the
# same way: by what it does not call. It shares load_pending_decisions (a
# SELECT) and the formatting helpers, and touches none of request_approvals,
# submit_paper_order, paper_broker, record_execution, or
# record_batch_monitoring. Its only effect that leaves this process is an
# HTTPS POST to api.telegram.org carrying text.
#
# There is deliberately NO read-back. Nothing here looks at replies, so no
# message anyone sends into that chat — including someone who is not the owner
# — can approve, size, or place anything. Sending is a one-way notification;
# approval stays where it already was, at the console in front of a human.
# request_approvals (chokepoint 1) is still a stub for exactly that reason.


NOTIFY_FOOTER = (
    "Replies to this chat are NOT read — this is a notification, not an approval "
    "prompt. Nothing has been sent to a broker.\n"
    "Review and answer at the console: python -m execution.approval_loop --dry-run"
)


def format_notification(decisions: list[PendingDecision]) -> str:
    """
    The whole pending batch as ONE plain-text message.

    One message rather than one per pick: a phone that buzzes eight times for
    eight candidates trains its owner to swipe the notification away, which is
    the opposite of what a human-in-the-loop step is for.

    Plain text, no markdown — the pick lines are full of %, +, - and | that a
    Telegram parse_mode would either mangle or reject, and the same string has
    to stay readable in a log file and a test failure.
    """
    if not decisions:
        return "No pending trade proposals."

    header = (
        f"Pulse — {len(decisions)} pending proposal(s) | "
        f"batch {_format_ts(decisions[0].ts)} | {MODE} mode"
    )
    body = "\n".join(format_pick_line(d) for d in decisions)
    return f"{header}\n\n{body}\n\n{NOTIFY_FOOTER}"


def run_telegram_setup(
    print_fn: Callable[[str], None] = print,
    fetch_fn: Callable[..., list[dict]] = telegram.fetch_updates,
) -> int:
    """
    One-time helper: find the chat id of whoever has messaged the bot.

    A bot cannot start a conversation on Telegram — the human has to message it
    first, and the chat id only becomes discoverable once they have. So this
    reads getUpdates and prints what it finds; it never sends anything, and it
    passes no offset, so it does not consume the updates it looked at.
    """
    token, configured_chat_id = telegram.credentials()
    if not token:
        print_fn(telegram.MISSING_TOKEN_MESSAGE)
        return 0

    try:
        updates = fetch_fn(token)
    except telegram.TelegramError as exc:
        print_fn(
            f"Telegram would not answer getUpdates: {exc}\n"
            "If that says Unauthorized, the token in .env is wrong or has been revoked — "
            "ask @BotFather for it again with /mytoken."
        )
        return 1

    chats = telegram.chat_candidates(updates)
    if not chats:
        print_fn(
            "No messages found for this bot yet.\n"
            "Open Telegram, find your bot, send it any message (\"hi\" is fine), then run "
            "this command again.\n"
            "Telegram only keeps undelivered updates for about 24 hours, so a conversation "
            "from days ago will not show up here."
        )
        return 0

    print_fn(f"Found {len(chats)} chat(s) that have messaged this bot (most recent first):\n")
    for chat in chats:
        print_fn(f"  chat id {chat['chat_id']}   {chat['name']} ({chat['kind']})")
        if chat["text"]:
            print_fn(f"      last message: {chat['text'][:80]}")
    print_fn("")

    best = chats[0]["chat_id"]
    if configured_chat_id == best:
        print_fn(f"TELEGRAM_CHAT_ID={best} is already what .env says — nothing to change.")
    else:
        if len(chats) > 1:
            print_fn("More than one chat showed up — pick the one that is your own conversation.")
        print_fn("Put this line in .env:\n")
        print_fn(f"  TELEGRAM_CHAT_ID={best}\n")
        if configured_chat_id:
            print_fn(f"(.env currently has TELEGRAM_CHAT_ID={configured_chat_id}.)")
    print_fn("Then: python -m execution.approval_loop --notify")
    return 0


def run_notify(
    print_fn: Callable[[str], None] = print,
    send_fn: Callable[..., dict] = telegram.send_message,
    engine: Engine | None = None,
) -> int:
    """
    Send the latest pending batch to the configured chat as one message.

    Prints the same message locally before sending it, so what landed on the
    phone can be checked against what left the machine.

    Missing credentials are a clean exit, not an error: this is a notification
    channel, and a machine with no phone attached to it should carry on quietly
    rather than fail a scheduled run.
    """
    token, chat_id = telegram.credentials()
    if not token:
        print_fn(telegram.MISSING_TOKEN_MESSAGE)
        return 0
    if not chat_id:
        print_fn(telegram.MISSING_CHAT_ID_MESSAGE)
        return 0

    decisions = load_pending_decisions(engine=engine)
    if not decisions:
        print_fn("No pending trade proposals — nothing to send.")
        return 0

    message = format_notification(decisions)
    print_fn(f"Sending this to Telegram chat {chat_id}:\n")
    print_fn(message)

    try:
        send_fn(message, token=token, chat_id=chat_id)
    except telegram.TelegramError as exc:
        print_fn(
            f"\nTelegram refused the message: {exc}\n"
            "If that says chat not found, the chat id is wrong — rerun "
            "--telegram-setup. Nothing was executed either way."
        )
        return 1

    print_fn(
        f"\nSent — {len(decisions)} proposal(s) delivered to chat {chat_id}. "
        "No order was placed and no row was changed; this was a message only."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint. Exactly one mode must be chosen, and not one of them can
    reach a broker: --dry-run reads and asks on the console, --telegram-setup
    reads getUpdates, --notify reads the pending batch and sends text. There is
    still no flag, env var, or argument here that calls submit_paper_order,
    which is the whole thing this module is arranged to make hard.
    """
    parser = argparse.ArgumentParser(
        prog="python -m execution.approval_loop",
        description=(
            "Preview the pending trade proposals, or push them to Telegram. "
            "None of these modes can place an order."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each pending proposal and ask y/n, without executing or writing anything.",
    )
    modes.add_argument(
        "--telegram-setup",
        action="store_true",
        help="Print the chat id of whoever has messaged the bot, to put in .env as TELEGRAM_CHAT_ID.",
    )
    modes.add_argument(
        "--notify",
        action="store_true",
        help="Send the latest pending proposals to the configured Telegram chat as one message. Send only.",
    )
    args = parser.parse_args(argv)

    if args.telegram_setup:
        return run_telegram_setup()
    if args.notify:
        return run_notify()

    run_dry_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
