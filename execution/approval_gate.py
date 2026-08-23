"""
The human step between "the engine wants to trade" and "an order goes out".

execution/trading_loop.py and execution/contradiction_monitor.py both know
how to decide and how to execute; this module is the pause in between: send
the batch of proposed opens/closes to a phone as one numbered Telegram
message, poll for "approve 1" / "reject 2" / "approve all" replies, and
hand back exactly which proposals a human said yes to.

Numbering is BATCH-LOCAL on purpose. At this seam no decisions rows exist
yet (they are logged after execution), so there are no database ids to
refer to. Each message numbers its proposals 1..N — closes first, then
opens — and replies refer to THIS message only. A stale "approve 3" about
last week's batch cannot hit this week's picks, because every request
starts its own numbering and ignores updates from before it was sent.

Fail-closed everywhere:
  - silence is never consent — an unanswered proposal is rejected at
    timeout (closes can be flipped to approve-on-timeout via
    APPROVAL_TIMEOUT_CLOSE_ACTION, a deliberate, documented exception:
    a close is risk-reducing);
  - Telegram unconfigured in telegram mode rejects everything and alerts,
    rather than trading unattended;
  - replies are only read from the configured chat (telegram.replies_from
    drops the rest), so a stranger who finds the bot cannot approve;
  - the getUpdates poll runs under a Postgres advisory lock, because the
    weekly cycle and the hourly contradiction monitor share one bot and
    getUpdates is single-consumer — if another gate holds the lock, this
    one rejects its whole batch and alerts instead of stealing replies.

APPROVAL_MODE=auto approves everything without touching the network — the
documented escape hatch back to unattended behavior.

Nothing in this module knows what a broker is. It returns an
ApprovalOutcome; acting on it is the caller's business.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import time
from collections.abc import Callable

from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine
from execution import telegram
from execution.exit_levels import ExitLevels
from execution.exit_levels import describe as describe_levels
from monitoring.alerts import send_slack_alert

logger = logging.getLogger(__name__)

# One fixed key for "someone is polling getUpdates for approvals" — the
# weekly cycle and the hourly monitor must never poll concurrently.
APPROVAL_LOCK_KEY = 903217

# Per-proposal outcome labels, written to decisions.approval_status.
APPROVED = "approved"
REJECTED = "rejected"
TIMEOUT = "timeout"
AUTO = "auto"

_REASON_LABELS = {
    "screen": "weekly screen pick",
    "out_of_book": "no longer in this cycle's book",
    "exit_rule": "hold-rule exit condition fired",
    "contradiction": "signals now contradict this position",
    "reactivation": "redeploying freed capital",
}


@dataclasses.dataclass
class ProposedTrade:
    """One open or close the engine wants a human to sign off on."""

    index: int  # batch-local 1..N, assigned by request_approval
    symbol: str
    action: str  # "open" | "close"
    side: str  # "long" | "short"
    # Approve-first: opens are proposed WITHOUT a size (None) — the human
    # decides which trades happen, then the caller allocates capital across
    # the approved subset and confirms the final sizes in a follow-up
    # message (send_followup). 0.0 for closes.
    target_position_pct: float | None = None
    predicted_return: float | None = None
    reason: str = "screen"  # screen | out_of_book | contradiction | reactivation
    # The screener/monitor's plain-English reasoning phases (see
    # monitoring/reasoning.py) — the same explanations Slack and the
    # dashboard already get. A human asked to approve a trade on a phone
    # deserves the "why", not just the ticker and the size.
    reasoning: list[dict] | None = None
    # For closes: how the position is doing right now, so "close this" is a
    # decision about a known P&L, not a mystery. None when unavailable.
    current_pnl_pct: float | None = None
    current_pnl_usd: float | None = None
    # The take-profit/stop-loss pair this pick will be held to, shown at
    # approval time. Approving a trade without knowing where it exits is
    # approving half a decision, and these are per-pick rather than the one
    # global pair, so they are not something a reader already knows.
    exit_levels: ExitLevels | None = None


@dataclasses.dataclass
class ApprovalOutcome:
    """What the human (or the policy standing in for one) decided."""

    approved: list[ProposedTrade]
    rejected: list[ProposedTrade]
    status: str  # "replied" | "timeout" | "unconfigured" | "auto" | "lock_busy" | "error" | "empty"
    # index -> APPROVED/REJECTED/TIMEOUT/AUTO, for decisions.approval_status.
    statuses: dict[int, str] = dataclasses.field(default_factory=dict)

    def approved_opens(self) -> list[ProposedTrade]:
        return [p for p in self.approved if p.action == "open"]

    def approved_closes(self) -> list[ProposedTrade]:
        return [p for p in self.approved if p.action == "close"]


# --------------------------------------------------------------------------
# Reply grammar — lifted from the original approval loop, generalized to
# plain known ids so it works on batch-local numbering.
# --------------------------------------------------------------------------

APPROVE_WORDS = frozenset({"approve", "approved", "approves", "yes", "ok", "okay", "accept"})
REJECT_WORDS = frozenset({"reject", "rejected", "rejects", "no", "skip", "deny", "decline"})


@dataclasses.dataclass
class ParsedReply:
    """What one chat message was understood to mean, if anything."""

    raw: str
    action: str | None = None  # "approve" | "reject" | None
    ids: list[int] = dataclasses.field(default_factory=list)
    unknown_ids: list[int] = dataclasses.field(default_factory=list)
    targets_all: bool = False

    @property
    def understood(self) -> bool:
        """A verb alone is not an instruction — ambiguity is never a yes."""
        return self.action is not None and bool(self.ids or self.unknown_ids or self.targets_all)

    @property
    def approves(self) -> bool:
        return self.action == "approve"


def parse_reply(text: str, known_ids: set[int]) -> ParsedReply:
    """
    Turn a chat message into an intent against this batch's numbers.

    Grammar, deliberately small: a verb, then either "all" or one or more
    numbers. Commas, extra spaces, "#" prefixes, a leading slash, and mixed
    case are all tolerated; anything else is ignored rather than guessed at.
    Fails closed by construction — an unrecognised message parses to
    .understood == False, and a caller that mishandles it approves nothing.
    """
    raw = str(text or "").strip()
    tokens = [t for t in re.split(r"[\s,]+", raw.lower()) if t]
    if not tokens:
        return ParsedReply(raw=raw)

    verb = tokens[0].lstrip("/")
    if verb in APPROVE_WORDS:
        action = "approve"
    elif verb in REJECT_WORDS:
        action = "reject"
    else:
        return ParsedReply(raw=raw)

    rest = tokens[1:]
    if "all" in rest:
        return ParsedReply(raw=raw, action=action, ids=sorted(known_ids), targets_all=True)

    found, unknown = [], []
    for token in rest:
        if not re.fullmatch(r"#?\d+", token):
            continue  # filler like "please" is ignored, not treated as a number
        number = int(token.lstrip("#"))
        (found if number in known_ids else unknown).append(number)

    return ParsedReply(raw=raw, action=action, ids=found, unknown_ids=unknown)


# --------------------------------------------------------------------------
# Formatting — one plain-text message a human reads on a phone
# --------------------------------------------------------------------------


def number_proposals(proposals: list[ProposedTrade]) -> list[ProposedTrade]:
    """Closes first, then opens, numbered 1..N. Order within each group is preserved."""
    ordered = [p for p in proposals if p.action == "close"] + [p for p in proposals if p.action != "close"]
    for i, proposal in enumerate(ordered, start=1):
        proposal.index = i
    return ordered


# A why-line longer than this stops being phone-readable — trimmed, because
# the reasoning phases were written for a dashboard, not a lock screen.
MAX_WHY_CHARS = 200


def short_why(p: ProposedTrade) -> str | None:
    """
    One phone-sized line of "why", built from the proposal's reasoning
    phases: the signals/forecast summaries (phases 2-3) when present,
    falling back to the selection story (phase 4). None if there is no
    reasoning at all — the line is simply omitted, never invented.
    """
    if not p.reasoning:
        return None
    summaries = [ph.get("summary", "") for ph in p.reasoning if ph.get("phase") in (2, 3) and ph.get("summary")]
    if not summaries:
        summaries = [ph.get("summary", "") for ph in p.reasoning if ph.get("summary")][:1]
    if not summaries:
        return None
    why = " ".join(summaries)
    if len(why) > MAX_WHY_CHARS:
        why = why[: MAX_WHY_CHARS - 1].rstrip() + "…"
    return why


def format_proposal_line(p: ProposedTrade) -> str:
    """
    One proposal, numbered so a reply can say "approve 2". First line is
    the decision (action, symbol, size, and for closes the current P&L);
    an indented second line says why, when the engine attached a reason.
    """
    label = _REASON_LABELS.get(p.reason, p.reason)
    if p.action == "close":
        line = f"{p.index}. CLOSE {p.side.upper()} {p.symbol} — {label}"
        if p.current_pnl_pct is not None:
            pnl = f" | P&L {p.current_pnl_pct:+.1%}"
            if p.current_pnl_usd is not None:
                pnl += f" (${p.current_pnl_usd:+,.0f})"
            line += pnl
    else:
        # No size on purpose: sizing happens AFTER approval, across only the
        # approved subset (the follow-up confirmation carries the numbers).
        line = f"{p.index}. OPEN {p.side.upper()} {p.symbol} — {label}"
        if p.predicted_return is not None:
            line += f" | expected {p.predicted_return:+.1%}"
        if p.exit_levels is not None:
            line += f"\n   Exits: {describe_levels(p.exit_levels)}"

    why = short_why(p)
    if why:
        line += f"\n   {why}"
    return line


def format_proposal_message(proposals: list[ProposedTrade], context: str, timeout_s: int) -> str:
    """
    The whole batch as ONE plain-text message (no Telegram markdown — the
    lines are full of %, +, - and | that parse modes mangle). One message,
    not one per pick: a phone that buzzes eight times trains its owner to
    swipe notifications away.
    """
    header = f"Trade proposals — {context} | paper mode | {len(proposals)} trade(s)"
    body = "\n".join(format_proposal_line(p) for p in proposals)
    minutes = max(1, round(timeout_s / 60))
    footer = (
        'Reply "approve 1", "reject 2 3", or "approve all" / "reject all".\n'
        "Sizes are decided after you answer: the approved picks share the "
        "deployable capital, weighted by conviction, and the final sizes are "
        "confirmed in a follow-up message.\n"
        "Numbers refer to THIS message only.\n"
        f"Anything unanswered in {minutes} min is treated as rejected. "
        "Paper account — no real money moves."
    )
    return f"{header}\n\n{body}\n\n{footer}"


def format_ack_message(outcome: ApprovalOutcome, proposals: list[ProposedTrade]) -> str:
    """Close the loop on the phone: say exactly what will and won't happen."""
    by_index = {p.index: p for p in proposals}
    lines = []
    for index in sorted(by_index):
        p = by_index[index]
        status = outcome.statuses.get(index, REJECTED)
        acted = p in outcome.approved
        if status == TIMEOUT:
            verdict = "no reply — closing anyway (timeout policy)" if acted else "no reply — rejected"
        else:
            verdict = "approved" if acted else "rejected"
        lines.append(f"{index}. {p.action.upper()} {p.symbol}: {verdict}")
    return (
        f"Heard you — acting on {len(outcome.approved)} of {len(proposals)} proposal(s).\n"
        + "\n".join(lines)
    )


# --------------------------------------------------------------------------
# The advisory lock — one getUpdates consumer at a time
# --------------------------------------------------------------------------


@contextlib.contextmanager
def poll_lock(engine=None):
    """
    Yields True if this process now holds the approval-poll lock, False if
    someone else does. Held for the duration of the with-block; session-level,
    so it survives across the poll's many round trips.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": APPROVAL_LOCK_KEY}).scalar())
        try:
            yield got
        finally:
            if got:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": APPROVAL_LOCK_KEY})


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def request_approval(
    proposals: list[ProposedTrade],
    *,
    context: str,
    timeout_s: int | None = None,
    poll_interval_s: float = 5,
    send_fn: Callable[..., dict] = telegram.send_message,
    fetch_fn: Callable[..., list[dict]] = telegram.fetch_updates,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    alert_fn: Callable[..., bool] = send_slack_alert,
    lock_factory: Callable[[], object] = poll_lock,
) -> ApprovalOutcome:
    """
    Show `proposals` to the human and block until every one has an answer,
    the timeout passes, or the gate discovers it cannot ask at all.

    Every path that cannot positively confirm a human said yes ends in
    rejection: no configuration, no lock, no reply, transport failure. The
    single exception is APPROVAL_TIMEOUT_CLOSE_ACTION=approve, which lets
    unanswered *closes* (risk-reducing by definition) through at timeout.
    """
    if not proposals:
        return ApprovalOutcome([], [], status="empty")

    proposals = number_proposals(proposals)
    timeout_s = settings.approval_timeout_s if timeout_s is None else timeout_s

    if settings.approval_mode == "auto":
        logger.info("APPROVAL_MODE=auto — approving all %d proposal(s) without asking.", len(proposals))
        return ApprovalOutcome(
            list(proposals), [], status=AUTO, statuses={p.index: AUTO for p in proposals}
        )

    token, chat_id = telegram.credentials()
    if not token or not chat_id:
        logger.warning("Telegram not configured in telegram approval mode — rejecting the whole batch.")
        alert_fn(
            f"Approval gate ({context}): Telegram is not configured but APPROVAL_MODE=telegram — "
            f"rejected all {len(proposals)} proposal(s). Set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
            "or set APPROVAL_MODE=auto deliberately.",
            severity="critical",
        )
        return ApprovalOutcome(
            [], list(proposals), status="unconfigured", statuses={p.index: REJECTED for p in proposals}
        )

    with lock_factory() as got_lock:
        if not got_lock:
            logger.warning("Another approval poll holds the lock — rejecting this batch rather than colliding.")
            alert_fn(
                f"Approval gate ({context}): another approval conversation is already in progress "
                f"(one Telegram bot, one poller) — rejected all {len(proposals)} proposal(s). "
                "They will be re-proposed by the next cycle.",
                severity="warning",
            )
            return ApprovalOutcome(
                [], list(proposals), status="lock_busy", statuses={p.index: REJECTED for p in proposals}
            )
        return _ask_and_poll(
            proposals, context=context, token=token, chat_id=chat_id, timeout_s=timeout_s,
            poll_interval_s=poll_interval_s, send_fn=send_fn, fetch_fn=fetch_fn, clock=clock,
            sleep_fn=sleep_fn, alert_fn=alert_fn,
        )


def _ask_and_poll(
    proposals, *, context, token, chat_id, timeout_s, poll_interval_s,
    send_fn, fetch_fn, clock, sleep_fn, alert_fn,
) -> ApprovalOutcome:
    known_ids = {p.index for p in proposals}

    # Offset hygiene, fetched BEFORE the proposal goes out: everything the
    # bot has already received is acknowledged past, so yesterday's
    # "approve 1" — or a reply meant for a previous batch — can never count
    # as an answer to this one.
    try:
        baseline = telegram.next_offset(fetch_fn(token, offset=None))
        send_fn(format_proposal_message(proposals, context, timeout_s), token=token, chat_id=chat_id)
    except telegram.TelegramError as exc:
        logger.warning("Could not send the proposal (%s) — rejecting the whole batch.", exc)
        alert_fn(
            f"Approval gate ({context}): could not reach Telegram to propose trades — "
            f"rejected all {len(proposals)} proposal(s). ({exc})",
            severity="critical",
        )
        return ApprovalOutcome(
            [], list(proposals), status="error", statuses={p.index: REJECTED for p in proposals}
        )

    answers: dict[int, bool] = {}
    offset = baseline
    deadline = clock() + timeout_s

    while clock() < deadline:
        try:
            updates = fetch_fn(token, offset=offset, poll_timeout=0)
        except telegram.TelegramError as exc:
            logger.warning("getUpdates failed mid-poll (%s) — will retry until the timeout.", exc)
            sleep_fn(poll_interval_s)
            continue

        offset = telegram.next_offset(updates) or offset

        for reply in telegram.replies_from(updates, chat_id):
            parsed = parse_reply(reply["text"], known_ids)
            if not parsed.understood:
                _try_send(
                    send_fn,
                    'Did not understand that. Reply "approve 1", "reject 2 3", or "approve all".',
                    token, chat_id,
                )
                continue
            # A later reply about the same number wins — people change their minds.
            for index in parsed.ids:
                answers[index] = parsed.approves

        if known_ids <= answers.keys():
            break
        sleep_fn(poll_interval_s)

    outcome = _settle(proposals, answers)
    _try_send(send_fn, format_ack_message(outcome, proposals), token, chat_id)
    logger.info(
        "Approval gate (%s): %d approved, %d rejected (status=%s).",
        context, len(outcome.approved), len(outcome.rejected), outcome.status,
    )
    return outcome


def _settle(proposals: list[ProposedTrade], answers: dict[int, bool]) -> ApprovalOutcome:
    """Turn the replies (and the silences) into a final verdict per proposal."""
    approved, rejected, statuses = [], [], {}
    timed_out = False
    close_action = settings.approval_timeout_close_action

    for p in proposals:
        if p.index in answers:
            if answers[p.index]:
                approved.append(p)
                statuses[p.index] = APPROVED
            else:
                rejected.append(p)
                statuses[p.index] = REJECTED
            continue

        timed_out = True
        statuses[p.index] = TIMEOUT
        # Silence is never consent — except, optionally, for closes, where
        # doing nothing means KEEPING a position the system has flagged.
        if p.action == "close" and close_action == "approve":
            approved.append(p)
        else:
            rejected.append(p)

    return ApprovalOutcome(approved, rejected, status=TIMEOUT if timed_out else "replied", statuses=statuses)


def _try_send(send_fn, message: str, token: str, chat_id: str) -> None:
    """Courtesy messages must never take the gate down."""
    try:
        send_fn(message, token=token, chat_id=chat_id)
    except telegram.TelegramError as exc:
        logger.warning("Could not send an acknowledgement to Telegram: %s", exc)


def send_followup(message: str, *, send_fn: Callable[..., dict] = telegram.send_message) -> None:
    """
    The post-approval confirmation channel: proposals go out size-less
    (approve-first), so once the caller has allocated capital across the
    approved subset, the final sizes are sent to the same phone as a
    follow-up. Best-effort by design — Telegram unconfigured (auto mode,
    tests, dev machines) or unreachable logs the message instead of raising;
    the allocation itself already happened and must not be rolled back by a
    messaging failure.
    """
    token, chat_id = telegram.credentials()
    if not token or not chat_id:
        logger.info("Approval follow-up (Telegram not configured): %s", message)
        return
    _try_send(send_fn, message, token, chat_id)
