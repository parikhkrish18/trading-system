"""
The approval gate, exercised entirely through injected functions — no
network, no database, no broker. What's being proven: the grammar reads
phone replies the way a human means them, and every path that cannot
positively confirm a "yes" ends in rejection.
"""
from __future__ import annotations

import contextlib

import pytest

from execution import approval_gate
from execution.approval_gate import (
    ApprovalOutcome,
    ProposedTrade,
    format_ack_message,
    format_proposal_message,
    number_proposals,
    parse_reply,
    request_approval,
)

KNOWN = {1, 2, 3}


def _open(symbol="NVDA", index=0, side="long", pred=0.031, reason="screen"):
    # No target_position_pct: opens are proposed size-less (approve-first) —
    # capital is allocated across the approved subset after the human answers.
    return ProposedTrade(
        index=index, symbol=symbol, action="open", side=side,
        predicted_return=pred, reason=reason,
    )


def _close(symbol="TSLA", index=0, side="long", reason="out_of_book"):
    return ProposedTrade(
        index=index, symbol=symbol, action="close", side=side,
        target_position_pct=0.0, reason=reason,
    )


# --------------------------------------------------------------------------
# Reply grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["approve 2", "APPROVE 2", "  approve   2  ", "/approve 2", "approve #2", "yes 2", "ok 2"],
)
def test_parse_reply_understands_the_usual_ways_to_approve_one_number(text):
    parsed = parse_reply(text, KNOWN)

    assert parsed.understood and parsed.approves
    assert parsed.ids == [2]
    assert parsed.targets_all is False


@pytest.mark.parametrize("text", ["reject 2", "no 2", "skip 2", "deny 2", "/reject 2"])
def test_parse_reply_understands_the_usual_ways_to_reject(text):
    parsed = parse_reply(text, KNOWN)

    assert parsed.understood and not parsed.approves
    assert parsed.ids == [2]


@pytest.mark.parametrize("text", ["approve 1 3", "approve 1,3", "approve 1, 3", "approve #1 #3"])
def test_parse_reply_takes_several_numbers_however_they_are_separated(text):
    assert parse_reply(text, KNOWN).ids == [1, 3]


def test_parse_reply_expands_all_to_the_whole_batch():
    parsed = parse_reply("approve all", KNOWN)

    assert parsed.targets_all is True
    assert parsed.ids == [1, 2, 3]


def test_parse_reply_separates_numbers_that_are_not_in_this_batch():
    """A number from some other conversation must not silently hit this batch."""
    parsed = parse_reply("approve 2 99", KNOWN)

    assert parsed.ids == [2]
    assert parsed.unknown_ids == [99]


@pytest.mark.parametrize(
    "text", ["", "   ", "hi", "what are these", "3", "2 approve", "maybe approve 2"]
)
def test_parse_reply_refuses_anything_outside_the_grammar(text):
    parsed = parse_reply(text, KNOWN)

    assert not parsed.understood
    assert parsed.ids == []


def test_parse_reply_treats_a_bare_verb_as_not_understood():
    """"approve" with nothing to approve is ambiguous, and ambiguity is never a yes."""
    assert not parse_reply("approve", KNOWN).understood


def test_parse_reply_ignores_filler_words_around_the_numbers():
    assert parse_reply("approve 2 please", KNOWN).ids == [2]


# --------------------------------------------------------------------------
# Numbering and formatting
# --------------------------------------------------------------------------


def test_number_proposals_puts_closes_first_then_opens():
    ordered = number_proposals([_open("NVDA"), _close("TSLA"), _open("XOM"), _close("KO")])

    assert [(p.index, p.symbol, p.action) for p in ordered] == [
        (1, "TSLA", "close"),
        (2, "KO", "close"),
        (3, "NVDA", "open"),
        (4, "XOM", "open"),
    ]


def test_proposal_message_numbers_every_line_and_explains_the_rules():
    proposals = number_proposals([_close("TSLA"), _open("NVDA")])
    message = format_proposal_message(proposals, "weekly cycle", timeout_s=900)

    assert "weekly cycle" in message
    assert "1. CLOSE LONG TSLA" in message
    assert "2. OPEN LONG NVDA — weekly screen pick | expected +3.1%" in message
    assert "THIS message only" in message
    assert "15 min" in message
    assert "Paper account" in message


def test_open_proposals_carry_no_size():
    """
    Approve-first: the human decides WHICH trades happen; sizes are computed
    afterwards over the approved subset and confirmed in a follow-up. A size
    in the proposal would be a promise the allocation step can't keep.
    """
    proposals = number_proposals([_open("NVDA"), _close("TSLA")])
    message = format_proposal_message(proposals, "weekly cycle", timeout_s=900)

    assert "of portfolio" not in message.split("Reply")[0]  # no size in any proposal line
    assert "Sizes are decided after you answer" in message


def test_proposal_message_is_plain_text_with_no_markdown():
    proposals = number_proposals([_open(), _close()])
    message = format_proposal_message(proposals, "weekly cycle", timeout_s=300)

    assert "*" not in message and "`" not in message and "_" not in message


def _phases(*summaries_by_phase):
    return [
        {"phase": phase, "title": f"Phase {phase}", "summary": summary, "lines": []}
        for phase, summary in summaries_by_phase
    ]


def test_proposal_line_includes_the_why_when_reasoning_is_attached():
    p = _open("NVDA", index=1)
    p.reasoning = _phases(
        (2, "Regime: trend. Strongest driver: 5-day price momentum."),
        (3, "+3.1% forecast, 100% model agreement."),
    )
    line = approval_gate.format_proposal_line(p)
    first, why = line.splitlines()
    assert first.startswith("1. OPEN LONG NVDA — weekly screen pick")
    assert "Regime: trend" in why
    assert "100% model agreement" in why


def test_proposal_line_without_reasoning_keeps_the_old_single_line_shape():
    line = approval_gate.format_proposal_line(_open("NVDA", index=1))
    assert "\n" not in line


def test_close_proposal_shows_current_pnl():
    p = _close("TSLA", index=1)
    p.current_pnl_pct = 0.042
    p.current_pnl_usd = 421.5
    line = approval_gate.format_proposal_line(p)
    assert "P&L +4.2%" in line
    assert "$+422" in line or "$+421" in line


def test_close_proposal_with_no_pnl_omits_the_field_rather_than_guessing():
    line = approval_gate.format_proposal_line(_close("TSLA", index=1))
    assert "P&L" not in line


def test_why_falls_back_to_the_selection_story_for_closes():
    p = _close("AAPL", index=1)
    p.reasoning = _phases((4, "AAPL was not one of this cycle's top picks."))
    line = approval_gate.format_proposal_line(p)
    assert "not one of this cycle's top picks" in line


def test_why_is_trimmed_to_stay_phone_readable():
    p = _open("NVDA", index=1)
    p.reasoning = _phases((2, "x" * 500))
    why_line = approval_gate.format_proposal_line(p).splitlines()[1]
    assert len(why_line) <= approval_gate.MAX_WHY_CHARS + 4  # indent + ellipsis
    assert why_line.endswith("…")


def test_message_with_reasoning_still_numbers_every_proposal_and_keeps_the_footer():
    """The reply grammar contract: numbered proposals, same instructions."""
    close = _close("TSLA")
    close.reasoning = _phases((2, "Contradiction detected via: news_sentiment."))
    close.current_pnl_pct = -0.021
    open_ = _open("NVDA")
    open_.reasoning = _phases((2, "Regime: trend."), (3, "+3.1% forecast, 100% model agreement."))
    proposals = number_proposals([close, open_])

    message = format_proposal_message(proposals, "weekly cycle", timeout_s=900)
    assert "1. CLOSE LONG TSLA" in message
    assert "2. OPEN LONG NVDA" in message
    assert 'Reply "approve 1", "reject 2 3", or "approve all"' in message
    assert "Numbers refer to THIS message only." in message


def test_ack_message_says_what_happened_to_each_number():
    proposals = number_proposals([_close("TSLA"), _open("NVDA")])
    outcome = ApprovalOutcome(
        approved=[proposals[0]], rejected=[proposals[1]], status="replied",
        statuses={1: "approved", 2: "rejected"},
    )

    ack = format_ack_message(outcome, proposals)

    assert "1. CLOSE TSLA: approved" in ack
    assert "2. OPEN NVDA: rejected" in ack


# --------------------------------------------------------------------------
# The gate — helpers
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Fetcher:
    """Hands the poll one scripted batch of updates per call, then nothing."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, token, offset=None, poll_timeout=0):
        self.calls.append({"offset": offset})
        return self.batches.pop(0) if self.batches else []


class Sender:
    def __init__(self):
        self.sent = []

    def __call__(self, message, token=None, chat_id=None):
        self.sent.append(message)
        return {}


class Alerts:
    def __init__(self):
        self.messages = []

    def __call__(self, message, severity="warning"):
        self.messages.append(message)
        return True


@contextlib.contextmanager
def _held_lock(got):
    yield got


def _free_lock():
    return _held_lock(True)


def _busy_lock():
    return _held_lock(False)


def _updates(*texts, chat_id="424242", start=100):
    return [
        {
            "update_id": start + i,
            "message": {"chat": {"id": int(chat_id), "type": "private"}, "text": t},
        }
        for i, t in enumerate(texts)
    ]


def _boom(*args, **kwargs):
    raise AssertionError("this path must not touch the network")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(approval_gate.settings, "approval_mode", "telegram")
    monkeypatch.setattr(approval_gate.settings, "approval_timeout_close_action", "reject")
    monkeypatch.setattr(approval_gate.settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(approval_gate.settings, "telegram_chat_id", "424242")


def _gate(proposals, fetcher, sender=None, alerts=None, clock=None, timeout_s=60, **kwargs):
    clock = clock or FakeClock()
    return request_approval(
        proposals,
        context="weekly cycle",
        timeout_s=timeout_s,
        poll_interval_s=5,
        send_fn=sender if sender is not None else Sender(),
        fetch_fn=fetcher,
        clock=clock,
        sleep_fn=clock.sleep,
        alert_fn=alerts if alerts is not None else Alerts(),
        lock_factory=_free_lock,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The gate — behavior
# --------------------------------------------------------------------------


def test_replied_batch_splits_approved_and_rejected(configured):
    # Pre-send baseline fetch returns nothing; then one poll answers both.
    fetcher = Fetcher([], _updates("approve 1", "reject 2"))

    outcome = _gate([_close("TSLA"), _open("NVDA")], fetcher)

    assert outcome.status == "replied"
    assert [p.symbol for p in outcome.approved] == ["TSLA"]
    assert [p.symbol for p in outcome.rejected] == ["NVDA"]
    assert outcome.statuses == {1: "approved", 2: "rejected"}


def test_approve_all_in_one_reply(configured):
    fetcher = Fetcher([], _updates("approve all"))

    outcome = _gate([_close(), _open()], fetcher)

    assert outcome.status == "replied"
    assert len(outcome.approved) == 2 and not outcome.rejected


def test_silence_rejects_everything_at_timeout(configured):
    outcome = _gate([_close(), _open()], Fetcher())

    assert outcome.status == "timeout"
    assert not outcome.approved
    assert len(outcome.rejected) == 2
    assert outcome.statuses == {1: "timeout", 2: "timeout"}


def test_partial_replies_settle_the_answered_and_reject_the_rest(configured):
    fetcher = Fetcher([], _updates("approve 2"))

    outcome = _gate([_close("TSLA"), _open("NVDA"), _open("XOM")], fetcher)

    assert outcome.status == "timeout"
    assert [p.symbol for p in outcome.approved] == ["NVDA"]
    assert {p.symbol for p in outcome.rejected} == {"TSLA", "XOM"}
    assert outcome.statuses == {1: "timeout", 2: "approved", 3: "timeout"}


def test_timeout_close_action_approve_lets_unanswered_closes_through(configured, monkeypatch):
    monkeypatch.setattr(approval_gate.settings, "approval_timeout_close_action", "approve")

    outcome = _gate([_close("TSLA"), _open("NVDA")], Fetcher())

    assert outcome.status == "timeout"
    assert [p.symbol for p in outcome.approved] == ["TSLA"]  # the close
    assert [p.symbol for p in outcome.rejected] == ["NVDA"]  # opens still fail closed
    assert outcome.statuses == {1: "timeout", 2: "timeout"}


def test_a_later_reply_overrides_an_earlier_one(configured):
    fetcher = Fetcher([], _updates("approve 1"), _updates("reject 1", start=200))

    outcome = _gate([_close("TSLA"), _open("NVDA")], fetcher)

    assert all(p.symbol != "TSLA" for p in outcome.approved)
    assert outcome.statuses[1] == "rejected"


def test_replies_from_another_chat_are_ignored(configured):
    fetcher = Fetcher([], _updates("approve all", chat_id="666"))

    outcome = _gate([_close(), _open()], fetcher)

    assert not outcome.approved
    assert outcome.status == "timeout"


def test_offset_isolation_a_stale_reply_from_before_the_proposal_never_counts(configured):
    # The bot's backlog already contains an old "approve 1" when the gate
    # starts. The baseline fetch sees it; the poll must start PAST it.
    stale = _updates("approve 1", start=50)
    fetcher = Fetcher(stale)

    outcome = _gate([_close("TSLA")], fetcher)

    assert outcome.status == "timeout"  # the stale yes was never read as an answer
    assert not outcome.approved
    assert fetcher.calls[0]["offset"] is None  # baseline look, consuming nothing
    assert all(c["offset"] == 51 for c in fetcher.calls[1:])  # polls start past the backlog


def test_auto_mode_approves_everything_without_touching_the_network(monkeypatch):
    monkeypatch.setattr(approval_gate.settings, "approval_mode", "auto")

    outcome = request_approval(
        [_close("TSLA"), _open("NVDA")],
        context="weekly cycle",
        send_fn=_boom,
        fetch_fn=_boom,
        lock_factory=_boom,
    )

    assert outcome.status == "auto"
    assert len(outcome.approved) == 2 and not outcome.rejected
    assert outcome.statuses == {1: "auto", 2: "auto"}


def test_unconfigured_telegram_rejects_everything_and_alerts(configured, monkeypatch):
    monkeypatch.setattr(approval_gate.settings, "telegram_bot_token", "")
    alerts = Alerts()

    outcome = request_approval(
        [_close(), _open()],
        context="weekly cycle",
        send_fn=_boom,
        fetch_fn=_boom,
        alert_fn=alerts,
        lock_factory=_boom,
    )

    assert outcome.status == "unconfigured"
    assert not outcome.approved and len(outcome.rejected) == 2
    assert any("not configured" in m for m in alerts.messages)


def test_busy_lock_rejects_the_batch_and_alerts(configured):
    alerts = Alerts()

    outcome = request_approval(
        [_close(), _open()],
        context="contradiction monitor",
        send_fn=_boom,
        fetch_fn=_boom,
        alert_fn=alerts,
        lock_factory=_busy_lock,
    )

    assert outcome.status == "lock_busy"
    assert not outcome.approved and len(outcome.rejected) == 2
    assert any("already in progress" in m for m in alerts.messages)


def test_send_failure_rejects_the_batch_and_alerts(configured):
    def failing_send(message, token=None, chat_id=None):
        raise approval_gate.telegram.TelegramError("sendMessage: chat not found")

    alerts = Alerts()
    outcome = _gate([_close(), _open()], Fetcher([]), sender=failing_send, alerts=alerts)

    assert outcome.status == "error"
    assert not outcome.approved and len(outcome.rejected) == 2
    assert any("could not reach Telegram" in m for m in alerts.messages)


def test_fetch_failures_mid_poll_are_retried_until_timeout_then_reject(configured):
    calls = {"n": 0}

    def flaky_fetch(token, offset=None, poll_timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return []  # the baseline fetch works
        raise approval_gate.telegram.TelegramError("getUpdates: down")

    outcome = _gate([_open()], flaky_fetch)

    assert outcome.status == "timeout"
    assert not outcome.approved
    assert calls["n"] > 2  # it kept trying rather than dying on the first failure


def test_empty_batch_asks_nobody(configured):
    outcome = request_approval(
        [], context="weekly cycle", send_fn=_boom, fetch_fn=_boom, lock_factory=_boom
    )

    assert outcome.status == "empty"
    assert not outcome.approved and not outcome.rejected


def test_gate_sends_a_final_ack_naming_each_verdict(configured):
    sender = Sender()
    fetcher = Fetcher([], _updates("approve 1", "reject 2"))

    _gate([_close("TSLA"), _open("NVDA")], fetcher, sender=sender)

    ack = sender.sent[-1]
    assert "1. CLOSE TSLA: approved" in ack
    assert "2. OPEN NVDA: rejected" in ack


def test_confused_replies_get_a_help_message_and_approve_nothing(configured):
    sender = Sender()
    fetcher = Fetcher([], _updates("lol what"))

    outcome = _gate([_open("NVDA")], fetcher, sender=sender)

    assert not outcome.approved
    assert any("Did not understand" in m for m in sender.sent)


def test_unknown_numbers_in_a_reply_change_nothing(configured):
    fetcher = Fetcher([], _updates("approve 99"))

    outcome = _gate([_open("NVDA")], fetcher)

    assert not outcome.approved  # 99 isn't in this batch; 1 was never answered
    assert outcome.status == "timeout"


def test_approved_opens_and_closes_helpers_split_by_action(configured):
    fetcher = Fetcher([], _updates("approve all"))

    outcome = _gate([_close("TSLA"), _open("NVDA")], fetcher)

    assert [p.symbol for p in outcome.approved_closes()] == ["TSLA"]
    assert [p.symbol for p in outcome.approved_opens()] == ["NVDA"]


# --------------------------------------------------------------------------
# The post-approval follow-up (final sizes)
# --------------------------------------------------------------------------


def test_send_followup_delivers_when_telegram_is_configured(configured):
    sent = []
    approval_gate.send_followup("Sizing the 2 approved pick(s)…", send_fn=lambda msg, *, token, chat_id: sent.append(msg))

    assert sent == ["Sizing the 2 approved pick(s)…"]


def test_send_followup_without_telegram_logs_instead_of_raising(monkeypatch):
    monkeypatch.setattr(approval_gate.settings, "telegram_bot_token", "")
    monkeypatch.setattr(approval_gate.settings, "telegram_chat_id", "")

    def _never(*a, **k):
        raise AssertionError("must not try to send without credentials")

    approval_gate.send_followup("sizes…", send_fn=_never)  # no exception is the assertion


def test_send_followup_swallows_transport_failures(configured):
    from execution import telegram

    def _boom(*a, **k):
        raise telegram.TelegramError("telegram is down")

    approval_gate.send_followup("sizes…", send_fn=_boom)  # the allocation already happened; messaging must not raise
