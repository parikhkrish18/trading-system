import contextlib
import datetime as dt
import pathlib
import re

import pandas as pd
import pytest

from execution import approval_loop
from execution.approval_loop import (
    MODE,
    ExecutionResult,
    PendingDecision,
    describe_reply,
    executed_fraction,
    format_candidate,
    format_dry_run_proposal,
    format_notification,
    format_pick_line,
    format_proposal,
    load_pending_decisions,
    main,
    paper_broker,
    paper_keys_present,
    parse_reply,
    record_batch_monitoring,
    record_execution,
    request_approvals,
    run_approval_loop,
    run_dry_run,
    run_listen,
    run_notify,
    run_telegram_setup,
    submit_paper_order,
    target_shares_for,
)


def _decision(decision_id=1, symbol="AAPL", forecast=0.021, regime="trend", target_position=0.125):
    return PendingDecision(
        decision_id=decision_id,
        ts=dt.datetime(2026, 7, 31, 14, 3, tzinfo=dt.UTC),
        symbol=symbol,
        forecast=forecast,
        regime=regime,
        target_position=target_position,
    )


class _FakeBroker:
    """Stands in for execution.broker_*.py — same three methods approval_loop uses."""

    def __init__(self, portfolio_value=100_000.0, positions=None):
        self.portfolio_value = portfolio_value
        self.positions = dict(positions or {})
        self.submitted = []

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_positions(self):
        return dict(self.positions)

    def submit_target_position(self, symbol, target_shares):
        self.submitted.append((symbol, target_shares))
        self.positions[symbol] = target_shares
        return {"symbol": symbol, "qty": target_shares}


class _FakeEngine:
    """Captures the params of every UPDATE the write-back issues."""

    def __init__(self):
        self.executed = []

    @contextlib.contextmanager
    def begin(self):
        yield self

    def execute(self, statement, params):
        self.executed.append(params)


def _submit_via_broker(broker, symbol, target_shares):
    """What submit_paper_order will be once un-stubbed — used to exercise the loop."""
    return broker.submit_target_position(symbol, target_shares)


# --- proposal formatting -------------------------------------------------


def test_format_candidate_shows_symbol_side_forecast_regime_and_target_pct():
    line = format_candidate(1, _decision(decision_id=42))

    assert line.startswith("1. AAPL LONG")
    assert "+12.50% of portfolio" in line
    assert "forecast +2.10%" in line
    assert "regime trend" in line
    assert "id 42" in line


def test_format_candidate_marks_negative_target_as_short():
    line = format_candidate(1, _decision(symbol="TSLA", forecast=-0.034, target_position=-0.08))

    assert "TSLA SHORT" in line
    assert "-8.00% of portfolio" in line
    assert "forecast -3.40%" in line


def test_format_proposal_lists_every_candidate_with_header_and_footer():
    decisions = [
        _decision(decision_id=1, symbol="AAPL"),
        _decision(decision_id=2, symbol="TSLA", target_position=-0.08, regime="chop"),
    ]

    message = format_proposal(decisions)

    assert "2026-07-31 14:03 UTC" in message  # the batch timestamp, not "now"
    assert "2 candidate(s)" in message
    assert f"{MODE} mode" in message
    assert "1. AAPL LONG" in message
    assert "2. TSLA SHORT" in message
    assert "regime chop" in message
    assert 'Reply "approve <id>" or "reject <id>"' in message
    assert "not explicitly approved is treated as rejected" in message


def test_format_proposal_with_no_candidates():
    assert format_proposal([]) == "No pending trade proposals."


# --- sizing --------------------------------------------------------------


def test_target_shares_rounds_toward_zero_so_rounding_never_grows_a_position():
    # 12.5% of 100k = $12,500 / $312.50 = exactly 40 shares
    assert target_shares_for(0.125, 100_000.0, 312.50) == 40.0
    # 12.5% of 100k / $300 = 41.67 -> 41, not 42
    assert target_shares_for(0.125, 100_000.0, 300.0) == 41.0
    # shorts round toward zero too: -41.67 -> -41
    assert target_shares_for(-0.125, 100_000.0, 300.0) == -41.0


def test_target_shares_is_zero_for_unusable_price_or_portfolio():
    assert target_shares_for(0.125, 100_000.0, 0.0) == 0.0
    assert target_shares_for(0.125, 0.0, 300.0) == 0.0


def test_executed_fraction_reflects_whole_share_rounding():
    assert executed_fraction(41.0, 300.0, 100_000.0) == pytest.approx(0.123)
    assert executed_fraction(-41.0, 300.0, 100_000.0) == pytest.approx(-0.123)
    assert executed_fraction(41.0, 300.0, 0.0) == 0.0


# --- DB read / write-back ------------------------------------------------


def test_load_pending_decisions_maps_rows(monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "id": 7,
                "ts": dt.datetime(2026, 7, 31, 14, 3, tzinfo=dt.UTC),
                "symbol": "MMM",
                "forecast": -0.03,
                "regime": "chop",
                "target_position": -0.05,
            }
        ]
    )
    captured = {}

    def fake_read_sql(statement, engine, params=None):
        captured["params"] = params
        return rows

    monkeypatch.setattr("execution.approval_loop.get_engine", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    decisions = load_pending_decisions()

    assert captured["params"] == {"mode": "paper"}  # never queries live rows
    assert len(decisions) == 1
    assert decisions[0].decision_id == 7
    assert decisions[0].symbol == "MMM"
    assert decisions[0].side == "short"


def test_record_execution_updates_the_row_in_paper_mode_only():
    engine = _FakeEngine()

    record_execution(7, 0.123, engine=engine)

    assert engine.executed == [{"executed_position": 0.123, "decision_id": 7, "mode": "paper"}]


def test_run_approval_loop_writes_executed_position_back_with_a_fake_broker(monkeypatch):
    decision = _decision(decision_id=7, symbol="AAPL", target_position=0.125)
    broker = _FakeBroker(portfolio_value=100_000.0)
    written = []

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr(
        "execution.approval_loop.record_execution",
        lambda decision_id, executed_position, engine=None: written.append((decision_id, executed_position)),
    )
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])

    results = run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: True},
        submit_fn=_submit_via_broker,
        price_fn=lambda symbols: {"AAPL": 300.0},
    )

    # 12.5% of $100k at $300 = 41 whole shares, submitted exactly once
    assert broker.submitted == [("AAPL", 41.0)]
    # ...and 41 * 300 / 100k = 12.3% written back, not the 12.5% that was asked for
    assert written == [(7, pytest.approx(0.123))]
    assert results[0].approved
    assert results[0].target_shares == 41.0
    assert results[0].executed_position == pytest.approx(0.123)


def test_run_approval_loop_records_zero_for_rejected_and_submits_nothing(monkeypatch):
    decision = _decision(decision_id=7)
    broker = _FakeBroker()
    written = []

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr(
        "execution.approval_loop.record_execution",
        lambda decision_id, executed_position, engine=None: written.append((decision_id, executed_position)),
    )
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])

    results = run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: False},
        submit_fn=_submit_via_broker,
        price_fn=lambda symbols: {"AAPL": 300.0},
    )

    assert broker.submitted == []
    assert written == [(7, 0.0)]  # decided-and-declined, not left pending
    assert not results[0].approved


def test_run_approval_loop_treats_a_missing_answer_as_rejected(monkeypatch):
    decision = _decision(decision_id=7)
    broker = _FakeBroker()

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr("execution.approval_loop.record_execution", lambda *a, **k: None)
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])

    results = run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {},  # human never answered
        submit_fn=_submit_via_broker,
        price_fn=lambda symbols: {"AAPL": 300.0},
    )

    assert broker.submitted == []
    assert not results[0].approved


def test_run_approval_loop_skips_approved_candidate_with_no_price(monkeypatch):
    decision = _decision(decision_id=7)
    broker = _FakeBroker()
    written = []

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr(
        "execution.approval_loop.record_execution",
        lambda decision_id, executed_position, engine=None: written.append((decision_id, executed_position)),
    )
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])

    results = run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: True},
        submit_fn=_submit_via_broker,
        price_fn=lambda symbols: {},  # no price stored for AAPL
    )

    assert broker.submitted == []
    assert written == []  # left pending so it can be re-proposed, not marked done
    assert results[0].note.startswith("skipped")


def test_run_approval_loop_records_zero_when_size_rounds_to_no_shares(monkeypatch):
    # 0.001% of a $1,000 book at $300/share is a fraction of one share
    decision = _decision(decision_id=7, target_position=0.00001)
    broker = _FakeBroker(portfolio_value=1_000.0)
    written = []

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr(
        "execution.approval_loop.record_execution",
        lambda decision_id, executed_position, engine=None: written.append((decision_id, executed_position)),
    )
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])

    run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: True},
        submit_fn=_submit_via_broker,
        price_fn=lambda symbols: {"AAPL": 300.0},
    )

    assert broker.submitted == []
    assert written == [(7, 0.0)]


def test_run_approval_loop_does_nothing_when_no_decisions_are_pending(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [])

    def _should_not_run(*args, **kwargs):
        raise AssertionError("nothing should be asked or submitted with an empty batch")

    assert run_approval_loop(broker=_FakeBroker(), request_fn=_should_not_run, submit_fn=_should_not_run) == []


# --- monitoring hookup ---------------------------------------------------


def test_record_batch_monitoring_snapshots_equity_and_runs_breakers(monkeypatch):
    broker = _FakeBroker(portfolio_value=100_000.0, positions={"AAPL": 41.0})
    snapshots = []
    breaker_calls = []

    monkeypatch.setattr(
        "execution.approval_loop.record_equity_snapshot",
        lambda equity_value, mode: snapshots.append((equity_value, mode)),
    )
    monkeypatch.setattr(
        "execution.approval_loop.load_equity_curve",
        lambda mode=None: pd.DataFrame({"equity_value": [100_000.0, 99_000.0]}),
    )
    monkeypatch.setattr(
        "execution.approval_loop.check_and_record_breakers",
        lambda **kwargs: breaker_calls.append(kwargs) or [],
    )

    triggered = record_batch_monitoring(
        broker,
        price_fn=lambda symbols: {"AAPL": 300.0},
        correlation_fn=lambda symbols: pd.DataFrame(),
    )

    assert snapshots == [(100_000.0, "paper")]
    assert triggered == []
    # breakers get position *values* in dollars, not share counts
    assert breaker_calls[0]["positions_by_symbol"] == {"AAPL": 41.0 * 300.0}
    assert breaker_calls[0]["portfolio_value"] == 100_000.0
    assert breaker_calls[0]["equity_curve"] == [100_000.0, 99_000.0]


# --- the remaining stub, and the now-live submit path --------------------


def test_request_approvals_is_still_a_stub():
    with pytest.raises(NotImplementedError, match="Telegram"):
        request_approvals("some proposal", [_decision()])


def test_submit_paper_order_places_the_order_through_the_broker():
    broker = _FakeBroker()

    order = submit_paper_order(broker, "AAPL", 41.0)

    assert broker.submitted == [("AAPL", 41.0)]
    assert order == {"symbol": "AAPL", "qty": 41.0}


def test_submit_paper_order_passes_through_a_no_op_from_the_broker():
    """Already at the target = no order; that's a None, not a failure."""

    class _AlreadyThere:
        def submit_target_position(self, symbol, target_shares):
            return None

    assert submit_paper_order(_AlreadyThere(), "AAPL", 41.0) is None


def test_run_approval_loop_still_cannot_execute_unattended(monkeypatch):
    """
    The safety property that survives un-stubbing the broker: with nothing
    injected the loop stops at the human step, so orders only ever follow an
    approval that a human actually gave.
    """
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    with pytest.raises(NotImplementedError, match="Telegram"):
        run_approval_loop(broker=_FakeBroker())


# --- paper-only construction ---------------------------------------------


def test_paper_broker_pins_mode_to_paper_and_never_confirms_live(monkeypatch):
    captured = {}

    class _SpyAlpaca:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("execution.approval_loop.AlpacaBroker", _SpyAlpaca)

    paper_broker()

    assert captured["kwargs"] == {"mode": "paper"}
    assert captured["args"] == ()  # nothing positional that could land on confirm_live


def test_module_never_mentions_live_mode_or_confirm_live():
    """
    A grep-level guard on the module's central safety claim: live must be
    unreachable from this file, not merely unreached by today's call graph.
    """
    source = pathlib.Path(approval_loop.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    # Strip docstrings, which discuss live precisely to explain its absence.
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)

    assert "confirm_live" not in code
    assert '"live"' not in code and "'live'" not in code
    assert approval_loop.MODE == "paper"


# --- keys-not-configured path --------------------------------------------


def test_run_approval_loop_exits_cleanly_when_no_paper_keys_are_configured(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])
    monkeypatch.setattr("execution.approval_loop.paper_keys_present", lambda: False)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("nothing should be asked, built or submitted without keys")

    monkeypatch.setattr("execution.approval_loop.paper_broker", _should_not_run)
    monkeypatch.setattr("execution.approval_loop.record_execution", _should_not_run)
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", _should_not_run)

    printed = []
    results = run_approval_loop(request_fn=_should_not_run, print_fn=printed.append)

    assert results == []
    message = "\n".join(printed)
    assert "ALPACA_PAPER_API_KEY" in message and "ALPACA_PAPER_SECRET_KEY" in message
    assert "not set" in message


def test_no_keys_check_is_skipped_when_the_caller_brings_its_own_broker(monkeypatch):
    """An injected broker needs no credentials of ours — that's the test path."""
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision(decision_id=7)])
    monkeypatch.setattr("execution.approval_loop.paper_keys_present", lambda: False)
    monkeypatch.setattr("execution.approval_loop.record_execution", lambda *a, **k: None)
    monkeypatch.setattr("execution.approval_loop.record_batch_monitoring", lambda broker, **kwargs: [])
    broker = _FakeBroker()

    run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: True},
        price_fn=lambda symbols: {"AAPL": 300.0},
    )

    assert broker.submitted == [("AAPL", 41.0)]


def test_paper_keys_present_requires_both_halves_of_the_pair(monkeypatch):
    from config.settings import settings as live_settings

    monkeypatch.setattr(live_settings, "alpaca_paper_api_key", "", raising=False)
    monkeypatch.setattr(live_settings, "alpaca_paper_secret_key", "", raising=False)
    assert not paper_keys_present()

    monkeypatch.setattr(live_settings, "alpaca_paper_api_key", "key", raising=False)
    assert not paper_keys_present()  # a key with no secret is not usable

    monkeypatch.setattr(live_settings, "alpaca_paper_secret_key", "secret", raising=False)
    assert paper_keys_present()


# --- approve -> order -> write-back -> monitoring, end to end ------------


def test_approving_places_the_order_writes_back_and_records_equity(monkeypatch):
    """
    The whole point of the feature in one pass, with the default submit_fn
    (not the test helper): an approval reaches the broker, the fill is
    stamped on the decision row, and the dashboard's equity and breaker
    panels get their snapshot.
    """
    decision = _decision(decision_id=7, symbol="AAPL", target_position=0.125)
    broker = _FakeBroker(portfolio_value=100_000.0)
    written, snapshots, breaker_calls = [], [], []

    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [decision])
    monkeypatch.setattr(
        "execution.approval_loop.record_execution",
        lambda decision_id, executed_position, engine=None: written.append((decision_id, executed_position)),
    )
    monkeypatch.setattr(
        "execution.approval_loop.record_equity_snapshot",
        lambda equity_value, mode: snapshots.append((equity_value, mode)),
    )
    monkeypatch.setattr(
        "execution.approval_loop.load_equity_curve",
        lambda mode=None: pd.DataFrame({"equity_value": [100_000.0]}),
    )
    monkeypatch.setattr(
        "execution.approval_loop.check_and_record_breakers",
        lambda **kwargs: breaker_calls.append(kwargs) or [],
    )
    results = run_approval_loop(
        broker=broker,
        request_fn=lambda message, decisions: {7: True},
        price_fn=lambda symbols: {"AAPL": 300.0},
        correlation_fn=lambda symbols: pd.DataFrame(),  # the breakers' correlation read is a real query
    )

    assert broker.submitted == [("AAPL", 41.0)]
    assert written == [(7, pytest.approx(0.123))]
    assert results[0].executed_position == pytest.approx(0.123)
    # monitoring hooks fire once per batch, in paper mode
    assert snapshots == [(100_000.0, "paper")]
    assert len(breaker_calls) == 1
    assert breaker_calls[0]["portfolio_value"] == 100_000.0


def test_execution_result_defaults_are_inert():
    result = ExecutionResult(_decision(), approved=False)
    assert result.target_shares == 0.0
    assert result.executed_position is None


# --- dry run -------------------------------------------------------------


class _Console:
    """Canned answers in, printed lines out — stands in for a terminal."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.prompts = []
        self.lines = []

    def input(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def print(self, line=""):
        self.lines.append(str(line))

    @property
    def text(self):
        return "\n".join(self.lines)


def test_format_dry_run_proposal_matches_the_telegram_style_one_liner():
    line = format_dry_run_proposal(
        _decision(symbol="TSLA", forecast=0.038, regime="trend", target_position=0.20)
    )

    assert line == (
        "🤖 Pulse proposes: LONG TSLA — 20.0% of portfolio | "
        "regime: trend | expected: +3.8% — Approve? [y/n] "
    )


def test_format_dry_run_proposal_shows_shorts_unsigned_with_a_signed_forecast():
    line = format_dry_run_proposal(
        _decision(symbol="XOM", forecast=-0.021, regime="chop", target_position=-0.08)
    )

    assert "SHORT XOM — 8.0% of portfolio" in line  # direction is in the word, not the size
    assert "expected: -2.1%" in line


def test_run_dry_run_prints_a_proposal_and_an_outcome_for_each_candidate():
    console = _Console(answers=["y", "n"])
    decisions = [_decision(decision_id=1, symbol="TSLA"), _decision(decision_id=2, symbol="KO")]

    summary = run_dry_run(decisions=decisions, input_fn=console.input, print_fn=console.print)

    assert [p.split(" — ")[0] for p in console.prompts] == [
        "🤖 Pulse proposes: LONG TSLA",
        "🤖 Pulse proposes: LONG KO",
    ]
    assert f"✅ approved — would execute in {MODE.upper()} mode (broker not connected)" in console.lines
    assert "⏭️ skipped" in console.lines
    assert (summary.approved, summary.skipped, summary.total) == (1, 1, 2)


def test_run_dry_run_summary_line_reports_the_counts():
    console = _Console(answers=["y", "y", "n"])
    decisions = [_decision(decision_id=i) for i in (1, 2, 3)]

    run_dry_run(decisions=decisions, input_fn=console.input, print_fn=console.print)

    assert "2 approved, 1 skipped, 3 total" in console.text
    assert "Nothing was executed" in console.text


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " y ", "﻿y"])
def test_run_dry_run_accepts_the_usual_spellings_of_yes(answer):
    console = _Console(answers=[answer])

    summary = run_dry_run(decisions=[_decision()], input_fn=console.input, print_fn=console.print)

    assert summary.approved == 1


@pytest.mark.parametrize("answer", ["n", "N", "no", "", "  ", "maybe", "yep"])
def test_run_dry_run_treats_anything_that_is_not_yes_as_a_skip(answer):
    """Fail closed, same as the real loop: silence and typos are never consent."""
    console = _Console(answers=[answer])

    summary = run_dry_run(decisions=[_decision()], input_fn=console.input, print_fn=console.print)

    assert (summary.approved, summary.skipped) == (0, 1)


def test_run_dry_run_treats_a_closed_console_as_a_skip():
    console = _Console(answers=[])  # every input() raises EOFError

    summary = run_dry_run(decisions=[_decision(), _decision(decision_id=2)], input_fn=console.input, print_fn=console.print)

    assert (summary.approved, summary.skipped) == (0, 2)


def test_run_dry_run_says_so_when_nothing_is_pending():
    console = _Console()

    summary = run_dry_run(decisions=[], input_fn=console.input, print_fn=console.print)

    assert "No pending trade proposals." in console.lines
    assert console.prompts == []  # never asks about an empty batch
    assert summary.total == 0


def test_run_dry_run_loads_the_pending_batch_when_none_is_passed(monkeypatch):
    console = _Console(answers=["y"])
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    summary = run_dry_run(input_fn=console.input, print_fn=console.print)

    assert summary.total == 1


def test_run_dry_run_never_touches_telegram_the_broker_or_the_database(monkeypatch):
    """The safety property: approving in a dry run still cannot cause a side effect."""
    console = _Console(answers=["y", "y"])

    def _forbidden(*args, **kwargs):
        raise AssertionError("dry run must not reach this")

    for name in (
        "request_approvals",
        "submit_paper_order",
        "paper_broker",
        "AlpacaBroker",
        "record_execution",
        "record_batch_monitoring",
        "run_approval_loop",
    ):
        monkeypatch.setattr(f"execution.approval_loop.{name}", _forbidden)

    summary = run_dry_run(
        decisions=[_decision(decision_id=1), _decision(decision_id=2)],
        input_fn=console.input,
        print_fn=console.print,
    )

    assert summary.approved == 2  # both approved, and still nothing happened


def test_main_requires_the_dry_run_flag_so_there_is_no_executing_mode(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.run_dry_run", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code != 0


def test_main_runs_the_dry_run_and_exits_clean(monkeypatch):
    calls = []
    monkeypatch.setattr("execution.approval_loop.run_dry_run", lambda: calls.append(True))

    assert main(["--dry-run"]) == 0
    assert calls == [True]


# --- Telegram: the send side only ----------------------------------------
#
# No real credentials appear anywhere below, and no test reaches the network:
# send_fn/fetch_fn are always injected.

FAKE_TOKEN = "123456:fake-test-token"  # not a real credential
FAKE_CHAT = "424242"


@pytest.fixture
def telegram_configured(monkeypatch):
    """Pretend .env has a bot token and a chat id."""
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", FAKE_CHAT)


class _Sender:
    """Captures what would have gone to Telegram."""

    def __init__(self, error=None):
        self.messages = []
        self.error = error

    def __call__(self, text, token=None, chat_id=None):
        if self.error:
            raise self.error
        self.messages.append({"text": text, "token": token, "chat_id": chat_id})
        return {"message_id": len(self.messages)}


def test_format_pick_line_is_the_same_string_the_dry_run_shows():
    """One formatter, so the phone and the console can never disagree."""
    decision = _decision(symbol="TSLA", forecast=0.038, regime="trend", target_position=0.20)

    line = format_pick_line(decision)

    assert line == (
        "🤖 Pulse proposes: LONG TSLA — 20.0% of portfolio | regime: trend | expected: +3.8%"
    )
    assert format_dry_run_proposal(decision) == f"{line} — Approve? [y/n] "


def test_format_notification_puts_every_pick_in_one_message():
    message = format_notification(
        [
            _decision(decision_id=1, symbol="TSLA", target_position=0.20),
            _decision(decision_id=2, symbol="XOM", target_position=-0.08),
            _decision(decision_id=3, symbol="KO", target_position=0.05),
        ]
    )

    assert message.startswith("Pulse — 3 pending proposal(s) | batch 2026-07-31 14:03 UTC | paper mode")
    assert "LONG TSLA" in message and "SHORT XOM" in message and "LONG KO" in message
    assert message.count("🤖 Pulse proposes:") == 3


def test_format_notification_is_plain_text_with_no_markdown_syntax():
    message = format_notification([_decision(symbol="TSLA", target_position=-0.08)])

    assert "*" not in message and "```" not in message and "<b>" not in message


def test_format_notification_on_an_empty_batch():
    assert format_notification([]) == "No pending trade proposals."


def test_run_notify_sends_one_message_containing_the_whole_batch(monkeypatch, telegram_configured):
    console = _Console()
    sender = _Sender()
    monkeypatch.setattr(
        "execution.approval_loop.load_pending_decisions",
        lambda engine=None: [_decision(decision_id=1, symbol="TSLA"), _decision(decision_id=2, symbol="KO")],
    )

    assert run_notify(print_fn=console.print, send_fn=sender) == 0

    assert len(sender.messages) == 1  # one buzz for the batch, not one per pick
    assert sender.messages[0]["chat_id"] == FAKE_CHAT
    assert sender.messages[0]["token"] == FAKE_TOKEN
    assert "TSLA" in sender.messages[0]["text"] and "KO" in sender.messages[0]["text"]
    assert "Sent — 2 proposal(s)" in console.text


def test_run_notify_echoes_the_message_locally_before_sending(monkeypatch, telegram_configured):
    console = _Console()
    sender = _Sender()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    run_notify(print_fn=console.print, send_fn=sender)

    assert sender.messages[0]["text"] in console.text


def test_run_notify_exits_cleanly_with_no_token(monkeypatch):
    console = _Console()
    sender = _Sender()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", "")
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", FAKE_CHAT)

    assert run_notify(print_fn=console.print, send_fn=sender) == 0
    assert sender.messages == []
    assert "TELEGRAM_BOT_TOKEN is not set" in console.text
    assert "@BotFather" in console.text


def test_run_notify_exits_cleanly_with_no_chat_id(monkeypatch):
    console = _Console()
    sender = _Sender()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", "")

    assert run_notify(print_fn=console.print, send_fn=sender) == 0
    assert sender.messages == []
    assert "--telegram-setup" in console.text


def test_run_notify_sends_nothing_when_no_batch_is_pending(monkeypatch, telegram_configured):
    console = _Console()
    sender = _Sender()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [])

    assert run_notify(print_fn=console.print, send_fn=sender) == 0
    assert sender.messages == []
    assert "nothing to send" in console.text


def test_run_notify_reports_a_telegram_failure_without_crashing(monkeypatch, telegram_configured):
    console = _Console()
    sender = _Sender(error=approval_loop.telegram.TelegramError("sendMessage: chat not found"))
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    assert run_notify(print_fn=console.print, send_fn=sender) == 1
    assert "chat not found" in console.text
    assert "Nothing was executed" in console.text


def test_run_notify_never_reaches_the_broker_or_writes_a_row(monkeypatch, telegram_configured):
    """The safety property: notifying is a message, and a message cannot trade."""
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    def _forbidden(*args, **kwargs):
        raise AssertionError("the notify path must not reach this")

    for name in (
        "request_approvals",
        "submit_paper_order",
        "paper_broker",
        "AlpacaBroker",
        "record_execution",
        "record_batch_monitoring",
        "run_approval_loop",
    ):
        monkeypatch.setattr(f"execution.approval_loop.{name}", _forbidden)

    assert run_notify(print_fn=_Console().print, send_fn=_Sender()) == 0


def test_run_telegram_setup_prints_the_chat_id_to_paste_into_env(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", "")
    updates = [{"message": {"chat": {"id": 424242, "first_name": "Neeraj", "type": "private"}, "text": "hi"}}]

    assert run_telegram_setup(print_fn=console.print, fetch_fn=lambda token: updates) == 0

    assert "chat id 424242" in console.text
    assert "TELEGRAM_CHAT_ID=424242" in console.text
    assert "--notify" in console.text


def test_run_telegram_setup_passes_the_configured_token_through(monkeypatch):
    seen = []
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", f"  {FAKE_TOKEN} ")
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", "")

    run_telegram_setup(print_fn=_Console().print, fetch_fn=lambda token: seen.append(token) or [])

    assert seen == [FAKE_TOKEN]  # stripped, so a pasted newline is not a mystery 404


def test_run_telegram_setup_notices_the_chat_id_is_already_configured(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_chat_id", "424242")
    updates = [{"message": {"chat": {"id": 424242, "first_name": "Neeraj", "type": "private"}, "text": "hi"}}]

    run_telegram_setup(print_fn=console.print, fetch_fn=lambda token: updates)

    assert "already what .env says" in console.text


def test_run_telegram_setup_tells_the_user_to_message_the_bot_first(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)

    assert run_telegram_setup(print_fn=console.print, fetch_fn=lambda token: []) == 0

    assert "send it any message" in console.text
    assert "24 hours" in console.text  # updates expire, so an old chat won't show


def test_run_telegram_setup_exits_cleanly_with_no_token(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", "")

    def _no_calls(token):
        raise AssertionError("must not call Telegram without a token")

    assert run_telegram_setup(print_fn=console.print, fetch_fn=_no_calls) == 0
    assert "TELEGRAM_BOT_TOKEN is not set" in console.text


def test_run_telegram_setup_explains_a_rejected_token(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", FAKE_TOKEN)

    def _unauthorized(token):
        raise approval_loop.telegram.TelegramError("getUpdates: HTTP 401 — Unauthorized")

    assert run_telegram_setup(print_fn=console.print, fetch_fn=_unauthorized) == 1
    assert "Unauthorized" in console.text
    assert "/mytoken" in console.text


def test_main_dispatches_the_telegram_modes(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.run_telegram_setup", lambda: 0)
    monkeypatch.setattr("execution.approval_loop.run_notify", lambda: 0)
    monkeypatch.setattr("execution.approval_loop.run_dry_run", _unreachable)

    assert main(["--telegram-setup"]) == 0
    assert main(["--notify"]) == 0


def test_main_passes_the_mode_exit_code_through(monkeypatch):
    """A refused send is a non-zero exit, so a scheduled run can notice."""
    monkeypatch.setattr("execution.approval_loop.run_notify", lambda: 1)

    assert main(["--notify"]) == 1


def test_main_refuses_two_modes_at_once(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.run_dry_run", lambda: None)
    monkeypatch.setattr("execution.approval_loop.run_notify", lambda: 0)

    with pytest.raises(SystemExit) as excinfo:
        main(["--dry-run", "--notify"])

    assert excinfo.value.code != 0


def test_no_cli_mode_can_reach_the_execution_path(monkeypatch):
    """
    The property the whole module is arranged around, restated now that there
    are three modes: none of them calls anything that places an order.
    """
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [])
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", "")

    for name in ("submit_paper_order", "paper_broker", "AlpacaBroker", "record_execution", "run_approval_loop"):
        monkeypatch.setattr(f"execution.approval_loop.{name}", _unreachable)

    assert main(["--dry-run"]) == 0
    assert main(["--telegram-setup"]) == 0
    assert main(["--notify"]) == 0


def _unreachable(*args, **kwargs):
    raise AssertionError("no CLI mode may reach this")


# --- the local echo must never be what stops a notification ---------------


def test_encodable_replaces_what_a_windows_console_cannot_print():
    """
    cp1252 is what Windows picks for a redirected stdout, and it has no 🤖 —
    which crashed --notify *before* it sent anything, under exactly the
    redirect-to-a-log-file use this mode is for.
    """
    safe = approval_loop._encodable(format_pick_line(_decision(symbol="TSLA")), "cp1252")

    safe.encode("cp1252")  # the point: this no longer raises
    assert "Pulse proposes" in safe and "TSLA" in safe


def test_encodable_leaves_a_utf8_console_untouched():
    line = format_pick_line(_decision())

    assert approval_loop._encodable(line, "utf-8") == line
    assert "🤖" in approval_loop._encodable(line, "utf-8")


def test_encodable_survives_an_unknown_or_missing_encoding():
    line = format_pick_line(_decision())

    assert approval_loop._encodable(line, None) == line
    assert approval_loop._encodable(line, "not-a-real-codec") == line


def test_notify_and_setup_default_to_the_safe_printer():
    """A default of plain print() is what made the crash reachable."""
    import inspect

    for fn in (run_notify, run_telegram_setup):
        assert inspect.signature(fn).parameters["print_fn"].default is approval_loop._print_encodable


def test_notification_lines_carry_the_id_a_reply_refers_to():
    message = format_notification([_decision(decision_id=7, symbol="XOM", target_position=-0.08)])

    assert "[7] 🤖 Pulse proposes: SHORT XOM" in message


def test_notification_footer_is_honest_about_what_a_reply_reaches():
    message = format_notification([_decision()])

    assert "No reply here can place an order" in message
    assert "--listen" in message  # replies do reach something; say which
    assert "--dry-run" in message  # and where the real decision happens


# --- reply grammar --------------------------------------------------------

BATCH = [
    _decision(decision_id=3, symbol="XOM", target_position=-0.071),
    _decision(decision_id=5, symbol="TSLA", target_position=0.20),
    _decision(decision_id=8, symbol="KO", target_position=0.021),
]


@pytest.mark.parametrize(
    "text",
    ["approve 5", "APPROVE 5", "  approve   5  ", "/approve 5", "approve #5", "yes 5", "ok 5"],
)
def test_parse_reply_understands_the_usual_ways_to_approve_one_pick(text):
    parsed = parse_reply(text, BATCH)

    assert parsed.understood and parsed.approves
    assert parsed.decision_ids == [5]
    assert parsed.targets_all is False


@pytest.mark.parametrize("text", ["reject 5", "no 5", "skip 5", "deny 5", "/reject 5"])
def test_parse_reply_understands_the_usual_ways_to_reject(text):
    parsed = parse_reply(text, BATCH)

    assert parsed.understood and not parsed.approves
    assert parsed.decision_ids == [5]


@pytest.mark.parametrize("text", ["approve 3 5", "approve 3,5", "approve 3, 5", "approve #3 #5"])
def test_parse_reply_takes_several_ids_however_they_are_separated(text):
    assert parse_reply(text, BATCH).decision_ids == [3, 5]


def test_parse_reply_expands_all_to_the_whole_batch():
    parsed = parse_reply("approve all", BATCH)

    assert parsed.targets_all is True
    assert parsed.decision_ids == [3, 5, 8]


def test_parse_reply_separates_ids_that_are_not_in_this_batch():
    """A stale id from yesterday's message must not silently hit today's pick."""
    parsed = parse_reply("approve 5 99", BATCH)

    assert parsed.decision_ids == [5]
    assert parsed.unknown_ids == [99]


@pytest.mark.parametrize(
    "text", ["", "   ", "hi", "what are these", "3", "5 approve", "maybe approve 5"]
)
def test_parse_reply_refuses_anything_outside_the_grammar(text):
    parsed = parse_reply(text, BATCH)

    assert not parsed.understood
    assert parsed.decision_ids == []


def test_parse_reply_treats_a_bare_verb_as_not_understood():
    """"approve" with nothing to approve is ambiguous, and ambiguity is never a yes."""
    parsed = parse_reply("approve", BATCH)

    assert not parsed.understood
    assert parsed.decision_ids == []


def test_parse_reply_ignores_filler_words_around_the_ids():
    assert parse_reply("approve 5 please", BATCH).decision_ids == [5]


def test_describe_reply_names_the_stock_behind_the_id():
    lines = "\n".join(describe_reply(parse_reply("approve 3", BATCH), BATCH))

    assert "APPROVE [3] XOM SHORT 7.1%" in lines


def test_describe_reply_explains_a_message_it_could_not_parse():
    lines = "\n".join(describe_reply(parse_reply("lol", BATCH), BATCH))

    assert "did not understand" in lines and "approve 3" in lines


def test_describe_reply_calls_out_an_id_from_another_batch():
    lines = "\n".join(describe_reply(parse_reply("approve 99", BATCH), BATCH))

    assert "[99] is not in this batch" in lines


# --- the listener ---------------------------------------------------------


def _updates(*texts, chat_id="424242", start=100):
    return [
        {
            "update_id": start + i,
            "message": {"chat": {"id": int(chat_id), "first_name": "Neeraj", "type": "private"}, "text": t},
        }
        for i, t in enumerate(texts)
    ]


class _Listener:
    """Hands the poll loop one batch of updates per call, then nothing."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, token, offset=None, poll_timeout=0):
        self.calls.append({"token": token, "offset": offset})
        return self.batches.pop(0) if self.batches else []


def test_run_listen_reports_what_each_reply_meant(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=console.print,
        fetch_fn=_Listener(_updates("approve 5", "reject 3")),
        send_fn=_Sender(),
        max_polls=1,
    )

    assert 'you said: "approve 5"' in console.text
    assert "APPROVE [5] TSLA LONG 20.0%" in console.text
    assert "REJECT [3] XOM SHORT 7.1%" in console.text
    assert "Heard 1 approval(s) and 1 rejection(s)" in console.text


def test_run_listen_records_nothing_and_says_so(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=console.print,
        fetch_fn=_Listener(_updates("approve all")),
        send_fn=_Sender(),
        max_polls=1,
    )

    assert "none of it was executed" in console.text.lower()


def test_run_listen_never_reaches_the_broker_or_writes_a_row(monkeypatch, telegram_configured):
    """The whole point of the choice to build understanding without action."""
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    def _forbidden(*args, **kwargs):
        raise AssertionError("the listener must not reach this")

    for name in (
        "request_approvals",
        "submit_paper_order",
        "paper_broker",
        "AlpacaBroker",
        "record_execution",
        "record_batch_monitoring",
        "run_approval_loop",
    ):
        monkeypatch.setattr(f"execution.approval_loop.{name}", _forbidden)

    assert (
        run_listen(
            print_fn=_Console().print,
            fetch_fn=_Listener(_updates("approve all")),
            send_fn=_Sender(),
            max_polls=1,
        )
        == 0
    )


def test_run_listen_ignores_a_stranger_messaging_the_bot(monkeypatch, telegram_configured):
    """A bot username is public. Only the configured chat is ever heard."""
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=console.print,
        fetch_fn=_Listener(_updates("approve all", chat_id="777777")),
        send_fn=_Sender(),
        max_polls=1,
    )

    assert "you said" not in console.text
    assert "Heard 0 approval(s)" in console.text


def test_run_listen_advances_the_offset_so_a_reply_is_read_once(monkeypatch, telegram_configured):
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)
    listener = _Listener(_updates("approve 5", start=100), [])

    run_listen(print_fn=_Console().print, fetch_fn=listener, send_fn=_Sender(), max_polls=2)

    assert listener.calls[0]["offset"] is None  # first poll takes whatever is queued
    assert listener.calls[1]["offset"] == 101  # then acknowledges past update_id 100


def test_run_listen_lets_a_later_reply_change_an_earlier_answer(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=console.print,
        fetch_fn=_Listener(_updates("approve 5", "reject 5")),
        send_fn=_Sender(),
        max_polls=1,
    )

    assert "Heard 0 approval(s) and 1 rejection(s) across 1 pick(s)" in console.text


def test_run_listen_acknowledges_back_to_the_phone(monkeypatch, telegram_configured):
    sender = _Sender()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=_Console().print,
        fetch_fn=_Listener(_updates("approve 5")),
        send_fn=sender,
        max_polls=1,
    )

    assert len(sender.messages) == 1
    assert "Heard you: approve" in sender.messages[0]["text"]
    assert "Nothing was executed" in sender.messages[0]["text"]


def test_run_listen_tells_the_phone_when_it_did_not_understand(monkeypatch, telegram_configured):
    sender = _Sender()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    run_listen(
        print_fn=_Console().print,
        fetch_fn=_Listener(_updates("lol wat")),
        send_fn=sender,
        max_polls=1,
    )

    assert "Did not understand" in sender.messages[0]["text"]


def test_run_listen_survives_a_failed_acknowledgement(monkeypatch, telegram_configured):
    """The courtesy message must not be able to take the listener down."""
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)
    sender = _Sender(error=approval_loop.telegram.TelegramError("sendMessage: blocked"))

    assert (
        run_listen(
            print_fn=console.print,
            fetch_fn=_Listener(_updates("approve 5")),
            send_fn=sender,
            max_polls=1,
        )
        == 0
    )
    assert "could not send the acknowledgement" in console.text
    assert "Heard 1 approval(s)" in console.text  # the reply still counted


def test_run_listen_exits_cleanly_with_no_credentials(monkeypatch):
    console = _Console()
    monkeypatch.setattr(approval_loop.telegram.settings, "telegram_bot_token", "")

    def _no_calls(*args, **kwargs):
        raise AssertionError("must not poll without a token")

    assert run_listen(print_fn=console.print, fetch_fn=_no_calls, send_fn=_no_calls) == 0
    assert "TELEGRAM_BOT_TOKEN is not set" in console.text


def test_run_listen_says_so_when_there_is_nothing_pending(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [])

    def _no_calls(*args, **kwargs):
        raise AssertionError("nothing to listen about")

    assert run_listen(print_fn=console.print, fetch_fn=_no_calls, send_fn=_no_calls) == 0
    assert "nothing to reply about" in console.text


def test_run_listen_reports_a_telegram_outage(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    def _down(token, offset=None, poll_timeout=0):
        raise approval_loop.telegram.TelegramError("getUpdates: could not reach api.telegram.org")

    assert run_listen(print_fn=console.print, fetch_fn=_down, send_fn=_Sender()) == 1
    assert "Telegram stopped answering" in console.text


def test_run_listen_stops_on_ctrl_c_without_a_traceback(monkeypatch, telegram_configured):
    console = _Console()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: BATCH)

    def _interrupt(token, offset=None, poll_timeout=0):
        raise KeyboardInterrupt

    assert run_listen(print_fn=console.print, fetch_fn=_interrupt, send_fn=_Sender()) == 0
    assert "Stopped listening" in console.text


def test_main_dispatches_listen(monkeypatch):
    monkeypatch.setattr("execution.approval_loop.run_listen", lambda: 0)
    monkeypatch.setattr("execution.approval_loop.run_dry_run", _unreachable)

    assert main(["--listen"]) == 0


def test_the_phone_still_gets_the_real_characters(monkeypatch, telegram_configured):
    """Sanitising is for the console only — the message itself is untouched."""
    sender = _Sender()
    monkeypatch.setattr("execution.approval_loop.load_pending_decisions", lambda engine=None: [_decision()])

    run_notify(print_fn=lambda line: None, send_fn=sender)

    assert "🤖" in sender.messages[0]["text"]
