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
    executed_fraction,
    format_candidate,
    format_dry_run_proposal,
    format_proposal,
    load_pending_decisions,
    main,
    paper_broker,
    paper_keys_present,
    record_batch_monitoring,
    record_execution,
    request_approvals,
    run_approval_loop,
    run_dry_run,
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
