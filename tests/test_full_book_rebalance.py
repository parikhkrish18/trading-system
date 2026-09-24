from __future__ import annotations

import pandas as pd
import pytest

from config.settings import settings
from execution import full_book_rebalance as fbr
from execution.approval_gate import ApprovalOutcome, number_proposals
from execution.exit_levels import ExitLevels
from models.screener import TradeCandidate

# Captured before _quiet_side_effects (below) monkeypatches fbr's own
# attribute to a no-op default for every test in this file -- the two
# direct unit tests of the real function need this original reference,
# not the per-test-patched one.
_real_drop_immediately_contradicted = fbr._drop_immediately_contradicted


class _Broker:
    def __init__(self, positions, portfolio_value=100_000.0):
        self.positions = dict(positions)
        self.portfolio_value = portfolio_value
        self.mode = "paper"
        self.targets = []

    def get_positions(self):
        return dict(self.positions)

    def get_portfolio_value(self):
        return self.portfolio_value

    def submit_target_position(self, symbol, target_shares):
        self.targets.append((symbol, target_shares))
        self.positions[symbol] = target_shares
        return {"symbol": symbol, "qty": target_shares}


def _approve_all(proposals, *, context, **kwargs):
    ordered = number_proposals(list(proposals))
    return ApprovalOutcome(
        list(ordered), [], status="auto", statuses={p.index: "auto" for p in ordered}
    )


def _approve_all_but_reject_closing(reject_symbols):
    """Approves every open proposal, but rejects the close proposal for the given symbols (kept as-is)."""

    def _fn(proposals, *, context, **kwargs):
        ordered = number_proposals(list(proposals))
        approved = [p for p in ordered if not (p.action == "close" and p.symbol in reject_symbols)]
        rejected = [p for p in ordered if p.action == "close" and p.symbol in reject_symbols]
        statuses = {p.index: ("rejected" if p in rejected else "auto") for p in ordered}
        return ApprovalOutcome(approved, rejected, status="auto", statuses=statuses)

    return _fn


def _candidate(symbol, conviction):
    return TradeCandidate(
        symbol=symbol,
        side="long",
        predicted_return=conviction,
        direction_agreement=0.9,
        conviction_score=conviction,
        target_position_pct=0.0,
        reasoning=[],
    )


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    monkeypatch.setattr(fbr, "send_followup", lambda *a, **k: None)
    monkeypatch.setattr(fbr, "replicate_to_clients", lambda *a, **k: None)
    monkeypatch.setattr(fbr, "_correlation_matrix", lambda *a, **k: pd.DataFrame())
    # Real usage hits the DB (news_events/prices/universe) for the
    # contradiction pre-check. Default to a no-op passthrough; tests of the
    # check itself override this back via monkeypatch per-case.
    monkeypatch.setattr(fbr, "_drop_immediately_contradicted", lambda engine, candidates: candidates)


def test_post_exit_rebalance_resizes_survivors_and_new_name_by_conviction(monkeypatch):
    """Freed capital is a whole-book optimization, not a replacement slot."""
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    monkeypatch.setattr(settings, "max_concentrated_positions", 3)
    broker = _Broker({"HIGH": 400.0, "LOW": 400.0})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.20)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["HIGH", "LOW", "NEW", "EXITED"])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"HIGH": 100.0, "LOW": 100.0, "NEW": 100.0})
    captured = {}

    def screen(feature_set_id, symbols, **kwargs):
        captured["symbols"] = symbols
        captured.update(kwargs)
        return [_candidate("HIGH", 0.60), _candidate("NEW", 0.30), _candidate("LOW", 0.10)]

    monkeypatch.setattr(fbr, "run_screen", screen)

    fbr.rebalance_after_exit(
        broker,
        engine=object(),
        excluded_symbols={"EXITED"},
        request_fn=_approve_all,
    )

    assert "EXITED" not in captured["symbols"]
    assert "HIGH" in captured["symbols"] and "LOW" in captured["symbols"]
    assert captured["total_deploy_pct"] == pytest.approx(1.0)
    targets = dict(broker.targets)
    assert targets["HIGH"] == pytest.approx(600.0)
    assert targets["NEW"] == pytest.approx(300.0)
    assert targets["LOW"] == pytest.approx(100.0)


def test_log_candidate_gets_target_shares_even_when_the_broker_read_back_is_stale(monkeypatch):
    """
    Regression test: a get_positions() read right after submitting a new
    position's order can still show 0 (unsettled), same class of race as
    the close-side settlement wait above. log_candidate must be told
    target_shares -- what was submitted -- not that stale read, or the
    round-trip trade reconstruction (which only starts an episode on a
    nonzero executed_position) never sees this position as opened at all.
    """

    class _StaleReadBroker(_Broker):
        def submit_target_position(self, symbol, target_shares):
            self.targets.append((symbol, target_shares))
            return {"symbol": symbol, "qty": target_shares}  # positions deliberately NOT updated yet

    broker = _StaleReadBroker({"KEEP": 400.0})
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.20)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["KEEP", "NEW"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("KEEP", 0.50), _candidate("NEW", 0.30)])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"KEEP": 100.0, "NEW": 100.0})

    logged = []
    fbr.rebalance_after_exit(
        broker,
        engine=object(),
        excluded_symbols=set(),
        request_fn=_approve_all,
        log_candidate=lambda candidate, executed_position, mode, target_shares, status: logged.append(
            (candidate.symbol, executed_position, target_shares)
        ),
    )

    new_row = next(row for row in logged if row[0] == "NEW")
    assert new_row[1] == new_row[2]  # executed_position == target_shares, not the stale 0 the broker read back


def test_full_book_rebalance_can_displace_an_existing_position(monkeypatch):
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    broker = _Broker({"KEEP": 400.0, "DROP": 400.0})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.20)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["KEEP", "DROP", "NEW1", "NEW2"])
    monkeypatch.setattr(
        fbr,
        "run_screen",
        lambda *a, **k: [_candidate("KEEP", 0.50), _candidate("NEW1", 0.30), _candidate("NEW2", 0.20)],
    )
    monkeypatch.setattr(
        fbr,
        "_latest_prices",
        lambda *a, **k: {"KEEP": 100.0, "DROP": 100.0, "NEW1": 100.0, "NEW2": 100.0},
    )

    closed = []
    fbr.rebalance_after_exit(
        broker,
        engine=object(),
        excluded_symbols={"EXITED"},
        request_fn=_approve_all,
        log_displaced_close=lambda symbol, status: closed.append(symbol),
    )

    assert ("DROP", 0.0) in broker.targets
    assert closed == ["DROP"]
    assert any(symbol == "NEW1" for symbol, _ in broker.targets)
    assert any(symbol == "NEW2" for symbol, _ in broker.targets)


def test_rebalance_waits_15_minutes_then_still_defers_if_the_exit_is_not_actually_flat(monkeypatch):
    broker = _Broker({"EXITED": 10.0, "KEEP": 20.0})
    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: pytest.fail("must not size before close settles"))
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: pytest.fail("must not screen before close settles"))
    slept = []
    monkeypatch.setattr(fbr.time, "sleep", lambda seconds: slept.append(seconds))

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols={"EXITED"}, request_fn=_approve_all)

    assert slept == [fbr._SETTLEMENT_WAIT_SECONDS]  # actually waited before giving up, not just deferred on the spot
    assert broker.targets == []


def test_rebalance_proceeds_once_the_exit_settles_during_the_wait(monkeypatch):
    """
    Regression test: the get_positions() read taken right after the close
    order can be stale even though the close already went through. A read
    taken 15 minutes later should be trusted -- this should NOT require a
    whole separate hourly check to pick the freed capital back up.
    """
    broker = _Broker({"EXITED": 10.0, "KEEP": 20.0})

    def _settle_during_the_wait(seconds):
        assert seconds == fbr._SETTLEMENT_WAIT_SECONDS
        del broker.positions["EXITED"]  # the close actually posted while we waited

    monkeypatch.setattr(fbr.time, "sleep", _settle_during_the_wait)
    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.20)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["KEEP", "NEW"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("KEEP", 0.50), _candidate("NEW", 0.30)])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"KEEP": 100.0, "NEW": 100.0})

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols={"EXITED"}, request_fn=_approve_all)

    assert any(symbol == "NEW" for symbol, _ in broker.targets)


# ---------- pre-open contradiction check on reactivation candidates ----------


def test_drop_immediately_contradicted_filters_out_a_symbol_flagged_by_current_signals(monkeypatch):
    """
    Regression test: a symbol closed by contradiction_monitor's hourly check
    for contradicting current sentiment/momentum must not be immediately
    reopened by a reactivation sourced from a stale persisted pool that has
    no way to know about that same news/momentum (live: P short closed at
    00:37 on sentiment+momentum contradiction, reopened at 00:58 from the
    persisted pool with no re-check).
    """
    from execution import contradiction_monitor as cm

    monkeypatch.setattr(cm, "_sector_by_symbol", lambda engine, symbols: {})

    def fake_reasons(engine, symbol, side, sector=None):
        if symbol == "P":
            return [{"signal": "news_sentiment", "value": 0.7, "detail": "P's own sentiment contradicts short"}]
        return []

    monkeypatch.setattr(cm, "_contradiction_reasons", fake_reasons)

    candidates = [_candidate("TECH", 0.50), TradeCandidate(
        symbol="P", side="short", predicted_return=-0.05, direction_agreement=0.9,
        conviction_score=0.05, target_position_pct=0.0, reasoning=[],
    )]

    kept = _real_drop_immediately_contradicted(object(), candidates)

    assert [c.symbol for c in kept] == ["TECH"]


def test_drop_immediately_contradicted_is_a_noop_on_an_empty_list():
    assert _real_drop_immediately_contradicted(object(), []) == []


def test_rebalance_skips_a_reactivation_candidate_that_would_immediately_recontradict(monkeypatch):
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    broker = _Broker({})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 1.0)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["TECH", "P"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("TECH", 0.60), _candidate("P", 0.20)])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"TECH": 100.0, "P": 100.0})
    monkeypatch.setattr(
        fbr, "_drop_immediately_contradicted", lambda engine, candidates: [c for c in candidates if c.symbol != "P"]
    )

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols=set(), request_fn=_approve_all)

    targets = {s for s, _ in broker.targets}
    assert "P" not in targets
    assert "TECH" in targets


def test_rebalance_no_ops_when_every_candidate_would_immediately_recontradict(monkeypatch):
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    broker = _Broker({})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 1.0)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["P"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("P", 0.20)])
    monkeypatch.setattr(fbr, "_drop_immediately_contradicted", lambda engine, candidates: [])

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols=set(), request_fn=_approve_all)

    assert broker.targets == []


# ---------- hold-state persistence (exit levels / miss counters) ----------


def test_reactivation_persists_the_new_positions_exit_levels(monkeypatch):
    """
    Regression test: a position opened via this path (the hourly
    contradiction monitor's mid-week reactivation, for example) previously
    never got its ATR/Donchian-derived take-profit/stop-loss recorded
    anywhere -- the dashboard read "No take-profit/stop-loss recorded" and
    execution/hold_rules.py's own stop/target check fell back to the
    generic global thresholds instead of this position's own levels.
    """
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    broker = _Broker({})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 1.0)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["A"])
    levels = ExitLevels(take_profit_pct=0.09, stop_loss_pct=0.045)
    candidate = _candidate("A", 0.5)
    candidate.exit_levels = levels
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [candidate])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"A": 100.0})
    monkeypatch.setattr(fbr.hold_rules, "load_exit_levels", lambda engine: {})
    monkeypatch.setattr(fbr.hold_rules, "load_missed_cycles", lambda engine: {})
    stored = {}

    def fake_store(engine, counts, levels_by_symbol=None):
        stored["counts"] = dict(counts)
        stored["levels"] = dict(levels_by_symbol or {})

    monkeypatch.setattr(fbr.hold_rules, "store_missed_cycles", fake_store)

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols=set(), request_fn=_approve_all)

    assert stored["counts"] == {"A": 0}
    assert stored["levels"]["A"] == levels


def test_kept_position_carries_forward_its_prior_exit_levels_and_miss_count(monkeypatch):
    """A position this rebalance doesn't touch must not lose its previously recorded levels/miss counter."""
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    monkeypatch.setattr(settings, "max_concentrated_positions", 3)
    broker = _Broker({"OTHER": 100.0})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.5)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["OTHER", "NEW"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("NEW", 0.5)])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"OTHER": 100.0, "NEW": 100.0})
    other_levels = ExitLevels(take_profit_pct=0.06, stop_loss_pct=0.03)
    monkeypatch.setattr(fbr.hold_rules, "load_exit_levels", lambda engine: {"OTHER": other_levels})
    monkeypatch.setattr(fbr.hold_rules, "load_missed_cycles", lambda engine: {"OTHER": 2})
    stored = {}

    def fake_store(engine, counts, levels_by_symbol=None):
        stored["counts"] = dict(counts)
        stored["levels"] = dict(levels_by_symbol or {})

    monkeypatch.setattr(fbr.hold_rules, "store_missed_cycles", fake_store)

    # OTHER isn't in the fresh candidate list, so it's proposed for closure
    # (displacement) by default -- reject that close to simulate it being
    # kept, same as a human declining a displacement they disagree with.
    fbr.rebalance_after_exit(
        broker, engine=object(), excluded_symbols=set(),
        request_fn=_approve_all_but_reject_closing({"OTHER"}),
    )

    assert stored["levels"].get("OTHER") == other_levels
    assert stored["counts"].get("OTHER") == 2


def test_closed_position_is_dropped_from_hold_state(monkeypatch):
    monkeypatch.setattr(settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(settings, "max_concentrated_position_pct", 0.70)
    broker = _Broker({"DROP": 100.0})

    monkeypatch.setattr(fbr, "_freed_capital_fraction", lambda *a, **k: 0.5)
    monkeypatch.setattr(fbr, "load_active_universe", lambda: ["DROP", "NEW"])
    monkeypatch.setattr(fbr, "run_screen", lambda *a, **k: [_candidate("NEW", 0.5)])
    monkeypatch.setattr(fbr, "_latest_prices", lambda *a, **k: {"DROP": 100.0, "NEW": 100.0})
    drop_levels = ExitLevels(take_profit_pct=0.06, stop_loss_pct=0.03)
    monkeypatch.setattr(fbr.hold_rules, "load_exit_levels", lambda engine: {"DROP": drop_levels})
    monkeypatch.setattr(fbr.hold_rules, "load_missed_cycles", lambda engine: {"DROP": 1})
    stored = {}

    def fake_store(engine, counts, levels_by_symbol=None):
        stored["counts"] = dict(counts)
        stored["levels"] = dict(levels_by_symbol or {})

    monkeypatch.setattr(fbr.hold_rules, "store_missed_cycles", fake_store)

    fbr.rebalance_after_exit(broker, engine=object(), excluded_symbols=set(), request_fn=_approve_all)

    assert "DROP" not in stored["counts"]
    assert "DROP" not in stored["levels"]
