"""
Reconciliation's job is to be believed. These cover the distinction it
exists to make — an order waiting to fill is not an order that failed —
because getting that wrong produced "3 of 3 positions diverged beyond
tolerance" on a completely successful run, and a warning that fires on
success stops being read.
"""
from execution.reconciliation import DIVERGED, FILLED, PARTIAL, QUEUED, REJECTED, reconcile_positions, summarize


def _order(status: str) -> dict:
    return {"id": "abc-123", "status": status}


# --------------------------------------------------------------------------
# The four cases that matter
# --------------------------------------------------------------------------


def test_a_queued_order_after_hours_is_not_a_warning():
    """
    The cycle runs when the market is shut, so the position is still zero
    and the order sits until the open. This is the normal, healthy state —
    it must not be reported as divergence.
    """
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={},  # nothing filled yet
        orders={"SPY": _order("accepted")},
    )

    assert results[0].outcome == QUEUED
    assert results[0].flagged is False


def test_a_rejected_order_is_a_warning():
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={},
        orders={"SPY": _order("rejected")},
    )

    assert results[0].outcome == REJECTED
    assert results[0].flagged is True


def test_a_partial_fill_is_a_warning_carrying_the_actual_gap():
    """Filled, but short — the number that matters is how short."""
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={"SPY": 60.0},
        orders={"SPY": _order("partially_filled")},
    )

    assert results[0].outcome == PARTIAL
    assert results[0].flagged is True
    assert results[0].diff_shares == -40.0


def test_a_clean_fill_is_silent():
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={"SPY": 100.0},
        orders={"SPY": _order("filled")},
    )

    assert results[0].outcome == FILLED
    assert results[0].flagged is False


# --------------------------------------------------------------------------
# Divergence with no order to explain it
# --------------------------------------------------------------------------


def test_a_wrong_position_with_no_order_is_flagged():
    """
    Nothing was submitted for this symbol and yet the holding is wrong.
    That is the case reconciliation was always meant to catch.
    """
    results = reconcile_positions(intended={"SPY": 100.0}, actual={"SPY": 40.0}, orders={})

    assert results[0].outcome == DIVERGED
    assert results[0].flagged is True


def test_an_unexpected_position_at_the_broker_is_flagged():
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={"SPY": 100.0, "TQQQ": 25.0},
        orders={"SPY": _order("filled")},
    )

    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["TQQQ"].flagged is True
    assert by_symbol["TQQQ"].intended_shares == 0.0


def test_a_matching_position_is_never_flagged_whatever_the_order_says():
    """
    A stale 'accepted' on an order whose position is already correct must
    not manufacture a warning about something that plainly worked.
    """
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={"SPY": 100.0},
        orders={"SPY": _order("accepted")},
    )

    assert results[0].flagged is False


def test_small_differences_stay_within_tolerance():
    results = reconcile_positions({"SPY": 100.0}, {"SPY": 101.0}, tolerance_pct=0.02)

    assert results[0].flagged is False


def test_tiny_position_rounding_dust_is_not_a_false_positive():
    """
    Regression test: the percentage tolerance breaks down as intended_shares
    approaches 0 -- intended=0.01, actual=0.011 is a trivial rounding
    difference (0.001 shares), but as a fraction of 0.01 that is a 10%
    "divergence", well past the 2% tolerance. Both the position and the gap
    are economically negligible, so this must not be flagged.
    """
    results = reconcile_positions(
        intended={"SPY": 0.01},
        actual={"SPY": 0.011},
        orders={},
    )

    assert results[0].outcome == FILLED
    assert results[0].flagged is False


def test_a_real_divergence_on_a_larger_position_still_flags_despite_the_dust_floor():
    """The dust floor must not swallow genuine problems on non-trivial size."""
    results = reconcile_positions(intended={"SPY": 100.0}, actual={"SPY": 40.0}, orders={})

    assert results[0].outcome == DIVERGED
    assert results[0].flagged is True


def test_an_unrecognised_status_is_treated_as_pending_not_failed():
    """
    Inventing a warning out of a status we don't understand is exactly the
    behaviour being fixed. Unknown means quiet.
    """
    results = reconcile_positions({"SPY": 100.0}, {}, orders={"SPY": _order("some_new_alpaca_state")})

    assert results[0].flagged is False


# --------------------------------------------------------------------------
# What the human reads
# --------------------------------------------------------------------------


def test_summary_of_a_queued_run_reads_as_normal_not_as_failure():
    results = reconcile_positions(
        intended={"SPY": 100.0, "QQQ": 50.0},
        actual={},
        orders={"SPY": _order("accepted"), "QQQ": _order("new")},
    )

    text = summarize(results)

    assert "queued" in text
    assert "diverged" not in text
    assert "attention" not in text


def test_summary_names_what_actually_went_wrong():
    results = reconcile_positions(
        intended={"SPY": 100.0},
        actual={},
        orders={"SPY": _order("rejected")},
    )

    text = summarize(results)

    assert "SPY" in text
    assert "attention" in text
    assert "rejected" in text


def test_summary_separates_a_real_problem_from_the_queue_around_it():
    results = reconcile_positions(
        intended={"SPY": 100.0, "QQQ": 50.0},
        actual={},
        orders={"SPY": _order("rejected"), "QQQ": _order("accepted")},
    )

    text = summarize(results)

    assert "1 of 2" in text
    assert "queued and waiting" in text


def test_summary_prints_share_counts_a_human_can_read():
    results = reconcile_positions(
        intended={"SPY": 32.063831455312794},
        actual={},
        orders={"SPY": _order("rejected")},
    )

    assert "32.063831455312794" not in summarize(results)
    assert "32.06" in summarize(results)


def test_all_clear_says_so_briefly():
    results = reconcile_positions({"SPY": 100.0}, {"SPY": 100.0}, orders={"SPY": _order("filled")})

    assert "reconciled" in summarize(results)
