from execution.reconciliation import reconcile_positions, summarize


def test_reconcile_positions_flags_beyond_tolerance():
    intended = {"SPY": 100.0, "QQQ": 50.0}
    actual = {"SPY": 100.0, "QQQ": 40.0}  # 20% short on QQQ
    results = reconcile_positions(intended, actual, tolerance_pct=0.02)

    by_symbol = {r.symbol: r for r in results}
    assert not by_symbol["SPY"].flagged
    assert by_symbol["QQQ"].flagged
    assert by_symbol["QQQ"].diff_shares == -10.0


def test_reconcile_positions_within_tolerance_not_flagged():
    intended = {"SPY": 100.0}
    actual = {"SPY": 101.0}  # 1% off
    results = reconcile_positions(intended, actual, tolerance_pct=0.02)
    assert not results[0].flagged


def test_reconcile_positions_handles_symbol_only_on_one_side():
    intended = {"SPY": 100.0}
    actual = {"SPY": 100.0, "TQQQ": 25.0}  # unexpected position at the broker
    results = reconcile_positions(intended, actual, tolerance_pct=0.02)
    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["TQQQ"].flagged
    assert by_symbol["TQQQ"].intended_shares == 0.0


def test_summarize_reports_all_clear():
    results = reconcile_positions({"SPY": 100.0}, {"SPY": 100.0})
    assert "All 1" in summarize(results)


def test_summarize_lists_flagged_symbols():
    results = reconcile_positions({"SPY": 100.0}, {"SPY": 50.0})
    text = summarize(results)
    assert "SPY" in text
    assert "1 of 1" in text
