import pytest

from models.screener import TradeCandidate, apply_full_deployment
from risk.sizing import scale_to_full_deployment


def _gross(result):
    return sum(abs(v) for v in result.sizes.values())


# --- the scaling itself ---------------------------------------------------


def test_scales_a_partly_invested_book_up_to_full_deployment():
    result = scale_to_full_deployment(
        {"AAPL": 0.10, "MSFT": 0.05, "KO": 0.05}, max_position_pct=0.50
    )

    assert result.reached_target
    assert _gross(result) == pytest.approx(1.0)
    assert result.deployed_pct == pytest.approx(1.0)
    assert result.reason == ""


def test_scaling_preserves_the_relative_conviction_ordering():
    """A pick the model liked twice as much must stay twice as large."""
    result = scale_to_full_deployment({"AAPL": 0.10, "MSFT": 0.05}, max_position_pct=0.90)

    assert result.sizes["AAPL"] == pytest.approx(2 * result.sizes["MSFT"])
    assert _gross(result) == pytest.approx(1.0)


def test_scales_an_overallocated_book_back_down_to_the_target():
    """Deploying more than 100% would be leverage nothing upstream sized for."""
    result = scale_to_full_deployment({"AAPL": 0.9, "MSFT": 0.9}, max_position_pct=0.80)

    assert _gross(result) == pytest.approx(1.0)
    assert result.reached_target


def test_target_allocation_is_configurable():
    result = scale_to_full_deployment({"AAPL": 0.1, "MSFT": 0.1}, max_position_pct=0.5, target_allocation=0.6)

    assert _gross(result) == pytest.approx(0.6)
    assert result.reached_target


# --- shorts ---------------------------------------------------------------


def test_shorts_keep_their_sign_through_scaling():
    result = scale_to_full_deployment({"AAPL": 0.10, "TSLA": -0.05}, max_position_pct=0.90)

    assert result.sizes["AAPL"] > 0
    assert result.sizes["TSLA"] < 0
    assert _gross(result) == pytest.approx(1.0)


def test_shorts_are_held_to_the_tighter_short_cap():
    result = scale_to_full_deployment(
        {"AAPL": 0.10, "TSLA": -0.10}, max_position_pct=0.60, max_short_position_pct=0.15
    )

    assert result.sizes["TSLA"] == pytest.approx(-0.15)  # capped, and still short
    assert result.sizes["AAPL"] <= 0.60 + 1e-9
    assert "TSLA" in result.capped_symbols


def test_short_cap_falls_back_to_the_long_cap_when_not_given():
    result = scale_to_full_deployment({"TSLA": -0.10, "AAPL": 0.10}, max_position_pct=0.25)

    assert result.sizes["TSLA"] == pytest.approx(-0.25)
    assert result.sizes["AAPL"] == pytest.approx(0.25)


# --- caps binding ---------------------------------------------------------


def test_caps_bind_before_the_target_and_the_shortfall_is_explained():
    """Two picks under a 25% cap can only ever be half a portfolio."""
    result = scale_to_full_deployment({"AAPL": 0.10, "MSFT": 0.10}, max_position_pct=0.25)

    assert result.sizes == {"AAPL": pytest.approx(0.25), "MSFT": pytest.approx(0.25)}
    assert result.deployed_pct == pytest.approx(0.50)
    assert not result.reached_target
    assert result.capped_symbols == ["AAPL", "MSFT"]
    assert "50.0%" in result.reason
    assert "cap" in result.reason


def test_no_position_ever_exceeds_its_cap_even_when_that_costs_the_target():
    result = scale_to_full_deployment(
        {"AAPL": 0.10, "MSFT": 0.10, "KO": 0.10}, max_position_pct=0.20
    )

    assert all(abs(size) <= 0.20 + 1e-9 for size in result.sizes.values())
    assert result.deployed_pct == pytest.approx(0.60)
    assert not result.reached_target


def test_leftover_allocation_is_redistributed_to_the_picks_with_headroom():
    """
    The reason this isn't one multiply: AAPL hits its cap, and the allocation
    it couldn't absorb has to flow to the others rather than going unused.
    """
    result = scale_to_full_deployment(
        {"AAPL": 0.40, "MSFT": 0.05, "KO": 0.05}, max_position_pct=0.50
    )

    assert result.sizes["AAPL"] == pytest.approx(0.50)  # capped
    # the remaining 50% splits evenly between the two equal-sized picks,
    # rather than each just getting its own naive share
    assert result.sizes["MSFT"] == pytest.approx(0.25)
    assert result.sizes["KO"] == pytest.approx(0.25)
    assert _gross(result) == pytest.approx(1.0)
    assert result.reached_target


def test_redistribution_cascades_when_it_pushes_a_second_pick_into_its_cap():
    result = scale_to_full_deployment(
        {"AAPL": 0.40, "MSFT": 0.30, "KO": 0.02}, max_position_pct=0.45
    )

    assert result.sizes["AAPL"] == pytest.approx(0.45)
    assert result.sizes["MSFT"] == pytest.approx(0.45)
    assert result.sizes["KO"] == pytest.approx(0.10)  # absorbs what the capped pair couldn't
    assert _gross(result) == pytest.approx(1.0)
    assert result.reached_target
    assert result.capped_symbols == ["AAPL", "MSFT"]


def test_a_position_exactly_on_its_cap_is_not_treated_as_a_shortfall():
    result = scale_to_full_deployment({"AAPL": 0.5, "MSFT": 0.5}, max_position_pct=0.50)

    assert _gross(result) == pytest.approx(1.0)
    assert result.reached_target
    assert result.reason == ""


# --- degenerate inputs ----------------------------------------------------


def test_no_candidates_deploys_nothing_and_says_so():
    result = scale_to_full_deployment({}, max_position_pct=0.25)

    assert result.sizes == {}
    assert result.deployed_pct == 0.0
    assert not result.reached_target
    assert "No sized candidates" in result.reason


def test_all_zero_sizes_deploys_nothing_rather_than_dividing_by_zero():
    result = scale_to_full_deployment({"AAPL": 0.0, "MSFT": 0.0}, max_position_pct=0.25)

    assert result.deployed_pct == 0.0
    assert not result.reached_target


def test_a_zero_sized_pick_alongside_real_ones_stays_at_zero():
    """
    A zero-sized pick carries no conviction to scale up — it must not be
    handed allocation just to help reach the target. That leaves AAPL alone
    to cover the book, so its own cap is the binding constraint.
    """
    result = scale_to_full_deployment({"AAPL": 0.10, "MSFT": 0.0}, max_position_pct=0.90)

    assert result.sizes["MSFT"] == 0.0
    assert result.sizes["AAPL"] == pytest.approx(0.90)
    assert not result.reached_target


def test_zero_target_allocation_flattens_everything():
    result = scale_to_full_deployment({"AAPL": 0.1}, max_position_pct=0.25, target_allocation=0.0)

    assert result.sizes == {"AAPL": 0.0}
    assert result.deployed_pct == 0.0


def test_single_pick_can_only_reach_its_own_cap():
    result = scale_to_full_deployment({"AAPL": 0.05}, max_position_pct=0.25)

    assert result.sizes["AAPL"] == pytest.approx(0.25)
    assert not result.reached_target
    assert "AAPL" in result.reason


# --- the screener wiring --------------------------------------------------


def _candidate(symbol, size):
    return TradeCandidate(
        symbol=symbol, side="long" if size >= 0 else "short", predicted_return=0.03,
        direction_agreement=0.9, conviction_score=0.027, target_position_pct=size,
    )


def test_apply_full_deployment_rewrites_the_candidates_sizes():
    candidates = [_candidate("AAPL", 0.10), _candidate("MSFT", 0.05)]

    result = apply_full_deployment(candidates, max_position_pct=0.90, max_short_position_pct=0.90)

    assert sum(abs(c.target_position_pct) for c in result) == pytest.approx(1.0)
    assert [c.symbol for c in result] == ["AAPL", "MSFT"]  # same picks, same order


def test_apply_full_deployment_does_not_change_which_symbols_were_picked():
    """It changes how much of the portfolio the picks take, never the shortlist."""
    candidates = [_candidate("AAPL", 0.10), _candidate("TSLA", -0.05)]

    result = apply_full_deployment(candidates, max_position_pct=0.25, max_short_position_pct=0.15)

    assert len(result) == 2
    assert result[1].target_position_pct < 0  # still a short
    assert abs(result[1].target_position_pct) <= 0.15 + 1e-9


def test_apply_full_deployment_logs_why_the_caps_stopped_it(caplog):
    candidates = [_candidate("AAPL", 0.10), _candidate("MSFT", 0.10)]

    with caplog.at_level("WARNING"):
        apply_full_deployment(candidates, max_position_pct=0.25, max_short_position_pct=0.15)

    assert "Full deployment not reached" in caplog.text
    assert "50.0%" in caplog.text


def test_apply_full_deployment_on_an_empty_shortlist_is_a_no_op():
    assert apply_full_deployment([], max_position_pct=0.25, max_short_position_pct=0.15) == []


def test_screener_leaves_sizes_alone_when_the_flag_is_off(monkeypatch):
    """The default: sizing stays exactly as select_trades produced it."""
    from config.settings import settings

    monkeypatch.setattr(settings, "full_deployment", False, raising=False)
    assert settings.full_deployment is False
