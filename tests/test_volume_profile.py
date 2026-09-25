from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.quant.volume_profile import VolumeProfile, compute_volume_profile, is_level_volume_confirmed


def _bars(prices: list[float], volumes: list[float]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """high == low == close == price for each bar, so typical_price is exact and unambiguous."""
    price = pd.Series(prices, dtype=float)
    volume = pd.Series(volumes, dtype=float)
    return price, price, price, volume


def test_poc_lands_at_the_price_cluster_carrying_the_most_volume():
    rng = np.random.default_rng(42)
    # Heavy cluster right at 100, thin scattered volume across the rest of the range.
    cluster = rng.normal(100.0, 0.3, size=500)
    noise = rng.uniform(90.0, 110.0, size=50)
    prices = np.concatenate([cluster, noise])
    volumes = np.concatenate([np.full(500, 1000.0), np.full(50, 10.0)])

    high, low, close, volume = _bars(list(prices), list(volumes))
    profile = compute_volume_profile(high, low, close, volume, bins=50, value_area_pct=0.70)

    assert profile is not None
    assert profile.poc == pytest.approx(100.0, abs=1.0)


def test_value_area_always_covers_at_least_the_target_fraction_of_volume():
    rng = np.random.default_rng(7)
    prices = rng.uniform(90.0, 110.0, size=300)
    volumes = rng.uniform(1.0, 100.0, size=300)

    high, low, close, volume = _bars(list(prices), list(volumes))
    profile = compute_volume_profile(high, low, close, volume, bins=40, value_area_pct=0.70)
    assert profile is not None

    in_value_area = (prices >= profile.val) & (prices <= profile.vah)
    covered_fraction = volumes[in_value_area].sum() / volumes.sum()
    assert covered_fraction >= 0.70 - 1e-9


def test_value_area_brackets_the_poc():
    rng = np.random.default_rng(3)
    prices = rng.uniform(90.0, 110.0, size=200)
    volumes = rng.uniform(1.0, 50.0, size=200)

    high, low, close, volume = _bars(list(prices), list(volumes))
    profile = compute_volume_profile(high, low, close, volume, bins=30, value_area_pct=0.70)

    assert profile is not None
    assert profile.val <= profile.poc <= profile.vah


def test_an_extreme_single_cluster_produces_a_tight_value_area():
    """90% of volume packed into one tight cluster should satisfy the 70% target from that cluster alone."""
    cluster = [100.0] * 90
    scattered = list(np.linspace(90.0, 110.0, 10))
    prices = cluster + scattered
    volumes = [100.0] * 90 + [1.0] * 10

    high, low, close, volume = _bars(prices, volumes)
    profile = compute_volume_profile(high, low, close, volume, bins=50, value_area_pct=0.70)

    assert profile is not None
    assert profile.poc == pytest.approx(100.0, abs=0.5)
    assert profile.vah - profile.val < 2.0  # tight band, not pulled wide by the scattered noise


def test_returns_none_for_a_flat_price_range():
    high, low, close, volume = _bars([100.0] * 20, [10.0] * 20)
    assert compute_volume_profile(high, low, close, volume) is None


def test_returns_none_for_fewer_than_two_bars():
    high, low, close, volume = _bars([100.0], [10.0])
    assert compute_volume_profile(high, low, close, volume) is None


def test_returns_none_for_zero_total_volume():
    high, low, close, volume = _bars([95.0, 100.0, 105.0], [0.0, 0.0, 0.0])
    assert compute_volume_profile(high, low, close, volume) is None


def test_returns_none_for_empty_input():
    empty = pd.Series([], dtype=float)
    assert compute_volume_profile(empty, empty, empty, empty) is None


# --------------------------------------------------------------------------
# is_level_volume_confirmed
# --------------------------------------------------------------------------


def test_is_level_volume_confirmed_true_within_the_value_area():
    profile = VolumeProfile(poc=100.0, val=95.0, vah=105.0)
    assert is_level_volume_confirmed(100.0, profile) is True
    assert is_level_volume_confirmed(95.0, profile) is True  # boundary inclusive
    assert is_level_volume_confirmed(105.0, profile) is True  # boundary inclusive


def test_is_level_volume_confirmed_false_outside_the_value_area():
    profile = VolumeProfile(poc=100.0, val=95.0, vah=105.0)
    assert is_level_volume_confirmed(110.0, profile) is False
    assert is_level_volume_confirmed(90.0, profile) is False


def test_is_level_volume_confirmed_fails_open_when_there_is_no_profile():
    assert is_level_volume_confirmed(90.0, None) is True


def test_is_level_volume_confirmed_fails_open_when_the_level_is_missing():
    profile = VolumeProfile(poc=100.0, val=95.0, vah=105.0)
    assert is_level_volume_confirmed(None, profile) is True
