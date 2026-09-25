"""
Volume profile -- where trading activity actually clustered across a recent
price range, as opposed to Donchian's rolling high/low (where price merely
REACHED). A Donchian edge printed by one thin, undefended spike looks
identical to one the market fought over all week; a volume profile tells
the two apart.

Needs genuine intraday bars (data/ingest/intraday_bars.py) -- the `prices`
table's one-volume-number-per-day daily bars carry no information on how
that volume distributed within the day, so this can never be computed from
them. See models/screener.py's _solidified_channel_distances for how this
gets used: not a competing support/resistance source, just a confirmation
check on the Donchian levels that already drive take-profit/stop-loss
sizing.

Shape deliberately differs from this module's siblings (donchian.py,
momentum.py, ...): those return a per-bar pd.Series feature, rolled over
the whole price history. A volume profile is a single aggregate over one
fixed window, in raw price terms -- the same shape models/screener.py's
own donchian_channel_levels already returns (support, resistance) in,
not a per-row model feature.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class VolumeProfile:
    """
    poc: Point of Control -- the price bucket with the most volume traded.
    val/vah: Value Area Low/High -- the price band around the POC holding
        value_area_pct of the window's total volume (see
        compute_volume_profile). val <= poc <= vah always.
    """

    poc: float
    val: float
    vah: float


def compute_volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile | None:
    """
    None when there isn't enough to compute a meaningful profile from:
    fewer than 2 usable bars, a flat price range (every bar's typical price
    identical -- nothing to bin), or zero total volume. Never raises.

    Each bar's volume is attributed to its typical price ((high+low+close)/3),
    not spread across the bar's own high/low -- standard practice at bar
    resolutions fine enough that the within-bar spread is small relative to
    the profile's own bucket width (true here: these are
    settings.volume_profile_bar_minutes-minute bars, not the daily bars
    that forced the coarser day-level approximation this module exists to
    avoid).

    Value area built by expanding outward from the POC bucket, at each step
    adding whichever neighboring bucket (immediately above the current
    included band, or immediately below it) carries more volume, until the
    included buckets' cumulative volume reaches `value_area_pct` of the
    total or no neighbors remain -- the standard algorithm, not a percentile
    cut, so the value area stays centered on where volume actually is
    rather than an arbitrary price-sorted slice.
    """
    typical_price = (high + low + close) / 3
    df = pd.DataFrame({"price": typical_price, "volume": volume}).dropna()
    df = df[df["volume"] > 0]
    if len(df) < 2:
        return None

    lo, hi = df["price"].min(), df["price"].max()
    if not (hi > lo):
        return None

    edges = np.linspace(lo, hi, bins + 1)
    # right=False + the manual top-edge clip: pd.cut's default right-closed
    # bins would silently drop the single bar sitting exactly at `hi`
    # (falls in no interval when right=False and hi is itself the last
    # edge) -- clip it into the last bucket instead of losing that volume.
    bucket_idx = np.clip(np.digitize(df["price"], edges, right=False) - 1, 0, bins - 1)
    bucket_volume = np.zeros(bins)
    np.add.at(bucket_volume, bucket_idx, df["volume"].to_numpy())

    total_volume = bucket_volume.sum()
    if total_volume <= 0:
        return None

    poc_idx = int(np.argmax(bucket_volume))
    cumulative = bucket_volume[poc_idx]
    target = value_area_pct * total_volume
    low_edge, high_edge = poc_idx, poc_idx
    while cumulative < target and (low_edge > 0 or high_edge < bins - 1):
        below_idx = low_edge - 1 if low_edge > 0 else None
        above_idx = high_edge + 1 if high_edge < bins - 1 else None
        below_vol = bucket_volume[below_idx] if below_idx is not None else -1.0
        above_vol = bucket_volume[above_idx] if above_idx is not None else -1.0
        # Ties favor the lower-price side -- an arbitrary but deterministic
        # choice; which side wins a tie has no principled answer here.
        if below_idx is not None and below_vol >= above_vol:
            low_edge = below_idx
            cumulative += below_vol
        elif above_idx is not None:
            high_edge = above_idx
            cumulative += above_vol

    # POC is a representative point price (a bucket's midpoint, the
    # conventional way to plot it), but val/vah are boundaries of the
    # included buckets, not their midpoints -- a level anywhere within the
    # outermost included bucket has real recorded volume behind it, and
    # using that bucket's midpoint here would understate the value area by
    # up to half a bucket's width on each side.
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    return VolumeProfile(poc=poc, val=float(edges[low_edge]), vah=float(edges[high_edge + 1]))


def is_level_volume_confirmed(level: float | None, profile: VolumeProfile | None) -> bool:
    """
    Whether `level` (a Donchian support or resistance price) falls within
    the profile's value area -- real evidence the market actually
    transacted size near this price, not just that price once reached it.

    Fails OPEN to True (confirmed) when there's no profile to check against
    (fetch failure, insufficient intraday data, or level itself missing) --
    this is a confirmation check layered on top of Donchian solidification
    that already worked before this existed; a data hiccup on this NEWER,
    optional signal must never silently reject an otherwise-good Donchian
    level, only skip grading it.
    """
    if level is None or profile is None:
        return True
    return profile.val <= level <= profile.vah
