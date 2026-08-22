"""The daily ingest entrypoint's symbol handling — chiefly that the market-regime proxy always rides along."""
from __future__ import annotations

from scripts.run_daily_ingest import _REGIME_PROXY, with_regime_proxy


def test_regime_proxy_is_appended_to_the_ingest_list():
    assert with_regime_proxy(["AAPL", "MSFT"]) == ["AAPL", "MSFT", _REGIME_PROXY]


def test_regime_proxy_is_not_duplicated():
    assert with_regime_proxy(["SPY", "AAPL"]) == ["SPY", "AAPL"]
