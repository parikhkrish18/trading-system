from dataclasses import dataclass

import pytest

from execution.broker_alpaca import AlpacaBroker


@dataclass
class _FakeAsset:
    shortable: bool = False
    easy_to_borrow: bool = False


class _FakeTradingClient:
    def __init__(self, asset: _FakeAsset):
        self._asset = asset

    def get_asset(self, symbol):
        return self._asset


def _make_broker(monkeypatch, asset: _FakeAsset) -> AlpacaBroker:
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_api_key", "key")
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_secret_key", "secret")
    monkeypatch.setattr(
        "execution.broker_alpaca.TradingClient", lambda *a, **k: _FakeTradingClient(asset)
    )
    return AlpacaBroker(mode="paper")


def test_is_shortable_true_when_both_flags_set(monkeypatch):
    broker = _make_broker(monkeypatch, _FakeAsset(shortable=True, easy_to_borrow=True))
    assert broker.is_shortable("AAPL") is True


def test_is_shortable_false_when_shortable_but_not_easy_to_borrow(monkeypatch):
    """Theoretically shortable ≠ shares actually available right now."""
    broker = _make_broker(monkeypatch, _FakeAsset(shortable=True, easy_to_borrow=False))
    assert broker.is_shortable("AAPL") is False


def test_is_shortable_false_when_asset_class_disallows_shorting(monkeypatch):
    broker = _make_broker(monkeypatch, _FakeAsset(shortable=False, easy_to_borrow=True))
    assert broker.is_shortable("AAPL") is False


def test_live_broker_requires_confirm_live(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_live_api_key", "key")
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_live_secret_key", "secret")
    with pytest.raises(RuntimeError, match="confirm_live"):
        AlpacaBroker(mode="live")
