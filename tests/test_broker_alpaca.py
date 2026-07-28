from dataclasses import dataclass

import pytest

from execution.broker_alpaca import AlpacaBroker


@dataclass
class _FakeAsset:
    shortable: bool = False
    easy_to_borrow: bool = False


@dataclass
class _FakePosition:
    symbol: str
    qty: float


class _FakeOrder:
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self):
        return self._data


class _FakeTradingClient:
    def __init__(self, asset: _FakeAsset | None = None, positions: dict | None = None):
        self._asset = asset
        self._positions = positions or {}
        self.submitted_orders = []

    def get_asset(self, symbol):
        return self._asset

    def get_all_positions(self):
        return [_FakePosition(symbol=s, qty=q) for s, q in self._positions.items()]

    def submit_order(self, order_request):
        self.submitted_orders.append(order_request)
        return _FakeOrder(symbol=order_request.symbol, qty=order_request.qty, side=str(order_request.side))


def _make_broker(monkeypatch, asset: _FakeAsset | None = None, positions: dict | None = None) -> tuple[AlpacaBroker, _FakeTradingClient]:
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_api_key", "key")
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_secret_key", "secret")
    client = _FakeTradingClient(asset, positions)
    monkeypatch.setattr("execution.broker_alpaca.TradingClient", lambda *a, **k: client)
    return AlpacaBroker(mode="paper"), client


def test_is_shortable_true_when_both_flags_set(monkeypatch):
    broker, _ = _make_broker(monkeypatch, asset=_FakeAsset(shortable=True, easy_to_borrow=True))
    assert broker.is_shortable("AAPL") is True


def test_is_shortable_false_when_shortable_but_not_easy_to_borrow(monkeypatch):
    """Theoretically shortable ≠ shares actually available right now."""
    broker, _ = _make_broker(monkeypatch, asset=_FakeAsset(shortable=True, easy_to_borrow=False))
    assert broker.is_shortable("AAPL") is False


def test_is_shortable_false_when_asset_class_disallows_shorting(monkeypatch):
    broker, _ = _make_broker(monkeypatch, asset=_FakeAsset(shortable=False, easy_to_borrow=True))
    assert broker.is_shortable("AAPL") is False


def test_submit_target_position_allows_fractional_shares_when_going_long(monkeypatch):
    broker, client = _make_broker(monkeypatch)
    broker.submit_target_position("AAPL", 3.5631)
    assert client.submitted_orders[0].qty == 3.5631


def test_submit_target_position_rounds_new_short_to_whole_shares(monkeypatch):
    """
    Alpaca rejects fractional quantities for orders that open/increase a
    short position ("fractional orders cannot be sold short") — this is a
    real API error hit live, not a hypothetical.
    """
    broker, client = _make_broker(monkeypatch)
    broker.submit_target_position("NVDA", -9.3613)
    assert client.submitted_orders[0].qty == 9.0


def test_submit_target_position_skips_short_rounding_to_zero(monkeypatch):
    broker, client = _make_broker(monkeypatch)
    result = broker.submit_target_position("XYZ", -0.4)
    assert result is None
    assert client.submitted_orders == []


def test_submit_target_position_allows_fractional_when_covering_a_short(monkeypatch):
    """Buying back toward flat/long isn't 'selling short' — fractional is fine."""
    broker, client = _make_broker(monkeypatch, positions={"NVDA": -9.0})
    broker.submit_target_position("NVDA", 2.5)  # BUY, covers the short and goes long
    assert client.submitted_orders[0].qty == 11.5


def test_submit_target_position_allows_fractional_when_trimming_a_long(monkeypatch):
    """Selling but staying net long (target_shares >= 0) isn't shorting either."""
    broker, client = _make_broker(monkeypatch, positions={"AAPL": 10.0})
    broker.submit_target_position("AAPL", 2.5)  # SELL, but ends up long, not short
    assert client.submitted_orders[0].qty == 7.5


def test_live_broker_requires_confirm_live(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_live_api_key", "key")
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_live_secret_key", "secret")
    with pytest.raises(RuntimeError, match="confirm_live"):
        AlpacaBroker(mode="live")
