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
    side: str = "long"
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


@dataclass
class _FakeClock:
    is_open: bool = True


@dataclass
class _FakeQuote:
    ask_price: float | None = 100.0
    bid_price: float | None = 99.5


class _FakeOrder:
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self):
        return self._data


class _FakeTradingClient:
    def __init__(self, asset: _FakeAsset | None = None, positions: dict | None = None, is_open: bool = True):
        self._asset = asset
        self._positions = positions or {}
        self._is_open = is_open
        self.submitted_orders = []

    def get_asset(self, symbol):
        return self._asset

    def get_all_positions(self):
        return [_FakePosition(symbol=s, qty=q) for s, q in self._positions.items()]

    def get_clock(self):
        return _FakeClock(is_open=self._is_open)

    def submit_order(self, order_request):
        self.submitted_orders.append(order_request)
        return _FakeOrder(symbol=order_request.symbol, qty=order_request.qty, side=str(order_request.side))


class _FakeDataClient:
    def __init__(self, quotes: dict | None = None):
        self._quotes = quotes or {}

    def get_stock_latest_quote(self, request):
        symbol = request.symbol_or_symbols
        return {symbol: self._quotes.get(symbol, _FakeQuote())}


def _make_broker(
    monkeypatch,
    asset: _FakeAsset | None = None,
    positions: dict | None = None,
    is_open: bool = True,
    quotes: dict | None = None,
) -> tuple[AlpacaBroker, _FakeTradingClient]:
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_api_key", "key")
    monkeypatch.setattr("execution.broker_alpaca.settings.alpaca_paper_secret_key", "secret")
    client = _FakeTradingClient(asset, positions, is_open=is_open)
    data_client = _FakeDataClient(quotes)
    monkeypatch.setattr("execution.broker_alpaca.TradingClient", lambda *a, **k: client)
    monkeypatch.setattr("execution.broker_alpaca.StockHistoricalDataClient", lambda *a, **k: data_client)
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


def test_submit_target_position_uses_limit_order_outside_rth(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, client = _make_broker(monkeypatch, is_open=False, quotes={"AAPL": _FakeQuote(ask_price=100.0, bid_price=99.5)})

    broker.submit_target_position("AAPL", 5.0)

    order = client.submitted_orders[0]
    assert order.extended_hours is True
    assert order.qty == 5.0  # long, whole number anyway here


def test_submit_target_position_extended_hours_buy_prices_above_ask(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, client = _make_broker(monkeypatch, is_open=False, quotes={"AAPL": _FakeQuote(ask_price=100.0, bid_price=99.5)})

    broker.submit_target_position("AAPL", 5.0)  # BUY

    order = client.submitted_orders[0]
    assert order.limit_price == pytest.approx(round(100.0 * 1.005, 2), abs=1e-6)


def test_submit_target_position_extended_hours_sell_prices_below_bid(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, client = _make_broker(
        monkeypatch, is_open=False, positions={"AAPL": 10.0}, quotes={"AAPL": _FakeQuote(ask_price=100.0, bid_price=99.5)}
    )

    broker.submit_target_position("AAPL", 5.0)  # SELL, trims long

    order = client.submitted_orders[0]
    assert order.limit_price == pytest.approx(round(99.5 * 0.995, 2), abs=1e-6)


def test_submit_target_position_extended_hours_rounds_fractional_to_whole_shares(monkeypatch):
    """
    Extended-hours orders must be whole shares even when going long (unlike
    RTH) — and must round DOWN, never up: rounding up an order that reduces
    an existing position (see the next test) can exceed what's actually
    held and gets rejected outright by Alpaca.
    """
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, client = _make_broker(monkeypatch, is_open=False, quotes={"AAPL": _FakeQuote()})

    broker.submit_target_position("AAPL", 5.7)

    assert client.submitted_orders[0].qty == 5.0


def test_submit_target_position_extended_hours_closing_order_never_exceeds_held_shares(monkeypatch):
    """
    Regression test: hit live. Closing a 30.8892-share position outside RTH
    was rounding the sell order up to 31 shares — more than actually held —
    and Alpaca rejected it with "insufficient qty available for order".
    """
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, client = _make_broker(monkeypatch, is_open=False, positions={"BAC": 30.8892}, quotes={"BAC": _FakeQuote()})

    broker.submit_target_position("BAC", 0.0)  # close to flat

    assert client.submitted_orders[0].qty == 30.0  # floored, not rounded up to 31


def test_submit_target_position_falls_back_to_market_order_when_extended_hours_disabled(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", False)
    broker, client = _make_broker(monkeypatch, is_open=False, quotes={"AAPL": _FakeQuote()})

    broker.submit_target_position("AAPL", 5.7)

    order = client.submitted_orders[0]
    assert not hasattr(order, "extended_hours") or order.extended_hours is None
    assert order.qty == 5.7  # old behavior: fractional market order, queues until next open


def test_submit_target_position_extended_hours_raises_without_a_quote(monkeypatch):
    monkeypatch.setattr("execution.broker_alpaca.settings.allow_extended_hours_trading", True)
    broker, _ = _make_broker(monkeypatch, is_open=False, quotes={"AAPL": _FakeQuote(ask_price=None, bid_price=None)})

    with pytest.raises(RuntimeError, match="No quote available"):
        broker.submit_target_position("AAPL", 5.0)


def test_get_positions_detailed_maps_all_fields(monkeypatch):
    broker, client = _make_broker(monkeypatch)
    client.get_all_positions = lambda: [
        _FakePosition(
            symbol="TSLA", qty=68.721, side="long", avg_entry_price=306.72,
            current_price=308.63, market_value=21209.36, cost_basis=21077.98,
            unrealized_pl=131.38, unrealized_plpc=0.00623,
        )
    ]

    result = broker.get_positions_detailed()

    assert len(result) == 1
    pos = result[0]
    assert pos["symbol"] == "TSLA"
    assert pos["side"] == "long"
    assert pos["qty"] == 68.721
    assert pos["unrealized_pl"] == pytest.approx(131.38)


def test_get_positions_detailed_empty_when_no_positions(monkeypatch):
    broker, _ = _make_broker(monkeypatch)
    assert broker.get_positions_detailed() == []
