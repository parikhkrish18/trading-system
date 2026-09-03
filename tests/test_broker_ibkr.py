"""
IBKRBroker exercised against a fake ib_insync IB() session -- no real TWS/
Gateway connection. Constructed via object.__new__ to skip __init__'s
_connect() call entirely (the same pattern the broker's own methods assume:
self.ib is the only thing they touch).
"""
from unittest.mock import patch

import pytest

from execution.broker_ibkr import IBKRBroker


class _FakeContract:
    def __init__(self, symbol):
        self.symbol = symbol


class _FakePosition:
    def __init__(self, symbol, position):
        self.contract = _FakeContract(symbol)
        self.position = position


class _FakeIBOrder:
    def __init__(self, order_id):
        self.orderId = order_id


class _FakeOrderStatus:
    def __init__(self, status="Submitted", filled=0.0):
        self.status = status
        self.filled = filled


class _FakeTrade:
    def __init__(self, order_id, status="Submitted"):
        self.order = _FakeIBOrder(order_id)
        self.orderStatus = _FakeOrderStatus(status)


class _FakeIB:
    def __init__(self, positions=None, trades=None):
        self._positions = dict(positions or {})
        self._trades = list(trades or [])
        self.placed = []
        self._next_order_id = 1

    def reqPositions(self):
        pass

    def sleep(self, seconds):
        pass

    def positions(self):
        return [_FakePosition(sym, qty) for sym, qty in self._positions.items()]

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        order_id = self._next_order_id
        self._next_order_id += 1
        trade = _FakeTrade(order_id)
        self.placed.append((contract, order, trade))
        self._trades.append(trade)
        return trade

    def trades(self):
        return list(self._trades)

    def openTrades(self):
        return list(self._trades)

    def cancelOrder(self, order):
        pass


def _make_broker(fake_ib: _FakeIB, mode: str = "paper") -> IBKRBroker:
    broker = object.__new__(IBKRBroker)
    broker.mode = mode
    broker.ib = fake_ib
    return broker


def test_submit_target_position_return_dict_carries_an_id_alias_of_order_id():
    """
    Regression test: trading_loop.py's _order_states reads order.get("id")
    to re-query get_order() after submission, but this broker used to return
    only "order_id" -- so the real order status was never re-queried for the
    default BROKER=ibkr configuration. "id" must be present and equal to
    "order_id", matching AlpacaBroker.submit_target_position's return shape
    (order.model_dump(), keyed by "id").
    """
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)

    result = broker.submit_target_position("AAPL", 10.0)

    assert result is not None
    assert result["id"] == result["order_id"]
    assert result["id"] == fake_ib.placed[0][2].order.orderId


def test_get_order_finds_a_trade_by_the_id_this_broker_returns():
    """
    End-to-end within this module: the "id" submit_target_position hands
    back must be exactly what get_order() can look up, since that's the
    whole point of the alias.
    """
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)

    submitted = broker.submit_target_position("AAPL", 10.0)
    fake_ib.placed[0][2].orderStatus = _FakeOrderStatus(status="Filled", filled=10.0)

    state = broker.get_order(submitted["id"])

    assert state == {"status": "filled", "filled": 10.0}


def test_submit_target_position_returns_none_when_no_trade_is_needed():
    fake_ib = _FakeIB(positions={"AAPL": 10.0})
    broker = _make_broker(fake_ib)

    assert broker.submit_target_position("AAPL", 10.0) is None
    assert fake_ib.placed == []


def test_submit_target_position_buy_side_from_flat():
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)

    result = broker.submit_target_position("AAPL", 15.0)

    assert result["side"] == "BUY"
    assert result["qty"] == 15
    contract, order, _trade = fake_ib.placed[0]
    assert contract.symbol == "AAPL"
    assert order.action == "BUY"


def test_submit_target_position_sell_side_reducing_a_long():
    fake_ib = _FakeIB(positions={"AAPL": 20.0})
    broker = _make_broker(fake_ib)

    result = broker.submit_target_position("AAPL", 5.0)

    assert result["side"] == "SELL"
    assert result["qty"] == 15  # abs(5 - 20)


def test_submit_target_position_qty_rounds_to_whole_shares():
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)

    result = broker.submit_target_position("AAPL", 10.4)

    assert result["qty"] == 10


def test_submit_target_position_skips_a_sub_share_delta():
    """delta < 1 share after rounding -> no order, same as the "no trade needed" case."""
    fake_ib = _FakeIB(positions={"AAPL": 10.0})
    broker = _make_broker(fake_ib)

    result = broker.submit_target_position("AAPL", 10.3)

    assert result is None
    assert fake_ib.placed == []


def test_get_order_returns_none_for_an_unknown_id():
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)
    assert broker.get_order("nonexistent-id") is None


def test_get_positions_reads_from_the_session():
    fake_ib = _FakeIB(positions={"AAPL": 10.0, "TSLA": -5.0})
    broker = _make_broker(fake_ib)

    positions = broker.get_positions()

    assert positions == {"AAPL": 10.0, "TSLA": -5.0}


def test_flatten_all_closes_every_open_position():
    fake_ib = _FakeIB(positions={"AAPL": 10.0, "TSLA": -5.0})
    broker = _make_broker(fake_ib)

    broker.flatten_all()

    actions_by_symbol = {c.symbol: o.action for c, o, _t in fake_ib.placed}
    # AAPL is long -> must SELL to close; TSLA is short -> must BUY to close.
    assert len(fake_ib.placed) == 2
    assert actions_by_symbol == {"AAPL": "SELL", "TSLA": "BUY"}


def test_flatten_all_skips_positions_already_at_zero():
    fake_ib = _FakeIB(positions={"AAPL": 0.0})
    broker = _make_broker(fake_ib)

    broker.flatten_all()

    assert fake_ib.placed == []


def test_flatten_all_cancels_open_orders_first():
    fake_ib = _FakeIB()
    broker = _make_broker(fake_ib)
    submitted = broker.submit_target_position("AAPL", 10.0)
    assert submitted is not None
    cancelled = []
    fake_ib.cancelOrder = lambda order: cancelled.append(order.orderId)

    fake_ib._positions = {}  # nothing left to close, just verifying the cancel step ran
    broker.flatten_all()

    assert cancelled == [fake_ib.placed[0][2].order.orderId]


def test_get_account_reports_mode_host_and_port_alongside_summary_items():
    fake_ib = _FakeIB()

    class _Item:
        def __init__(self, tag, value, currency="USD"):
            self.tag = tag
            self.value = value
            self.currency = currency

    fake_ib.accountSummary = lambda: [_Item("NetLiquidation", "150000.00")]
    broker = _make_broker(fake_ib, mode="paper")
    broker.host = "127.0.0.1"
    broker.port = 7497

    account = broker.get_account()

    assert account["NetLiquidation"] == "150000.00"
    assert account["mode"] == "paper"
    assert account["host"] == "127.0.0.1"
    assert account["port"] == 7497


def test_get_portfolio_value_prefers_usd_net_liquidation():
    fake_ib = _FakeIB()

    class _Item:
        def __init__(self, tag, value, currency):
            self.tag = tag
            self.value = value
            self.currency = currency

    fake_ib.accountSummary = lambda: [
        _Item("NetLiquidation", "999999", "EUR"),
        _Item("NetLiquidation", "150000.00", "USD"),
    ]
    broker = _make_broker(fake_ib)

    assert broker.get_portfolio_value() == 150000.00


def test_get_portfolio_value_falls_back_to_any_currency_if_no_usd_row():
    fake_ib = _FakeIB()

    class _Item:
        def __init__(self, tag, value, currency):
            self.tag = tag
            self.value = value
            self.currency = currency

    fake_ib.accountSummary = lambda: [_Item("NetLiquidation", "88000.0", "GBP")]
    broker = _make_broker(fake_ib)

    assert broker.get_portfolio_value() == 88000.0


def test_get_portfolio_value_raises_when_net_liquidation_is_entirely_absent():
    fake_ib = _FakeIB()
    fake_ib.accountSummary = lambda: []
    broker = _make_broker(fake_ib)

    with pytest.raises(RuntimeError, match="NetLiquidation"):
        broker.get_portfolio_value()


# ---------------------------------------------------------------------
# Construction / mode safety -- these hit __init__ directly (not the
# object.__new__ bypass used above), so _connect() must be patched out.
# ---------------------------------------------------------------------


def test_constructor_rejects_an_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        IBKRBroker(mode="sandbox")


def test_constructor_refuses_live_mode_without_explicit_confirmation():
    """Same deliberate-friction pattern as broker_alpaca.py: live trading must be opted into explicitly."""
    with pytest.raises(RuntimeError, match="confirm_live"):
        IBKRBroker(mode="live", confirm_live=False)


def test_constructor_live_mode_with_confirmation_attempts_to_connect():
    with patch.object(IBKRBroker, "_connect", return_value=None) as mock_connect:
        broker = IBKRBroker(mode="live", confirm_live=True)
        mock_connect.assert_called_once()
        assert broker.mode == "live"


def test_connect_retries_then_raises_a_clear_error_when_nothing_is_listening():
    broker = object.__new__(IBKRBroker)
    broker.host = "127.0.0.1"
    broker.port = 1  # nothing listens on port 1
    broker.client_id = 99
    broker.mode = "paper"

    class _FailingIB:
        def __init__(self):
            self.connect_attempts = 0

        def connect(self, host, port, clientId):
            self.connect_attempts += 1
            raise ConnectionRefusedError("no TWS/Gateway listening")

    broker.ib = _FailingIB()

    with pytest.raises(RuntimeError, match="Could not connect to IBKR"):
        broker._connect(retries=2, delay=0)

    assert broker.ib.connect_attempts == 2
