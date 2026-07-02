"""Tests for the broker layer: paper fills and the guarded Binance client."""

from __future__ import annotations

import pytest
from cbe_broker import (
    SIDE_BUY,
    SIDE_SELL,
    BinanceBroker,
    OrderResult,
    PaperBroker,
    _trim_qty,
)
from cbe_strategies import FLAT, LONG, SHORT


class TestPaperBroker:
    def test_buy_creates_long_position(self):
        b = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
        b.market_order("BTCUSDT", SIDE_BUY, 0.1, mark_price=20_000.0)
        pos = b.get_position("BTCUSDT")
        assert pos.qty == pytest.approx(0.1)
        assert pos.direction == LONG
        assert b.cash == pytest.approx(10_000.0 - 2_000.0)

    def test_sell_creates_short_position(self):
        b = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
        b.market_order("BTCUSDT", SIDE_SELL, 0.1, mark_price=20_000.0)
        pos = b.get_position("BTCUSDT")
        assert pos.qty == pytest.approx(-0.1)
        assert pos.direction == SHORT
        assert b.cash == pytest.approx(12_000.0)  # short adds cash

    def test_fees_and_slippage_applied(self):
        b = PaperBroker(cash=10_000.0, fee_bps=10.0, slippage_bps=5.0)
        result = b.market_order("BTCUSDT", SIDE_BUY, 1.0, mark_price=100.0)
        assert result.price == pytest.approx(100.0 * 1.0005)  # buy slips up
        # cash reduced by notional + fee
        assert b.cash < 10_000.0 - 100.0

    def test_equity_marks_position(self):
        b = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
        b.market_order("BTCUSDT", SIDE_BUY, 0.1, mark_price=20_000.0)
        # price rises to 25k -> equity = 8000 cash + 0.1*25000
        assert b.get_equity(25_000.0) == pytest.approx(8_000.0 + 2_500.0)

    def test_flip_through_zero_resets_avg_price(self):
        b = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
        b.market_order("BTCUSDT", SIDE_BUY, 0.1, mark_price=100.0)
        b.market_order("BTCUSDT", SIDE_SELL, 0.3, mark_price=200.0)  # now net short 0.2
        pos = b.get_position("BTCUSDT")
        assert pos.qty == pytest.approx(-0.2)
        assert pos.avg_price == pytest.approx(200.0)

    def test_non_positive_qty_skipped(self):
        b = PaperBroker()
        result = b.market_order("BTCUSDT", SIDE_BUY, 0.0, mark_price=100.0)
        assert result.status == "SKIPPED"

    def test_supports_short_flag(self):
        assert PaperBroker().supports_short is True


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Records signed requests and returns canned payloads."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def request(self, method, url, params=None, headers=None, timeout=None):
        assert timeout is not None
        assert headers and "X-MBX-APIKEY" in headers
        assert "signature" in params and "timestamp" in params
        self.requests.append({"method": method, "url": url, "params": dict(params)})
        return _FakeResponse(self.payloads.pop(0) if self.payloads else {})


class TestBinanceBrokerGuards:
    def test_live_requires_credentials(self):
        with pytest.raises(ValueError, match="requires api_key"):
            BinanceBroker(live=True)

    def test_no_short_support(self):
        assert BinanceBroker().supports_short is False

    def test_dry_run_does_not_send_order(self):
        session = _FakeSession([])
        broker = BinanceBroker(api_key="k", api_secret="s", live=False, session=session)
        result = broker.market_order("BTCUSDT", SIDE_BUY, 0.01, mark_price=30_000.0)
        assert result.status == "DRY_RUN"
        assert session.requests == []  # nothing sent

    def test_live_order_is_signed_and_sent(self):
        payload = {
            "status": "FILLED",
            "executedQty": "0.01",
            "fills": [{"price": "30000.0", "qty": "0.01"}],
        }
        session = _FakeSession([payload])
        broker = BinanceBroker(api_key="k", api_secret="s", live=True, session=session)
        result = broker.market_order("BTCUSDT", SIDE_BUY, 0.01, mark_price=29_900.0)
        assert result.status == "FILLED"
        assert result.price == pytest.approx(30_000.0)  # from fills, not mark
        assert session.requests[0]["method"] == "POST"
        assert session.requests[0]["params"]["type"] == "MARKET"

    def test_signature_is_deterministic(self):
        broker = BinanceBroker(api_key="k", api_secret="secret")
        sig1 = broker._sign({"a": 1, "b": 2})
        sig2 = broker._sign({"a": 1, "b": 2})
        assert sig1 == sig2 and len(sig1) == 64  # sha256 hex

    def test_get_equity_sums_quote_balance(self):
        payload = {"balances": [{"asset": "USDT", "free": "500.0", "locked": "0.0"}]}
        session = _FakeSession([payload])
        broker = BinanceBroker(api_key="k", api_secret="s", session=session)
        assert broker.get_equity(30_000.0) == pytest.approx(500.0)


class TestTrimQty:
    def test_trims_trailing_zeros(self):
        assert _trim_qty(0.010000) == "0.01"
        assert _trim_qty(1.0) == "1"

    def test_precision(self):
        assert _trim_qty(0.123456789) == "0.123457"


def test_order_result_defaults():
    r = OrderResult("BTCUSDT", SIDE_BUY, 1.0, 100.0, 100.0, "FILLED")
    assert r.raw == {} and r.reason == ""
    assert FLAT == 0  # sanity on shared constants
