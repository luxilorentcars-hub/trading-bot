"""Tests for the arbitrage module: triangle math, YES/NO math, scan, CLI."""

from __future__ import annotations

import pytest
from cbe_arbitrage import (
    BookTop,
    PaperArbLedger,
    Triangle,
    fetch_binance_book_tops,
    fetch_polymarket_book_top,
    format_opportunity,
    scan_triangles,
    triangular_edges,
    yes_no_edge,
)

TRI = Triangle("BTCUSDT", "ETHBTC", "ETHUSDT")


def make_books(btc=100.0, eth_btc=0.5, eth=50.0, spread=0.0, qty=10.0):
    """Books where mid prices are exactly consistent unless overridden."""
    half = spread / 2.0
    return {
        "BTCUSDT": BookTop("BTCUSDT", btc * (1 - half), qty, btc * (1 + half), qty),
        "ETHBTC": BookTop("ETHBTC", eth_btc * (1 - half), qty, eth_btc * (1 + half), qty),
        "ETHUSDT": BookTop("ETHUSDT", eth * (1 - half), qty, eth * (1 + half), qty),
    }


class TestTriangularMath:
    def test_efficient_market_no_edge(self):
        books = make_books(spread=0.001)
        results = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=10
        )
        assert all(o.edge_bps < 0 for o in results)

    def test_forward_edge_exact(self):
        # ETH overpriced in USDT: buy BTC@100, buy ETH@0.5 BTC, sell ETH@51
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 10.0, 51.1, 10.0)
        fwd, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        # gross multiplier = 51 / (100 * 0.5) = 1.02 -> +200 bps
        assert fwd.edge_bps == pytest.approx(200.0)
        assert fwd.is_actionable

    def test_backward_edge_detected(self):
        # ETH underpriced in USDT: buy ETH@49, sell for BTC@0.5, sell BTC@100
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 48.9, 10.0, 49.0, 10.0)
        _, bwd = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        # gross = (0.5 * 100) / 49 = 1.0204 -> ~+204 bps
        assert bwd.edge_bps == pytest.approx((50.0 / 49.0 - 1.0) * 10_000, rel=1e-9)

    def test_fees_kill_marginal_edge(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 50.1, 10.0, 50.2, 10.0)  # +20 bps gross
        no_fee, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        with_fee, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=10.0
        )
        assert no_fee.edge_bps > 0
        assert with_fee.edge_bps < 0  # 3 legs x 10 bps > 20 bps gross

    def test_max_notional_limited_by_thinnest_leg(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 0.1, 51.1, 10.0)  # only 0.1 ETH bid
        fwd, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        # 0.1 ETH * 0.5 BTC/ETH * 100 USDT/BTC = 5 USDT entry capital
        assert fwd.max_notional == pytest.approx(5.0)
        assert fwd.expected_profit == pytest.approx(5.0 * 0.02)

    def test_crossed_book_rejected(self):
        bad = BookTop("BTCUSDT", 101.0, 1.0, 100.0, 1.0)
        with pytest.raises(ValueError, match="crossed"):
            bad.validate()


class TestScan:
    def test_scan_filters_and_sorts(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 10.0, 51.1, 10.0)
        hits = scan_triangles([TRI], books, fee_bps=0.0, min_edge_bps=1.0)
        assert len(hits) == 1  # only the forward direction is positive
        assert hits[0].description.endswith("(forward)")

    def test_scan_skips_missing_books(self):
        books = make_books()
        del books["ETHBTC"]
        assert scan_triangles([TRI], books) == []

    def test_min_edge_threshold(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 10.0, 51.1, 10.0)  # +200 bps
        assert scan_triangles([TRI], books, fee_bps=0.0, min_edge_bps=500.0) == []


class TestYesNo:
    def test_edge_when_asks_sum_below_dollar(self):
        yes = BookTop("YES", 0.54, 100.0, 0.55, 100.0)
        no = BookTop("NO", 0.41, 80.0, 0.42, 80.0)
        opp = yes_no_edge(yes, no, fee_bps=0.0, market="test")
        # cost 0.97 -> $0.03 per pair, 80 pairs (thinner NO side)
        assert opp.expected_profit == pytest.approx(0.03 * 80.0)
        assert opp.edge_bps == pytest.approx(0.03 / 0.97 * 10_000, rel=1e-9)
        assert opp.is_actionable

    def test_no_edge_when_sum_above_dollar(self):
        yes = BookTop("YES", 0.59, 100.0, 0.60, 100.0)
        no = BookTop("NO", 0.44, 80.0, 0.45, 80.0)
        opp = yes_no_edge(yes, no)
        assert opp.edge_bps < 0
        assert not opp.is_actionable

    def test_fee_reduces_payout(self):
        yes = BookTop("YES", 0.49, 10.0, 0.50, 10.0)
        no = BookTop("NO", 0.48, 10.0, 0.49, 10.0)  # cost 0.99
        no_fee = yes_no_edge(yes, no, fee_bps=0.0)
        with_fee = yes_no_edge(yes, no, fee_bps=200.0)  # 2% fee -> payout 0.98
        assert no_fee.is_actionable
        assert not with_fee.is_actionable


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        assert timeout is not None
        self.calls.append({"url": url, "params": dict(params or {})})
        return _FakeResponse(self.payload)


class TestFetchers:
    def test_binance_book_tops(self):
        payload = [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "100.0",
                "bidQty": "2.0",
                "askPrice": "100.1",
                "askQty": "3.0",
            }
        ]
        session = _FakeSession(payload)
        books = fetch_binance_book_tops(["btcusdt"], session=session)
        assert books["BTCUSDT"].ask == pytest.approx(100.1)
        assert '"BTCUSDT"' in session.calls[0]["params"]["symbols"]

    def test_polymarket_book_top(self):
        payload = {
            "bids": [{"price": "0.53", "size": "10"}, {"price": "0.54", "size": "5"}],
            "asks": [{"price": "0.56", "size": "7"}, {"price": "0.55", "size": "3"}],
        }
        session = _FakeSession(payload)
        top = fetch_polymarket_book_top("tok123", session=session)
        assert top.bid == pytest.approx(0.54)  # best (highest) bid
        assert top.ask == pytest.approx(0.55)  # best (lowest) ask
        assert top.ask_qty == pytest.approx(3.0)

    def test_polymarket_empty_book_raises(self):
        session = _FakeSession({"bids": [], "asks": []})
        with pytest.raises(ValueError, match="empty book"):
            fetch_polymarket_book_top("tok123", session=session)


class TestLedgerAndCli:
    def test_ledger_caps_deployment(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 1000.0, 51.1, 1000.0)
        fwd, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        ledger = PaperArbLedger(capital_per_trade=100.0)
        profit = ledger.execute(fwd, now_ms=1)
        assert profit == pytest.approx(100.0 * 0.02)  # capped at 100, not max_notional
        assert ledger.realized_profit == pytest.approx(profit)
        assert ledger.executions[0]["ts"] == 1

    def test_ledger_ignores_non_actionable(self):
        books = make_books(spread=0.001)
        fwd, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=10.0
        )
        ledger = PaperArbLedger()
        assert ledger.execute(fwd) == 0.0
        assert ledger.executions == []

    def test_format_opportunity_contains_legs(self):
        books = make_books()
        books["ETHUSDT"] = BookTop("ETHUSDT", 51.0, 10.0, 51.1, 10.0)
        fwd, _ = triangular_edges(
            TRI, books["BTCUSDT"], books["ETHBTC"], books["ETHUSDT"], fee_bps=0.0
        )
        text = format_opportunity(fwd)
        assert "BUY  BTCUSDT" in text and "SELL ETHUSDT" in text

    def test_cli_synthetic_run(self, capsys):
        from run_arbitrage import main

        rc = main(["--synthetic", "--polls", "30", "--seed", "11", "--min-edge-bps", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PAPER MODE" in out
        # 30 seeded polls with 20% injection probability must find something
        assert "PAPER FILL" in out

    def test_cli_polymarket_requires_tokens(self):
        from run_arbitrage import main

        with pytest.raises(SystemExit, match="yes-token"):
            main(["--polymarket"])

    def test_cli_bad_triangle_spec(self):
        from run_arbitrage import main

        with pytest.raises(SystemExit, match="3 symbols"):
            main(["--triangle", "BTCUSDT,ETHBTC", "--synthetic"])
