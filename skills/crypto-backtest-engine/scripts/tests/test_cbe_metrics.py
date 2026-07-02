"""Unit tests for cbe_metrics."""

from __future__ import annotations

import pytest
from cbe_metrics import compute_metrics, equity_returns, max_drawdown, ulcer_index


class TestMaxDrawdown:
    def test_known_drawdown(self):
        equity = [100.0, 120.0, 90.0, 95.0, 130.0]
        dd, duration = max_drawdown(equity)
        assert dd == pytest.approx(90.0 / 120.0 - 1.0)  # -25%
        assert duration == 2  # bars 90 and 95 under the 120 peak

    def test_monotonic_rise_no_drawdown(self):
        dd, duration = max_drawdown([100.0, 110.0, 120.0])
        assert dd == 0.0
        assert duration == 0


class TestEquityReturns:
    def test_simple(self):
        assert equity_returns([100.0, 110.0]) == [pytest.approx(0.10)]

    def test_zero_guard(self):
        assert equity_returns([0.0, 50.0]) == [0.0]


class TestUlcerIndex:
    def test_flat_curve_is_zero(self):
        assert ulcer_index([100.0] * 10) == pytest.approx(0.0)

    def test_drawdown_increases_index(self):
        calm = ulcer_index([100.0, 101.0, 102.0, 103.0])
        rough = ulcer_index([100.0, 80.0, 60.0, 100.0])
        assert rough > calm


class TestComputeMetrics:
    def make(self, equity, pnls=None, pnl_pcts=None, bpy=8760.0):
        pnls = pnls if pnls is not None else []
        pnl_pcts = pnl_pcts if pnl_pcts is not None else []
        return compute_metrics(equity, pnls, pnl_pcts, bpy, exposure=0.5)

    def test_total_return(self):
        m = self.make([10_000.0, 11_000.0, 12_000.0])
        assert m["total_return_pct"] == pytest.approx(20.0)
        assert m["net_profit"] == pytest.approx(2000.0)

    def test_win_rate_profit_factor(self):
        m = self.make(
            [10_000.0, 10_100.0, 10_050.0, 10_200.0],
            pnls=[100.0, -50.0, 150.0],
            pnl_pcts=[0.01, -0.005, 0.015],
        )
        assert m["num_trades"] == 3
        assert m["win_rate_pct"] == pytest.approx(66.67, abs=0.01)
        assert m["profit_factor"] == pytest.approx(250.0 / 50.0)
        assert m["max_consecutive_wins"] == 1
        assert m["max_consecutive_losses"] == 1

    def test_sharpe_sign(self):
        rising = [10_000.0 * (1.001**i) for i in range(200)]
        falling = [10_000.0 * (0.999**i) for i in range(200)]
        assert self.make(rising)["sharpe"] > 0
        assert self.make(falling)["sharpe"] < 0

    def test_cagr_annualization(self):
        # exactly one year of daily bars doubling -> CAGR 100%
        equity = [10_000.0 * (2.0 ** (i / 364)) for i in range(365)]
        m = self.make(equity, bpy=365.0)
        assert m["cagr_pct"] == pytest.approx(100.0, rel=0.02)

    def test_calmar_negative_dd(self):
        equity = [10_000.0, 12_000.0, 9_000.0, 13_000.0]
        m = self.make(equity)
        assert m["max_drawdown_pct"] == pytest.approx(-25.0)
        assert m["calmar"] > 0

    def test_requires_two_points(self):
        with pytest.raises(ValueError):
            self.make([10_000.0])

    def test_sqn_zero_without_variance(self):
        m = self.make([10_000.0, 10_100.0], pnls=[50.0], pnl_pcts=[0.005])
        assert m["sqn"] == 0.0  # single trade -> no std -> undefined
