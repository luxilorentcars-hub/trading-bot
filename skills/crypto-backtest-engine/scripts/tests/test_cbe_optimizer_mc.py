"""Tests for grid search, walk-forward analysis, and Monte Carlo."""

from __future__ import annotations

import pytest
from cbe_engine import BacktestConfig
from cbe_monte_carlo import monte_carlo_trades
from cbe_optimizer import expand_grid, grid_search, walk_forward


class TestExpandGrid:
    def test_cartesian_product(self):
        combos = expand_grid({"fast": [10, 20], "slow": [50, 100]})
        assert len(combos) == 4
        assert {"fast": 10, "slow": 50} in combos

    def test_empty_grid(self):
        assert expand_grid({}) == [{}]


class TestGridSearch:
    def test_sorted_best_first(self, synthetic_candles):
        results = grid_search(
            synthetic_candles,
            "sma_cross",
            {"fast": [5, 10], "slow": [30, 60]},
            BacktestConfig(),
            metric="sharpe",
            min_trades=1,
        )
        assert len(results) >= 2
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_combos_skipped(self, synthetic_candles):
        # fast >= slow raises in the constructor and must be skipped silently
        results = grid_search(
            synthetic_candles,
            "sma_cross",
            {"fast": [50], "slow": [30, 100]},
            min_trades=1,
        )
        assert all(r.params["fast"] < r.params["slow"] for r in results)

    def test_unknown_metric_raises(self, synthetic_candles):
        with pytest.raises(KeyError):
            grid_search(
                synthetic_candles,
                "sma_cross",
                {"fast": [5], "slow": [30]},
                metric="alpha_magic",
                min_trades=1,
            )

    def test_min_trades_filter(self, synthetic_candles):
        # a 2000-bar series can't produce 500 sma-cross trades
        results = grid_search(
            synthetic_candles,
            "sma_cross",
            {"fast": [5], "slow": [30]},
            min_trades=500,
        )
        assert results == []


class TestWalkForward:
    def test_splits_and_stitching(self, synthetic_candles):
        result = walk_forward(
            synthetic_candles,
            "sma_cross",
            {"fast": [5, 10], "slow": [30, 60]},
            BacktestConfig(),
            n_splits=3,
            min_trades=1,
        )
        assert len(result.splits) <= 3
        assert result.splits, "expected at least one split to optimize"
        # OOS equity stitched over the test regions only
        assert len(result.oos_equity) > 0
        # test ranges must not overlap and must be chronological
        ranges = [s.test_range for s in result.splits]
        for (a_start, a_end), (b_start, _) in zip(ranges, ranges[1:]):
            assert a_end <= b_start
        # every split trained on data strictly before its test window
        for s in result.splits:
            assert s.train_range[1] == s.test_range[0]

    def test_requires_enough_data(self):
        from cbe_data import generate_synthetic_ohlcv

        tiny = generate_synthetic_ohlcv(30, seed=1)
        with pytest.raises(ValueError, match="not enough data"):
            walk_forward(tiny, "sma_cross", {"fast": [5], "slow": [10]}, n_splits=5)


class TestMonteCarlo:
    def test_deterministic_for_seed(self):
        pnls = [0.02, -0.01, 0.03, -0.02, 0.01] * 10
        a = monte_carlo_trades(pnls, n_sims=200, seed=7)
        b = monte_carlo_trades(pnls, n_sims=200, seed=7)
        assert a == b

    def test_percentiles_ordered(self):
        pnls = [0.05, -0.03, 0.02, -0.01, 0.04, -0.02] * 8
        result = monte_carlo_trades(pnls, n_sims=500, seed=3)
        assert result.final_equity_p5 <= result.final_equity_p50 <= result.final_equity_p95
        assert result.max_drawdown_p95_pct <= result.max_drawdown_p50_pct <= 0.0

    def test_all_winning_trades_never_lose(self):
        result = monte_carlo_trades([0.01] * 30, n_sims=100, seed=1)
        assert result.prob_loss_pct == 0.0
        assert result.prob_ruin_pct == 0.0

    def test_empty_trades_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_trades([])
