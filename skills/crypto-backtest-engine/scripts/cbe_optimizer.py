"""Parameter optimization and walk-forward analysis.

``grid_search`` exhaustively evaluates a parameter grid on one data set.
``walk_forward`` guards against overfitting: it re-optimizes on rolling
in-sample windows and stitches together only OUT-OF-SAMPLE results. The
walk-forward efficiency (OOS/IS performance ratio) is the headline
robustness number — below ~0.5 usually means the edge is curve-fit.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass, field

from cbe_data import Candle
from cbe_engine import BacktestConfig, run_backtest
from cbe_strategies import build_strategy

DEFAULT_METRIC = "sharpe"


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """{'fast': [10, 20], 'slow': [50]} -> [{'fast':10,'slow':50}, ...]"""
    if not param_grid:
        return [{}]
    keys = sorted(param_grid)
    combos = itertools.product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in combos]


@dataclass
class GridResult:
    params: dict
    score: float
    metrics: dict


def grid_search(
    candles: list[Candle],
    strategy_name: str,
    param_grid: dict[str, list],
    config: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
    min_trades: int = 3,
) -> list[GridResult]:
    """Evaluate every combo; return results sorted best-first by ``metric``.

    Combos that raise (invalid params, e.g. fast >= slow) or produce fewer
    than ``min_trades`` trades are skipped — too few trades means the score
    is statistical noise.
    """
    results: list[GridResult] = []
    for params in expand_grid(param_grid):
        try:
            strategy = build_strategy(strategy_name, params)
        except ValueError:
            continue  # invalid combo (constructor guard)
        run = run_backtest(candles, strategy, deepcopy(config) if config else None)
        if run.metrics["num_trades"] < min_trades:
            continue
        if metric not in run.metrics:
            raise KeyError(f"metric '{metric}' not found; available: {sorted(run.metrics)}")
        results.append(GridResult(params=params, score=run.metrics[metric], metrics=run.metrics))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


@dataclass
class WalkForwardSplit:
    split: int
    train_range: tuple[int, int]  # candle index range [start, end)
    test_range: tuple[int, int]
    best_params: dict
    train_score: float
    test_metrics: dict


@dataclass
class WalkForwardResult:
    splits: list[WalkForwardSplit] = field(default_factory=list)
    oos_equity: list[tuple[int, float]] = field(default_factory=list)
    oos_return_pct: float = 0.0
    avg_train_score: float = 0.0
    avg_test_score: float = 0.0
    walk_forward_efficiency: float = 0.0


def walk_forward(
    candles: list[Candle],
    strategy_name: str,
    param_grid: dict[str, list],
    config: BacktestConfig | None = None,
    n_splits: int = 4,
    train_bars: int | None = None,
    metric: str = DEFAULT_METRIC,
    anchored: bool = False,
    min_trades: int = 3,
) -> WalkForwardResult:
    """Rolling re-optimization with out-of-sample stitching.

    The series after the initial training window is cut into ``n_splits``
    test chunks. For each chunk, the grid is optimized on the preceding
    ``train_bars`` bars (or everything before it when ``anchored``), and
    the winning params are evaluated on the untouched chunk.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    n = len(candles)
    if train_bars is None:
        train_bars = n // (n_splits + 1)
    test_len = (n - train_bars) // n_splits
    if train_bars < 10 or test_len < 10:
        raise ValueError(
            f"not enough data: {n} bars for train_bars={train_bars}, n_splits={n_splits}"
        )

    result = WalkForwardResult()
    initial_capital = (config or BacktestConfig()).initial_capital
    equity = initial_capital
    for split in range(n_splits):
        test_start = train_bars + split * test_len
        test_end = n if split == n_splits - 1 else test_start + test_len
        train_start = 0 if anchored else test_start - train_bars
        train_slice = candles[train_start:test_start]
        test_slice = candles[test_start:test_end]

        ranked = grid_search(train_slice, strategy_name, param_grid, config, metric, min_trades)
        if not ranked:
            continue  # no combo produced enough trades in this window
        best = ranked[0]
        strategy = build_strategy(strategy_name, best.params)
        test_run = run_backtest(test_slice, strategy, deepcopy(config) if config else None)

        # stitch OOS equity by compounding chunk returns
        chunk_scale = equity / initial_capital
        for ts, eq in test_run.equity_curve:
            result.oos_equity.append((ts, eq * chunk_scale))
        equity = result.oos_equity[-1][1]

        result.splits.append(
            WalkForwardSplit(
                split=split,
                train_range=(train_start, test_start),
                test_range=(test_start, test_end),
                best_params=best.params,
                train_score=best.score,
                test_metrics=test_run.metrics,
            )
        )

    if result.splits:
        result.avg_train_score = sum(s.train_score for s in result.splits) / len(result.splits)
        result.avg_test_score = sum(s.test_metrics[metric] for s in result.splits) / len(
            result.splits
        )
        result.oos_return_pct = (equity / initial_capital - 1.0) * 100.0
        if result.avg_train_score != 0:
            result.walk_forward_efficiency = result.avg_test_score / abs(result.avg_train_score)
    return result
