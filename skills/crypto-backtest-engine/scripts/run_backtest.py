#!/usr/bin/env python3
"""CLI entry point for the crypto backtest engine.

Examples:
  # Quick offline sanity check on synthetic data
  python run_backtest.py --synthetic 5000 --strategy sma_cross \
      --params '{"fast": 20, "slow": 100}'

  # Real BTC data from Binance (public API, cached to CSV)
  python run_backtest.py --symbol BTCUSDT --interval 4h --max-bars 8000 \
      --strategy donchian_breakout --trailing-stop 0.08 --monte-carlo 2000

  # Grid search + walk-forward robustness check
  python run_backtest.py --csv data/btc_1h.csv --strategy sma_cross \
      --optimize '{"fast": [10, 20, 50], "slow": [100, 200]}' --walk-forward 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cbe_data import bars_per_year, candles_from_csv, generate_synthetic_ohlcv, load_or_fetch
from cbe_engine import BacktestConfig, run_backtest
from cbe_monte_carlo import monte_carlo_trades
from cbe_optimizer import grid_search, walk_forward
from cbe_report import format_report, result_to_dict
from cbe_strategies import DEFAULT_GRIDS, STRATEGY_REGISTRY, build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto backtest engine")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbol", help="Binance symbol to fetch, e.g. BTCUSDT")
    source.add_argument("--csv", help="Local OHLCV CSV path")
    source.add_argument("--synthetic", type=int, metavar="N_BARS", help="Synthetic data bars")

    parser.add_argument("--interval", default="1h", help="Bar interval (default 1h)")
    parser.add_argument("--max-bars", type=int, default=20_000, help="Max bars to fetch")
    parser.add_argument("--cache-dir", default="data_cache", help="CSV cache dir for fetches")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic data")

    parser.add_argument(
        "--strategy",
        default="sma_cross",
        choices=sorted(STRATEGY_REGISTRY),
        help="Strategy name",
    )
    parser.add_argument("--params", default="{}", help="Strategy params as JSON")

    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument(
        "--sizing",
        default="fixed_fraction",
        choices=["fixed_fraction", "risk_per_trade", "vol_target"],
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--vol-target", type=float, default=0.40)
    parser.add_argument("--stop-loss", type=float, default=None, help="e.g. 0.05 = 5%%")
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--trailing-stop", type=float, default=None)
    parser.add_argument("--short-funding-bps", type=float, default=0.0)

    parser.add_argument(
        "--optimize",
        default=None,
        metavar="GRID_JSON",
        help='Grid search, e.g. \'{"fast": [10, 20], "slow": [50, 100]}\'; '
        '"default" uses the built-in grid for the strategy',
    )
    parser.add_argument("--metric", default="sharpe", help="Ranking metric for optimization")
    parser.add_argument("--top", type=int, default=5, help="Show top-N grid results")
    parser.add_argument(
        "--walk-forward", type=int, default=0, metavar="N_SPLITS", help="Walk-forward splits"
    )
    parser.add_argument(
        "--monte-carlo", type=int, default=0, metavar="N_SIMS", help="Monte Carlo simulations"
    )
    parser.add_argument("--json-out", default=None, help="Write full JSON results to path")
    return parser


def load_candles(args: argparse.Namespace) -> list:
    if args.synthetic is not None:
        return generate_synthetic_ohlcv(args.synthetic, interval=args.interval, seed=args.seed)
    if args.csv is not None:
        return candles_from_csv(args.csv)
    return load_or_fetch(
        args.symbol, args.interval, cache_dir=args.cache_dir, max_bars=args.max_bars
    )


def build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        sizing_mode=args.sizing,
        fraction=args.fraction,
        risk_pct=args.risk_pct,
        vol_target_annual=args.vol_target,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        short_funding_daily_bps=args.short_funding_bps,
        bars_per_year=bars_per_year(args.interval),
    )


def resolve_grid(args: argparse.Namespace) -> dict[str, list]:
    if args.optimize == "default":
        return DEFAULT_GRIDS.get(args.strategy, {})
    return json.loads(args.optimize)


def main(argv: list | None = None) -> str:
    args = build_parser().parse_args(argv)
    candles = load_candles(args)
    if not candles:
        raise SystemExit("no candles loaded")
    config = build_config(args)
    params = json.loads(args.params)

    grid_lines: list = []
    if args.optimize:
        grid = resolve_grid(args)
        ranked = grid_search(candles, args.strategy, grid, config, metric=args.metric)
        if not ranked:
            raise SystemExit("grid search produced no valid results (too few trades?)")
        grid_lines.append(f"-- Grid search: top {min(args.top, len(ranked))} by {args.metric} --")
        for r in ranked[: args.top]:
            grid_lines.append(
                f"  {r.params}  {args.metric}={r.score:.3f}  "
                f"return={r.metrics['total_return_pct']:+.2f}%  trades={r.metrics['num_trades']}"
            )
        params = ranked[0].params  # final run uses the winner

    strategy = build_strategy(args.strategy, params)
    result = run_backtest(candles, strategy, config)

    walk = None
    if args.walk_forward > 0:
        grid = resolve_grid(args) if args.optimize else DEFAULT_GRIDS.get(args.strategy, {})
        walk = walk_forward(
            candles, args.strategy, grid, config, n_splits=args.walk_forward, metric=args.metric
        )

    mc = None
    if args.monte_carlo > 0 and result.trades:
        mc = monte_carlo_trades(
            [t.pnl_pct for t in result.trades],
            initial_capital=config.initial_capital,
            n_sims=args.monte_carlo,
        )

    report = format_report(result, monte_carlo=mc, walk=walk)
    if grid_lines:
        report = "\n".join(grid_lines) + "\n\n" + report

    if args.json_out:
        payload = result_to_dict(result, monte_carlo=mc, walk=walk)
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        report += f"\nJSON results written to {args.json_out}"
    return report


if __name__ == "__main__":
    print(main())
