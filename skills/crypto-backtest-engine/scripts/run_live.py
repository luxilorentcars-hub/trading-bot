#!/usr/bin/env python3
"""CLI for live/paper trading with the crypto backtest engine's strategies.

SAFETY: paper mode is the default. Real order placement requires ALL of:
  --broker binance  --live  --i-understand-live
and BINANCE_API_KEY / BINANCE_API_SECRET in the environment. Without every
one of these, orders are simulated (paper) or dry-run (logged, not sent).

Examples:
  # Paper trade one poll on the latest closed 1h BTC bar (synthetic feed)
  python run_live.py --synthetic --strategy sma_cross \
      --params '{"fast":20,"slow":100}'

  # Paper trade against live Binance data (read-only price feed), loop hourly
  python run_live.py --symbol BTCUSDT --interval 1h --strategy donchian_breakout \
      --broker paper --capital 5000 --loop --poll-seconds 3600

  # REAL orders (spot, long/flat only) — every guard must be set explicitly
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python run_live.py \
      --symbol BTCUSDT --interval 1h --strategy sma_cross \
      --broker binance --live --i-understand-live \
      --max-notional 200 --state state/btc.json --loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cbe_broker import BinanceBroker, PaperBroker
from cbe_data import INTERVAL_MS, generate_synthetic_ohlcv, load_or_fetch
from cbe_live import LiveConfig, LiveTrader, format_decision
from cbe_strategies import STRATEGY_REGISTRY, build_strategy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live/paper crypto trader")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
    p.add_argument("--synthetic", action="store_true", help="Use synthetic feed (offline demo)")
    p.add_argument("--window-bars", type=int, default=400)

    p.add_argument("--strategy", default="sma_cross", choices=sorted(STRATEGY_REGISTRY))
    p.add_argument("--params", default="{}", help="Strategy params as JSON")

    p.add_argument("--broker", default="paper", choices=["paper", "binance"])
    p.add_argument("--capital", type=float, default=10_000.0, help="Paper broker starting cash")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    p.add_argument(
        "--sizing", default="fixed_fraction", choices=["fixed_fraction", "risk_per_trade"]
    )
    p.add_argument("--fraction", type=float, default=0.95)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--stop-loss", type=float, default=None)
    p.add_argument("--max-notional", type=float, default=None, help="Hard cap per position")
    p.add_argument("--min-order-notional", type=float, default=10.0)
    p.add_argument("--allow-short", action="store_true", help="Only honored by shorting brokers")

    p.add_argument("--state", default=None, help="JSON state file (restart-safe)")
    p.add_argument("--loop", action="store_true", help="Poll continuously")
    p.add_argument(
        "--poll-seconds", type=int, default=None, help="Loop interval (default: bar length)"
    )
    p.add_argument("--max-polls", type=int, default=0, help="Stop after N polls (0 = unlimited)")

    # Real-order guards — ALL required for live placement.
    p.add_argument("--live", action="store_true", help="Send REAL orders (binance broker only)")
    p.add_argument(
        "--i-understand-live",
        action="store_true",
        help="Required acknowledgement that --live places real orders with real funds",
    )
    return p


def make_broker(args: argparse.Namespace):
    if args.broker == "paper":
        return PaperBroker(cash=args.capital, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    # binance
    live = args.live
    if live and not args.i_understand_live:
        raise SystemExit("Refusing --live without --i-understand-live. Real orders use real funds.")
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if live and (not api_key or not api_secret):
        raise SystemExit(
            "--live requires BINANCE_API_KEY and BINANCE_API_SECRET in the environment."
        )
    return BinanceBroker(api_key=api_key, api_secret=api_secret, live=live)


def load_window(args: argparse.Namespace):
    if args.synthetic:
        return generate_synthetic_ohlcv(args.window_bars + 5, interval=args.interval, seed=7)
    return load_or_fetch(args.symbol, args.interval, max_bars=args.window_bars + 5)


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LiveConfig(
        symbol=args.symbol,
        interval=args.interval,
        sizing_mode=args.sizing,
        fraction=args.fraction,
        risk_pct=args.risk_pct,
        stop_loss_pct=args.stop_loss,
        max_notional=args.max_notional,
        min_order_notional=args.min_order_notional,
        allow_short=args.allow_short,
        window_bars=args.window_bars,
    )
    broker = make_broker(args)
    strategy = build_strategy(args.strategy, json.loads(args.params))
    trader = LiveTrader(strategy, broker, config, state_path=args.state)

    mode = (
        "LIVE (REAL ORDERS)"
        if (args.broker == "binance" and args.live)
        else ("paper" if args.broker == "paper" else "binance dry-run")
    )
    print(
        f"# {args.strategy} on {args.symbol} {args.interval} | broker={args.broker} | mode={mode}"
    )

    poll_seconds = args.poll_seconds or INTERVAL_MS[args.interval] // 1000
    polls = 0
    while True:
        window = load_window(args)
        decision = trader.step(window, bar_closed=True)
        if decision is not None:
            print(format_decision(decision))
        else:
            print("# no new closed bar")
        polls += 1
        if not args.loop or (args.max_polls and polls >= args.max_polls):
            break
        time.sleep(poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
