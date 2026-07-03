#!/usr/bin/env python3
"""CLI for the market-neutral arbitrage scanner (paper only — no real orders).

Examples:
  # Offline demo: synthetic books with occasional injected arbitrage
  python run_arbitrage.py --synthetic --polls 20

  # Scan live Binance books for a triangle, once
  python run_arbitrage.py --triangle BTCUSDT,ETHBTC,ETHUSDT

  # Continuous scan, 5s cadence, paper-executing every hit >= 3 bps
  python run_arbitrage.py --triangle BTCUSDT,ETHBTC,ETHUSDT \
      --loop --poll-seconds 5 --min-edge-bps 3 --capital 500

  # Polymarket YES/NO check (public book endpoint; lawful-access regions only)
  python run_arbitrage.py --polymarket --yes-token <id> --no-token <id>
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cbe_arb_dashboard import write_dashboard
from cbe_arbitrage import (
    DEFAULT_ALT_UNIVERSE,
    BookTop,
    PaperArbLedger,
    Triangle,
    build_triangles,
    fetch_binance_book_tops,
    fetch_polymarket_book_top,
    format_opportunity,
    scan_triangles,
    universe_symbols,
    yes_no_edge,
)

DEFAULT_TRIANGLE = "BTCUSDT,ETHBTC,ETHUSDT"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Market-neutral arbitrage scanner (paper only)")
    p.add_argument(
        "--triangle",
        action="append",
        default=None,
        metavar="A,B,C",
        help=f"Triangle as X/QUOTE,Y/X,Y/QUOTE (default {DEFAULT_TRIANGLE}); repeatable",
    )
    p.add_argument("--synthetic", action="store_true", help="Offline demo with synthetic books")
    p.add_argument("--seed", type=int, default=11, help="Seed for synthetic books")

    # Universe mode: auto-build triangles from alts bridged through BTC/ETH/BNB.
    p.add_argument(
        "--assets",
        default=None,
        help="Comma-separated alt symbols to auto-triangulate (e.g. SOL,XRP,ADA); "
        "'default' uses a built-in liquid-alt universe",
    )
    p.add_argument("--bridges", default="BTC,ETH", help="Bridge assets (default BTC,ETH)")
    p.add_argument("--quote", default="USDT", help="Quote/accounting currency (default USDT)")

    p.add_argument("--dashboard", default=None, help="Write a live HTML dashboard to this path")

    p.add_argument("--polymarket", action="store_true", help="Scan a Polymarket YES/NO pair")
    p.add_argument("--yes-token", default=None, help="Polymarket CLOB token id for YES")
    p.add_argument("--no-token", default=None, help="Polymarket CLOB token id for NO")
    p.add_argument("--market-name", default="market", help="Label for the Polymarket pair")

    p.add_argument("--fee-bps", type=float, default=10.0, help="Taker fee per leg (bps)")
    p.add_argument("--min-edge-bps", type=float, default=1.0)
    p.add_argument("--capital", type=float, default=1_000.0, help="Paper capital per execution")

    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--polls", type=int, default=1, help="Number of polls (ignored with --loop)")
    return p


def parse_triangles(args) -> list:
    """Resolve the triangle set: explicit --triangle, --assets universe, or default."""
    if args.assets is not None:
        alts = (
            DEFAULT_ALT_UNIVERSE
            if args.assets == "default"
            else [s.strip().upper() for s in args.assets.split(",") if s.strip()]
        )
        bridges = [s.strip().upper() for s in args.bridges.split(",") if s.strip()]
        triangles = build_triangles(alts, bridges, args.quote)
        if not triangles:
            raise SystemExit("--assets produced no valid triangles")
        return triangles
    triangles = []
    for spec in args.triangle or [DEFAULT_TRIANGLE]:
        parts = [s.strip().upper() for s in spec.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--triangle must have 3 symbols, got: {spec}")
        triangles.append(Triangle(*parts))
    return triangles


def synthetic_books(triangle: Triangle, rng: random.Random) -> dict:
    """Mostly-efficient synthetic books; ~20% of snapshots embed a real edge."""
    btc = rng.uniform(20_000, 40_000)
    eth_btc = rng.uniform(0.05, 0.08)
    eth = btc * eth_btc
    spread = 0.0004
    skew = 1.0 + rng.uniform(0.004, 0.008) * (1 if rng.random() < 0.5 else -1)
    if rng.random() < 0.20:
        eth *= skew  # misprice the Y/QUOTE leg -> creates a loop edge
    qty = lambda: rng.uniform(0.5, 5.0)  # noqa: E731 - tiny local factory
    return {
        triangle.leg_a: BookTop(
            triangle.leg_a, btc * (1 - spread), qty(), btc * (1 + spread), qty()
        ),
        triangle.leg_b: BookTop(
            triangle.leg_b, eth_btc * (1 - spread), qty(), eth_btc * (1 + spread), qty()
        ),
        triangle.leg_c: BookTop(
            triangle.leg_c, eth * (1 - spread), qty(), eth * (1 + spread), qty()
        ),
    }


def scan_once(args, triangles, ledger, rng, poll_index: int) -> int:
    if args.synthetic:
        books = {}
        for tri in triangles:
            books.update(synthetic_books(tri, rng))
    else:
        books = fetch_binance_book_tops(universe_symbols(triangles))
    hits = scan_triangles(triangles, books, fee_bps=args.fee_bps, min_edge_bps=args.min_edge_bps)
    for opp in hits:
        print(format_opportunity(opp))
        profit = ledger.execute(opp, now_ms=poll_index)
        print(f"  PAPER FILL: +{profit:.4f} (cumulative {ledger.realized_profit:+.4f})")
    if not hits:
        print(f"# poll {poll_index}: no edge >= {args.min_edge_bps} bps")
    return len(hits)


def scan_polymarket(args) -> int:
    if not args.yes_token or not args.no_token:
        raise SystemExit("--polymarket requires --yes-token and --no-token")
    yes_book = fetch_polymarket_book_top(args.yes_token)
    no_book = fetch_polymarket_book_top(args.no_token)
    opp = yes_no_edge(yes_book, no_book, fee_bps=args.fee_bps, market=args.market_name)
    print(format_opportunity(opp))
    if not opp.is_actionable:
        print("# no YES/NO edge at current asks")
    return 1 if opp.is_actionable else 0


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("# PAPER MODE — this scanner never places real orders")
    if args.polymarket:
        scan_polymarket(args)
        return 0

    triangles = parse_triangles(args)
    print(f"# scanning {len(triangles)} triangle(s)")
    ledger = PaperArbLedger(capital_per_trade=args.capital)
    rng = random.Random(args.seed)
    timeline: list = []
    poll = 0
    while True:
        scan_once(args, triangles, ledger, rng, poll)
        timeline.append((poll, ledger.realized_profit))
        if args.dashboard:
            _write_dashboard(args, triangles, ledger, timeline, poll)
        poll += 1
        if args.loop:
            time.sleep(args.poll_seconds)
        elif poll >= args.polls:
            break
    if ledger.executions:
        print(
            f"# paper summary: {len(ledger.executions)} executions, "
            f"total profit {ledger.realized_profit:+.4f} on {args.capital:.0f}/trade"
        )
    if args.dashboard:
        print(f"# dashboard written to {args.dashboard} — open it in your browser")
    return 0


def _write_dashboard(args, triangles, ledger, timeline, poll) -> None:
    meta = {
        "title": "Crypto Arbitrage Scanner",
        "subtitle": f"{len(triangles)} triangles | bridges {args.bridges} | quote {args.quote} "
        f"| fee {args.fee_bps} bps/leg | min edge {args.min_edge_bps} bps",
        "polls": poll + 1,
        "refresh_seconds": args.poll_seconds if args.loop else None,
        "top": 15,
        "recent": 15,
    }
    write_dashboard(args.dashboard, timeline, ledger.executions, ledger.realized_profit, meta)


if __name__ == "__main__":
    raise SystemExit(main())
