"""Test helpers for crypto-backtest-engine tests."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbe_data import Candle  # noqa: E402


def make_trend_candles(
    n: int, start: float = 100.0, step: float = 1.0, ts0: int = 1_600_000_000_000
) -> list[Candle]:
    """Strictly rising (or falling) series with tiny wicks — easy to reason about."""
    candles = []
    price = start
    for i in range(n):
        o = price
        c = price + step
        hi = max(o, c) + 0.25
        lo = max(0.01, min(o, c) - 0.25)
        candles.append(Candle(ts=ts0 + i * 3_600_000, open=o, high=hi, low=lo, close=c, volume=1.0))
        price = c
    return candles
