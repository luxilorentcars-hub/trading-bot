"""Data layer for the crypto backtest engine.

Three interchangeable OHLCV sources:
  1. Binance public REST API (no API key required) with pagination + CSV caching
  2. Local CSV files (timestamp_ms,open,high,low,close,volume)
  3. Deterministic synthetic data (regime-switching GBM) for tests and offline work

All sources return ``list[Candle]`` sorted by timestamp.
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4

MS_PER_MINUTE = 60_000
INTERVAL_MS = {
    "1m": MS_PER_MINUTE,
    "3m": 3 * MS_PER_MINUTE,
    "5m": 5 * MS_PER_MINUTE,
    "15m": 15 * MS_PER_MINUTE,
    "30m": 30 * MS_PER_MINUTE,
    "1h": 60 * MS_PER_MINUTE,
    "2h": 120 * MS_PER_MINUTE,
    "4h": 240 * MS_PER_MINUTE,
    "6h": 360 * MS_PER_MINUTE,
    "12h": 720 * MS_PER_MINUTE,
    "1d": 1440 * MS_PER_MINUTE,
    "1w": 7 * 1440 * MS_PER_MINUTE,
}

MS_PER_YEAR = 365 * 24 * 60 * MS_PER_MINUTE  # crypto trades 24/7


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. ``ts`` is the bar open time in epoch milliseconds."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def bars_per_year(interval: str) -> float:
    """Number of bars in a year for a 24/7 crypto market."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unknown interval '{interval}'. Valid: {sorted(INTERVAL_MS)}")
    return MS_PER_YEAR / INTERVAL_MS[interval]


def validate_candles(candles: list[Candle]) -> None:
    """Raise ValueError on unsorted timestamps or inconsistent OHLC bars."""
    for i, c in enumerate(candles):
        if c.high < c.low:
            raise ValueError(f"Bar {i}: high {c.high} < low {c.low}")
        if not (c.low <= c.open <= c.high and c.low <= c.close <= c.high):
            raise ValueError(f"Bar {i}: open/close outside high-low range")
        if i > 0 and c.ts <= candles[i - 1].ts:
            raise ValueError(f"Bar {i}: timestamps not strictly increasing")


# ---------------------------------------------------------------------------
# Synthetic data (deterministic, seeded)
# ---------------------------------------------------------------------------

# (annualized drift, annualized volatility) per market regime
_REGIMES = {
    "bull": (1.2, 0.55),
    "bear": (-0.9, 0.85),
    "chop": (0.0, 0.40),
}


def generate_synthetic_ohlcv(
    n_bars: int,
    interval: str = "1h",
    seed: int = 42,
    start_price: float = 30_000.0,
    start_ts: int = 1_577_836_800_000,  # 2020-01-01 UTC
    regime_switch_prob: float = 0.002,
) -> list[Candle]:
    """Regime-switching GBM with realistic wicks and volume.

    Deterministic for a given seed, so tests never need the network.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    rng = random.Random(seed)
    interval_ms = INTERVAL_MS[interval]
    dt = interval_ms / MS_PER_YEAR

    regime = rng.choice(sorted(_REGIMES))
    price = start_price
    candles: list[Candle] = []
    for i in range(n_bars):
        if rng.random() < regime_switch_prob:
            regime = rng.choice(sorted(_REGIMES))
        drift, vol = _REGIMES[regime]
        bar_open = price
        ret = drift * dt + vol * (dt**0.5) * rng.gauss(0.0, 1.0)
        bar_close = bar_open * max(0.2, 1.0 + ret)
        body_hi = max(bar_open, bar_close)
        body_lo = min(bar_open, bar_close)
        wick = abs(rng.gauss(0.0, 0.4)) * vol * (dt**0.5) * bar_open
        bar_high = body_hi + wick * rng.random()
        bar_low = max(0.01, body_lo - wick * rng.random())
        volume = abs(rng.gauss(1.0, 0.35)) * (1.0 + 8.0 * abs(ret))
        candles.append(
            Candle(
                ts=start_ts + i * interval_ms,
                open=round(bar_open, 2),
                high=round(bar_high, 2),
                low=round(bar_low, 2),
                close=round(bar_close, 2),
                volume=round(volume, 4),
            )
        )
        price = bar_close
    return candles


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

CSV_HEADER = ["timestamp_ms", "open", "high", "low", "close", "volume"]


def candles_to_csv(candles: list[Candle], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for c in candles:
            writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])


def candles_from_csv(path: str) -> list[Candle]:
    candles: list[Candle] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "close" not in reader.fieldnames:
            raise ValueError(f"CSV {path} missing OHLCV header (expected {CSV_HEADER})")
        ts_field = "timestamp_ms" if "timestamp_ms" in reader.fieldnames else "timestamp"
        for row in reader:
            candles.append(
                Candle(
                    ts=int(float(row[ts_field])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                )
            )
    candles.sort(key=lambda c: c.ts)
    validate_candles(candles)
    return candles


# ---------------------------------------------------------------------------
# Binance public API
# ---------------------------------------------------------------------------


def fetch_binance_klines(
    symbol: str,
    interval: str = "1h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_bars: int = 50_000,
    session: object = None,
) -> list[Candle]:
    """Fetch OHLCV klines from Binance's public REST API (no key needed).

    Paginates in blocks of 1000 and retries transient failures with
    exponential backoff. ``session`` accepts any object with a
    ``get(url, params=, timeout=)`` method (mockable in tests).
    """
    import requests  # local import keeps offline paths dependency-free

    if interval not in INTERVAL_MS:
        raise ValueError(f"Unknown interval '{interval}'. Valid: {sorted(INTERVAL_MS)}")
    http = session if session is not None else requests
    candles: list[Candle] = []
    cursor = start_ms
    while len(candles) < max_bars:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(MAX_KLINES_PER_REQUEST, max_bars - len(candles)),
        }
        if cursor is not None:
            params["startTime"] = cursor
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = _get_with_retry(http, params)
        if not rows:
            break
        for row in rows:
            candles.append(
                Candle(
                    ts=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        if len(rows) < MAX_KLINES_PER_REQUEST:
            break
        cursor = candles[-1].ts + INTERVAL_MS[interval]
        if end_ms is not None and cursor > end_ms:
            break
    del candles[max_bars:]  # defensive: server may ignore the limit param
    validate_candles(candles)
    return candles


def _get_with_retry(http: object, params: dict) -> list:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = http.get(  # type: ignore[attr-defined]
                BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Binance klines request failed after {MAX_RETRIES} retries: {last_error}")


def load_or_fetch(
    symbol: str,
    interval: str,
    cache_dir: str = "data_cache",
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_bars: int = 50_000,
) -> list[Candle]:
    """CSV-cached Binance fetch: hits the network only on cache miss."""
    cache_path = Path(cache_dir) / f"{symbol.upper()}_{interval}_{start_ms}_{end_ms}.csv"
    if cache_path.exists():
        return candles_from_csv(str(cache_path))
    candles = fetch_binance_klines(symbol, interval, start_ms, end_ms, max_bars)
    if candles:
        candles_to_csv(candles, str(cache_path))
    return candles
