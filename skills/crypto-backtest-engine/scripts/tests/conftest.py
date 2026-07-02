"""Shared fixtures for crypto-backtest-engine tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent
for path in (str(SCRIPTS_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cbe_data import Candle, generate_synthetic_ohlcv  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_candles() -> list[Candle]:
    """2000 deterministic 1h bars shared across tests."""
    return generate_synthetic_ohlcv(2000, interval="1h", seed=42)
