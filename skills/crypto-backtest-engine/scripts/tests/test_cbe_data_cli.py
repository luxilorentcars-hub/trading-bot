"""Tests for the data layer (synthetic, CSV, mocked Binance) and the CLI."""

from __future__ import annotations

import json

import pytest
from cbe_data import (
    INTERVAL_MS,
    Candle,
    bars_per_year,
    candles_from_csv,
    candles_to_csv,
    fetch_binance_klines,
    generate_synthetic_ohlcv,
    validate_candles,
)


class TestSynthetic:
    def test_deterministic_for_seed(self):
        a = generate_synthetic_ohlcv(500, seed=7)
        b = generate_synthetic_ohlcv(500, seed=7)
        assert a == b

    def test_different_seeds_differ(self):
        assert generate_synthetic_ohlcv(100, seed=1) != generate_synthetic_ohlcv(100, seed=2)

    def test_valid_ohlc(self):
        candles = generate_synthetic_ohlcv(1000, seed=42)
        validate_candles(candles)  # raises on inconsistency

    def test_timestamps_spaced_by_interval(self):
        candles = generate_synthetic_ohlcv(10, interval="4h", seed=1)
        assert candles[1].ts - candles[0].ts == INTERVAL_MS["4h"]

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            generate_synthetic_ohlcv(0)


class TestBarsPerYear:
    def test_hourly(self):
        assert bars_per_year("1h") == pytest.approx(8760.0)

    def test_daily(self):
        assert bars_per_year("1d") == pytest.approx(365.0)

    def test_unknown_interval(self):
        with pytest.raises(ValueError):
            bars_per_year("13m")


class TestCsvRoundtrip:
    def test_roundtrip(self, tmp_path):
        candles = generate_synthetic_ohlcv(50, seed=3)
        path = tmp_path / "ohlcv.csv"
        candles_to_csv(candles, str(path))
        loaded = candles_from_csv(str(path))
        assert loaded == candles

    def test_missing_header_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="missing OHLCV header"):
            candles_from_csv(str(path))


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Serves canned kline pages and records requests."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        assert timeout is not None, "requests must always set a timeout"
        self.calls.append(dict(params or {}))
        return _FakeResponse(self.pages.pop(0) if self.pages else [])


def _kline_row(ts, price):
    return [ts, str(price), str(price + 1), str(price - 1), str(price), "12.5", 0, 0, 0, 0, 0, 0]


class TestBinanceFetch:
    def test_single_page(self):
        page = [_kline_row(1_600_000_000_000 + i * 3_600_000, 100 + i) for i in range(5)]
        session = _FakeSession([page])
        candles = fetch_binance_klines("btcusdt", "1h", session=session)
        assert len(candles) == 5
        assert candles[0] == Candle(1_600_000_000_000, 100.0, 101.0, 99.0, 100.0, 12.5)
        assert session.calls[0]["symbol"] == "BTCUSDT"

    def test_pagination_advances_cursor(self):
        page1 = [_kline_row(1_600_000_000_000 + i * 3_600_000, 100) for i in range(1000)]
        page2 = [_kline_row(page1[-1][0] + 3_600_000, 100)]
        session = _FakeSession([page1, page2])
        candles = fetch_binance_klines("BTCUSDT", "1h", start_ms=1_600_000_000_000, session=session)
        assert len(candles) == 1001
        assert session.calls[1]["startTime"] == page1[-1][0] + 3_600_000

    def test_max_bars_respected(self):
        page = [_kline_row(1_600_000_000_000 + i * 3_600_000, 100) for i in range(1000)]
        session = _FakeSession([page])
        candles = fetch_binance_klines("BTCUSDT", "1h", max_bars=300, session=session)
        assert len(candles) == 300
        assert session.calls[0]["limit"] == 300

    def test_unknown_interval_rejected(self):
        with pytest.raises(ValueError):
            fetch_binance_klines("BTCUSDT", "7m", session=_FakeSession([]))


class TestCli:
    def test_synthetic_end_to_end(self):
        from run_backtest import main

        report = main(
            [
                "--synthetic",
                "1500",
                "--strategy",
                "sma_cross",
                "--params",
                '{"fast": 10, "slow": 40}',
            ]
        )
        assert "CRYPTO BACKTEST REPORT" in report
        assert "sma_cross" in report
        assert "Sharpe" in report

    def test_optimize_walk_forward_monte_carlo(self, tmp_path):
        from run_backtest import main

        out = tmp_path / "results.json"
        report = main(
            [
                "--synthetic",
                "1500",
                "--strategy",
                "sma_cross",
                "--optimize",
                '{"fast": [5, 10], "slow": [30, 60]}',
                "--walk-forward",
                "2",
                "--monte-carlo",
                "200",
                "--json-out",
                str(out),
            ]
        )
        assert "Grid search" in report
        assert "Walk-forward analysis" in report
        assert "Monte Carlo" in report
        payload = json.loads(out.read_text())
        assert payload["metrics"]["num_trades"] >= 1
        assert "walk_forward" in payload
        assert "monte_carlo" in payload

    def test_default_grid_keyword(self):
        from run_backtest import main

        report = main(
            [
                "--synthetic",
                "1200",
                "--strategy",
                "momentum",
                "--optimize",
                "default",
                "--top",
                "3",
            ]
        )
        assert "Grid search" in report
