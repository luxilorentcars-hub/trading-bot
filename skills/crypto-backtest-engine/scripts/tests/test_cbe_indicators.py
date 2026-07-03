"""Unit tests for cbe_indicators."""

from __future__ import annotations

import pytest
from cbe_indicators import (
    atr,
    bollinger,
    donchian,
    ema,
    roc,
    rolling_std,
    rsi,
    sma,
    true_range,
)


class TestSma:
    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sma(values, 3)
        assert result[:2] == [None, None]
        assert result[2] == pytest.approx(2.0)
        assert result[4] == pytest.approx(4.0)

    def test_alignment_length(self):
        assert len(sma([1.0] * 10, 4)) == 10

    def test_period_longer_than_series(self):
        assert sma([1.0, 2.0], 5) == [None, None]

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            sma([1.0], 0)


class TestEma:
    def test_seeded_with_sma(self):
        values = [2.0, 4.0, 6.0, 8.0]
        result = ema(values, 3)
        assert result[2] == pytest.approx(4.0)  # SMA of first 3
        alpha = 2.0 / 4.0
        assert result[3] == pytest.approx(alpha * 8.0 + (1 - alpha) * 4.0)

    def test_tracks_trend(self):
        rising = [float(i) for i in range(1, 51)]
        result = ema(rising, 10)
        assert result[-1] < rising[-1]  # lags
        assert result[-1] > result[20]  # but follows


class TestRsi:
    def test_all_gains_is_100(self):
        rising = [float(i) for i in range(1, 30)]
        result = rsi(rising, 14)
        assert result[-1] == pytest.approx(100.0)

    def test_all_losses_near_zero(self):
        falling = [float(i) for i in range(30, 1, -1)]
        result = rsi(falling, 14)
        assert result[-1] == pytest.approx(0.0, abs=1e-9)

    def test_bounds(self, synthetic_candles):
        closes = [c.close for c in synthetic_candles]
        for value in rsi(closes, 14):
            if value is not None:
                assert 0.0 <= value <= 100.0


class TestAtrTrueRange:
    def test_true_range_gap(self):
        highs = [10.0, 20.0]
        lows = [9.0, 19.0]
        closes = [9.5, 19.5]
        tr = true_range(highs, lows, closes)
        assert tr[1] == pytest.approx(20.0 - 9.5)  # gap dominates H-L

    def test_atr_positive(self, synthetic_candles):
        highs = [c.high for c in synthetic_candles]
        lows = [c.low for c in synthetic_candles]
        closes = [c.close for c in synthetic_candles]
        result = atr(highs, lows, closes, 14)
        assert result[13] is not None
        assert all(v > 0 for v in result if v is not None)


class TestBollinger:
    def test_band_ordering(self, synthetic_candles):
        closes = [c.close for c in synthetic_candles]
        mid, upper, lower = bollinger(closes, 20, 2.0)
        for m, u, lo in zip(mid, upper, lower):
            if m is not None:
                assert lo <= m <= u

    def test_constant_series_zero_width(self):
        closes = [50.0] * 30
        mid, upper, lower = bollinger(closes, 20, 2.0)
        assert upper[-1] == pytest.approx(mid[-1])
        assert lower[-1] == pytest.approx(mid[-1])


class TestDonchian:
    def test_excludes_current_bar(self):
        highs = [10.0, 11.0, 12.0, 99.0]
        lows = [9.0, 10.0, 11.0, 1.0]
        upper, lower = donchian(highs, lows, 3)
        # channel at bar 3 built from bars 0..2 only
        assert upper[3] == pytest.approx(12.0)
        assert lower[3] == pytest.approx(9.0)

    def test_warmup_none(self):
        upper, lower = donchian([1.0] * 5, [1.0] * 5, 3)
        assert upper[:3] == [None, None, None]


class TestRocRollingStd:
    def test_roc(self):
        values = [100.0, 110.0, 121.0]
        result = roc(values, 1)
        assert result[1] == pytest.approx(0.10)
        assert result[2] == pytest.approx(0.10)

    def test_rolling_std_constant_is_zero(self):
        result = rolling_std([5.0] * 10, 4)
        assert result[-1] == pytest.approx(0.0)
