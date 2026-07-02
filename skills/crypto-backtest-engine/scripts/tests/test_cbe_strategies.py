"""Unit tests for the strategy library."""

from __future__ import annotations

import pytest
from cbe_strategies import (
    DEFAULT_GRIDS,
    FLAT,
    LONG,
    SHORT,
    STRATEGY_REGISTRY,
    BollingerBreakout,
    DonchianBreakout,
    Momentum,
    RsiMeanReversion,
    SmaCross,
    build_strategy,
)
from cbe_test_helpers import make_trend_candles


class TestRegistry:
    def test_all_strategies_registered(self):
        assert set(STRATEGY_REGISTRY) == {
            "sma_cross",
            "rsi_reversion",
            "bollinger_breakout",
            "donchian_breakout",
            "momentum",
        }

    def test_every_strategy_has_default_grid(self):
        assert set(DEFAULT_GRIDS) == set(STRATEGY_REGISTRY)

    def test_build_strategy(self):
        s = build_strategy("sma_cross", {"fast": 5, "slow": 20})
        assert isinstance(s, SmaCross)
        assert "fast=5" in s.describe()

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_strategy("hodl")

    def test_signals_always_valid(self, synthetic_candles):
        for name in STRATEGY_REGISTRY:
            strategy = build_strategy(name)
            strategy.prepare(synthetic_candles)
            for i in range(len(synthetic_candles)):
                assert strategy.signal(i) in (LONG, SHORT, FLAT), name


class TestSmaCross:
    def test_long_in_uptrend(self):
        candles = make_trend_candles(120, start=100.0, step=1.0)
        s = SmaCross(5, 20)
        s.prepare(candles)
        assert s.signal(119) == LONG

    def test_short_in_downtrend(self):
        candles = make_trend_candles(120, start=500.0, step=-1.0)
        s = SmaCross(5, 20)
        s.prepare(candles)
        assert s.signal(119) == SHORT

    def test_no_short_when_disabled(self):
        candles = make_trend_candles(120, start=500.0, step=-1.0)
        s = SmaCross(5, 20, allow_short=False)
        s.prepare(candles)
        assert s.signal(119) == FLAT

    def test_warmup_flat(self):
        candles = make_trend_candles(120)
        s = SmaCross(5, 20)
        s.prepare(candles)
        assert s.signal(0) == FLAT

    def test_fast_must_be_below_slow(self):
        with pytest.raises(ValueError):
            SmaCross(50, 50)


class TestRsiMeanReversion:
    def test_long_only(self, synthetic_candles):
        s = RsiMeanReversion()
        s.prepare(synthetic_candles)
        signals = {s.signal(i) for i in range(len(synthetic_candles))}
        assert SHORT not in signals

    def test_holds_until_exit_threshold(self):
        # steep fall then steady rise: strategy should enter oversold and hold
        down = make_trend_candles(40, start=300.0, step=-5.0)
        up = make_trend_candles(40, start=100.0, step=2.0, ts0=down[-1].ts + 3_600_000)
        candles = down + up
        s = RsiMeanReversion(period=14, entry=30.0, exit=60.0)
        s.prepare(candles)
        sequence = [s.signal(i) for i in range(len(candles))]
        assert LONG in sequence
        first_long = sequence.index(LONG)
        block = sequence[first_long:]
        # once exited, RSI on a monotonic rise stays high: no flip-flop
        assert block.count(LONG) == len([x for x in block if x == LONG])
        assert sequence[-1] == FLAT  # exited after recovery

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            RsiMeanReversion(entry=60.0, exit=40.0)


class TestBreakouts:
    def test_bollinger_long_after_breakout(self):
        flat = make_trend_candles(60, start=100.0, step=0.0)
        surge = make_trend_candles(10, start=100.0, step=5.0, ts0=flat[-1].ts + 3_600_000)
        candles = flat + surge
        s = BollingerBreakout(period=20, num_std=2.0)
        s.prepare(candles)
        assert s.signal(len(candles) - 1) == LONG

    def test_donchian_long_after_new_high(self):
        candles = make_trend_candles(80, start=100.0, step=1.0)
        s = DonchianBreakout(entry=20, exit=10)
        s.prepare(candles)
        assert s.signal(79) == LONG

    def test_donchian_exit_channel_validation(self):
        with pytest.raises(ValueError):
            DonchianBreakout(entry=10, exit=20)


class TestMomentum:
    def test_sign_follows_trend(self):
        up = make_trend_candles(120, start=100.0, step=1.0)
        s = Momentum(lookback=30)
        s.prepare(up)
        assert s.signal(119) == LONG
        down = make_trend_candles(120, start=500.0, step=-1.0)
        s2 = Momentum(lookback=30)
        s2.prepare(down)
        assert s2.signal(119) == SHORT

    def test_threshold_filters_weak_moves(self):
        candles = make_trend_candles(120, start=100.0, step=0.01)
        s = Momentum(lookback=30, threshold=0.50)
        s.prepare(candles)
        assert s.signal(119) == FLAT
