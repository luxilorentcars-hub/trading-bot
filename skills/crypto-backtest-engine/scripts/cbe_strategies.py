"""Strategy library for the crypto backtest engine.

A strategy converts price history into a *target direction* per bar:
+1 (long), -1 (short), or 0 (flat). The engine executes the target at the
NEXT bar's open, so ``signal(i)`` may only use data up to and including
bar i — the alignment guarantees no look-ahead by construction.

Register new strategies in ``STRATEGY_REGISTRY`` to expose them to the
CLI and the optimizer. ``DEFAULT_GRIDS`` provides sane parameter grids.
"""

from __future__ import annotations

from cbe_data import Candle
from cbe_indicators import bollinger, donchian, roc, rsi, sma

LONG = 1
SHORT = -1
FLAT = 0


class Strategy:
    """Base class. Subclasses implement ``prepare`` and ``signal``."""

    name = "base"

    def __init__(self, **params: object) -> None:
        self.params = params

    def prepare(self, candles: list[Candle]) -> None:
        """Precompute indicators over the full series (called once)."""
        raise NotImplementedError

    def signal(self, i: int) -> int:
        """Target direction after bar i closes: LONG, SHORT or FLAT."""
        raise NotImplementedError

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({args})"


class SmaCross(Strategy):
    """Golden/death cross: long while fast SMA > slow SMA, short otherwise."""

    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = True) -> None:
        super().__init__(fast=fast, slow=slow, allow_short=allow_short)
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def prepare(self, candles: list[Candle]) -> None:
        closes = [c.close for c in candles]
        self._fast = sma(closes, self.fast)
        self._slow = sma(closes, self.slow)

    def signal(self, i: int) -> int:
        f, s = self._fast[i], self._slow[i]
        if f is None or s is None:
            return FLAT
        if f > s:
            return LONG
        return SHORT if self.allow_short else FLAT


class RsiMeanReversion(Strategy):
    """Buy oversold dips, exit once RSI normalizes. Long-only by design."""

    name = "rsi_reversion"

    def __init__(self, period: int = 14, entry: float = 30.0, exit: float = 55.0) -> None:
        super().__init__(period=period, entry=entry, exit=exit)
        if not entry < exit:
            raise ValueError("entry threshold must be below exit threshold")
        self.period = period
        self.entry = entry
        self.exit = exit
        self._holding = False

    def prepare(self, candles: list[Candle]) -> None:
        self._rsi = rsi([c.close for c in candles], self.period)
        self._holding = False

    def signal(self, i: int) -> int:
        value = self._rsi[i]
        if value is None:
            return FLAT
        if self._holding:
            if value >= self.exit:
                self._holding = False
                return FLAT
            return LONG
        if value <= self.entry:
            self._holding = True
            return LONG
        return FLAT


class BollingerBreakout(Strategy):
    """Volatility breakout: enter on band break, exit on middle-band cross."""

    name = "bollinger_breakout"

    def __init__(self, period: int = 20, num_std: float = 2.0, allow_short: bool = True) -> None:
        super().__init__(period=period, num_std=num_std, allow_short=allow_short)
        self.period = period
        self.num_std = num_std
        self.allow_short = allow_short
        self._position = FLAT

    def prepare(self, candles: list[Candle]) -> None:
        closes = [c.close for c in candles]
        self._closes = closes
        self._mid, self._upper, self._lower = bollinger(closes, self.period, self.num_std)
        self._position = FLAT

    def signal(self, i: int) -> int:
        mid, upper, lower = self._mid[i], self._upper[i], self._lower[i]
        if mid is None or upper is None or lower is None:
            return FLAT
        close = self._closes[i]
        if self._position == LONG and close < mid:
            self._position = FLAT
        elif self._position == SHORT and close > mid:
            self._position = FLAT
        if self._position == FLAT:
            if close > upper:
                self._position = LONG
            elif close < lower and self.allow_short:
                self._position = SHORT
        return self._position


class DonchianBreakout(Strategy):
    """Turtle-style channel breakout with a shorter exit channel."""

    name = "donchian_breakout"

    def __init__(self, entry: int = 20, exit: int = 10, allow_short: bool = True) -> None:
        super().__init__(entry=entry, exit=exit, allow_short=allow_short)
        if exit > entry:
            raise ValueError("exit channel must be <= entry channel")
        self.entry_n = entry
        self.exit_n = exit
        self.allow_short = allow_short
        self._position = FLAT

    def prepare(self, candles: list[Candle]) -> None:
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        self._closes = [c.close for c in candles]
        self._entry_hi, self._entry_lo = donchian(highs, lows, self.entry_n)
        self._exit_hi, self._exit_lo = donchian(highs, lows, self.exit_n)
        self._position = FLAT

    def signal(self, i: int) -> int:
        entry_hi, entry_lo = self._entry_hi[i], self._entry_lo[i]
        exit_hi, exit_lo = self._exit_hi[i], self._exit_lo[i]
        if entry_hi is None or entry_lo is None or exit_hi is None or exit_lo is None:
            return FLAT
        close = self._closes[i]
        if self._position == LONG and close < exit_lo:
            self._position = FLAT
        elif self._position == SHORT and close > exit_hi:
            self._position = FLAT
        if self._position == FLAT:
            if close > entry_hi:
                self._position = LONG
            elif close < entry_lo and self.allow_short:
                self._position = SHORT
        return self._position


class Momentum(Strategy):
    """Time-series momentum: hold the sign of the trailing return."""

    name = "momentum"

    def __init__(
        self, lookback: int = 90, threshold: float = 0.0, allow_short: bool = True
    ) -> None:
        super().__init__(lookback=lookback, threshold=threshold, allow_short=allow_short)
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.lookback = lookback
        self.threshold = threshold
        self.allow_short = allow_short

    def prepare(self, candles: list[Candle]) -> None:
        self._roc = roc([c.close for c in candles], self.lookback)

    def signal(self, i: int) -> int:
        value = self._roc[i]
        if value is None:
            return FLAT
        if value > self.threshold:
            return LONG
        if value < -self.threshold and self.allow_short:
            return SHORT
        return FLAT


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    SmaCross.name: SmaCross,
    RsiMeanReversion.name: RsiMeanReversion,
    BollingerBreakout.name: BollingerBreakout,
    DonchianBreakout.name: DonchianBreakout,
    Momentum.name: Momentum,
}

DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    SmaCross.name: {"fast": [10, 20, 50], "slow": [50, 100, 200]},
    RsiMeanReversion.name: {"period": [7, 14, 21], "entry": [20, 30], "exit": [50, 60]},
    BollingerBreakout.name: {"period": [20, 40], "num_std": [1.5, 2.0, 2.5]},
    DonchianBreakout.name: {"entry": [20, 55], "exit": [10, 20]},
    Momentum.name: {"lookback": [30, 90, 180], "threshold": [0.0, 0.05]},
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    """Instantiate a registered strategy, skipping invalid param combos."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Valid: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**(params or {}))
