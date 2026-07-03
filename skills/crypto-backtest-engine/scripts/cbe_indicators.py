"""Technical indicators for the crypto backtest engine.

Every function returns a list aligned with its input: index i holds the
indicator value computed from data up to and including bar i, or ``None``
during the warmup window. Pure Python, O(n) implementations.
"""

from __future__ import annotations

Series = "list[Optional[float]]"


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average (running-sum, O(n))."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average seeded with the SMA of the first window."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    """Wilder's RSI (smoothed gains/losses)."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        avg_gain += max(change, 0.0)
        avg_loss += max(-change, 0.0)
    avg_gain /= period
    avg_loss /= period
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """True range: max(H-L, |H-prevC|, |L-prevC|)."""
    out: list[float] = []
    for i in range(len(highs)):
        if i == 0:
            out.append(highs[0] - lows[0])
        else:
            prev_close = closes[i - 1]
            out.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - prev_close),
                    abs(lows[i] - prev_close),
                )
            )
    return out


def atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float | None]:
    """Wilder's average true range."""
    _check_period(period)
    tr = true_range(highs, lows, closes)
    out: list[float | None] = [None] * len(tr)
    if len(tr) < period:
        return out
    prev = sum(tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(tr)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def rolling_std(values: list[float], period: int) -> list[float | None]:
    """Rolling population standard deviation (Welford-style running sums)."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    total = 0.0
    total_sq = 0.0
    for i, v in enumerate(values):
        total += v
        total_sq += v * v
        if i >= period:
            old = values[i - period]
            total -= old
            total_sq -= old * old
        if i >= period - 1:
            mean = total / period
            variance = max(0.0, total_sq / period - mean * mean)
            out[i] = variance**0.5
    return out


def bollinger(
    values: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Bollinger bands: (middle, upper, lower)."""
    mid = sma(values, period)
    std = rolling_std(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        if mid[i] is not None and std[i] is not None:
            upper[i] = mid[i] + num_std * std[i]
            lower[i] = mid[i] - num_std * std[i]
    return mid, upper, lower


def donchian(
    highs: list[float], lows: list[float], period: int = 20
) -> tuple[list[float | None], list[float | None]]:
    """Donchian channel over the *prior* ``period`` bars (excludes bar i).

    Excluding the current bar means a close above the upper band is a true
    breakout of previous highs rather than a self-referential trigger.
    """
    _check_period(period)
    upper: list[float | None] = [None] * len(highs)
    lower: list[float | None] = [None] * len(lows)
    for i in range(period, len(highs)):
        window_high = max(highs[i - period : i])
        window_low = min(lows[i - period : i])
        upper[i] = window_high
        lower[i] = window_low
    return upper, lower


def roc(values: list[float], period: int) -> list[float | None]:
    """Rate of change: values[i] / values[i-period] - 1."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    for i in range(period, len(values)):
        base = values[i - period]
        if base != 0:
            out[i] = values[i] / base - 1.0
    return out


def returns(values: list[float]) -> list[float]:
    """Simple bar-to-bar returns; first element is 0.0."""
    out = [0.0]
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append(values[i] / prev - 1.0 if prev != 0 else 0.0)
    return out


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
