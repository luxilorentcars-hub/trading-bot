"""Performance metrics for the crypto backtest engine.

Computes a full statistics suite from an equity curve and a trade log:
returns, risk-adjusted ratios, drawdown analytics, and trade statistics
(including Van Tharp's SQN). All annualization uses ``bars_per_year``
(crypto trades 24/7, so 1h bars => 8760 bars/year).
"""

from __future__ import annotations

import math


def equity_returns(equity: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        out.append(equity[i] / prev - 1.0 if prev > 0 else 0.0)
    return out


def max_drawdown(equity: list[float]) -> tuple[float, int]:
    """Return (max drawdown as negative fraction, longest bars-under-water)."""
    peak = float("-inf")
    max_dd = 0.0
    underwater = 0
    max_underwater = 0
    for value in equity:
        if value >= peak:
            peak = value
            underwater = 0
        else:
            underwater += 1
            max_underwater = max(max_underwater, underwater)
            if peak > 0:
                max_dd = min(max_dd, value / peak - 1.0)
    return max_dd, max_underwater


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = _mean(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _percentile(sorted_xs: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_xs[int(k)]
    return sorted_xs[lo] * (hi - k) + sorted_xs[hi] * (k - lo)


def sharpe_ratio(bar_returns: list[float], bars_per_year: float) -> float:
    std = _std(bar_returns)
    if std == 0.0:
        return 0.0
    return _mean(bar_returns) / std * math.sqrt(bars_per_year)


def sortino_ratio(bar_returns: list[float], bars_per_year: float) -> float:
    downside = [r for r in bar_returns if r < 0]
    if not downside:
        return 0.0 if _mean(bar_returns) <= 0 else float("inf")
    downside_dev = (sum(r * r for r in downside) / len(bar_returns)) ** 0.5
    if downside_dev == 0.0:
        return 0.0
    return _mean(bar_returns) / downside_dev * math.sqrt(bars_per_year)


def ulcer_index(equity: list[float]) -> float:
    """RMS of percentage drawdowns — penalizes deep AND long drawdowns."""
    if not equity:
        return 0.0
    peak = equity[0]
    squared_dd_sum = 0.0
    for value in equity:
        peak = max(peak, value)
        dd_pct = (value / peak - 1.0) * 100.0 if peak > 0 else 0.0
        squared_dd_sum += dd_pct * dd_pct
    return (squared_dd_sum / len(equity)) ** 0.5


def compute_metrics(
    equity: list[float],
    trade_pnls: list[float],
    trade_pnl_pcts: list[float],
    bars_per_year: float,
    exposure: float = 0.0,
    initial_capital: float | None = None,
) -> dict:
    """Full metric suite. ``equity`` is the per-bar equity curve."""
    if len(equity) < 2:
        raise ValueError("equity curve needs at least 2 points")
    initial = initial_capital if initial_capital is not None else equity[0]
    final = equity[-1]
    n_bars = len(equity)
    bar_returns = equity_returns(equity)

    total_return = final / initial - 1.0
    years = n_bars / bars_per_year
    cagr = (final / initial) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else 0.0
    max_dd, dd_duration = max_drawdown(equity)
    vol_annual = _std(bar_returns) * math.sqrt(bars_per_year)

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    net_profit = final - initial

    sorted_returns = sorted(bar_returns)
    pnl_std = _std(trade_pnl_pcts)
    sqn = math.sqrt(len(trade_pnl_pcts)) * _mean(trade_pnl_pcts) / pnl_std if pnl_std > 0 else 0.0

    return {
        "initial_capital": round(initial, 2),
        "final_equity": round(final, 2),
        "net_profit": round(net_profit, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_annual_pct": round(vol_annual * 100, 2),
        "sharpe": round(sharpe_ratio(bar_returns, bars_per_year), 3),
        "sortino": round(sortino_ratio(bar_returns, bars_per_year), 3),
        "calmar": round(cagr / abs(max_dd), 3) if max_dd < 0 else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_bars": dd_duration,
        "ulcer_index": round(ulcer_index(equity), 3),
        "var_95_bar_pct": round(_percentile(sorted_returns, 0.05) * 100, 3),
        "recovery_factor": round(net_profit / (abs(max_dd) * initial), 3) if max_dd < 0 else 0.0,
        "exposure_pct": round(exposure * 100, 2),
        "num_trades": len(trade_pnls),
        "win_rate_pct": round(len(wins) / len(trade_pnls) * 100, 2) if trade_pnls else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else 0.0,
        "avg_trade_pnl": round(_mean(trade_pnls), 2),
        "avg_win": round(_mean(wins), 2) if wins else 0.0,
        "avg_loss": round(_mean(losses), 2) if losses else 0.0,
        "payoff_ratio": round(_mean(wins) / abs(_mean(losses)), 3)
        if wins and losses and _mean(losses) != 0
        else 0.0,
        "expectancy_pct": round(_mean(trade_pnl_pcts) * 100, 3),
        "sqn": round(sqn, 3),
        "max_consecutive_wins": _max_streak(trade_pnls, positive=True),
        "max_consecutive_losses": _max_streak(trade_pnls, positive=False),
        "bars": n_bars,
        "years": round(years, 3),
    }


def _max_streak(pnls: list[float], positive: bool) -> int:
    best = 0
    current = 0
    for p in pnls:
        hit = p > 0 if positive else p <= 0
        current = current + 1 if hit else 0
        best = max(best, current)
    return best
