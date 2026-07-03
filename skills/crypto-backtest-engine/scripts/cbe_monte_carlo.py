"""Monte Carlo robustness analysis on the trade log.

Bootstraps the sequence of per-trade returns (sampling with replacement)
to answer: "if the same edge produced these trades in a different order,
what equity paths were plausible?" Reports percentile bands for final
equity and max drawdown, plus probability of loss and of ruin.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    n_sims: int
    n_trades: int
    final_equity_p5: float
    final_equity_p50: float
    final_equity_p95: float
    max_drawdown_p50_pct: float
    max_drawdown_p95_pct: float  # 95th percentile WORST drawdown
    prob_loss_pct: float  # P(final equity < initial)
    prob_ruin_pct: float  # P(drawdown beyond ruin threshold)
    ruin_threshold_pct: float


def _percentile(sorted_xs: list[float], pct: float) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_xs[int(k)]
    return sorted_xs[lo] * (hi - k) + sorted_xs[hi] * (k - lo)


def monte_carlo_trades(
    trade_pnl_pcts: list[float],
    initial_capital: float = 10_000.0,
    n_sims: int = 1000,
    seed: int = 7,
    ruin_drawdown_pct: float = 50.0,
) -> MonteCarloResult:
    """Bootstrap trade returns; deterministic for a given seed.

    ``trade_pnl_pcts`` are per-trade returns on entry notional (e.g. 0.02
    for +2%); each simulated path compounds them on full equity.
    """
    if not trade_pnl_pcts:
        raise ValueError("no trades to bootstrap")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    rng = random.Random(seed)
    n_trades = len(trade_pnl_pcts)
    finals: list[float] = []
    max_dds: list[float] = []
    ruined = 0
    for _ in range(n_sims):
        equity = initial_capital
        peak = equity
        worst_dd = 0.0
        for _ in range(n_trades):
            r = trade_pnl_pcts[rng.randrange(n_trades)]
            equity *= max(0.0, 1.0 + r)
            peak = max(peak, equity)
            if peak > 0:
                worst_dd = min(worst_dd, equity / peak - 1.0)
        finals.append(equity)
        max_dds.append(worst_dd)
        if worst_dd <= -ruin_drawdown_pct / 100.0:
            ruined += 1

    finals.sort()
    max_dds.sort()  # most negative (worst) first
    losses = sum(1 for f in finals if f < initial_capital)
    return MonteCarloResult(
        n_sims=n_sims,
        n_trades=n_trades,
        final_equity_p5=round(_percentile(finals, 0.05), 2),
        final_equity_p50=round(_percentile(finals, 0.50), 2),
        final_equity_p95=round(_percentile(finals, 0.95), 2),
        max_drawdown_p50_pct=round(_percentile(max_dds, 0.50) * 100, 2),
        max_drawdown_p95_pct=round(_percentile(max_dds, 0.05) * 100, 2),
        prob_loss_pct=round(losses / n_sims * 100, 2),
        prob_ruin_pct=round(ruined / n_sims * 100, 2),
        ruin_threshold_pct=ruin_drawdown_pct,
    )
