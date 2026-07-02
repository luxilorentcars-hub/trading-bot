"""Human-readable and JSON reporting for backtest results."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from cbe_engine import BacktestResult
from cbe_monte_carlo import MonteCarloResult
from cbe_optimizer import WalkForwardResult

_RULE = "=" * 64


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def format_report(
    result: BacktestResult,
    monte_carlo: MonteCarloResult | None = None,
    walk: WalkForwardResult | None = None,
    max_trades_shown: int = 10,
) -> str:
    """Multi-section plain-text report."""
    m = result.metrics
    first_ts = result.equity_curve[0][0]
    last_ts = result.equity_curve[-1][0]
    lines = [
        _RULE,
        "CRYPTO BACKTEST REPORT",
        _RULE,
        f"Strategy        : {result.strategy}",
        f"Period          : {_fmt_ts(first_ts)} -> {_fmt_ts(last_ts)} UTC "
        f"({m['bars']} bars, {m['years']} years)",
        f"Costs           : fee {result.config.fee_bps} bps/side, "
        f"slippage {result.config.slippage_bps} bps",
        f"Sizing          : {result.config.sizing_mode}",
        "",
        "-- Performance --",
        f"Net profit      : {m['net_profit']:>12.2f}  ({m['total_return_pct']:+.2f}%)",
        f"CAGR            : {m['cagr_pct']:>11.2f}%   Volatility: {m['volatility_annual_pct']:.2f}%",
        f"Sharpe          : {m['sharpe']:>12.3f}   Sortino   : {m['sortino']:.3f}",
        f"Calmar          : {m['calmar']:>12.3f}   SQN       : {m['sqn']:.3f}",
        "",
        "-- Risk --",
        f"Max drawdown    : {m['max_drawdown_pct']:>11.2f}%  "
        f"(longest {m['max_drawdown_duration_bars']} bars underwater)",
        f"Ulcer index     : {m['ulcer_index']:>12.3f}   VaR95/bar : {m['var_95_bar_pct']:.3f}%",
        f"Recovery factor : {m['recovery_factor']:>12.3f}   Exposure  : {m['exposure_pct']:.2f}%",
        "",
        "-- Trades --",
        f"Trades          : {m['num_trades']:>6}   Win rate: {m['win_rate_pct']:.2f}%   "
        f"Profit factor: {m['profit_factor']:.3f}",
        f"Expectancy      : {m['expectancy_pct']:>6.3f}% per trade   Payoff: {m['payoff_ratio']:.3f}",
        f"Avg win/loss    : {m['avg_win']:.2f} / {m['avg_loss']:.2f}   "
        f"Streaks: +{m['max_consecutive_wins']} / -{m['max_consecutive_losses']}",
    ]

    if result.trades:
        lines += ["", f"-- Last {min(max_trades_shown, len(result.trades))} trades --"]
        for t in result.trades[-max_trades_shown:]:
            side = "LONG " if t.direction == 1 else "SHORT"
            lines.append(
                f"{_fmt_ts(t.entry_ts)}  {side}  in {t.entry_price:>12.2f} "
                f"out {t.exit_price:>12.2f}  pnl {t.pnl:>10.2f} ({t.pnl_pct * 100:+.2f}%)  "
                f"[{t.exit_reason}]"
            )

    if walk is not None and walk.splits:
        lines += [
            "",
            "-- Walk-forward analysis --",
            f"Splits          : {len(walk.splits)}   OOS return: {walk.oos_return_pct:+.2f}%",
            f"Avg IS score    : {walk.avg_train_score:.3f}   Avg OOS score: {walk.avg_test_score:.3f}",
            f"WF efficiency   : {walk.walk_forward_efficiency:.3f}  "
            f"({'ROBUST' if walk.walk_forward_efficiency >= 0.5 else 'LIKELY OVERFIT'})",
        ]
        for s in walk.splits:
            lines.append(
                f"  split {s.split}: params {s.best_params}  "
                f"IS {s.train_score:.3f} -> OOS {s.test_metrics.get('sharpe', 0.0):.3f}"
            )

    if monte_carlo is not None:
        mc = monte_carlo
        lines += [
            "",
            f"-- Monte Carlo ({mc.n_sims} sims, {mc.n_trades} trades bootstrapped) --",
            f"Final equity    : p5 {mc.final_equity_p5:.2f} | p50 {mc.final_equity_p50:.2f} "
            f"| p95 {mc.final_equity_p95:.2f}",
            f"Max drawdown    : median {mc.max_drawdown_p50_pct:.2f}% | "
            f"worst-5% {mc.max_drawdown_p95_pct:.2f}%",
            f"P(loss)         : {mc.prob_loss_pct:.2f}%   "
            f"P(ruin >{mc.ruin_threshold_pct:.0f}% DD): {mc.prob_ruin_pct:.2f}%",
        ]

    lines.append(_RULE)
    return "\n".join(lines)


def result_to_dict(
    result: BacktestResult,
    monte_carlo: MonteCarloResult | None = None,
    walk: WalkForwardResult | None = None,
) -> dict:
    """JSON-serializable summary (equity curve downsampled to <= 1000 pts)."""
    curve = result.equity_curve
    step = max(1, len(curve) // 1000)
    payload = {
        "strategy": result.strategy,
        "config": asdict(result.config),
        "metrics": result.metrics,
        "exposure": result.exposure,
        "trades": [asdict(t) for t in result.trades],
        "equity_curve": [{"ts": ts, "equity": round(eq, 2)} for ts, eq in curve[::step]],
    }
    if monte_carlo is not None:
        payload["monte_carlo"] = asdict(monte_carlo)
    if walk is not None:
        payload["walk_forward"] = {
            "oos_return_pct": walk.oos_return_pct,
            "avg_train_score": walk.avg_train_score,
            "avg_test_score": walk.avg_test_score,
            "walk_forward_efficiency": walk.walk_forward_efficiency,
            "splits": [
                {
                    "split": s.split,
                    "train_range": list(s.train_range),
                    "test_range": list(s.test_range),
                    "best_params": s.best_params,
                    "train_score": s.train_score,
                    "test_metrics": s.test_metrics,
                }
                for s in walk.splits
            ],
        }
    return payload
