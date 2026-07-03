"""Event-driven backtest engine for crypto strategies.

Execution model (deliberately pessimistic, no look-ahead by construction):
  * A signal computed on the close of bar i is executed at the OPEN of
    bar i+1, with slippage applied against the trader.
  * Stops / take-profits / trailing stops are evaluated intrabar against
    the bar's high/low. Gaps fill at the open. If a stop and a take-profit
    are both touched in the same bar, the STOP is assumed to fill first.
  * Fees are charged on every fill (entry and exit). Shorts can accrue a
    daily funding/borrow cost.
  * No leverage: position notional is capped at current equity.

Position sizing modes:
  * ``fixed_fraction`` — notional = equity * fraction
  * ``risk_per_trade`` — qty = (equity * risk_pct%) / stop distance
    (stop distance from ``stop_loss_pct`` or ATR * ``atr_stop_mult``)
  * ``vol_target``     — notional scaled so realized annual vol ≈ target
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cbe_data import Candle
from cbe_indicators import atr as compute_atr
from cbe_indicators import returns as bar_returns
from cbe_indicators import rolling_std
from cbe_metrics import compute_metrics
from cbe_strategies import FLAT, LONG, SHORT, Strategy

VALID_SIZING_MODES = ("fixed_fraction", "risk_per_trade", "vol_target")

EXIT_SIGNAL = "signal"
EXIT_STOP_LOSS = "stop_loss"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TRAILING_STOP = "trailing_stop"
EXIT_END_OF_DATA = "end_of_data"


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 10.0  # taker fee per fill (0.10% = Binance spot default)
    slippage_bps: float = 5.0  # adverse price movement per fill
    sizing_mode: str = "fixed_fraction"
    fraction: float = 1.0  # fixed_fraction: share of equity per position
    risk_pct: float = 1.0  # risk_per_trade: % of equity risked per trade
    atr_period: int = 14
    atr_stop_mult: float = 2.0  # stop distance in ATRs when stop_loss_pct unset
    vol_target_annual: float = 0.40  # vol_target: 40% annualized
    vol_lookback: int = 30
    stop_loss_pct: float | None = None  # 0.05 => exit 5% against entry
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    short_funding_daily_bps: float = 0.0  # daily cost of holding a short
    bars_per_year: float = 8760.0  # 1h bars, 24/7 market

    def validate(self) -> None:
        if self.sizing_mode not in VALID_SIZING_MODES:
            raise ValueError(
                f"sizing_mode '{self.sizing_mode}' invalid; expected one of {VALID_SIZING_MODES}"
            )
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        for name in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass
class Trade:
    direction: int  # LONG or SHORT
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float  # net of fees and funding
    pnl_pct: float  # net pnl / entry notional
    fees: float
    funding: float
    bars_held: int
    exit_reason: str
    mae_pct: float  # max adverse excursion (negative or 0)
    mfe_pct: float  # max favorable excursion (positive or 0)


@dataclass
class _OpenPosition:
    direction: int
    qty: float
    entry_price: float
    entry_ts: int
    entry_index: int
    entry_fee: float
    peak_price: float  # favorable extreme, drives trailing stop
    worst_price: float  # adverse extreme, drives MAE
    funding: float = 0.0


@dataclass
class BacktestResult:
    strategy: str
    config: BacktestConfig
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    exposure: float = 0.0
    metrics: dict = field(default_factory=dict)


def run_backtest(
    candles: list[Candle], strategy: Strategy, config: BacktestConfig | None = None
) -> BacktestResult:
    """Run one strategy over one series and return curve, trades, metrics."""
    config = config or BacktestConfig()
    config.validate()
    if len(candles) < 3:
        raise ValueError("need at least 3 candles to backtest")

    strategy.prepare(candles)
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    atr_series = compute_atr(highs, lows, closes, config.atr_period)
    vol_series = rolling_std(bar_returns(closes), config.vol_lookback)
    bar_ms = candles[1].ts - candles[0].ts
    bar_days = bar_ms / 86_400_000.0

    result = BacktestResult(strategy=strategy.describe(), config=config)
    realized_equity = config.initial_capital
    position: _OpenPosition | None = None
    pending_target: int | None = None
    bars_in_market = 0

    for i, bar in enumerate(candles):
        # -- 1. Execute the target decided on the previous close, at this open.
        if pending_target is not None:
            current_dir = position.direction if position else FLAT
            if pending_target != current_dir:
                if position is not None:
                    fill = _apply_slippage(bar.open, -position.direction, config.slippage_bps)
                    realized_equity += _close_position(
                        result, position, fill, bar.ts, i, EXIT_SIGNAL
                    )
                    position = None
                if pending_target != FLAT:
                    position = _open_position(
                        pending_target,
                        bar,
                        i,
                        realized_equity,
                        config,
                        atr_series[i - 1] if i > 0 else None,
                        vol_series[i - 1] if i > 0 else None,
                    )
            pending_target = None

        # -- 2. Intrabar protective exits (stop first when ambiguous).
        if position is not None:
            position.funding += _funding_cost(position, config, bar_days)
            exit_fill, reason = _check_protective_exits(position, bar, config)
            _update_excursions(position, bar)
            if exit_fill is not None and reason is not None:
                realized_equity += _close_position(result, position, exit_fill, bar.ts, i, reason)
                position = None

        # -- 3. Mark to market on the close.
        equity = realized_equity
        if position is not None:
            equity += _unrealized_pnl(position, bar.close)
            bars_in_market += 1
        result.equity_curve.append((bar.ts, equity))

        # -- 4. Ask the strategy for next bar's target (uses data <= bar i).
        target = strategy.signal(i)
        if target not in (LONG, SHORT, FLAT):
            raise ValueError(f"strategy returned invalid signal {target!r} at bar {i}")
        pending_target = target

    # Force-close any open position on the last close.
    if position is not None:
        last = candles[-1]
        fill = _apply_slippage(last.close, -position.direction, config.slippage_bps)
        realized_equity += _close_position(
            result, position, fill, last.ts, len(candles) - 1, EXIT_END_OF_DATA
        )
        result.equity_curve[-1] = (last.ts, realized_equity)

    result.exposure = bars_in_market / len(candles)
    result.metrics = compute_metrics(
        [eq for _, eq in result.equity_curve],
        [t.pnl for t in result.trades],
        [t.pnl_pct for t in result.trades],
        config.bars_per_year,
        exposure=result.exposure,
        initial_capital=config.initial_capital,
    )
    return result


# ---------------------------------------------------------------------------
# Fill mechanics
# ---------------------------------------------------------------------------


def _apply_slippage(price: float, trade_side: int, slippage_bps: float) -> float:
    """Adverse fill: buying pays up, selling receives less.

    ``trade_side`` is +1 when buying (opening long / closing short) and -1
    when selling (opening short / closing long).
    """
    return price * (1.0 + trade_side * slippage_bps / 10_000.0)


def _open_position(
    direction: int,
    bar: Candle,
    index: int,
    equity: float,
    config: BacktestConfig,
    prev_atr: float | None,
    prev_vol: float | None,
) -> _OpenPosition | None:
    fill = _apply_slippage(bar.open, direction, config.slippage_bps)
    notional = _position_notional(equity, fill, config, prev_atr, prev_vol)
    if notional <= 0:
        return None
    qty = notional / fill
    entry_fee = notional * config.fee_bps / 10_000.0
    return _OpenPosition(
        direction=direction,
        qty=qty,
        entry_price=fill,
        entry_ts=bar.ts,
        entry_index=index,
        entry_fee=entry_fee,
        peak_price=fill,
        worst_price=fill,
    )


def _position_notional(
    equity: float,
    fill: float,
    config: BacktestConfig,
    prev_atr: float | None,
    prev_vol: float | None,
) -> float:
    if equity <= 0:
        return 0.0
    if config.sizing_mode == "fixed_fraction":
        return equity * config.fraction
    if config.sizing_mode == "risk_per_trade":
        if config.stop_loss_pct is not None:
            stop_distance = fill * config.stop_loss_pct
        elif prev_atr is not None and prev_atr > 0:
            stop_distance = prev_atr * config.atr_stop_mult
        else:
            return equity * config.fraction  # warmup fallback
        risk_amount = equity * config.risk_pct / 100.0
        return min(risk_amount / stop_distance * fill, equity)  # cap: no leverage
    # vol_target
    if prev_vol is None or prev_vol <= 0:
        return equity * config.fraction
    realized_annual = prev_vol * config.bars_per_year**0.5
    scale = min(1.0, config.vol_target_annual / realized_annual)
    return equity * scale


def _funding_cost(position: _OpenPosition, config: BacktestConfig, bar_days: float) -> float:
    if position.direction != SHORT or config.short_funding_daily_bps == 0.0:
        return 0.0
    notional = position.qty * position.entry_price
    return notional * config.short_funding_daily_bps / 10_000.0 * bar_days


def _check_protective_exits(
    position: _OpenPosition, bar: Candle, config: BacktestConfig
) -> tuple[float | None, str | None]:
    """Evaluate stop-loss, trailing stop, take-profit against this bar.

    Returns (fill_price_with_slippage, reason) or (None, None). Stops use
    the trailing peak as of the PREVIOUS bar, so a favorable move inside
    this bar cannot ratchet the stop before the bar is over.
    """
    d = position.direction
    stop_candidates: list[float] = []
    stop_reason = EXIT_STOP_LOSS
    if config.stop_loss_pct is not None:
        stop_candidates.append(position.entry_price * (1.0 - d * config.stop_loss_pct))
    if config.trailing_stop_pct is not None:
        trail = position.peak_price * (1.0 - d * config.trailing_stop_pct)
        if (
            not stop_candidates
            or (d == LONG and trail > stop_candidates[0])
            or (d == SHORT and trail < stop_candidates[0])
        ):
            stop_reason = EXIT_TRAILING_STOP
        stop_candidates.append(trail)
    stop_price = None
    if stop_candidates:
        stop_price = max(stop_candidates) if d == LONG else min(stop_candidates)
        if stop_reason == EXIT_STOP_LOSS and config.trailing_stop_pct is not None:
            # the tighter of the two won; recheck which one it was
            fixed = position.entry_price * (1.0 - d * config.stop_loss_pct)
            stop_reason = EXIT_STOP_LOSS if stop_price == fixed else EXIT_TRAILING_STOP

    # Stop first (conservative) — check touch, then gap-aware fill price.
    if stop_price is not None:
        touched = bar.low <= stop_price if d == LONG else bar.high >= stop_price
        if touched:
            gap = (d == LONG and bar.open <= stop_price) or (d == SHORT and bar.open >= stop_price)
            raw_fill = bar.open if gap else stop_price
            return _apply_slippage(raw_fill, -d, config.slippage_bps), stop_reason

    if config.take_profit_pct is not None:
        tp_price = position.entry_price * (1.0 + d * config.take_profit_pct)
        touched = bar.high >= tp_price if d == LONG else bar.low <= tp_price
        if touched:
            gap = (d == LONG and bar.open >= tp_price) or (d == SHORT and bar.open <= tp_price)
            raw_fill = bar.open if gap else tp_price
            return _apply_slippage(raw_fill, -d, config.slippage_bps), EXIT_TAKE_PROFIT
    return None, None


def _update_excursions(position: _OpenPosition, bar: Candle) -> None:
    if position.direction == LONG:
        position.peak_price = max(position.peak_price, bar.high)
        position.worst_price = min(position.worst_price, bar.low)
    else:
        position.peak_price = min(position.peak_price, bar.low)
        position.worst_price = max(position.worst_price, bar.high)


def _unrealized_pnl(position: _OpenPosition, price: float) -> float:
    gross = position.qty * (price - position.entry_price) * position.direction
    return gross - position.entry_fee - position.funding


def _close_position(
    result: BacktestResult,
    position: _OpenPosition,
    fill: float,
    ts: int,
    index: int,
    reason: str,
) -> float:
    """Record the trade; return net pnl to add to realized equity."""
    d = position.direction
    gross = position.qty * (fill - position.entry_price) * d
    exit_fee = position.qty * fill * result.config.fee_bps / 10_000.0
    fees = position.entry_fee + exit_fee
    net = gross - fees - position.funding
    entry_notional = position.qty * position.entry_price
    entry = position.entry_price
    result.trades.append(
        Trade(
            direction=d,
            entry_ts=position.entry_ts,
            exit_ts=ts,
            entry_price=round(entry, 8),
            exit_price=round(fill, 8),
            qty=round(position.qty, 10),
            pnl=round(net, 6),
            pnl_pct=net / entry_notional if entry_notional else 0.0,
            fees=round(fees, 6),
            funding=round(position.funding, 6),
            bars_held=index - position.entry_index,
            exit_reason=reason,
            mae_pct=min(0.0, (position.worst_price - entry) / entry * d),
            mfe_pct=max(0.0, (position.peak_price - entry) / entry * d),
        )
    )
    return net
