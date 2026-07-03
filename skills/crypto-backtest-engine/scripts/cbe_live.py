"""Live/paper trading loop that reuses the backtest strategy code.

Signal parity with the backtester is the whole point: on every poll the
trader replays the strategy over the full rolling window (``prepare`` then
``signal(i)`` for each bar), exactly as the backtest engine does. The
target from the LAST CLOSED bar drives a market order that reconciles the
current position toward the target — the live analogue of the backtest's
"fill at next bar's open".

Safety model:
  * Acts only on CLOSED candles (the in-progress bar is dropped).
  * Position sizing is capped at equity (no leverage) and by ``max_notional``.
  * State (last processed bar, target) persists to JSON so a restart does
    not re-trade a bar already acted on.
  * The broker decides whether an order is real or a dry-run; this loop is
    identical in paper and live mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cbe_broker import SIDE_BUY, SIDE_SELL, Broker, OrderResult
from cbe_data import Candle
from cbe_strategies import FLAT, LONG, SHORT, Strategy


@dataclass
class LiveConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    sizing_mode: str = "fixed_fraction"  # or "risk_per_trade"
    fraction: float = 0.95  # share of equity per position (leave a buffer)
    risk_pct: float = 1.0  # risk_per_trade: % of equity risked to the stop
    stop_loss_pct: float | None = None  # required for risk_per_trade sizing
    max_notional: float | None = None  # hard cap on any single position notional
    min_order_notional: float = 10.0  # skip dust orders below exchange minimum
    allow_short: bool = False  # spot brokers cannot short
    window_bars: int = 400  # rolling history fed to the strategy each poll

    def validate(self) -> None:
        if self.sizing_mode not in ("fixed_fraction", "risk_per_trade"):
            raise ValueError(f"invalid sizing_mode {self.sizing_mode!r}")
        if not 0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if self.sizing_mode == "risk_per_trade" and not self.stop_loss_pct:
            raise ValueError("risk_per_trade sizing requires stop_loss_pct")
        if self.window_bars < 50:
            raise ValueError("window_bars too small for indicator warmup")


@dataclass
class LiveDecision:
    ts: int
    bar_close: float
    current_direction: int
    target_direction: int
    target_qty: float
    order: OrderResult | None = None
    note: str = ""


@dataclass
class LiveState:
    last_bar_ts: int = 0
    target_direction: int = FLAT
    decisions: int = 0

    @classmethod
    def load(cls, path: str | None) -> LiveState:
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            return cls(**{k: data[k] for k in ("last_bar_ts", "target_direction", "decisions")})
        return cls()

    def save(self, path: str | None) -> None:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(asdict(self), indent=2))


class LiveTrader:
    """Drives one strategy on one symbol through a broker."""

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        config: LiveConfig,
        state_path: str | None = None,
    ) -> None:
        config.validate()
        self.strategy = strategy
        self.broker = broker
        self.config = config
        self.state_path = state_path
        self.state = LiveState.load(state_path)

    # -- signal (identical replay to the backtest) ----------------------

    def compute_target(self, closed_candles: list[Candle]) -> int:
        """Replay the strategy over the window; return the last bar's target."""
        if len(closed_candles) < 2:
            return FLAT
        self.strategy.prepare(closed_candles)
        target = FLAT
        for i in range(len(closed_candles)):
            target = self.strategy.signal(i)
        if target == SHORT and not (self.config.allow_short and self.broker.supports_short):
            return FLAT  # spot / short disabled: SHORT collapses to FLAT
        return target

    # -- sizing (capped, no leverage) -----------------------------------

    def target_qty(self, direction: int, equity: float, price: float) -> float:
        if direction == FLAT or price <= 0:
            return 0.0
        cfg = self.config
        if cfg.sizing_mode == "risk_per_trade":
            stop_distance = price * cfg.stop_loss_pct
            risk_amount = equity * cfg.risk_pct / 100.0
            notional = min(risk_amount / stop_distance * price, equity * cfg.fraction)
        else:
            notional = equity * cfg.fraction
        if cfg.max_notional is not None:
            notional = min(notional, cfg.max_notional)
        return notional / price

    # -- one poll -------------------------------------------------------

    def step(self, candles: list[Candle], bar_closed: bool = True) -> LiveDecision | None:
        """Process the latest data. ``candles`` is the rolling window ending
        with the most recent bar. If ``bar_closed`` is False the last bar is
        still forming and is dropped before any decision."""
        closed = candles if bar_closed else candles[:-1]
        if len(closed) < 2:
            return None
        window = closed[-self.config.window_bars :]
        last = window[-1]

        if last.ts <= self.state.last_bar_ts:
            return None  # already acted on this bar (idempotent restart-safe)

        target_dir = self.compute_target(window)
        price = last.close
        equity = self.broker.get_equity(price)
        pos = self.broker.get_position(self.config.symbol)
        desired_qty = self.target_qty(target_dir, equity, price) * target_dir  # signed
        delta = desired_qty - pos.qty

        decision = LiveDecision(
            ts=last.ts,
            bar_close=price,
            current_direction=pos.direction,
            target_direction=target_dir,
            target_qty=abs(desired_qty),
        )

        if abs(delta) * price < self.config.min_order_notional:
            decision.note = "no material change; order skipped"
        else:
            side = SIDE_BUY if delta > 0 else SIDE_SELL
            decision.order = self.broker.market_order(self.config.symbol, side, abs(delta), price)

        self.state.last_bar_ts = last.ts
        self.state.target_direction = target_dir
        self.state.decisions += 1
        self.state.save(self.state_path)
        return decision


def format_decision(decision: LiveDecision) -> str:
    dir_name = {LONG: "LONG", SHORT: "SHORT", FLAT: "FLAT"}
    when = datetime.fromtimestamp(decision.ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    parts = [
        f"[{when} UTC] close={decision.bar_close:.2f}",
        f"{dir_name[decision.current_direction]} -> {dir_name[decision.target_direction]}",
    ]
    if decision.order is not None:
        o = decision.order
        parts.append(f"{o.status} {o.side} {o.qty:.6f} @ {o.price:.2f} ({o.notional:.2f})")
        if o.reason:
            parts.append(f"({o.reason})")
    if decision.note:
        parts.append(decision.note)
    return "  ".join(parts)
