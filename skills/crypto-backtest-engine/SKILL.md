---
name: crypto-backtest-engine
description: Event-driven crypto backtesting engine with realistic execution costs, protective exits, position sizing, grid search, walk-forward analysis, and Monte Carlo robustness testing, plus a paper/live trading layer that reuses the same strategy code. Use when backtesting crypto trading strategies on Binance OHLCV data, local CSV files, or synthetic data; when optimizing strategy parameters; when stress-testing a strategy's robustness; or when paper-trading or live-trading a validated strategy through a broker. Complements backtest-expert (methodology) with an executable engine.
---

# Crypto Backtest Engine

Executable, dependency-light backtesting engine for crypto strategies. Where
`backtest-expert` teaches the methodology, this skill runs the numbers: it
simulates strategies bar by bar with pessimistic execution assumptions and
reports the full statistics suite plus overfitting guards.

## Core design decisions

- **No look-ahead by construction**: a signal computed on the close of bar
  *i* fills at the *open of bar i+1*, with slippage applied against the
  trader. Sizing inputs (ATR, realized vol) use data up to bar *i-1*.
- **Pessimistic fills**: stops/take-profits are evaluated intrabar against
  high/low; gaps fill at the open; when a stop and a take-profit are both
  touched in one bar, the stop is assumed to fill first.
- **Costs everywhere**: taker fees per fill, slippage per fill, optional
  daily funding cost on shorts. No leverage — notional is capped at equity.

## Quick start

```bash
cd skills/crypto-backtest-engine/scripts

# Offline sanity check on deterministic synthetic data
python3 run_backtest.py --synthetic 5000 --strategy sma_cross \
    --params '{"fast": 20, "slow": 100}'

# Real BTC data from the Binance public API (no key needed, CSV-cached)
python3 run_backtest.py --symbol BTCUSDT --interval 4h --max-bars 8000 \
    --strategy donchian_breakout --trailing-stop 0.08 --monte-carlo 2000

# Grid search + walk-forward robustness check + JSON export
python3 run_backtest.py --csv data/btc_1h.csv --strategy sma_cross \
    --optimize '{"fast": [10, 20, 50], "slow": [100, 200]}' \
    --walk-forward 4 --json-out results.json
```

## Components

| Module | Role |
|---|---|
| `cbe_data.py` | Binance public API fetch (paginated, retried, cached), CSV I/O, seeded synthetic OHLCV |
| `cbe_indicators.py` | SMA, EMA, RSI (Wilder), ATR, Bollinger, Donchian, ROC, rolling std — O(n), warmup-aligned |
| `cbe_strategies.py` | Strategy base class + registry: `sma_cross`, `rsi_reversion`, `bollinger_breakout`, `donchian_breakout`, `momentum` |
| `cbe_engine.py` | Event-driven simulator: fees, slippage, funding, stop-loss / take-profit / trailing stop, 3 sizing modes |
| `cbe_metrics.py` | Sharpe, Sortino, Calmar, SQN, max DD + duration, Ulcer index, VaR, profit factor, expectancy, streaks |
| `cbe_optimizer.py` | Grid search + walk-forward analysis with walk-forward efficiency (OOS/IS) |
| `cbe_monte_carlo.py` | Trade-bootstrap simulation: equity/drawdown percentile bands, P(loss), P(ruin) |
| `cbe_report.py` | Plain-text report + JSON export |
| `run_backtest.py` | Backtest CLI wiring everything together |
| `cbe_broker.py` | Broker interface + `PaperBroker` (simulated) + guarded `BinanceBroker` (real spot) |
| `cbe_live.py` | `LiveTrader`: replays the strategy over a rolling window and reconciles the position toward the target |
| `run_live.py` | Paper/live trading CLI (paper by default; real orders behind explicit guards) |

## Position sizing modes

- `fixed_fraction` — notional = equity × fraction (default: all-in)
- `risk_per_trade` — qty sized so the stop distance risks `--risk-pct` of
  equity (stop from `--stop-loss` or ATR × multiplier), capped at equity
- `vol_target` — notional scaled so realized annualized vol ≈ `--vol-target`

## Interpreting the robustness sections

- **Walk-forward efficiency** below ~0.5 means out-of-sample performance is
  less than half of in-sample: treat the strategy as curve-fit.
- **Monte Carlo p5 final equity** below initial capital, or a meaningful
  P(ruin), means the trade distribution cannot support the position size —
  reduce sizing before questioning the signal.
- Follow the `backtest-expert` skill for the full "beat it to death"
  methodology: parameter neighborhoods, regime slicing, cost stress tests.

## Writing a new strategy

Subclass `Strategy` in `cbe_strategies.py`, implement `prepare(candles)`
(precompute indicators) and `signal(i)` (return `LONG`, `SHORT`, or `FLAT`
using data up to bar *i* only), then add the class to `STRATEGY_REGISTRY`
and a parameter grid to `DEFAULT_GRIDS`. The engine handles execution,
costs, exits, and sizing — strategies stay pure signal logic.

## Paper & live trading

The live layer reuses the exact same `Strategy` classes as the backtest, so
signals are identical to what you validated. On each poll the trader replays
the strategy over a rolling window of **closed** candles and places a market
order that reconciles the current position toward the target — the live
analogue of the backtest's "fill at next bar's open".

**Safety model — paper by default.** Real orders require *all* of
`--broker binance --live --i-understand-live` plus `BINANCE_API_KEY` /
`BINANCE_API_SECRET` in the environment. Missing any one → orders are
simulated (paper) or dry-run (logged, never sent). Position size is capped
at equity (no leverage) and by `--max-notional`; `--state` persists the last
processed bar so a restart never re-trades a bar.

```bash
cd skills/crypto-backtest-engine/scripts

# Paper trade against synthetic data (fully offline)
python3 run_live.py --synthetic --strategy sma_cross --params '{"fast":20,"slow":100}'

# Paper trade against live Binance prices, loop each bar, persist state
python3 run_live.py --symbol BTCUSDT --interval 1h --strategy donchian_breakout \
    --broker paper --capital 5000 --state state/btc.json --loop

# REAL spot orders (long/flat only) — every guard must be explicit
BINANCE_API_KEY=... BINANCE_API_SECRET=... python3 run_live.py \
    --symbol BTCUSDT --interval 1h --strategy sma_cross \
    --broker binance --live --i-understand-live \
    --max-notional 200 --state state/btc.json --loop
```

Binance spot cannot short, so a SHORT target collapses to FLAT there; use the
`PaperBroker` (or a futures adapter, not yet implemented) for short exposure.
Always paper-trade a strategy that passed walk-forward before going live.

## Testing

```bash
python3 -m pytest skills/crypto-backtest-engine/scripts/tests/ -v
```

The test suite (135 tests) covers execution semantics (next-open fills,
gap handling, stop-vs-TP priority), cost accounting, sizing caps, metric
math against hand-computed values, optimizer stitching, Monte Carlo
determinism, paper-broker fills, live position reconciliation, restart-safe
state, the real-order safety guards, and a mocked Binance client — no
network access required.
