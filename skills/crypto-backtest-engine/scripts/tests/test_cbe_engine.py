"""Engine correctness tests: no look-ahead, cost accounting, exits, sizing."""

from __future__ import annotations

import pytest
from cbe_data import Candle
from cbe_engine import (
    EXIT_END_OF_DATA,
    EXIT_SIGNAL,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_TRAILING_STOP,
    BacktestConfig,
    run_backtest,
)
from cbe_strategies import FLAT, LONG, SHORT, Strategy
from cbe_test_helpers import make_trend_candles


class ScriptedStrategy(Strategy):
    """Emits a pre-defined target per bar index — full control in tests."""

    name = "scripted"

    def __init__(self, script: dict, default: int = FLAT) -> None:
        super().__init__()
        self.script = script
        self.default = default

    def prepare(self, candles) -> None:
        pass

    def signal(self, i: int) -> int:
        return self.script.get(i, self.default)


def no_cost_config(**overrides) -> BacktestConfig:
    base = {"fee_bps": 0.0, "slippage_bps": 0.0, "initial_capital": 10_000.0}
    base.update(overrides)
    return BacktestConfig(**base)


class TestExecutionModel:
    def test_signal_fills_at_next_bar_open(self):
        candles = make_trend_candles(10, start=100.0, step=1.0)
        # flat until bar 3's close, long thereafter
        strategy = ScriptedStrategy({0: FLAT, 1: FLAT, 2: FLAT}, default=LONG)
        result = run_backtest(candles, strategy, no_cost_config())
        assert len(result.trades) == 1
        trade = result.trades[0]
        # signal at close of bar 3 -> filled at bar 4's open (no look-ahead)
        assert trade.entry_price == pytest.approx(candles[4].open)
        assert trade.entry_ts == candles[4].ts

    def test_flat_strategy_never_trades(self, synthetic_candles):
        result = run_backtest(synthetic_candles, ScriptedStrategy({}), no_cost_config())
        assert result.trades == []
        assert result.exposure == 0.0
        equities = [eq for _, eq in result.equity_curve]
        assert all(eq == pytest.approx(10_000.0) for eq in equities)

    def test_long_profits_on_uptrend(self):
        candles = make_trend_candles(50, start=100.0, step=1.0)
        result = run_backtest(candles, ScriptedStrategy({}, default=LONG), no_cost_config())
        assert result.metrics["net_profit"] > 0

    def test_short_profits_on_downtrend(self):
        candles = make_trend_candles(50, start=200.0, step=-1.0)
        result = run_backtest(candles, ScriptedStrategy({}, default=SHORT), no_cost_config())
        assert result.metrics["net_profit"] > 0

    def test_open_position_closed_at_end_of_data(self):
        candles = make_trend_candles(20)
        result = run_backtest(candles, ScriptedStrategy({}, default=LONG), no_cost_config())
        assert result.trades[-1].exit_reason == EXIT_END_OF_DATA

    def test_reversal_closes_then_opens(self):
        candles = make_trend_candles(30)
        script = {i: LONG if i < 15 else SHORT for i in range(30)}
        result = run_backtest(candles, ScriptedStrategy(script), no_cost_config())
        assert len(result.trades) == 2
        assert result.trades[0].direction == LONG
        assert result.trades[0].exit_reason == EXIT_SIGNAL
        assert result.trades[1].direction == SHORT

    def test_invalid_signal_rejected(self):
        candles = make_trend_candles(10)
        with pytest.raises(ValueError, match="invalid signal"):
            run_backtest(candles, ScriptedStrategy({0: 2}), no_cost_config())


class TestCosts:
    def test_fees_and_slippage_reduce_pnl(self):
        candles = make_trend_candles(50)
        strategy = ScriptedStrategy({}, default=LONG)
        free = run_backtest(candles, strategy, no_cost_config())
        costly = run_backtest(
            candles, ScriptedStrategy({}, default=LONG), no_cost_config(fee_bps=25, slippage_bps=10)
        )
        assert costly.metrics["net_profit"] < free.metrics["net_profit"]
        assert costly.trades[0].fees > 0

    def test_slippage_adverse_on_entry(self):
        candles = make_trend_candles(10)
        result = run_backtest(
            candles,
            ScriptedStrategy({0: FLAT, 1: FLAT, 2: FLAT}, default=LONG),
            no_cost_config(slippage_bps=100),
        )
        assert result.trades[0].entry_price == pytest.approx(candles[4].open * 1.01)

    def test_short_funding_cost_accrues(self):
        candles = make_trend_candles(50, start=500.0, step=-1.0)
        strategy = ScriptedStrategy({}, default=SHORT)
        no_funding = run_backtest(candles, strategy, no_cost_config())
        with_funding = run_backtest(
            candles,
            ScriptedStrategy({}, default=SHORT),
            no_cost_config(short_funding_daily_bps=50.0),
        )
        assert with_funding.trades[0].funding > 0
        assert with_funding.metrics["net_profit"] < no_funding.metrics["net_profit"]


class TestProtectiveExits:
    def test_stop_loss_triggers(self):
        candles = make_trend_candles(30, start=200.0, step=-2.0)
        result = run_backtest(
            candles,
            ScriptedStrategy({0: LONG}, default=LONG),
            no_cost_config(stop_loss_pct=0.03),
        )
        assert result.trades[0].exit_reason == EXIT_STOP_LOSS
        # loss is bounded near the stop (small tolerance for gap fills)
        assert result.trades[0].pnl_pct >= -0.05

    def test_take_profit_triggers(self):
        candles = make_trend_candles(30, start=100.0, step=2.0)
        result = run_backtest(
            candles,
            ScriptedStrategy({0: LONG}, default=LONG),
            no_cost_config(take_profit_pct=0.05),
        )
        assert result.trades[0].exit_reason == EXIT_TAKE_PROFIT
        assert result.trades[0].pnl > 0

    def test_trailing_stop_locks_in_gains(self):
        # up 30 bars then straight down: trailing stop should exit near top
        up = make_trend_candles(30, start=100.0, step=2.0)
        down = make_trend_candles(30, start=160.0, step=-3.0, ts0=up[-1].ts + 3_600_000)
        candles = up + down
        result = run_backtest(
            candles,
            ScriptedStrategy({0: LONG}, default=LONG),
            no_cost_config(trailing_stop_pct=0.05),
        )
        trade = result.trades[0]
        assert trade.exit_reason == EXIT_TRAILING_STOP
        assert trade.pnl > 0  # captured most of the uptrend

    def test_stop_beats_take_profit_same_bar(self):
        # one wide bar touches both stop (low) and target (high) -> stop wins
        candles = [
            Candle(ts=1_600_000_000_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=1),
            Candle(ts=1_600_003_600_000, open=100.0, high=120.0, low=80.0, close=110.0, volume=1),
            Candle(ts=1_600_007_200_000, open=110.0, high=111.0, low=109.0, close=110.0, volume=1),
        ]
        result = run_backtest(
            candles,
            ScriptedStrategy({0: LONG}, default=LONG),
            no_cost_config(stop_loss_pct=0.05, take_profit_pct=0.05),
        )
        assert result.trades[0].exit_reason == EXIT_STOP_LOSS

    def test_gap_through_stop_fills_at_open(self):
        candles = [
            Candle(ts=1_600_000_000_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=1),
            Candle(ts=1_600_003_600_000, open=100.0, high=101.0, low=99.0, close=100.0, volume=1),
            # gap down far through the 5% stop
            Candle(ts=1_600_007_200_000, open=80.0, high=81.0, low=79.0, close=80.0, volume=1),
            Candle(ts=1_600_010_800_000, open=80.0, high=81.0, low=79.0, close=80.0, volume=1),
        ]
        result = run_backtest(
            candles,
            ScriptedStrategy({0: LONG}, default=LONG),
            no_cost_config(stop_loss_pct=0.05),
        )
        trade = result.trades[0]
        assert trade.exit_reason == EXIT_STOP_LOSS
        assert trade.exit_price == pytest.approx(80.0)  # open fill, not stop price


class TestSizing:
    def test_fixed_fraction_halves_notional(self):
        candles = make_trend_candles(30)
        full = run_backtest(
            candles, ScriptedStrategy({}, default=LONG), no_cost_config(fraction=1.0)
        )
        half = run_backtest(
            candles, ScriptedStrategy({}, default=LONG), no_cost_config(fraction=0.5)
        )
        assert half.trades[0].qty == pytest.approx(full.trades[0].qty * 0.5)

    def test_risk_per_trade_caps_at_equity(self):
        candles = make_trend_candles(60)
        config = no_cost_config(sizing_mode="risk_per_trade", risk_pct=1.0, stop_loss_pct=0.05)
        result = run_backtest(candles, ScriptedStrategy({}, default=LONG), config)
        trade = result.trades[0]
        notional = trade.qty * trade.entry_price
        assert notional <= config.initial_capital * 1.0001
        # risk = qty * stop distance should be ~1% of equity (or capped below)
        risk = trade.qty * trade.entry_price * 0.05
        assert risk <= config.initial_capital * 0.0101

    def test_vol_target_reduces_exposure_in_high_vol(self, synthetic_candles):
        # stay flat past the vol-lookback warmup so sizing sees realized vol
        warmup = {i: FLAT for i in range(100)}
        strategy_a = ScriptedStrategy(dict(warmup), default=LONG)
        strategy_b = ScriptedStrategy(dict(warmup), default=LONG)
        full = run_backtest(synthetic_candles, strategy_a, no_cost_config())
        scaled = run_backtest(
            synthetic_candles,
            strategy_b,
            no_cost_config(sizing_mode="vol_target", vol_target_annual=0.10),
        )
        # a 10% vol target on a crypto series must size well below all-in
        assert scaled.trades[0].qty < full.trades[0].qty

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            BacktestConfig(sizing_mode="martingale").validate()
        with pytest.raises(ValueError):
            BacktestConfig(fraction=0.0).validate()
        with pytest.raises(ValueError):
            BacktestConfig(stop_loss_pct=-1.0).validate()


class TestResultIntegrity:
    def test_equity_curve_matches_candles(self, synthetic_candles):
        from cbe_strategies import SmaCross

        result = run_backtest(synthetic_candles, SmaCross(10, 30), BacktestConfig())
        assert len(result.equity_curve) == len(synthetic_candles)
        assert result.metrics["num_trades"] == len(result.trades)

    def test_trade_pnls_sum_to_final_equity(self, synthetic_candles):
        from cbe_strategies import SmaCross

        result = run_backtest(synthetic_candles, SmaCross(10, 30), BacktestConfig())
        final = result.equity_curve[-1][1]
        expected = result.config.initial_capital + sum(t.pnl for t in result.trades)
        assert final == pytest.approx(expected, rel=1e-9)

    def test_mae_mfe_signs(self, synthetic_candles):
        from cbe_strategies import DonchianBreakout

        result = run_backtest(synthetic_candles, DonchianBreakout(20, 10), BacktestConfig())
        for trade in result.trades:
            assert trade.mae_pct <= 0.0
            assert trade.mfe_pct >= 0.0
