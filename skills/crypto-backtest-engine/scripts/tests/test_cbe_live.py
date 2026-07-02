"""Tests for the live trader: signal parity, reconciliation, sizing, safety."""

from __future__ import annotations

import pytest
from cbe_broker import PaperBroker
from cbe_live import LiveConfig, LiveState, LiveTrader
from cbe_strategies import FLAT, LONG, SHORT, Strategy
from cbe_test_helpers import make_trend_candles


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, script: dict, default: int = FLAT) -> None:
        super().__init__()
        self.script = script
        self.default = default

    def prepare(self, candles) -> None:
        self._n = len(candles)

    def signal(self, i: int) -> int:
        return self.script.get(i, self.default)


def paper_trader(strategy, **cfg_overrides):
    cfg = LiveConfig(window_bars=50, min_order_notional=1.0, **cfg_overrides)
    broker = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
    return LiveTrader(strategy, broker, cfg), broker


class TestConfigValidation:
    def test_bad_sizing_mode(self):
        with pytest.raises(ValueError):
            LiveConfig(sizing_mode="yolo").validate()

    def test_risk_per_trade_needs_stop(self):
        with pytest.raises(ValueError, match="stop_loss_pct"):
            LiveConfig(sizing_mode="risk_per_trade").validate()

    def test_fraction_bounds(self):
        with pytest.raises(ValueError):
            LiveConfig(fraction=1.5).validate()

    def test_window_too_small(self):
        with pytest.raises(ValueError):
            LiveConfig(window_bars=10).validate()


class TestSignalParity:
    def test_replays_to_last_bar_target(self):
        candles = make_trend_candles(60)
        # long on the whole window -> target LONG
        trader, _ = paper_trader(ScriptedStrategy({}, default=LONG))
        assert trader.compute_target(candles) == LONG

    def test_short_collapses_to_flat_on_spot(self):
        candles = make_trend_candles(60)
        # allow_short False + paper... but paper supports short; disable via config
        trader, _ = paper_trader(ScriptedStrategy({}, default=SHORT), allow_short=False)
        assert trader.compute_target(candles) == FLAT

    def test_short_allowed_when_broker_supports_and_config_on(self):
        candles = make_trend_candles(60)
        trader, _ = paper_trader(ScriptedStrategy({}, default=SHORT), allow_short=True)
        assert trader.compute_target(candles) == SHORT


class TestStep:
    def test_opens_long_position(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(ScriptedStrategy({}, default=LONG), fraction=0.5)
        decision = trader.step(candles)
        assert decision is not None
        assert decision.target_direction == LONG
        assert decision.order is not None and decision.order.side == "BUY"
        pos = broker.get_position("BTCUSDT")
        assert pos.qty > 0
        # sized to ~50% of equity
        assert pos.qty * candles[-1].close == pytest.approx(5_000.0, rel=0.02)

    def test_idempotent_same_bar(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(ScriptedStrategy({}, default=LONG))
        first = trader.step(candles)
        second = trader.step(candles)  # same last bar
        assert first is not None
        assert second is None  # already acted on this bar

    def test_drops_in_progress_bar(self):
        candles = make_trend_candles(60)
        trader, _ = paper_trader(ScriptedStrategy({}, default=LONG))
        # with bar_closed=False the last (forming) bar is ignored;
        # decision acts on candles[-2]
        decision = trader.step(candles, bar_closed=False)
        assert decision is not None
        assert decision.ts == candles[-2].ts

    def test_reconciles_long_to_flat(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(ScriptedStrategy({}, default=LONG))
        trader.step(candles)
        assert broker.get_position("BTCUSDT").qty > 0
        # next bar: strategy flips to FLAT -> should sell everything
        more = make_trend_candles(61)
        trader.strategy = ScriptedStrategy({}, default=FLAT)
        decision = trader.step(more)
        assert decision.target_direction == FLAT
        assert broker.get_position("BTCUSDT").qty == pytest.approx(0.0, abs=1e-9)

    def test_no_material_change_skips_order(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(ScriptedStrategy({}, default=LONG))
        trader.step(candles)
        n_orders = len(broker.order_log)
        # same target, next bar, tiny price move -> delta below min_order_notional
        trader2 = LiveTrader(
            ScriptedStrategy({}, default=LONG),
            broker,
            LiveConfig(window_bars=50, min_order_notional=1e9),  # huge threshold
        )
        more = make_trend_candles(61)
        decision = trader2.step(more)
        assert "skipped" in decision.note
        assert len(broker.order_log) == n_orders  # no new order

    def test_max_notional_cap(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(
            ScriptedStrategy({}, default=LONG), fraction=1.0, max_notional=1_000.0
        )
        decision = trader.step(candles)
        assert decision.order.notional <= 1_000.0 * 1.001


class TestSizing:
    def test_risk_per_trade_respects_risk(self):
        candles = make_trend_candles(60)
        trader, broker = paper_trader(
            ScriptedStrategy({}, default=LONG),
            sizing_mode="risk_per_trade",
            risk_pct=1.0,
            stop_loss_pct=0.05,
            fraction=1.0,
        )
        decision = trader.step(candles)
        qty = decision.target_qty
        price = candles[-1].close
        # risk = qty * stop_distance should be ~1% of equity
        risk = qty * price * 0.05
        assert risk == pytest.approx(100.0, rel=0.05)

    def test_flat_target_zero_qty(self):
        trader, _ = paper_trader(ScriptedStrategy({}, default=FLAT))
        assert trader.target_qty(FLAT, 10_000.0, 100.0) == 0.0


class TestStatePersistence:
    def test_state_roundtrip(self, tmp_path):
        path = str(tmp_path / "state.json")
        candles = make_trend_candles(60)
        trader, _ = paper_trader(ScriptedStrategy({}, default=LONG))
        trader.state_path = path
        trader.step(candles)
        reloaded = LiveState.load(path)
        assert reloaded.last_bar_ts == candles[-1].ts
        assert reloaded.target_direction == LONG
        assert reloaded.decisions == 1

    def test_restart_does_not_retrade(self, tmp_path):
        path = str(tmp_path / "state.json")
        candles = make_trend_candles(60)
        strategy = ScriptedStrategy({}, default=LONG)
        broker = PaperBroker(cash=10_000.0, fee_bps=0.0, slippage_bps=0.0)
        cfg = LiveConfig(window_bars=50, min_order_notional=1.0)
        t1 = LiveTrader(strategy, broker, cfg, state_path=path)
        t1.step(candles)
        orders_after_first = len(broker.order_log)
        # fresh trader loads persisted state -> same bar is skipped
        t2 = LiveTrader(ScriptedStrategy({}, default=LONG), broker, cfg, state_path=path)
        assert t2.step(candles) is None
        assert len(broker.order_log) == orders_after_first


class TestCliSafety:
    def test_live_without_ack_refused(self):
        from run_live import main

        with pytest.raises(SystemExit, match="i-understand-live"):
            main(
                [
                    "--symbol",
                    "BTCUSDT",
                    "--broker",
                    "binance",
                    "--live",
                    "--max-polls",
                    "1",
                ]
            )

    def test_live_without_keys_refused(self, monkeypatch):
        from run_live import main

        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
        with pytest.raises(SystemExit, match="BINANCE_API_KEY"):
            main(
                [
                    "--broker",
                    "binance",
                    "--live",
                    "--i-understand-live",
                    "--max-polls",
                    "1",
                ]
            )

    def test_paper_synthetic_runs(self, capsys):
        from run_live import main

        rc = main(
            [
                "--synthetic",
                "--strategy",
                "sma_cross",
                "--params",
                '{"fast":10,"slow":30}',
                "--broker",
                "paper",
                "--max-polls",
                "1",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "paper" in out
