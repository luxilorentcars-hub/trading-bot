---
layout: default
title: "Crypto Backtest Engine"
grand_parent: English
parent: Skill Guides
nav_order: 13
lang_peer: /ja/skills/crypto-backtest-engine/
permalink: /en/skills/crypto-backtest-engine/
generated: true
---

# Crypto Backtest Engine
{: .no_toc }

Event-driven crypto backtesting engine with realistic execution costs, protective exits, position sizing, grid search, walk-forward analysis, and Monte Carlo robustness testing, plus a paper/live trading layer that reuses the same strategy code. Use when backtesting crypto trading strategies on Binance OHLCV data, local CSV files, or synthetic data; when optimizing strategy parameters; when stress-testing a strategy's robustness; or when paper-trading or live-trading a validated strategy through a broker. Complements backtest-expert (methodology) with an executable engine.
{: .fs-6 .fw-300 }

<span class="badge badge-free">No API</span>

[View Source on GitHub](https://github.com/tradermonty/claude-trading-skills/tree/main/skills/crypto-backtest-engine){: .btn .fs-5 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 1. Overview

# Crypto Backtest Engine

---

## 2. Prerequisites

- OHLCV klines from the Binance public REST API (no key required); CSV and synthetic data also supported; Real spot order placement via signed Binance REST; paper broker is the default, real orders are guarded behind explicit flags and API keys; Local OHLCV CSV files
- Python 3.9+ recommended

---

## 3. Quick Start

Invoke this skill by describing your analysis needs to Claude.

---

## 4. Workflow

See the skill's SKILL.md for the complete workflow.

---

## 5. Resources

**Scripts:**

- `skills/crypto-backtest-engine/scripts/cbe_broker.py`
- `skills/crypto-backtest-engine/scripts/cbe_data.py`
- `skills/crypto-backtest-engine/scripts/cbe_engine.py`
- `skills/crypto-backtest-engine/scripts/cbe_indicators.py`
- `skills/crypto-backtest-engine/scripts/cbe_live.py`
- `skills/crypto-backtest-engine/scripts/cbe_metrics.py`
- `skills/crypto-backtest-engine/scripts/cbe_monte_carlo.py`
- `skills/crypto-backtest-engine/scripts/cbe_optimizer.py`
- `skills/crypto-backtest-engine/scripts/cbe_report.py`
- `skills/crypto-backtest-engine/scripts/cbe_strategies.py`
- `skills/crypto-backtest-engine/scripts/run_backtest.py`
- `skills/crypto-backtest-engine/scripts/run_live.py`
