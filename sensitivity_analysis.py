#!/usr/bin/env python3
"""
PHANTOM Strategy Input Sensitivity Analysis

Varies ONE strategy parameter at a time while holding all others constant.
Uses fixed risk params (1ct, ATR 2.5, RR 1.5, RTH) so we only see signal impact.
"""

import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig, StopLossMode, TakeProfitMode, PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy

import pandas as pd

INITIAL_CAPITAL = 100000.0

# Fixed risk config — isolate strategy signal impact
RISK = RiskConfig(
    position_mode=PositionSizeMode.FIXED_CONTRACTS,
    fixed_contracts=1,
    use_stop_loss=True,
    sl_mode=StopLossMode.ATR,
    sl_atr_mult=2.5,
    sl_atr_period=14,
    use_take_profit=True,
    tp_mode=TakeProfitMode.RISK_REWARD,
    tp_rr_ratio=1.5,
    use_trailing=False,
    use_breakeven=False,
    use_session_filter=True,
    session_start_hour=9,
    session_start_minute=30,
    session_end_hour=16,
    session_end_minute=0,
    close_at_session_end=True,
    use_max_hold=False,
    commission_per_contract=1.24,
    slippage_ticks=1,
    tick_size=0.25,
    point_value=50.0,
)

# Baseline strategy params (pivot_right=1 per user request)
BASELINE = {
    "cvd_smooth": 1,
    "cvd_anchor_bars": 390,
    "pivot_left": 10,
    "pivot_right": 1,
    "max_level_age": 100,
    "max_levels": 30,
    "sweep_ticks": 2,
    "tick_size": 0.25,
    "min_strength": 0.3,
    "strength_period": 100,
    "use_momentum_filter": False,
    "momentum_bars": 5,
    "momentum_tolerance": 0.02,
    "signal_cooldown": 5,
    "allow_longs": True,
    "allow_shorts": True,
}


def run_test(combo, data_dict):
    """Run a single test."""
    params, label = combo
    try:
        strategy = PhantomCVDStrategy(params=params)
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=RISK)
        report = engine.run(data_dict, strategy, verbose=False)
        return {
            "label": label,
            "net_profit": report.net_profit,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe": report.sharpe_ratio,
            "avg_trade": report.avg_trade,
            "expectancy": report.expectancy,
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def print_sweep(title, results):
    """Print a parameter sweep table."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Setting':<30s} {'Trades':>6s} {'WR':>6s} {'PF':>6s} {'Net P&L':>12s} {'AvgTrd':>8s} {'MaxDD':>6s} {'Sharpe':>7s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*8} {'-'*6} {'-'*7}")

    for r in results:
        if "error" in r:
            print(f"  {r['label']:<30s}  ERROR: {r['error']}")
            continue
        marker = " ***" if r["net_profit"] > 0 else ""
        print(f"  {r['label']:<30s} {r['total_trades']:>6d} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f}"
              f" ${r['net_profit']:>10,.2f} ${r['avg_trade']:>7.2f} {r['max_drawdown_pct']:>5.1f}% {r['sharpe']:>6.2f}{marker}")


def main():
    n_workers = max(cpu_count() - 1, 1)

    # Load 1 year of data (2024 — recent, high-quality)
    print("Loading 2024 data...")
    loader = DataLoader()
    data = loader.load_csv("data/es_1min/ES_2024.csv")
    data = data[data["close"] >= 1000]
    data = data[~data.index.duplicated(keep='first')]
    print(f"  {len(data)} bars | {data.index[0]} -> {data.index[-1]}")

    all_tests = []

    # =========================================================
    # 1. BASELINE (pivot_right=1)
    # =========================================================
    all_tests.append((dict(BASELINE), "BASELINE (pr=1)"))

    # =========================================================
    # 2. CVD SMOOTHING
    # =========================================================
    for val in [1, 3, 5, 8, 10, 15, 20]:
        p = dict(BASELINE)
        p["cvd_smooth"] = val
        all_tests.append((p, f"cvd_smooth={val}"))

    # =========================================================
    # 3. PIVOT LEFT (swing detection lookback)
    # =========================================================
    for val in [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50]:
        p = dict(BASELINE)
        p["pivot_left"] = val
        all_tests.append((p, f"pivot_left={val}"))

    # =========================================================
    # 4. MAX LEVEL AGE (how long to keep swing levels)
    # =========================================================
    for val in [30, 50, 75, 100, 150, 200, 300, 390, 500]:
        p = dict(BASELINE)
        p["max_level_age"] = val
        all_tests.append((p, f"max_level_age={val}"))

    # =========================================================
    # 5. SWEEP TICKS (ticks past level to count as sweep)
    # =========================================================
    for val in [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]:
        p = dict(BASELINE)
        p["sweep_ticks"] = val
        all_tests.append((p, f"sweep_ticks={val}"))

    # =========================================================
    # 6. MIN STRENGTH (divergence strength filter)
    # =========================================================
    for val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = dict(BASELINE)
        p["min_strength"] = val
        all_tests.append((p, f"min_strength={val}"))

    # =========================================================
    # 7. STRENGTH PERIOD (CVD range lookback for normalization)
    # =========================================================
    for val in [20, 30, 50, 75, 100, 150, 200, 300]:
        p = dict(BASELINE)
        p["strength_period"] = val
        all_tests.append((p, f"strength_period={val}"))

    # =========================================================
    # 8. SIGNAL COOLDOWN
    # =========================================================
    for val in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        p = dict(BASELINE)
        p["signal_cooldown"] = val
        all_tests.append((p, f"signal_cooldown={val}"))

    # =========================================================
    # 9. MOMENTUM FILTER (ON/OFF + bars + tolerance)
    # =========================================================
    # First: momentum OFF vs ON with different bar counts
    all_tests.append((dict(BASELINE), "momentum=OFF"))
    for bars in [3, 5, 7, 10, 15, 20]:
        p = dict(BASELINE)
        p["use_momentum_filter"] = True
        p["momentum_bars"] = bars
        p["momentum_tolerance"] = 0.02
        all_tests.append((p, f"momentum=ON bars={bars} tol=0.02"))

    # Momentum ON with different tolerances (fixed bars=5)
    for tol in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        p = dict(BASELINE)
        p["use_momentum_filter"] = True
        p["momentum_bars"] = 5
        p["momentum_tolerance"] = tol
        all_tests.append((p, f"momentum=ON bars=5 tol={tol}"))

    # =========================================================
    # 10. CVD ANCHOR BARS (reset period)
    # =========================================================
    for val in [100, 200, 390, 500, 780, 0]:
        p = dict(BASELINE)
        p["cvd_anchor_bars"] = val
        label = f"cvd_anchor={val}" if val > 0 else "cvd_anchor=OFF(0)"
        all_tests.append((p, label))

    # =========================================================
    # 11. COMBINED: best of each (will test after seeing results)
    # =========================================================
    # Placeholder combos with promising param stacks
    combos_to_test = [
        {"cvd_smooth": 5, "min_strength": 0.2, "sweep_ticks": 3, "signal_cooldown": 10},
        {"cvd_smooth": 10, "min_strength": 0.1, "sweep_ticks": 1, "signal_cooldown": 3},
        {"cvd_smooth": 5, "pivot_left": 15, "sweep_ticks": 3, "max_level_age": 200},
        {"cvd_smooth": 8, "pivot_left": 7, "min_strength": 0.1, "signal_cooldown": 10},
        {"cvd_smooth": 3, "pivot_left": 12, "sweep_ticks": 2, "min_strength": 0.4, "signal_cooldown": 7},
    ]
    for i, overrides in enumerate(combos_to_test):
        p = dict(BASELINE)
        p.update(overrides)
        desc = ", ".join(f"{k}={v}" for k, v in overrides.items())
        all_tests.append((p, f"COMBO_{i+1}: {desc}"))

    total = len(all_tests)
    print(f"\nRunning {total} tests with {n_workers} workers...")

    # Run all in parallel
    start = time.time()
    worker_fn = partial(run_test, data_dict=data)
    results = []
    with Pool(n_workers) as pool:
        for r in pool.imap(worker_fn, all_tests, chunksize=2):
            results.append(r)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed/total:.1f}s per test)")

    # =========================================================
    # PRINT RESULTS BY GROUP
    # =========================================================

    # Baseline
    print_sweep("BASELINE", [r for r in results if r["label"].startswith("BASELINE")])

    # Group by parameter
    groups = [
        ("CVD SMOOTHING (cvd_smooth)", "cvd_smooth="),
        ("PIVOT LEFT BARS (swing detection lookback)", "pivot_left="),
        ("MAX SWING LEVEL AGE (bars to keep levels)", "max_level_age="),
        ("SWEEP THRESHOLD (ticks past level)", "sweep_ticks="),
        ("MIN DIVERGENCE STRENGTH", "min_strength="),
        ("STRENGTH LOOKBACK PERIOD", "strength_period="),
        ("SIGNAL COOLDOWN BARS", "signal_cooldown="),
        ("MOMENTUM FILTER", "momentum="),
        ("CVD ANCHOR/RESET PERIOD", "cvd_anchor="),
        ("COMBINED PARAMETER STACKS", "COMBO_"),
    ]

    for title, prefix in groups:
        group = [r for r in results if r["label"].startswith(prefix)]
        if group:
            print_sweep(title, group)

    # =========================================================
    # SUMMARY: Best value for each parameter
    # =========================================================
    print(f"\n{'='*90}")
    print(f"  SUMMARY: BEST VALUE FOR EACH PARAMETER (by Net P&L)")
    print(f"{'='*90}")

    for title, prefix in groups:
        group = [r for r in results if r["label"].startswith(prefix) and "error" not in r and r["total_trades"] >= 10]
        if group:
            best = max(group, key=lambda x: x["net_profit"])
            print(f"  {title:<45s} → {best['label']:<30s} Net: ${best['net_profit']:>8,.2f}  PF: {best['profit_factor']:.2f}  WR: {best['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
