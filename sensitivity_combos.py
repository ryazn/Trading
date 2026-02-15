#!/usr/bin/env python3
"""
Test combined best params from sensitivity analysis.
Uses insights: min_strength=0.6, pivot_left=20, sweep_ticks=10, momentum bars=15
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

BASELINE = {
    "cvd_smooth": 1, "cvd_anchor_bars": 390,
    "pivot_left": 10, "pivot_right": 1,
    "max_level_age": 100, "max_levels": 30,
    "sweep_ticks": 2, "tick_size": 0.25,
    "min_strength": 0.3, "strength_period": 100,
    "use_momentum_filter": False, "momentum_bars": 5, "momentum_tolerance": 0.02,
    "signal_cooldown": 5, "allow_longs": True, "allow_shorts": True,
}


def run_test(combo, data_dict):
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
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def print_results(title, results):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"  {'Config':<50s} {'Trades':>6s} {'WR':>6s} {'PF':>6s} {'Net P&L':>12s} {'AvgTrd':>8s} {'MaxDD':>6s}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*8} {'-'*6}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<50s}  ERROR")
            continue
        m = " ***" if r["net_profit"] > 0 else ""
        print(f"  {r['label']:<50s} {r['total_trades']:>6d} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f}"
              f" ${r['net_profit']:>10,.2f} ${r['avg_trade']:>7.2f} {r['max_drawdown_pct']:>5.1f}%{m}")


def main():
    n_workers = max(cpu_count() - 1, 1)

    # Load BOTH years for train/val
    loader = DataLoader()

    print("Loading 2024 (test year)...")
    d24 = loader.load_csv("data/es_1min/ES_2024.csv")
    d24 = d24[d24["close"] >= 1000]
    d24 = d24[~d24.index.duplicated(keep='first')]

    print("Loading 2025 (validation)...")
    d25 = loader.load_csv("data/es_1min/ES_2025.csv")
    d25 = d25[d25["close"] >= 1000]
    d25 = d25[~d25.index.duplicated(keep='first')]

    print("Loading 2023 (additional validation)...")
    d23 = loader.load_csv("data/es_1min/ES_2023.csv")
    d23 = d23[d23["close"] >= 1000]
    d23 = d23[~d23.index.duplicated(keep='first')]

    # Build combined test configs
    tests = []

    # Baseline
    tests.append((dict(BASELINE), "A. BASELINE"))

    # Stack the winners from sensitivity analysis
    # min_strength=0.6 was the ONLY profitable param
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    tests.append((p, "B. min_strength=0.6 only"))

    # min_strength=0.5 (more trades)
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    tests.append((p, "C. min_strength=0.5 only"))

    # Best PL: pivot_left=20
    p = dict(BASELINE)
    p["pivot_left"] = 20
    tests.append((p, "D. pivot_left=20 only"))

    # sweep_ticks=10 (fewer but cleaner sweeps)
    p = dict(BASELINE)
    p["sweep_ticks"] = 10
    tests.append((p, "E. sweep_ticks=10 only"))

    # sweep_ticks=1 (was 2nd best)
    p = dict(BASELINE)
    p["sweep_ticks"] = 1
    tests.append((p, "F. sweep_ticks=1 only"))

    # momentum bars=15
    p = dict(BASELINE)
    p["use_momentum_filter"] = True
    p["momentum_bars"] = 15
    p["momentum_tolerance"] = 0.02
    tests.append((p, "G. momentum ON bars=15"))

    # === COMBINED STACKS ===

    # Stack: strength 0.6 + pivot 20
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    tests.append((p, "H. strength=0.6 + pivot=20"))

    # Stack: strength 0.5 + pivot 15
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["pivot_left"] = 15
    tests.append((p, "I. strength=0.5 + pivot=15"))

    # Stack: strength 0.5 + sweep 1
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["sweep_ticks"] = 1
    tests.append((p, "J. strength=0.5 + sweep=1"))

    # Stack: strength 0.6 + sweep 1
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["sweep_ticks"] = 1
    tests.append((p, "K. strength=0.6 + sweep=1"))

    # Stack: strength 0.5 + pivot 20 + sweep 1
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["pivot_left"] = 20
    p["sweep_ticks"] = 1
    tests.append((p, "L. str=0.5 + piv=20 + swp=1"))

    # Stack: strength 0.6 + pivot 20 + sweep 1
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 1
    tests.append((p, "M. str=0.6 + piv=20 + swp=1"))

    # Stack: all best + momentum
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["pivot_left"] = 20
    p["sweep_ticks"] = 1
    p["use_momentum_filter"] = True
    p["momentum_bars"] = 15
    p["momentum_tolerance"] = 0.02
    tests.append((p, "N. str=0.5+piv=20+swp=1+mom15"))

    # Stack: aggressive filtering
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 1
    p["signal_cooldown"] = 7
    p["max_level_age"] = 50
    tests.append((p, "O. str=0.6+piv=20+swp=1+cd=7+age=50"))

    # Stack: very selective
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 7
    p["signal_cooldown"] = 7
    tests.append((p, "P. str=0.6+piv=20+swp=7+cd=7"))

    # Widest net + high strength
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["pivot_left"] = 12
    p["sweep_ticks"] = 1
    p["max_level_age"] = 200
    p["strength_period"] = 200
    tests.append((p, "Q. str=0.5+piv=12+swp=1+age200+sp200"))

    # Fine-tune around strength 0.55
    for s in [0.45, 0.5, 0.55, 0.6, 0.65]:
        p = dict(BASELINE)
        p["min_strength"] = s
        p["pivot_left"] = 20
        p["sweep_ticks"] = 1
        tests.append((p, f"R. piv=20+swp=1 str={s}"))

    total = len(tests)
    print(f"\nRunning {total} configs on 3 years...")

    # Run on 2024
    worker_fn = partial(run_test, data_dict=d24)
    with Pool(n_workers) as pool:
        r24 = list(pool.imap(worker_fn, tests, chunksize=2))
    print_results("2024 RESULTS", r24)

    # Run on 2025
    worker_fn = partial(run_test, data_dict=d25)
    with Pool(n_workers) as pool:
        r25 = list(pool.imap(worker_fn, tests, chunksize=2))
    print_results("2025 RESULTS", r25)

    # Run on 2023
    worker_fn = partial(run_test, data_dict=d23)
    with Pool(n_workers) as pool:
        r23 = list(pool.imap(worker_fn, tests, chunksize=2))
    print_results("2023 RESULTS", r23)

    # Cross-year comparison
    print(f"\n{'='*110}")
    print(f"  CROSS-YEAR COMPARISON (Net P&L)")
    print(f"{'='*110}")
    print(f"  {'Config':<50s} {'2023':>12s} {'2024':>12s} {'2025':>12s} {'TOTAL':>12s} {'Consistent':>10s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i in range(total):
        label = tests[i][1]
        n23 = r23[i].get("net_profit", 0) if "error" not in r23[i] else 0
        n24 = r24[i].get("net_profit", 0) if "error" not in r24[i] else 0
        n25 = r25[i].get("net_profit", 0) if "error" not in r25[i] else 0
        total_pnl = n23 + n24 + n25

        # Check consistency: profitable in 2+ years
        pos_years = sum(1 for x in [n23, n24, n25] if x > 0)
        consistent = "YES" if pos_years >= 2 else "NO"

        m = " ***" if total_pnl > 0 else ""
        print(f"  {label:<50s} ${n23:>10,.2f} ${n24:>10,.2f} ${n25:>10,.2f} ${total_pnl:>10,.2f} {consistent:>10s}{m}")


if __name__ == "__main__":
    main()
