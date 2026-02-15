#!/usr/bin/env python3
"""
Quick V2 optimizer: test trend/vol filter variations with the winning risk params.
Fixed 1ct, ATR 2.5 SL, RR 1.5 TP, RTH session filter.
"""

import sys
import time
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig, StopLossMode, TakeProfitMode, PositionSizeMode,
)
from strategies.phantom_cvd_v2 import PhantomCVDStrategyV2

import pandas as pd

INITIAL_CAPITAL = 100000.0


def load_data(years):
    loader = DataLoader()
    frames = []
    for year in years:
        path = f"data/es_1min/ES_{year}.csv"
        df = loader.load_csv(path)
        df = df[df["close"] >= 1000]
        frames.append(df)
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


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


def run_combo(combo, data_dict):
    params, combo_id = combo
    try:
        strategy = PhantomCVDStrategyV2(params=params)
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=RISK)
        report = engine.run(data_dict, strategy, verbose=False)
        return {
            "combo_id": combo_id,
            **{k: v for k, v in params.items() if k not in [
                "cvd_smooth", "cvd_anchor_bars", "max_levels", "tick_size",
                "allow_longs", "allow_shorts", "momentum_bars", "momentum_tolerance",
            ]},
            "net_profit": report.net_profit,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe": report.sharpe_ratio,
            "avg_trade": report.avg_trade,
        }
    except Exception:
        return None


def main():
    n_workers = max(cpu_count() - 1, 1)
    print(f"V2 Quick Optimizer | Workers: {n_workers}")

    print("\nLoading TRAIN (2021-2023)...")
    train = load_data([2021, 2022, 2023])
    print(f"  {len(train)} bars")

    print("Loading VAL (2024-2025)...")
    val = load_data([2024, 2025])
    print(f"  {len(val)} bars")

    # Build combos
    combos = []
    cid = 0

    for pivot_left in [8, 10, 12, 15]:
        for pivot_right in [1, 2, 3]:
            for min_strength in [0.2, 0.3, 0.4, 0.5]:
                for sweep_ticks in [1, 2, 3]:
                    for signal_cooldown in [3, 5, 10]:
                        for max_level_age in [80, 100, 150, 200]:
                            for use_trend in [False, True]:
                                for trend_mode in (["with_trend", "ranging"] if use_trend else [""]):
                                    for use_vol in [False, True]:
                                        params = {
                                            "cvd_smooth": 1,
                                            "cvd_anchor_bars": 390,
                                            "max_levels": 30,
                                            "tick_size": 0.25,
                                            "allow_longs": True,
                                            "allow_shorts": True,
                                            "pivot_left": pivot_left,
                                            "pivot_right": pivot_right,
                                            "min_strength": min_strength,
                                            "sweep_ticks": sweep_ticks,
                                            "signal_cooldown": signal_cooldown,
                                            "max_level_age": max_level_age,
                                            "strength_period": 100,
                                            "use_momentum_filter": False,
                                            "momentum_bars": 5,
                                            "momentum_tolerance": 0.02,
                                            "use_trend_filter": use_trend,
                                            "trend_ema_period": 200,
                                            "trend_mode": trend_mode if use_trend else "with_trend",
                                            "trend_atr_band": 3.0,
                                            "use_vol_filter": use_vol,
                                            "vol_atr_period": 14,
                                            "vol_lookback": 200,
                                            "vol_min_pctile": 0.15,
                                            "vol_max_pctile": 0.85,
                                            "use_dynamic_sweep": False,
                                        }
                                        combos.append((params, cid))
                                        cid += 1

    # Random sample to keep manageable
    import random
    random.seed(42)
    if len(combos) > 600:
        combos = random.sample(combos, 600)
        combos = [(p, i) for i, (p, _) in enumerate(combos)]

    total = len(combos)
    print(f"\nTesting {total} V2 combos...")

    # TRAIN
    start = time.time()
    worker_fn = partial(run_combo, data_dict=train)
    results = []
    done = 0
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(worker_fn, combos, chunksize=4):
            done += 1
            if r is not None:
                results.append(r)
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                n_prof = len([x for x in results if x["net_profit"] > 0 and x["total_trades"] >= 20])
                print(f"  [{done}/{total}] {n_prof} profitable | {rate:.1f}/sec")

    print(f"\nTrain done in {(time.time()-start)/60:.1f} min")

    # Sort by net profit
    viable = [r for r in results if r["total_trades"] >= 20]
    viable.sort(key=lambda x: x["net_profit"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP 15 V2 CONFIGS (TRAIN 2021-2023)")
    print(f"{'='*80}")
    show_keys = ["pivot_left", "pivot_right", "min_strength", "sweep_ticks",
                 "signal_cooldown", "max_level_age", "use_trend_filter",
                 "trend_mode", "use_vol_filter"]
    for i, r in enumerate(viable[:15]):
        params_str = ", ".join(f"{k}={r[k]}" for k in show_keys if k in r)
        print(f"\n  #{i+1}")
        print(f"  {params_str}")
        print(f"  Trades: {r['total_trades']:>4d}  WR: {r['win_rate']:>5.1f}%  PF: {r['profit_factor']:>5.2f}"
              f"  Net: ${r['net_profit']:>10,.2f}  DD: {r['max_drawdown_pct']:.1f}%  Sharpe: {r['sharpe']:.2f}")

    # VALIDATE top 20
    top_n = min(20, len(viable))
    print(f"\n{'='*60}")
    print(f"  Validating top {top_n} on 2024-2025...")
    print(f"{'='*60}")

    val_combos = [(viable[i], i) for i in range(top_n)]
    # Need to reconstruct full params for V2...
    val_combos_full = []
    for i in range(top_n):
        r = viable[i]
        params = {
            "cvd_smooth": 1,
            "cvd_anchor_bars": 390,
            "max_levels": 30,
            "tick_size": 0.25,
            "allow_longs": True,
            "allow_shorts": True,
            "pivot_left": r["pivot_left"],
            "pivot_right": r["pivot_right"],
            "min_strength": r["min_strength"],
            "sweep_ticks": r["sweep_ticks"],
            "signal_cooldown": r["signal_cooldown"],
            "max_level_age": r["max_level_age"],
            "strength_period": 100,
            "use_momentum_filter": False,
            "momentum_bars": 5,
            "momentum_tolerance": 0.02,
            "use_trend_filter": r.get("use_trend_filter", False),
            "trend_ema_period": 200,
            "trend_mode": r.get("trend_mode", "with_trend"),
            "trend_atr_band": 3.0,
            "use_vol_filter": r.get("use_vol_filter", False),
            "vol_atr_period": 14,
            "vol_lookback": 200,
            "vol_min_pctile": 0.15,
            "vol_max_pctile": 0.85,
            "use_dynamic_sweep": False,
        }
        val_combos_full.append((params, i))

    val_fn = partial(run_combo, data_dict=val)
    val_results = []
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(val_fn, val_combos_full, chunksize=1):
            if r is not None:
                val_results.append(r)

    # Combine
    print(f"\n{'='*80}")
    print(f"  TRAIN vs VALIDATION")
    print(f"{'='*80}")

    combined = []
    for vr in val_results:
        cid = vr["combo_id"]
        tr = viable[cid]
        combined.append({"train": tr, "val": vr})

    combined.sort(key=lambda x: (
        x["val"]["net_profit"] > 0 and x["train"]["net_profit"] > 0,
        min(x["train"]["net_profit"], x["val"]["net_profit"]),
    ), reverse=True)

    for i, c in enumerate(combined[:10]):
        tr = c["train"]
        vr = c["val"]
        params_str = ", ".join(f"{k}={tr[k]}" for k in show_keys if k in tr)
        status = "PASS" if tr["net_profit"] > 0 and vr["net_profit"] > 0 else "FAIL"
        print(f"\n  #{i+1} [{status}]")
        print(f"  {params_str}")
        print(f"  TRAIN | Trades: {tr['total_trades']:>4d}  WR: {tr['win_rate']:>5.1f}%  PF: {tr['profit_factor']:>5.2f}"
              f"  Net: ${tr['net_profit']:>10,.2f}  DD: {tr['max_drawdown_pct']:.1f}%")
        print(f"  VAL   | Trades: {vr['total_trades']:>4d}  WR: {vr['win_rate']:>5.1f}%  PF: {vr['profit_factor']:>5.2f}"
              f"  Net: ${vr['net_profit']:>10,.2f}  DD: {vr['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    main()
