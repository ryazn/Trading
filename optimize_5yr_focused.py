#!/usr/bin/env python3
"""
PHANTOM Strategy 5-Year Focused Optimizer

Narrow search around the promising parameter zone identified in phase 1.
Tests structural variations: swept_level SL, different max_hold values,
trailing stops, breakeven, and tighter grids.
"""

import sys
import itertools
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig,
    StopLossMode,
    TakeProfitMode,
    TrailingMode,
    PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy

INITIAL_CAPITAL = 100000.0


def load_data(csv_dir, years, min_price=1000.0):
    """Load specific years of data."""
    loader = DataLoader()
    import pandas as pd
    frames = []
    for year in years:
        path = Path(csv_dir) / f"ES_{year}.csv"
        if path.exists():
            print(f"  Loading {path.name}...")
            df = loader.load_csv(str(path))
            if min_price > 0:
                mask = df["close"] >= min_price
                removed = (~mask).sum()
                df = df[mask]
                if removed > 0:
                    print(f"    Price filter: removed {removed} bars")
            frames.append(df)
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    after = len(combined)
    if before != after:
        print(f"  Deduplicated: {before} -> {after}")
    print(f"  Total bars: {len(combined)}  |  Range: {combined.index[0]} -> {combined.index[-1]}")
    return combined


def run_single(combo, data_dict):
    """Run a single parameter combo."""
    s_params, risk_config, combo_id, label = combo

    base_strat = {
        "cvd_smooth": 1,
        "cvd_anchor_bars": 390,
        "max_levels": 30,
        "tick_size": 0.25,
        "allow_longs": True,
        "allow_shorts": True,
    }
    full_params = {**base_strat, **s_params}

    try:
        strategy = PhantomCVDStrategy(params=full_params)
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=risk_config)
        report = engine.run(data_dict, strategy, verbose=False)

        return {
            "combo_id": combo_id,
            "label": label,
            **s_params,
            "sl_mode": str(risk_config.sl_mode.value),
            "sl_atr_mult": risk_config.sl_atr_mult,
            "tp_rr_ratio": risk_config.tp_rr_ratio,
            "use_session": risk_config.use_session_filter,
            "use_max_hold": risk_config.use_max_hold,
            "max_hold_bars": risk_config.max_hold_bars,
            "use_trailing": risk_config.use_trailing,
            "trail_atr_mult": risk_config.trail_atr_mult,
            "use_breakeven": risk_config.use_breakeven,
            "use_momentum": s_params.get("use_momentum_filter", False),
            "net_profit": report.net_profit,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe": report.sharpe_ratio,
            "expectancy": report.expectancy,
            "avg_trade": report.avg_trade,
        }
    except Exception as e:
        return None


def build_combos():
    """Build focused parameter combinations."""
    combos = []
    combo_id = 0

    # =====================================================
    # GROUP A: Fine-tune around best zone (ATR stop)
    # Best zone: pivot_left=10, pivot_right=2, strength_period=100
    # =====================================================
    for pivot_left in [8, 10, 12, 15]:
        for pivot_right in [1, 2, 3]:
            for min_strength in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                for sweep_ticks in [1, 2, 3]:
                    for signal_cooldown in [3, 5, 10, 15]:
                        for max_level_age in [80, 100, 150]:
                            for strength_period in [50, 80, 100]:
                                s_params = {
                                    "pivot_left": pivot_left,
                                    "pivot_right": pivot_right,
                                    "min_strength": min_strength,
                                    "sweep_ticks": sweep_ticks,
                                    "signal_cooldown": signal_cooldown,
                                    "max_level_age": max_level_age,
                                    "strength_period": strength_period,
                                    "use_momentum_filter": False,
                                }

                                for sl_mult in [1.5, 2.0, 2.5]:
                                    for tp_rr in [1.5, 2.0, 2.5, 3.0]:
                                        for use_hold in [False, True]:
                                            risk = RiskConfig(
                                                position_mode=PositionSizeMode.RISK_PER_TRADE,
                                                risk_pct=1.0,
                                                use_stop_loss=True,
                                                sl_mode=StopLossMode.ATR,
                                                sl_atr_mult=sl_mult,
                                                sl_atr_period=14,
                                                use_take_profit=True,
                                                tp_mode=TakeProfitMode.RISK_REWARD,
                                                tp_rr_ratio=tp_rr,
                                                use_trailing=False,
                                                use_breakeven=False,
                                                use_session_filter=True,
                                                session_start_hour=9,
                                                session_start_minute=30,
                                                session_end_hour=16,
                                                session_end_minute=0,
                                                close_at_session_end=True,
                                                use_max_hold=use_hold,
                                                max_hold_bars=120,
                                                commission_per_contract=1.24,
                                                slippage_ticks=1,
                                                tick_size=0.25,
                                                point_value=50.0,
                                            )
                                            combos.append((s_params, risk, combo_id, "A_finetune"))
                                            combo_id += 1

    # That's way too many - let's use random sampling
    import random
    random.seed(42)

    # Keep first 800 combos randomly
    if len(combos) > 800:
        combos = random.sample(combos, 800)
        combo_id = 800

    # =====================================================
    # GROUP B: Swept-level stop loss (uses the actual swept level)
    # =====================================================
    for pivot_left in [10, 15, 20]:
        for pivot_right in [1, 2]:
            for min_strength in [0.3, 0.5, 0.7]:
                for sweep_ticks in [2, 3]:
                    for signal_cooldown in [5, 10]:
                        for max_level_age in [100, 200]:
                            s_params = {
                                "pivot_left": pivot_left,
                                "pivot_right": pivot_right,
                                "min_strength": min_strength,
                                "sweep_ticks": sweep_ticks,
                                "signal_cooldown": signal_cooldown,
                                "max_level_age": max_level_age,
                                "strength_period": 100,
                                "use_momentum_filter": False,
                            }

                            for offset_ticks in [2, 3, 5]:
                                for tp_rr in [1.5, 2.0, 3.0]:
                                    risk = RiskConfig(
                                        position_mode=PositionSizeMode.RISK_PER_TRADE,
                                        risk_pct=1.0,
                                        use_stop_loss=True,
                                        sl_mode=StopLossMode.SWEPT_LEVEL,
                                        sl_swept_offset_ticks=offset_ticks,
                                        sl_atr_period=14,
                                        use_take_profit=True,
                                        tp_mode=TakeProfitMode.RISK_REWARD,
                                        tp_rr_ratio=tp_rr,
                                        use_trailing=False,
                                        use_breakeven=False,
                                        use_session_filter=True,
                                        session_start_hour=9,
                                        session_start_minute=30,
                                        session_end_hour=16,
                                        session_end_minute=0,
                                        close_at_session_end=True,
                                        use_max_hold=True,
                                        max_hold_bars=120,
                                        commission_per_contract=1.24,
                                        slippage_ticks=1,
                                        tick_size=0.25,
                                        point_value=50.0,
                                    )
                                    combos.append((s_params, risk, combo_id, "B_swept_sl"))
                                    combo_id += 1

    # =====================================================
    # GROUP C: Trailing stop (no fixed TP)
    # =====================================================
    for pivot_left in [10, 15]:
        for pivot_right in [1, 2]:
            for min_strength in [0.3, 0.5]:
                for sweep_ticks in [2, 3]:
                    for signal_cooldown in [5, 10]:
                        s_params = {
                            "pivot_left": pivot_left,
                            "pivot_right": pivot_right,
                            "min_strength": min_strength,
                            "sweep_ticks": sweep_ticks,
                            "signal_cooldown": signal_cooldown,
                            "max_level_age": 100,
                            "strength_period": 100,
                            "use_momentum_filter": False,
                        }

                        for sl_mult in [2.0, 2.5]:
                            for trail_mult in [1.0, 1.5, 2.0]:
                                risk = RiskConfig(
                                    position_mode=PositionSizeMode.RISK_PER_TRADE,
                                    risk_pct=1.0,
                                    use_stop_loss=True,
                                    sl_mode=StopLossMode.ATR,
                                    sl_atr_mult=sl_mult,
                                    sl_atr_period=14,
                                    use_take_profit=False,
                                    use_trailing=True,
                                    trail_mode=TrailingMode.ATR,
                                    trail_atr_mult=trail_mult,
                                    use_breakeven=False,
                                    use_session_filter=True,
                                    session_start_hour=9,
                                    session_start_minute=30,
                                    session_end_hour=16,
                                    session_end_minute=0,
                                    close_at_session_end=True,
                                    use_max_hold=True,
                                    max_hold_bars=120,
                                    commission_per_contract=1.24,
                                    slippage_ticks=1,
                                    tick_size=0.25,
                                    point_value=50.0,
                                )
                                combos.append((s_params, risk, combo_id, "C_trail"))
                                combo_id += 1

    # =====================================================
    # GROUP D: Momentum filter ON
    # =====================================================
    for pivot_left in [10, 15]:
        for pivot_right in [1, 2]:
            for min_strength in [0.3, 0.5]:
                for sweep_ticks in [2, 3]:
                    for signal_cooldown in [5, 10]:
                        s_params = {
                            "pivot_left": pivot_left,
                            "pivot_right": pivot_right,
                            "min_strength": min_strength,
                            "sweep_ticks": sweep_ticks,
                            "signal_cooldown": signal_cooldown,
                            "max_level_age": 100,
                            "strength_period": 100,
                            "use_momentum_filter": True,
                            "momentum_bars": 5,
                            "momentum_tolerance": 0.02,
                        }

                        for sl_mult in [2.0, 2.5]:
                            for tp_rr in [1.5, 2.0, 3.0]:
                                risk = RiskConfig(
                                    position_mode=PositionSizeMode.RISK_PER_TRADE,
                                    risk_pct=1.0,
                                    use_stop_loss=True,
                                    sl_mode=StopLossMode.ATR,
                                    sl_atr_mult=sl_mult,
                                    sl_atr_period=14,
                                    use_take_profit=True,
                                    tp_mode=TakeProfitMode.RISK_REWARD,
                                    tp_rr_ratio=tp_rr,
                                    use_trailing=False,
                                    use_breakeven=False,
                                    use_session_filter=True,
                                    session_start_hour=9,
                                    session_start_minute=30,
                                    session_end_hour=16,
                                    session_end_minute=0,
                                    close_at_session_end=True,
                                    use_max_hold=True,
                                    max_hold_bars=120,
                                    commission_per_contract=1.24,
                                    slippage_ticks=1,
                                    tick_size=0.25,
                                    point_value=50.0,
                                )
                                combos.append((s_params, risk, combo_id, "D_momentum"))
                                combo_id += 1

    # =====================================================
    # GROUP E: Breakeven + TP (hybrid exit)
    # =====================================================
    for pivot_left in [10, 15]:
        for pivot_right in [1, 2]:
            for min_strength in [0.3, 0.5]:
                for sweep_ticks in [2, 3]:
                    for signal_cooldown in [5, 10]:
                        s_params = {
                            "pivot_left": pivot_left,
                            "pivot_right": pivot_right,
                            "min_strength": min_strength,
                            "sweep_ticks": sweep_ticks,
                            "signal_cooldown": signal_cooldown,
                            "max_level_age": 100,
                            "strength_period": 100,
                            "use_momentum_filter": False,
                        }

                        for sl_mult in [2.0, 2.5]:
                            for tp_rr in [2.0, 3.0]:
                                for be_trigger in [8, 12, 20]:
                                    risk = RiskConfig(
                                        position_mode=PositionSizeMode.RISK_PER_TRADE,
                                        risk_pct=1.0,
                                        use_stop_loss=True,
                                        sl_mode=StopLossMode.ATR,
                                        sl_atr_mult=sl_mult,
                                        sl_atr_period=14,
                                        use_take_profit=True,
                                        tp_mode=TakeProfitMode.RISK_REWARD,
                                        tp_rr_ratio=tp_rr,
                                        use_trailing=False,
                                        use_breakeven=True,
                                        be_trigger_points=be_trigger,
                                        be_offset=1.0,
                                        use_session_filter=True,
                                        session_start_hour=9,
                                        session_start_minute=30,
                                        session_end_hour=16,
                                        session_end_minute=0,
                                        close_at_session_end=True,
                                        use_max_hold=True,
                                        max_hold_bars=120,
                                        commission_per_contract=1.24,
                                        slippage_ticks=1,
                                        tick_size=0.25,
                                        point_value=50.0,
                                    )
                                    combos.append((s_params, risk, combo_id, "E_breakeven"))
                                    combo_id += 1

    # =====================================================
    # GROUP F: Fixed-contracts mode (removes position sizing noise)
    # =====================================================
    for pivot_left in [10, 15]:
        for pivot_right in [1, 2]:
            for min_strength in [0.3, 0.5, 0.7]:
                for sweep_ticks in [2, 3]:
                    for signal_cooldown in [5, 10]:
                        s_params = {
                            "pivot_left": pivot_left,
                            "pivot_right": pivot_right,
                            "min_strength": min_strength,
                            "sweep_ticks": sweep_ticks,
                            "signal_cooldown": signal_cooldown,
                            "max_level_age": 100,
                            "strength_period": 100,
                            "use_momentum_filter": False,
                        }

                        for sl_mult in [2.0, 2.5]:
                            for tp_rr in [1.5, 2.0, 3.0]:
                                risk = RiskConfig(
                                    position_mode=PositionSizeMode.FIXED_CONTRACTS,
                                    fixed_contracts=1,
                                    use_stop_loss=True,
                                    sl_mode=StopLossMode.ATR,
                                    sl_atr_mult=sl_mult,
                                    sl_atr_period=14,
                                    use_take_profit=True,
                                    tp_mode=TakeProfitMode.RISK_REWARD,
                                    tp_rr_ratio=tp_rr,
                                    use_trailing=False,
                                    use_breakeven=False,
                                    use_session_filter=True,
                                    session_start_hour=9,
                                    session_start_minute=30,
                                    session_end_hour=16,
                                    session_end_minute=0,
                                    close_at_session_end=True,
                                    use_max_hold=True,
                                    max_hold_bars=120,
                                    commission_per_contract=1.24,
                                    slippage_ticks=1,
                                    tick_size=0.25,
                                    point_value=50.0,
                                )
                                combos.append((s_params, risk, combo_id, "F_fixed_1ct"))
                                combo_id += 1

    return combos


def main():
    n_workers = max(cpu_count() - 1, 1)
    print(f"PHANTOM 5-Year FOCUSED Optimizer  |  Workers: {n_workers}")
    print("=" * 60)

    # Load data
    print("\n--- Loading TRAIN set (2021-2023) ---")
    train_data = load_data("data/es_1min", [2021, 2022, 2023])

    print("\n--- Loading VALIDATION set (2024-2025) ---")
    val_data = load_data("data/es_1min", [2024, 2025])

    # Build combos
    combos = build_combos()
    total = len(combos)

    # Count by group
    groups = {}
    for _, _, _, label in combos:
        groups[label] = groups.get(label, 0) + 1
    for g, n in groups.items():
        print(f"  {g}: {n} combos")
    print(f"  TOTAL: {total} combos")
    est_time = total / n_workers * 25 / 60
    print(f"  Estimated time: ~{est_time:.0f} minutes")

    # =====================================================
    # PHASE 1: TRAIN
    # =====================================================
    print(f"\n{'='*60}")
    print("  PHASE 1: Training (2021-2023)")
    print(f"{'='*60}\n")

    start = time.time()
    worker_fn = partial(run_single, data_dict=train_data)

    results = []
    done = 0
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, combos, chunksize=4):
            done += 1
            if result is not None:
                results.append(result)
            if done % 200 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                n_profitable = len([r for r in results if r["net_profit"] > 0 and r["total_trades"] >= 20])
                print(f"  [{done}/{total}] {n_profitable} profitable | "
                      f"{rate:.1f}/sec | ETA: {eta:.1f} min")

    train_time = time.time() - start
    print(f"\nPhase 1 done in {train_time/60:.1f} minutes")

    # Filter
    viable = [r for r in results if r["total_trades"] >= 20]
    profitable = [r for r in viable if r["net_profit"] > 0]
    print(f"Total viable (20+ trades): {len(viable)}")
    print(f"Profitable: {len(profitable)}")

    # Show top by group
    for group_name in sorted(groups.keys()):
        group_results = sorted(
            [r for r in viable if r["label"] == group_name],
            key=lambda x: x["net_profit"], reverse=True
        )
        if group_results:
            best = group_results[0]
            print(f"\n  Best {group_name}: "
                  f"Trades={best['total_trades']}, WR={best['win_rate']:.1f}%, "
                  f"PF={best['profit_factor']:.2f}, Net=${best['net_profit']:,.2f}, "
                  f"DD={best['max_drawdown_pct']:.1f}%")

    # Sort all viable by net profit
    viable.sort(key=lambda x: x["net_profit"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  TOP 15 BY NET PROFIT (TRAIN 2021-2023)")
    print(f"{'='*100}")
    for i, r in enumerate(viable[:15]):
        keys_to_show = ["pivot_left", "pivot_right", "min_strength", "sweep_ticks",
                        "signal_cooldown", "max_level_age", "sl_mode", "sl_atr_mult",
                        "tp_rr_ratio", "use_session", "use_max_hold", "use_trailing",
                        "use_breakeven", "use_momentum"]
        params_str = ", ".join(f"{k}={r[k]}" for k in keys_to_show if k in r)
        print(f"\n  #{i+1} [{r['label']}]")
        print(f"  {params_str}")
        print(f"  Trades: {r['total_trades']:>4d}  WR: {r['win_rate']:>5.1f}%  PF: {r['profit_factor']:>5.2f}"
              f"  Net: ${r['net_profit']:>10,.2f}  DD: {r['max_drawdown_pct']:.1f}%  Sharpe: {r['sharpe']:.2f}")

    # =====================================================
    # PHASE 2: VALIDATE TOP 30
    # =====================================================
    top_n = min(30, len(viable))
    top_train = viable[:top_n]

    print(f"\n{'='*60}")
    print(f"  PHASE 2: Validating top {top_n} on 2024-2025")
    print(f"{'='*60}\n")

    val_combos = []
    for i, r in enumerate(top_train):
        s_params = {
            "pivot_left": r["pivot_left"],
            "pivot_right": r["pivot_right"],
            "min_strength": r["min_strength"],
            "sweep_ticks": r["sweep_ticks"],
            "signal_cooldown": r["signal_cooldown"],
            "max_level_age": r["max_level_age"],
            "strength_period": r.get("strength_period", 100),
            "use_momentum_filter": r.get("use_momentum", False),
            "momentum_bars": 5,
            "momentum_tolerance": 0.02,
        }
        # Rebuild risk config from the result
        sl_mode = StopLossMode.ATR
        if r.get("sl_mode") == "swept_level":
            sl_mode = StopLossMode.SWEPT_LEVEL

        pos_mode = PositionSizeMode.RISK_PER_TRADE
        if r.get("label") == "F_fixed_1ct":
            pos_mode = PositionSizeMode.FIXED_CONTRACTS

        risk = RiskConfig(
            position_mode=pos_mode,
            risk_pct=1.0,
            fixed_contracts=1,
            use_stop_loss=True,
            sl_mode=sl_mode,
            sl_atr_mult=r.get("sl_atr_mult", 2.0),
            sl_atr_period=14,
            sl_swept_offset_ticks=3,
            use_take_profit=r.get("tp_rr_ratio", 0) > 0,
            tp_mode=TakeProfitMode.RISK_REWARD,
            tp_rr_ratio=r.get("tp_rr_ratio", 2.0),
            use_trailing=r.get("use_trailing", False),
            trail_mode=TrailingMode.ATR,
            trail_atr_mult=r.get("trail_atr_mult", 1.5),
            use_breakeven=r.get("use_breakeven", False),
            be_trigger_points=r.get("be_trigger_points", 15),
            be_offset=1.0,
            use_session_filter=True,
            session_start_hour=9,
            session_start_minute=30,
            session_end_hour=16,
            session_end_minute=0,
            close_at_session_end=True,
            use_max_hold=r.get("use_max_hold", False),
            max_hold_bars=r.get("max_hold_bars", 120),
            commission_per_contract=1.24,
            slippage_ticks=1,
            tick_size=0.25,
            point_value=50.0,
        )
        val_combos.append((s_params, risk, i, r["label"]))

    start = time.time()
    val_fn = partial(run_single, data_dict=val_data)

    val_results = []
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(val_fn, val_combos, chunksize=1):
            if result is not None:
                val_results.append(result)

    val_time = time.time() - start
    print(f"Phase 2 done in {val_time:.1f} seconds")

    # Combine
    print(f"\n{'='*100}")
    print(f"  FINAL: TRAIN vs VALIDATION")
    print(f"{'='*100}")

    combined = []
    for vr in val_results:
        cid = vr["combo_id"]
        tr = top_train[cid]
        combined.append({
            "train": tr,
            "val": vr,
            "val_profitable": vr["net_profit"] > 0,
            "both_profitable": tr["net_profit"] > 0 and vr["net_profit"] > 0,
        })

    combined.sort(key=lambda x: (
        x["both_profitable"],
        min(x["train"]["profit_factor"], x["val"]["profit_factor"]),
    ), reverse=True)

    for i, c in enumerate(combined[:15]):
        tr = c["train"]
        vr = c["val"]
        keys_to_show = ["pivot_left", "pivot_right", "min_strength", "sweep_ticks",
                        "signal_cooldown", "sl_mode", "sl_atr_mult", "tp_rr_ratio",
                        "use_trailing", "use_breakeven", "use_momentum"]
        params_str = ", ".join(f"{k}={tr[k]}" for k in keys_to_show if k in tr)
        status = "PASS" if c["both_profitable"] else ("PARTIAL" if c["val_profitable"] or tr["net_profit"] > 0 else "FAIL")
        print(f"\n  #{i+1} [{tr['label']}] {status}")
        print(f"  {params_str}")
        print(f"  TRAIN | Trades: {tr['total_trades']:>4d}  WR: {tr['win_rate']:>5.1f}%  PF: {tr['profit_factor']:>5.2f}"
              f"  Net: ${tr['net_profit']:>10,.2f}  DD: {tr['max_drawdown_pct']:.1f}%")
        print(f"  VAL   | Trades: {vr['total_trades']:>4d}  WR: {vr['win_rate']:>5.1f}%  PF: {vr['profit_factor']:>5.2f}"
              f"  Net: ${vr['net_profit']:>10,.2f}  DD: {vr['max_drawdown_pct']:.1f}%")

    # Export best
    passed = [c for c in combined if c["both_profitable"]]
    if passed:
        best = passed[0]
        tr = best["train"]
        vr = best["val"]
        print(f"\n{'='*60}")
        print("  BEST CONFIG -> config/phantom_5yr_optimized.yaml")
        print(f"{'='*60}")

        sl_mode_str = tr.get("sl_mode", "atr")
        use_trail = tr.get("use_trailing", False)
        use_be = tr.get("use_breakeven", False)
        use_mom = tr.get("use_momentum", False)
        use_hold = tr.get("use_max_hold", False)
        is_fixed = tr.get("label") == "F_fixed_1ct"

        config_yaml = f"""# ================================================
# PHANTOM CVD Strategy - 5-YEAR OPTIMIZED
# ================================================
# Grid search: train 2021-2023, validated 2024-2025
# Train: {tr['total_trades']} trades, PF {tr['profit_factor']:.2f}, WR {tr['win_rate']:.1f}%, Net ${tr['net_profit']:,.2f}
# Val:   {vr['total_trades']} trades, PF {vr['profit_factor']:.2f}, WR {vr['win_rate']:.1f}%, Net ${vr['net_profit']:,.2f}
# Group: {tr['label']}

data:
  source: "csv"
  csv_path: "data/es_1min"

strategy:
  cvd_smooth: 1
  cvd_anchor_bars: 390
  pivot_left: {tr['pivot_left']}
  pivot_right: {tr['pivot_right']}
  max_level_age: {tr['max_level_age']}
  max_levels: 30
  sweep_ticks: {tr['sweep_ticks']}
  tick_size: 0.25
  min_strength: {tr['min_strength']}
  strength_period: {tr.get('strength_period', 100)}
  use_momentum_filter: {str(use_mom).lower()}
  momentum_bars: 5
  momentum_tolerance: 0.02
  signal_cooldown: {tr['signal_cooldown']}
  allow_longs: true
  allow_shorts: true

risk:
  initial_capital: 100000.0
  position_mode: "{'fixed_contracts' if is_fixed else 'risk_per_trade'}"
  {'fixed_contracts: 1' if is_fixed else 'risk_pct: 1.0'}
  use_stop_loss: true
  sl_mode: "{sl_mode_str}"
  sl_atr_mult: {tr.get('sl_atr_mult', 2.0)}
  sl_atr_period: 14
  sl_swept_offset_ticks: 3
  use_take_profit: {str(tr.get('tp_rr_ratio', 0) > 0).lower()}
  tp_mode: "risk_reward"
  tp_rr_ratio: {tr.get('tp_rr_ratio', 2.0)}
  use_trailing: {str(use_trail).lower()}
  trail_mode: "atr"
  trail_atr_mult: {tr.get('trail_atr_mult', 1.5)}
  use_breakeven: {str(use_be).lower()}
  be_trigger_points: 15.0
  be_offset: 1.0
  use_session_filter: true
  session_start_hour: 9
  session_start_minute: 30
  session_end_hour: 16
  session_end_minute: 0
  close_at_session_end: true
  use_max_drawdown: false
  use_daily_loss_limit: false
  use_max_hold: {str(use_hold).lower()}
  max_hold_bars: 120
  commission_per_contract: 1.24
  slippage_ticks: 1
  tick_size: 0.25
  point_value: 50.0
"""
        Path("config/phantom_5yr_optimized.yaml").write_text(config_yaml)
        print("  Written!")
    else:
        print("\n  NO CONFIG PASSED BOTH TRAIN + VALIDATION")
        print("  Strategy needs fundamental improvements (see quant analysis)")

        # Still export the least-bad config for analysis
        if combined:
            best = combined[0]
            tr = best["train"]
            vr = best["val"]
            sl_mode_str = tr.get("sl_mode", "atr")
            is_fixed = tr.get("label") == "F_fixed_1ct"
            config_yaml = f"""# ================================================
# PHANTOM CVD Strategy - 5-YEAR BEST ATTEMPT
# ================================================
# NOTE: This config was NOT profitable on both train and validation.
# The strategy needs fundamental improvements.
# Train: {tr['total_trades']} trades, PF {tr['profit_factor']:.2f}, WR {tr['win_rate']:.1f}%, Net ${tr['net_profit']:,.2f}
# Val:   {vr['total_trades']} trades, PF {vr['profit_factor']:.2f}, WR {vr['win_rate']:.1f}%, Net ${vr['net_profit']:,.2f}

data:
  source: "csv"
  csv_path: "data/es_1min"

strategy:
  cvd_smooth: 1
  cvd_anchor_bars: 390
  pivot_left: {tr['pivot_left']}
  pivot_right: {tr['pivot_right']}
  max_level_age: {tr['max_level_age']}
  max_levels: 30
  sweep_ticks: {tr['sweep_ticks']}
  tick_size: 0.25
  min_strength: {tr['min_strength']}
  strength_period: {tr.get('strength_period', 100)}
  use_momentum_filter: false
  signal_cooldown: {tr['signal_cooldown']}
  allow_longs: true
  allow_shorts: true

risk:
  initial_capital: 100000.0
  position_mode: "{'fixed_contracts' if is_fixed else 'risk_per_trade'}"
  {'fixed_contracts: 1' if is_fixed else 'risk_pct: 1.0'}
  use_stop_loss: true
  sl_mode: "{sl_mode_str}"
  sl_atr_mult: {tr.get('sl_atr_mult', 2.0)}
  sl_atr_period: 14
  use_take_profit: true
  tp_mode: "risk_reward"
  tp_rr_ratio: {tr.get('tp_rr_ratio', 2.0)}
  use_trailing: false
  use_breakeven: false
  use_session_filter: true
  session_start_hour: 9
  session_start_minute: 30
  session_end_hour: 16
  session_end_minute: 0
  close_at_session_end: true
  use_max_hold: {str(tr.get('use_max_hold', False)).lower()}
  max_hold_bars: 120
  commission_per_contract: 1.24
  slippage_ticks: 1
  tick_size: 0.25
  point_value: 50.0
"""
            Path("config/phantom_5yr_optimized.yaml").write_text(config_yaml)
            print("  Wrote best-attempt config for analysis.")


if __name__ == "__main__":
    main()
