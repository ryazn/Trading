#!/usr/bin/env python3
"""
PHANTOM Strategy 5-Year Optimizer

Smart parallel grid search with train/validation split.
  Train:  2021-2023  (in-sample)
  Test:   2024-2025  (out-of-sample)

Uses multiprocessing for 16-core parallelism.
"""

import sys
import itertools
import time
import argparse
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

# =====================================================
# PARAMETER GRID
# =====================================================

# Phase 1: Coarse grid (train set)
STRATEGY_GRID = {
    "pivot_left":       [10, 15, 20, 30],
    "pivot_right":      [1, 2, 3],
    "min_strength":     [0.3, 0.5, 0.7],
    "sweep_ticks":      [2, 3, 5],
    "signal_cooldown":  [5, 10, 20],
    "max_level_age":    [100, 200, 390],
    "strength_period":  [30, 50, 100],
}

RISK_GRID = {
    "sl_atr_mult":      [2.0, 2.5, 3.0],
    "tp_rr_ratio":      [1.5, 2.0, 3.0],
    "use_session_filter": [False, True],
    "use_max_hold":     [False, True],
}

# Fixed strategy params
BASE_STRATEGY = {
    "cvd_smooth": 1,
    "cvd_anchor_bars": 390,
    "max_levels": 30,
    "tick_size": 0.25,
    "use_momentum_filter": False,
    "momentum_bars": 5,
    "momentum_tolerance": 0.02,
    "allow_longs": True,
    "allow_shorts": True,
}

INITIAL_CAPITAL = 100000.0


def build_risk_config(risk_overrides: dict) -> RiskConfig:
    """Build a RiskConfig from override dict."""
    use_session = risk_overrides.get("use_session_filter", False)
    use_max_hold = risk_overrides.get("use_max_hold", False)

    return RiskConfig(
        position_mode=PositionSizeMode.RISK_PER_TRADE,
        risk_pct=1.0,
        use_stop_loss=True,
        sl_mode=StopLossMode.ATR,
        sl_atr_mult=risk_overrides.get("sl_atr_mult", 2.0),
        sl_atr_period=14,
        use_take_profit=True,
        tp_mode=TakeProfitMode.RISK_REWARD,
        tp_rr_ratio=risk_overrides.get("tp_rr_ratio", 2.0),
        use_trailing=False,
        use_breakeven=False,
        # Session filter: RTH 9:30-16:00 ET
        use_session_filter=use_session,
        session_start_hour=9,
        session_start_minute=30,
        session_end_hour=16,
        session_end_minute=0,
        close_at_session_end=use_session,
        # Max hold
        use_max_hold=use_max_hold,
        max_hold_bars=120,  # 2 hours for mean-reversion
        # Costs
        commission_per_contract=1.24,
        slippage_ticks=1,
        tick_size=0.25,
        point_value=50.0,
    )


def run_single_combo(combo, data_dict):
    """Run a single parameter combo. Designed for multiprocessing."""
    strat_params, risk_overrides, combo_id = combo

    s_params = {**BASE_STRATEGY, **strat_params}
    risk_config = build_risk_config(risk_overrides)

    try:
        strategy = PhantomCVDStrategy(params=s_params)
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=risk_config)
        report = engine.run(data_dict, strategy, verbose=False)

        return {
            "combo_id": combo_id,
            **strat_params,
            **{k: v for k, v in risk_overrides.items()},
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


def run_validation(combo, data_dict):
    """Run validation on a specific combo."""
    return run_single_combo(combo, data_dict)


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
    if not frames:
        raise ValueError(f"No data found for years {years}")
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    # Deduplicate
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    after = len(combined)
    if before != after:
        print(f"  Deduplicated: {before} -> {after}")
    print(f"  Total bars: {len(combined)}  |  Range: {combined.index[0]} -> {combined.index[-1]}")
    return combined


def build_combos():
    """Build all parameter combinations."""
    strat_keys = list(STRATEGY_GRID.keys())
    strat_combos = list(itertools.product(*[STRATEGY_GRID[k] for k in strat_keys]))

    risk_keys = list(RISK_GRID.keys())
    risk_combos = list(itertools.product(*[RISK_GRID[k] for k in risk_keys]))

    all_combos = []
    combo_id = 0
    for s_vals in strat_combos:
        s_params = {strat_keys[i]: s_vals[i] for i in range(len(strat_keys))}
        for r_vals in risk_combos:
            r_params = {risk_keys[i]: r_vals[i] for i in range(len(risk_keys))}
            all_combos.append((s_params, r_params, combo_id))
            combo_id += 1

    return all_combos


def score_result(r):
    """
    Composite score balancing profitability, consistency, and risk.
    Higher is better.
    """
    if r is None or r["total_trades"] < 20:
        return -999

    pf = r["profit_factor"]
    wr = r["win_rate"] / 100
    sharpe = r["sharpe"]
    dd = r["max_drawdown_pct"]
    trades = r["total_trades"]
    expectancy = r["expectancy"]
    net = r["net_profit"]

    if pf <= 0 or net <= 0:
        return -999

    # Penalize too few trades (not enough data) and too many (overtrading)
    trade_score = min(trades / 50, 1.0)  # ramp up to 50 trades
    if trades > 2000:
        trade_score *= 0.5  # penalize overtrading

    # Core score: profit factor × win rate × Sharpe
    core = (pf * wr * max(sharpe, 0.01))

    # Drawdown penalty
    dd_penalty = max(1.0 - dd / 30, 0.1)

    # Expectancy bonus (normalized)
    exp_bonus = max(expectancy / 100, 0) if expectancy > 0 else 0

    return core * dd_penalty * trade_score + exp_bonus


def print_results(results, param_keys, title, top_n=10):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    for i, r in enumerate(results[:top_n]):
        print(f"\n  --- #{i+1} (Score: {score_result(r):.3f}) ---")
        params_str = ", ".join(f"{k}={r[k]}" for k in param_keys if k in r)
        print(f"  {params_str}")
        print(f"  Trades: {r['total_trades']:>4d}  |  Win Rate: {r['win_rate']:>5.1f}%  |  PF: {r['profit_factor']:>6.2f}")
        print(f"  Net P&L: ${r['net_profit']:>10,.2f}  |  Avg Trade: ${r['avg_trade']:>8,.2f}  |  Max DD: {r['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe: {r['sharpe']:>6.2f}  |  Expectancy: ${r['expectancy']:>8,.2f}")


def main():
    parser = argparse.ArgumentParser(description="PHANTOM 5-Year Optimizer")
    parser.add_argument("--data-dir", type=str, default="data/es_1min",
                        help="Directory with ES_YYYY.csv files")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0 = auto)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: random sample of 500 combos")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else max(cpu_count() - 1, 1)
    print(f"PHANTOM 5-Year Optimizer  |  Workers: {n_workers}")
    print("=" * 60)

    # =====================================================
    # LOAD DATA
    # =====================================================
    print("\n--- Loading TRAIN set (2021-2023) ---")
    train_data = load_data(args.data_dir, [2021, 2022, 2023])

    print("\n--- Loading VALIDATION set (2024-2025) ---")
    val_data = load_data(args.data_dir, [2024, 2025])

    # =====================================================
    # BUILD COMBOS
    # =====================================================
    all_combos = build_combos()
    total = len(all_combos)

    if args.quick:
        import random
        random.seed(42)
        sample_size = min(500, total)
        all_combos = random.sample(all_combos, sample_size)
        # Re-number
        all_combos = [(s, r, i) for i, (s, r, _) in enumerate(all_combos)]
        total = len(all_combos)

    strat_keys = list(STRATEGY_GRID.keys())
    risk_keys = list(RISK_GRID.keys())
    all_keys = strat_keys + risk_keys

    print(f"\nStrategy combos: {len(list(itertools.product(*STRATEGY_GRID.values())))}")
    print(f"Risk combos: {len(list(itertools.product(*RISK_GRID.values())))}")
    print(f"Total combos to test: {total}")
    est_time = total / n_workers * 25 / 60  # ~25s per run on 3yr data
    print(f"Estimated time: ~{est_time:.0f} minutes")

    # =====================================================
    # PHASE 1: TRAIN SET OPTIMIZATION
    # =====================================================
    print(f"\n{'='*60}")
    print("  PHASE 1: Training (2021-2023)")
    print(f"{'='*60}\n")

    start = time.time()
    worker_fn = partial(run_single_combo, data_dict=train_data)

    results = []
    done = 0
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, all_combos, chunksize=4):
            done += 1
            if result is not None:
                results.append(result)
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{total}] {len(results)} viable | "
                      f"{rate:.1f} combos/sec | ETA: {eta:.1f} min")

    train_time = time.time() - start
    print(f"\nPhase 1 done: {len(results)} viable results in {train_time/60:.1f} minutes")

    # Filter viable (minimum trades)
    viable = [r for r in results if r["total_trades"] >= 20 and r["net_profit"] > 0]
    print(f"Profitable with 20+ trades: {len(viable)}")

    if not viable:
        print("\nNo profitable configurations found on training data.")
        print("Showing top 10 by least negative P&L:")
        all_with_trades = [r for r in results if r["total_trades"] >= 10]
        all_with_trades.sort(key=lambda x: x["net_profit"], reverse=True)
        print_results(all_with_trades, all_keys, "BEST OF LOSING CONFIGS (TRAIN)", top_n=10)
        return

    # Rank by composite score
    viable.sort(key=score_result, reverse=True)

    print_results(viable, all_keys, "TOP 10 BY COMPOSITE SCORE (TRAIN)", top_n=10)

    # Also show by other metrics
    by_pf = sorted(viable, key=lambda x: x["profit_factor"], reverse=True)
    print_results(by_pf, all_keys, "TOP 5 BY PROFIT FACTOR (TRAIN)", top_n=5)

    by_net = sorted(viable, key=lambda x: x["net_profit"], reverse=True)
    print_results(by_net, all_keys, "TOP 5 BY NET PROFIT (TRAIN)", top_n=5)

    by_sharpe = sorted(viable, key=lambda x: x["sharpe"], reverse=True)
    print_results(by_sharpe, all_keys, "TOP 5 BY SHARPE (TRAIN)", top_n=5)

    # =====================================================
    # PHASE 2: VALIDATION
    # =====================================================
    top_n = min(30, len(viable))
    top_configs = viable[:top_n]

    print(f"\n{'='*60}")
    print(f"  PHASE 2: Validating top {top_n} on 2024-2025")
    print(f"{'='*60}\n")

    # Rebuild combos for validation
    val_combos = []
    for i, r in enumerate(top_configs):
        s_params = {k: r[k] for k in strat_keys}
        r_params = {k: r[k] for k in risk_keys}
        val_combos.append((s_params, r_params, i))

    start = time.time()
    val_fn = partial(run_single_combo, data_dict=val_data)

    val_results = []
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(val_fn, val_combos, chunksize=1):
            if result is not None:
                val_results.append(result)

    val_time = time.time() - start
    print(f"Phase 2 done in {val_time:.1f} seconds")

    # Match train + validation results
    combined = []
    for vr in val_results:
        cid = vr["combo_id"]
        tr = top_configs[cid]
        combined.append({
            "train": tr,
            "val": vr,
            "train_score": score_result(tr),
            "val_score": score_result(vr),
            "combined_score": score_result(tr) * 0.4 + score_result(vr) * 0.6,
            # Forward test: val must also be profitable
            "val_profitable": vr["net_profit"] > 0,
        })

    # Sort by combined score, prioritizing val profitability
    combined.sort(key=lambda x: (x["val_profitable"], x["combined_score"]), reverse=True)

    print(f"\n{'='*100}")
    print(f"  FINAL RESULTS: TRAIN vs VALIDATION")
    print(f"{'='*100}")

    for i, c in enumerate(combined[:10]):
        tr = c["train"]
        vr = c["val"]
        params_str = ", ".join(f"{k}={tr[k]}" for k in all_keys)
        print(f"\n  --- #{i+1} (Combined Score: {c['combined_score']:.3f}) ---")
        print(f"  {params_str}")
        print(f"  TRAIN  | Trades: {tr['total_trades']:>4d}  WR: {tr['win_rate']:>5.1f}%  PF: {tr['profit_factor']:>5.2f}"
              f"  Net: ${tr['net_profit']:>10,.2f}  DD: {tr['max_drawdown_pct']:.1f}%  Sharpe: {tr['sharpe']:.2f}")
        print(f"  VAL    | Trades: {vr['total_trades']:>4d}  WR: {vr['win_rate']:>5.1f}%  PF: {vr['profit_factor']:>5.2f}"
              f"  Net: ${vr['net_profit']:>10,.2f}  DD: {vr['max_drawdown_pct']:.1f}%  Sharpe: {vr['sharpe']:.2f}")
        status = "PASS" if c["val_profitable"] else "FAIL"
        print(f"  STATUS: {status}")

    # =====================================================
    # EXPORT BEST CONFIG
    # =====================================================
    if combined and combined[0]["val_profitable"]:
        best = combined[0]["train"]
        print(f"\n{'='*60}")
        print("  BEST CONFIG (writing to config/phantom_5yr_optimized.yaml)")
        print(f"{'='*60}")

        config_yaml = f"""# ================================================
# PHANTOM CVD Strategy - 5-YEAR OPTIMIZED
# ================================================
# Optimized via grid search on 2021-2023, validated on 2024-2025
# Train: {combined[0]['train']['total_trades']} trades, PF {combined[0]['train']['profit_factor']:.2f}, WR {combined[0]['train']['win_rate']:.1f}%, Net ${combined[0]['train']['net_profit']:,.2f}
# Val:   {combined[0]['val']['total_trades']} trades, PF {combined[0]['val']['profit_factor']:.2f}, WR {combined[0]['val']['win_rate']:.1f}%, Net ${combined[0]['val']['net_profit']:,.2f}

data:
  source: "csv"
  csv_path: "data/es_1min"

strategy:
  cvd_smooth: 1
  cvd_anchor_bars: 390
  pivot_left: {best['pivot_left']}
  pivot_right: {best['pivot_right']}
  max_level_age: {best['max_level_age']}
  max_levels: 30
  sweep_ticks: {best['sweep_ticks']}
  tick_size: 0.25
  min_strength: {best['min_strength']}
  strength_period: {best['strength_period']}
  use_momentum_filter: false
  momentum_bars: 5
  momentum_tolerance: 0.02
  signal_cooldown: {best['signal_cooldown']}
  allow_longs: true
  allow_shorts: true

risk:
  initial_capital: 100000.0
  position_mode: "risk_per_trade"
  risk_pct: 1.0
  use_stop_loss: true
  sl_mode: "atr"
  sl_atr_mult: {best['sl_atr_mult']}
  sl_atr_period: 14
  use_take_profit: true
  tp_mode: "risk_reward"
  tp_rr_ratio: {best['tp_rr_ratio']}
  use_trailing: false
  use_breakeven: false
  use_session_filter: {str(best['use_session_filter']).lower()}
  session_start_hour: 9
  session_start_minute: 30
  session_end_hour: 16
  session_end_minute: 0
  close_at_session_end: {str(best['use_session_filter']).lower()}
  use_max_drawdown: false
  use_daily_loss_limit: false
  use_max_hold: {str(best['use_max_hold']).lower()}
  max_hold_bars: 120
  commission_per_contract: 1.24
  slippage_ticks: 1
  tick_size: 0.25
  point_value: 50.0
"""
        config_path = Path("config/phantom_5yr_optimized.yaml")
        config_path.write_text(config_yaml)
        print(f"  Written to {config_path}")
    else:
        print("\nNo config passed validation. Strategy may need fundamental changes.")


if __name__ == "__main__":
    main()
