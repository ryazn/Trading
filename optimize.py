#!/usr/bin/env python3
"""
PHANTOM Strategy Optimizer

Grid search across strategy and risk parameters to find the best
configuration on historical data.

Usage:
    python optimize.py --csv "CME_MINI_ES1!, 1_6378e.csv"
    python optimize.py --csv-dir data/es_1min
    python optimize.py --sample 10000
"""

import sys
import itertools
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig,
    StopLossMode,
    TakeProfitMode,
    PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy


def run_single(data, strategy_params, risk_config, initial_capital):
    """Run a single backtest and return key metrics."""
    strategy = PhantomCVDStrategy(params=strategy_params)
    engine = BacktestEngine(initial_capital=initial_capital, risk_config=risk_config)
    report = engine.run(data, strategy, verbose=False)
    return {
        "net_profit": report.net_profit,
        "total_trades": report.total_trades,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "max_drawdown_pct": report.max_drawdown_pct,
        "sharpe": report.sharpe_ratio,
        "expectancy": report.expectancy,
        "avg_trade": report.avg_trade,
    }


def main():
    parser = argparse.ArgumentParser(description="PHANTOM Strategy Optimizer")
    parser.add_argument("--csv", type=str, help="Path to CSV data file")
    parser.add_argument("--csv-dir", type=str, help="Directory of yearly CSV files (e.g. data/es_1min)")
    parser.add_argument("--sample", type=int, default=0, help="Generate N bars of sample data")
    args = parser.parse_args()

    loader = DataLoader()

    if args.csv_dir:
        print(f"Loading CSV files from {args.csv_dir}...")
        data = loader.load_directory(args.csv_dir)
    elif args.csv:
        print(f"Loading {args.csv}...")
        data = loader.load_csv(args.csv)
    else:
        bars = args.sample if args.sample > 0 else 5000
        print(f"Generating {bars} bars of sample data...")
        data = loader.generate_sample_data(bars=bars)

    stats = loader.get_stats()
    print(f"  Bars: {stats['bars']}  |  Price: {stats['price_range']}")

    initial_capital = 100000.0

    # =====================================================
    # PARAMETER GRID - Tune these ranges for your market
    # =====================================================

    # Strategy parameters to sweep
    strategy_grid = {
        "pivot_left":       [5, 10, 15, 20],
        "pivot_right":      [1, 2, 3],
        "min_strength":     [0.1, 0.3, 0.5],
        "sweep_ticks":      [1, 2, 3],
        "signal_cooldown":  [3, 5, 10],
        "max_level_age":    [100, 200, 300],
        "strength_period":  [30, 50, 80],
    }

    # Risk parameters to sweep
    risk_grid = {
        "sl_atr_mult":  [1.5, 2.0, 2.5, 3.0],
        "tp_rr_ratio":  [1.5, 2.0, 2.5, 3.0],
    }

    # Fixed strategy params
    base_strategy = {
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

    # Build all strategy param combos
    strat_keys = list(strategy_grid.keys())
    strat_combos = list(itertools.product(*[strategy_grid[k] for k in strat_keys]))

    risk_keys = list(risk_grid.keys())
    risk_combos = list(itertools.product(*[risk_grid[k] for k in risk_keys]))

    total_combos = len(strat_combos) * len(risk_combos)
    print(f"\nOptimizing {len(strat_combos)} strategy combos x {len(risk_combos)} risk combos = {total_combos} total")
    print("This may take a while...\n")

    results = []
    best_pf = 0.0
    best_result = None
    tested = 0

    for s_vals in strat_combos:
        s_params = {**base_strategy}
        for i, k in enumerate(strat_keys):
            s_params[k] = s_vals[i]

        for r_vals in risk_combos:
            risk_overrides = {}
            for i, k in enumerate(risk_keys):
                risk_overrides[k] = r_vals[i]

            risk_config = RiskConfig(
                position_mode=PositionSizeMode.RISK_PER_TRADE,
                risk_pct=1.0,
                use_stop_loss=True,
                sl_mode=StopLossMode.ATR,
                sl_atr_mult=risk_overrides.get("sl_atr_mult", 2.0),
                sl_atr_period=14,
                use_take_profit=True,
                tp_mode=TakeProfitMode.RISK_REWARD,
                tp_rr_ratio=risk_overrides.get("tp_rr_ratio", 2.0),
                commission_per_contract=1.24,
                slippage_ticks=1,
                tick_size=0.25,
                point_value=50.0,
            )

            try:
                metrics = run_single(data, s_params, risk_config, initial_capital)
            except Exception:
                continue

            tested += 1

            entry = {
                **{k: s_vals[i] for i, k in enumerate(strat_keys)},
                **{k: r_vals[i] for i, k in enumerate(risk_keys)},
                **metrics,
            }
            results.append(entry)

            # Track best by profit factor (with minimum trade count)
            if metrics["total_trades"] >= 3 and metrics["profit_factor"] > best_pf:
                best_pf = metrics["profit_factor"]
                best_result = entry

            if tested % 500 == 0:
                print(f"  Tested {tested}/{total_combos}...")

    print(f"\nTested {tested} parameter combinations\n")

    # =====================================================
    # RESULTS
    # =====================================================

    # Filter to configs with at least 3 trades
    viable = [r for r in results if r["total_trades"] >= 3]
    print(f"Viable configurations (3+ trades): {len(viable)}")

    if not viable:
        print("\nNo viable configurations found. Try wider parameter ranges or more data.")
        return

    # Sort by different criteria
    print("\n" + "=" * 80)
    print("  TOP 5 BY PROFIT FACTOR (minimum 3 trades)")
    print("=" * 80)
    by_pf = sorted(viable, key=lambda x: x["profit_factor"], reverse=True)[:5]
    _print_results(by_pf, strat_keys + risk_keys)

    print("\n" + "=" * 80)
    print("  TOP 5 BY NET PROFIT")
    print("=" * 80)
    by_profit = sorted(viable, key=lambda x: x["net_profit"], reverse=True)[:5]
    _print_results(by_profit, strat_keys + risk_keys)

    print("\n" + "=" * 80)
    print("  TOP 5 BY SHARPE RATIO")
    print("=" * 80)
    by_sharpe = sorted(viable, key=lambda x: x["sharpe"], reverse=True)[:5]
    _print_results(by_sharpe, strat_keys + risk_keys)

    print("\n" + "=" * 80)
    print("  TOP 5 BY WIN RATE (minimum 3 trades)")
    print("=" * 80)
    by_wr = sorted(viable, key=lambda x: x["win_rate"], reverse=True)[:5]
    _print_results(by_wr, strat_keys + risk_keys)

    # Best overall
    if best_result:
        print("\n" + "=" * 80)
        print("  BEST OVERALL (highest profit factor with 3+ trades)")
        print("=" * 80)
        _print_results([best_result], strat_keys + risk_keys)


def _print_results(results, param_keys):
    for i, r in enumerate(results):
        print(f"\n  --- #{i+1} ---")
        print(f"  Params: ", end="")
        params_str = ", ".join(f"{k}={r[k]}" for k in param_keys)
        print(params_str)
        print(f"  Trades: {r['total_trades']:>4d}  |  Win Rate: {r['win_rate']:>5.1f}%  |  PF: {r['profit_factor']:>6.2f}")
        print(f"  Net P&L: ${r['net_profit']:>10,.2f}  |  Avg Trade: ${r['avg_trade']:>8,.2f}  |  Max DD: {r['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe: {r['sharpe']:>6.2f}  |  Expectancy: ${r['expectancy']:>8,.2f}")


if __name__ == "__main__":
    main()
