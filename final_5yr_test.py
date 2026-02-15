#!/usr/bin/env python3
"""Final 5-year test of top configs from sensitivity analysis."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig, StopLossMode, TakeProfitMode, PositionSizeMode,
)
from backtester.visualization import BacktestChart
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


def load_5yr():
    loader = DataLoader()
    frames = []
    for year in [2021, 2022, 2023, 2024, 2025]:
        df = loader.load_csv(f"data/es_1min/ES_{year}.csv")
        df = df[df["close"] >= 1000]
        frames.append(df)
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def run_and_report(name, params, data, risk=RISK, chart_suffix=None):
    strategy = PhantomCVDStrategy(params=params)
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=risk)
    start = time.time()
    report = engine.run(data, strategy, verbose=False)
    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"  {name}  ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Net P&L:     ${report.net_profit:>10,.2f}  ({report.net_profit/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"  Trades:      {report.total_trades:>6d}  (Long: {report.long_trades}, Short: {report.short_trades})")
    print(f"  Win Rate:    {report.win_rate:>6.1f}%")
    print(f"  Profit Factor: {report.profit_factor:>6.2f}")
    print(f"  Sharpe:      {report.sharpe_ratio:>6.2f}")
    print(f"  Max DD:      {report.max_drawdown_pct:>6.2f}%")
    print(f"  Avg Trade:   ${report.avg_trade:>8.2f}")
    print(f"  Avg Win:     ${report.avg_win:>8.2f}")
    print(f"  Avg Loss:    ${report.avg_loss:>8.2f}")
    print(f"  Expectancy:  ${report.expectancy:>8.2f}")

    if report.total_trades > 0:
        sl = sum(1 for t in report.trades if t.exit_reason == "Stop Loss")
        tp = sum(1 for t in report.trades if t.exit_reason == "Take Profit")
        sess = sum(1 for t in report.trades if t.exit_reason == "Session Close")
        rev = sum(1 for t in report.trades if t.exit_reason == "Reverse Signal")
        print(f"  Exits:       SL={sl} ({100*sl/report.total_trades:.0f}%) | "
              f"TP={tp} ({100*tp/report.total_trades:.0f}%) | "
              f"Sess={sess} | Rev={rev}")
        print(f"  Long WR:     {report.long_win_rate:.1f}%  |  Short WR: {report.short_win_rate:.1f}%")
        print(f"  Long P&L:    ${report.long_net_profit:>10,.2f}  |  Short P&L: ${report.short_net_profit:>10,.2f}")

    if chart_suffix:
        chart = BacktestChart(data, report, engine.signals)
        path = f"output/backtest_{chart_suffix}.html"
        chart.plot_backtest(save_path=path)
        print(f"  Chart:       {path}")

    return report


def main():
    print("Loading 5 years of ES 1-min data...")
    data = load_5yr()
    print(f"  {len(data)} bars | {data.index[0]} -> {data.index[-1]}")

    # 1. Previous best (baseline)
    run_and_report("PREVIOUS BEST (baseline, str=0.3)", dict(BASELINE), data)

    # 2. Config K: strength=0.6 + sweep=1 (top performer)
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["sweep_ticks"] = 1
    run_and_report("CONFIG K: strength=0.6 + sweep=1", p, data, chart_suffix="config_K")

    # 3. Config B: just strength=0.6
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    run_and_report("CONFIG B: strength=0.6 only", p, data, chart_suffix="config_B")

    # 4. Config Q: str=0.5+piv=12+swp=1+age200+sp200
    p = dict(BASELINE)
    p["min_strength"] = 0.5
    p["pivot_left"] = 12
    p["sweep_ticks"] = 1
    p["max_level_age"] = 200
    p["strength_period"] = 200
    run_and_report("CONFIG Q: str=0.5+piv=12+swp=1+age200+sp200", p, data, chart_suffix="config_Q")

    # 5. Config P: str=0.6+piv=20+swp=7+cd=7
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 7
    p["signal_cooldown"] = 7
    run_and_report("CONFIG P: str=0.6+piv=20+swp=7+cd=7", p, data, chart_suffix="config_P")

    # 6. Config M: str=0.6+piv=20+swp=1
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 1
    run_and_report("CONFIG M: str=0.6+piv=20+swp=1", p, data)


if __name__ == "__main__":
    main()
