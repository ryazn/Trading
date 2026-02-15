#!/usr/bin/env python3
"""Generate reasonably-sized charts for the top configs (last 2 weeks only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.visualization import BacktestChart
from backtester.risk_manager import (
    RiskConfig, StopLossMode, TakeProfitMode, PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy

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
    use_session_filter=True,
    session_start_hour=9,
    session_start_minute=30,
    session_end_hour=16,
    session_end_minute=0,
    close_at_session_end=True,
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


def generate_chart(name, params, data, filename):
    print(f"\n  Running {name}...")
    strategy = PhantomCVDStrategy(params=params)
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, risk_config=RISK)
    report = engine.run(data, strategy, verbose=False)

    print(f"    Net: ${report.net_profit:,.2f} | Trades: {report.total_trades} | "
          f"WR: {report.win_rate:.1f}% | PF: {report.profit_factor:.2f}")

    chart = BacktestChart(data, report, engine.signals)

    # Last 2 weeks only
    two_weeks_ago = data.index[-1] - pd.Timedelta(weeks=2)
    start_idx = data.index.searchsorted(two_weeks_ago)
    bar_range = (start_idx, len(data))

    path = f"output/{filename}"
    chart.plot_backtest(
        title=f"PHANTOM CVD — {name} (Last 2 Weeks)",
        show_volume=True,
        show_equity=True,
        show_cvd=True,
        cvd_values=strategy.cvd,
        bar_range=bar_range,
        save_path=path,
    )
    print(f"    Chart saved: {path}")
    return report


def main():
    Path("output").mkdir(exist_ok=True)

    print("Loading 5 years of ES 1-min data...")
    data = load_5yr()
    print(f"  {len(data)} bars")

    # Config B: strength=0.6 (best overall)
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    generate_chart("Config B (str=0.6)", p, data, "backtest_config_B.html")

    # Config K: strength=0.6 + sweep=1
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["sweep_ticks"] = 1
    generate_chart("Config K (str=0.6, swp=1)", p, data, "backtest_config_K.html")

    # Config P: str=0.6+piv=20+swp=7+cd=7 (quality king)
    p = dict(BASELINE)
    p["min_strength"] = 0.6
    p["pivot_left"] = 20
    p["sweep_ticks"] = 7
    p["signal_cooldown"] = 7
    generate_chart("Config P (str=0.6, piv=20, swp=7)", p, data, "backtest_config_P.html")

    print("\nDone! Charts are in output/")


if __name__ == "__main__":
    main()
