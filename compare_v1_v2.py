#!/usr/bin/env python3
"""
Compare PHANTOM V1 vs V2 on 5 years of ES data.
Tests the impact of trend filter + volatility filter.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig, StopLossMode, TakeProfitMode, PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy
from strategies.phantom_cvd_v2 import PhantomCVDStrategyV2

import pandas as pd


def load_5yr():
    loader = DataLoader()
    frames = []
    for year in [2021, 2022, 2023, 2024, 2025]:
        path = f"data/es_1min/ES_{year}.csv"
        df = loader.load_csv(path)
        df = df[df["close"] >= 1000]
        frames.append(df)
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def make_risk(session_filter=True):
    return RiskConfig(
        position_mode=PositionSizeMode.FIXED_CONTRACTS,
        fixed_contracts=1,
        use_stop_loss=True,
        sl_mode=StopLossMode.ATR,
        sl_atr_mult=2.0,
        sl_atr_period=14,
        use_take_profit=True,
        tp_mode=TakeProfitMode.RISK_REWARD,
        tp_rr_ratio=2.0,
        use_trailing=False,
        use_breakeven=False,
        use_session_filter=session_filter,
        session_start_hour=9,
        session_start_minute=30,
        session_end_hour=16,
        session_end_minute=0,
        close_at_session_end=session_filter,
        use_max_hold=True,
        max_hold_bars=120,
        commission_per_contract=1.24,
        slippage_ticks=1,
        tick_size=0.25,
        point_value=50.0,
    )


def run_and_report(name, strategy, risk, data):
    engine = BacktestEngine(initial_capital=100000.0, risk_config=risk)
    start = time.time()
    report = engine.run(data, strategy, verbose=False)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"  {name}  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Net P&L:    ${report.net_profit:>10,.2f}")
    print(f"  Trades:     {report.total_trades:>6d}")
    print(f"  Win Rate:   {report.win_rate:>6.1f}%")
    print(f"  PF:         {report.profit_factor:>6.2f}")
    print(f"  Sharpe:     {report.sharpe_ratio:>6.2f}")
    print(f"  Max DD:     {report.max_drawdown_pct:>6.2f}%")
    print(f"  Avg Trade:  ${report.avg_trade:>8.2f}")
    print(f"  Expectancy: ${report.expectancy:>8.2f}")

    if report.total_trades > 0:
        sl_exits = sum(1 for t in report.trades if t.exit_reason == "Stop Loss")
        tp_exits = sum(1 for t in report.trades if t.exit_reason == "Take Profit")
        other = report.total_trades - sl_exits - tp_exits
        print(f"  Exits:      SL={sl_exits} ({100*sl_exits/report.total_trades:.0f}%) | "
              f"TP={tp_exits} ({100*tp_exits/report.total_trades:.0f}%) | Other={other}")

    return report


def main():
    print("Loading 5 years of ES 1-min data...")
    data = load_5yr()
    print(f"  {len(data)} bars | {data.index[0]} -> {data.index[-1]}")

    risk = make_risk(session_filter=True)

    # Base params shared by V1 and V2
    base_params = {
        "pivot_left": 10,
        "pivot_right": 2,
        "min_strength": 0.3,
        "sweep_ticks": 2,
        "signal_cooldown": 5,
        "max_level_age": 100,
        "strength_period": 100,
    }

    # ============================================
    # TEST 1: V1 baseline (RTH session filter only)
    # ============================================
    v1 = PhantomCVDStrategy(params=base_params)
    run_and_report("V1: Baseline (RTH session filter)", v1, risk, data)

    # ============================================
    # TEST 2: V2 with trend filter (with_trend)
    # ============================================
    v2_trend = PhantomCVDStrategyV2(params={
        **base_params,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",
        "use_vol_filter": False,
    })
    run_and_report("V2: + Trend filter (with_trend, 200 EMA)", v2_trend, risk, data)

    # ============================================
    # TEST 3: V2 with vol filter only
    # ============================================
    v2_vol = PhantomCVDStrategyV2(params={
        **base_params,
        "use_trend_filter": False,
        "use_vol_filter": True,
        "vol_min_pctile": 0.15,
        "vol_max_pctile": 0.85,
    })
    run_and_report("V2: + Vol filter only (15-85th pctile)", v2_vol, risk, data)

    # ============================================
    # TEST 4: V2 with both filters
    # ============================================
    v2_both = PhantomCVDStrategyV2(params={
        **base_params,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",
        "use_vol_filter": True,
        "vol_min_pctile": 0.15,
        "vol_max_pctile": 0.85,
    })
    run_and_report("V2: + Trend + Vol filters", v2_both, risk, data)

    # ============================================
    # TEST 5: V2 ranging mode (only trade near EMA)
    # ============================================
    v2_ranging = PhantomCVDStrategyV2(params={
        **base_params,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "ranging",
        "trend_atr_band": 3.0,
        "use_vol_filter": True,
        "vol_min_pctile": 0.15,
        "vol_max_pctile": 0.85,
    })
    run_and_report("V2: Ranging mode (near EMA) + Vol filter", v2_ranging, risk, data)

    # ============================================
    # TEST 6: V2 near_ema mode + tighter strength
    # ============================================
    v2_tight = PhantomCVDStrategyV2(params={
        **base_params,
        "min_strength": 0.5,
        "signal_cooldown": 10,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",
        "use_vol_filter": True,
        "vol_min_pctile": 0.20,
        "vol_max_pctile": 0.80,
    })
    run_and_report("V2: Tight (strength=0.5, cd=10, trend+vol)", v2_tight, risk, data)

    # ============================================
    # TEST 7: V2 with dynamic sweep threshold
    # ============================================
    v2_dynamic = PhantomCVDStrategyV2(params={
        **base_params,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",
        "use_vol_filter": True,
        "vol_min_pctile": 0.15,
        "vol_max_pctile": 0.85,
        "use_dynamic_sweep": True,
        "dynamic_sweep_atr_frac": 0.1,
    })
    run_and_report("V2: Dynamic sweep + Trend + Vol", v2_dynamic, risk, data)

    # ============================================
    # TEST 8: Best V1 params from optimizer (#2)
    # ============================================
    v1_best = PhantomCVDStrategy(params={
        "pivot_left": 10,
        "pivot_right": 2,
        "min_strength": 0.3,
        "sweep_ticks": 2,
        "signal_cooldown": 5,
        "max_level_age": 100,
        "strength_period": 100,
    })
    risk_best = RiskConfig(
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
    run_and_report("V1: Best optimizer config (1ct, ATR 2.5, RR 1.5)", v1_best, risk_best, data)

    # ============================================
    # TEST 9: V2 with best risk params from above
    # ============================================
    v2_best = PhantomCVDStrategyV2(params={
        "pivot_left": 10,
        "pivot_right": 2,
        "min_strength": 0.3,
        "sweep_ticks": 2,
        "signal_cooldown": 5,
        "max_level_age": 100,
        "strength_period": 100,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",
        "use_vol_filter": True,
        "vol_min_pctile": 0.15,
        "vol_max_pctile": 0.85,
    })
    run_and_report("V2: Best risk + Trend + Vol", v2_best, risk_best, data)


if __name__ == "__main__":
    main()
