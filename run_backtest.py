#!/usr/bin/env python3
"""
PHANTOM Strategy Backtester - Main entry point.

Usage:
    python run_backtest.py                              # Run with defaults
    python run_backtest.py --config config/phantom_config.yaml  # Run with config
    python run_backtest.py --symbol SPY --period 5d     # Quick yfinance run
    python run_backtest.py --csv data/ES_1min.csv       # Run on CSV data
    python run_backtest.py --sample 10000               # Run on synthetic data
    python run_backtest.py --verbose                    # Print each trade
    python run_backtest.py --no-chart                   # Skip chart generation
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import (
    RiskConfig,
    StopLossMode,
    TakeProfitMode,
    TrailingMode,
    PositionSizeMode,
)
from backtester.visualization import BacktestChart
from strategies.phantom_cvd import PhantomCVDStrategy


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_risk_config(cfg: dict) -> RiskConfig:
    """Build RiskConfig from config dict."""
    risk = cfg.get("risk", {})

    sl_mode_map = {
        "fixed_points": StopLossMode.FIXED_POINTS,
        "percentage": StopLossMode.PERCENTAGE,
        "atr": StopLossMode.ATR,
        "swept_level": StopLossMode.SWEPT_LEVEL,
    }
    tp_mode_map = {
        "fixed_points": TakeProfitMode.FIXED_POINTS,
        "percentage": TakeProfitMode.PERCENTAGE,
        "atr": TakeProfitMode.ATR,
        "risk_reward": TakeProfitMode.RISK_REWARD,
    }
    trail_mode_map = {
        "fixed_points": TrailingMode.FIXED_POINTS,
        "percentage": TrailingMode.PERCENTAGE,
        "atr": TrailingMode.ATR,
    }
    pos_mode_map = {
        "percent_equity": PositionSizeMode.PERCENT_EQUITY,
        "fixed_contracts": PositionSizeMode.FIXED_CONTRACTS,
        "risk_per_trade": PositionSizeMode.RISK_PER_TRADE,
    }

    return RiskConfig(
        position_mode=pos_mode_map.get(risk.get("position_mode", "risk_per_trade"), PositionSizeMode.RISK_PER_TRADE),
        equity_pct=risk.get("equity_pct", 100.0),
        fixed_contracts=risk.get("fixed_contracts", 1),
        risk_pct=risk.get("risk_pct", 1.0),
        use_stop_loss=risk.get("use_stop_loss", True),
        sl_mode=sl_mode_map.get(risk.get("sl_mode", "atr"), StopLossMode.ATR),
        sl_fixed_points=risk.get("sl_fixed_points", 20.0),
        sl_percentage=risk.get("sl_percentage", 1.0),
        sl_atr_mult=risk.get("sl_atr_mult", 2.0),
        sl_atr_period=risk.get("sl_atr_period", 14),
        sl_swept_offset_ticks=risk.get("sl_swept_offset_ticks", 2.0),
        use_take_profit=risk.get("use_take_profit", True),
        tp_mode=tp_mode_map.get(risk.get("tp_mode", "risk_reward"), TakeProfitMode.RISK_REWARD),
        tp_fixed_points=risk.get("tp_fixed_points", 40.0),
        tp_percentage=risk.get("tp_percentage", 2.0),
        tp_atr_mult=risk.get("tp_atr_mult", 3.0),
        tp_rr_ratio=risk.get("tp_rr_ratio", 2.0),
        use_trailing=risk.get("use_trailing", False),
        trail_mode=trail_mode_map.get(risk.get("trail_mode", "atr"), TrailingMode.ATR),
        trail_fixed_points=risk.get("trail_fixed_points", 15.0),
        trail_percentage=risk.get("trail_percentage", 1.0),
        trail_atr_mult=risk.get("trail_atr_mult", 1.5),
        use_breakeven=risk.get("use_breakeven", False),
        be_trigger_points=risk.get("be_trigger_points", 15.0),
        be_offset=risk.get("be_offset", 1.0),
        use_session_filter=risk.get("use_session_filter", False),
        session_start_hour=risk.get("session_start_hour", 9),
        session_start_minute=risk.get("session_start_minute", 30),
        session_end_hour=risk.get("session_end_hour", 16),
        session_end_minute=risk.get("session_end_minute", 0),
        close_at_session_end=risk.get("close_at_session_end", False),
        use_max_drawdown=risk.get("use_max_drawdown", False),
        max_drawdown_pct=risk.get("max_drawdown_pct", 5.0),
        use_daily_loss_limit=risk.get("use_daily_loss_limit", False),
        daily_loss_pct=risk.get("daily_loss_pct", 2.0),
        use_max_hold=risk.get("use_max_hold", False),
        max_hold_bars=risk.get("max_hold_bars", 50),
        commission_per_contract=risk.get("commission_per_contract", 1.24),
        slippage_ticks=risk.get("slippage_ticks", 1),
        tick_size=risk.get("tick_size", 0.25),
        point_value=risk.get("point_value", 50.0),
    )


def main():
    parser = argparse.ArgumentParser(description="PHANTOM Strategy Backtester")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--csv", type=str, help="Path to CSV data file")
    parser.add_argument("--csv-dir", type=str, help="Directory of yearly CSV files (e.g. data/es_1min)")
    parser.add_argument("--min-price", type=float, default=1000.0, help="Price floor filter for multi-contract data (default: 1000)")
    parser.add_argument("--symbol", type=str, help="yfinance ticker symbol")
    parser.add_argument("--period", type=str, default="5d", help="yfinance period")
    parser.add_argument("--interval", type=str, default="1m", help="yfinance interval")
    parser.add_argument("--sample", type=int, help="Generate N bars of sample data")
    parser.add_argument("--verbose", action="store_true", help="Print trade details")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    # Load configuration
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config("config/phantom_config.yaml") if Path("config/phantom_config.yaml").exists() else {"data": {}, "strategy": {}, "risk": {}}

    # Override config with CLI args
    data_cfg = cfg.get("data", {})
    strategy_cfg = cfg.get("strategy", {})

    # --- Load Data ---
    loader = DataLoader()
    print("=" * 60)
    print("  PHANTOM CVD Strategy Backtester")
    print("=" * 60)

    if args.csv_dir:
        print(f"\nLoading CSV files from {args.csv_dir}...")
        data = loader.load_directory(args.csv_dir, min_price=args.min_price)
    elif args.csv:
        print(f"\nLoading CSV data from {args.csv}...")
        data = loader.load_csv(args.csv, min_price=args.min_price)
    elif args.symbol:
        print(f"\nDownloading {args.symbol} ({args.interval}) via yfinance...")
        data = loader.download(args.symbol, period=args.period, interval=args.interval)
    elif args.sample:
        print(f"\nGenerating {args.sample} bars of sample data...")
        data = loader.generate_sample_data(bars=args.sample)
    elif data_cfg.get("source") == "csv":
        csv_path = data_cfg.get("csv_path", "data/ES_1min.csv")
        print(f"\nLoading CSV data from {csv_path}...")
        data = loader.load_csv(csv_path, min_price=args.min_price)
    elif data_cfg.get("source") == "yfinance":
        symbol = data_cfg.get("symbol", "ES=F")
        print(f"\nDownloading {symbol} via yfinance...")
        data = loader.download(
            symbol,
            period=data_cfg.get("period", "5d"),
            interval=data_cfg.get("interval", "1m"),
        )
    else:
        sample_bars = data_cfg.get("sample_bars", 5000)
        print(f"\nGenerating {sample_bars} bars of sample data...")
        data = loader.generate_sample_data(
            bars=sample_bars,
            start_price=data_cfg.get("sample_start_price", 5000.0),
            volatility=data_cfg.get("sample_volatility", 0.001),
        )

    stats = loader.get_stats()
    print(f"  Bars: {stats['bars']}  |  Range: {stats['start']} → {stats['end']}")
    print(f"  Price Range: {stats['price_range']}  |  Avg Volume: {stats['avg_volume']}")

    # --- Build Strategy ---
    strategy = PhantomCVDStrategy(params=strategy_cfg)

    # --- Build Risk Config ---
    risk_config = build_risk_config(cfg)
    initial_capital = cfg.get("risk", {}).get("initial_capital", 100000.0)

    # --- Run Backtest ---
    has_real_cvd = "cvd" in data.columns and data["cvd"].notna().sum() > 0
    print(f"\nCVD Source: {'TradingView (real)' if has_real_cvd else 'Estimated (close vs open)'}")
    print("Running backtest...")
    engine = BacktestEngine(initial_capital=initial_capital, risk_config=risk_config)
    report = engine.run(data, strategy, verbose=args.verbose)

    # --- Print Results ---
    print(report.summary())

    # --- Generate Charts ---
    if not args.no_chart:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        print("\nGenerating charts...")
        chart = BacktestChart(data, report, engine.signals)

        # Main backtest chart - last 2 weeks only for easy comparison with TradingView
        two_weeks_ago = data.index[-1] - pd.Timedelta(weeks=2)
        start_idx = data.index.searchsorted(two_weeks_ago)
        bar_range_2w = (start_idx, len(data))
        print(f"  Chart showing last 2 weeks: bars {start_idx} to {len(data)} ({len(data) - start_idx} bars)")

        fig = chart.plot_backtest(
            title="PHANTOM CVD Strategy Backtest (Last 2 Weeks)",
            show_volume=True,
            show_equity=True,
            show_cvd=True,
            cvd_values=strategy.cvd,
            bar_range=bar_range_2w,
            save_path=str(output_dir / "backtest_chart.html"),
        )

        # Performance dashboard
        chart.plot_performance_dashboard(
            save_path=str(output_dir / "performance_dashboard.html"),
        )

        # Trade log
        trades_df = report.to_dataframe()
        if not trades_df.empty:
            trades_df.to_csv(str(output_dir / "trades.csv"), index=False)
            print(f"Trade log saved to {output_dir / 'trades.csv'}")

        print(f"\nAll outputs saved to {output_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
