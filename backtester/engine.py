"""
Core backtesting engine.
Processes bar data through a strategy with risk management.
"""

import pandas as pd
import numpy as np
from typing import Optional
from backtester.risk_manager import RiskManager, RiskConfig, Trade
from backtester.performance import PerformanceReport


class BacktestEngine:
    """
    TradingView-inspired backtesting engine.

    Processes bars sequentially, feeds them to a strategy,
    manages positions through the risk manager, and collects results.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_config = risk_config or RiskConfig()
        self.risk_manager = RiskManager(self.risk_config)

        # State
        self.equity: float = initial_capital
        self.equity_curve: list[float] = []
        self.bar_times: list = []
        self.signals: list[dict] = []

    def run(
        self,
        data: pd.DataFrame,
        strategy,
        verbose: bool = False,
    ) -> PerformanceReport:
        """
        Run a backtest on the given data using the provided strategy.

        Args:
            data: DataFrame with columns [open, high, low, close, volume] and datetime index
            strategy: Strategy instance with generate_signals() method
            verbose: Print trade details as they happen
        """
        self.equity = self.initial_capital
        self.equity_curve = []
        self.bar_times = []
        self.signals = []
        self.risk_manager.reset(self.initial_capital)

        # Let strategy initialize with full data
        strategy.initialize(data)

        opens = data["open"].values
        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values
        volumes = data["volume"].values
        timestamps = data.index

        n_bars = len(data)

        for i in range(1, n_bars):
            timestamp = timestamps[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_volume = volumes[i]

            # Calculate ATR from available history
            lookback = min(i, self.risk_config.sl_atr_period + 1)
            atr = self.risk_manager.calc_atr(
                highs[i - lookback:i + 1],
                lows[i - lookback:i + 1],
                closes[i - lookback:i + 1],
            )

            # --- Update existing position ---
            trade = self.risk_manager.update_position(
                high=bar_high,
                low=bar_low,
                close=bar_close,
                bar_index=i,
                atr=atr,
                timestamp=timestamp,
            )

            if trade is not None:
                self.equity += trade.pnl
                if verbose:
                    direction = "LONG" if trade.direction == 1 else "SHORT"
                    print(
                        f"[{timestamp}] CLOSE {direction} | "
                        f"Entry: {trade.entry_price:.2f} → Exit: {trade.exit_price:.2f} | "
                        f"P&L: ${trade.pnl:.2f} | Reason: {trade.exit_reason} | "
                        f"Equity: ${self.equity:.2f}"
                    )

            # --- Get strategy signal ---
            signal = strategy.generate_signal(i)

            if signal is not None and signal["direction"] != 0:
                can_enter = self.risk_manager.can_trade(self.equity, timestamp)
                has_position = self.risk_manager.position is not None

                sig_dir = signal["direction"]  # 1 = long, -1 = short

                # Close opposite position if needed
                if has_position and self.risk_manager.position.direction != sig_dir:
                    close_trade = self.risk_manager.force_close(
                        bar_close, i, "Reverse Signal", str(timestamp)
                    )
                    if close_trade is not None:
                        self.equity += close_trade.pnl
                        if verbose:
                            direction = "LONG" if close_trade.direction == 1 else "SHORT"
                            print(
                                f"[{timestamp}] REVERSE CLOSE {direction} | "
                                f"P&L: ${close_trade.pnl:.2f} | Equity: ${self.equity:.2f}"
                            )
                    has_position = False

                # Enter new position
                if not has_position and can_enter:
                    swept_level = signal.get("swept_level")
                    pos = self.risk_manager.open_position(
                        direction=sig_dir,
                        entry_price=bar_close,
                        bar_index=i,
                        equity=self.equity,
                        atr=atr,
                        swept_level=swept_level,
                        timestamp=str(timestamp),
                    )

                    self.signals.append({
                        "bar_index": i,
                        "timestamp": timestamp,
                        "direction": sig_dir,
                        "price": bar_close,
                        "strength": signal.get("strength", 0),
                        "swept_level": swept_level,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                    })

                    if verbose:
                        direction = "LONG" if sig_dir == 1 else "SHORT"
                        sl_str = f"{pos.stop_loss:.2f}" if pos.stop_loss else "None"
                        tp_str = f"{pos.take_profit:.2f}" if pos.take_profit else "None"
                        print(
                            f"[{timestamp}] ENTER {direction} @ {pos.entry_price:.2f} | "
                            f"Qty: {pos.quantity} | SL: {sl_str} | TP: {tp_str}"
                        )

            # Track equity
            # Mark-to-market: include unrealized P&L
            unrealized = 0.0
            if self.risk_manager.position is not None:
                pos = self.risk_manager.position
                price_diff = (bar_close - pos.entry_price) * pos.direction
                unrealized = price_diff * pos.quantity * self.risk_config.point_value

            self.equity_curve.append(self.equity + unrealized)
            self.bar_times.append(timestamp)

        # Close any open position at end
        if self.risk_manager.position is not None:
            final_trade = self.risk_manager.force_close(
                closes[-1], n_bars - 1, "End of Data", str(timestamps[-1])
            )
            if final_trade:
                self.equity += final_trade.pnl
                if verbose:
                    print(f"[END] Closed remaining position | P&L: ${final_trade.pnl:.2f}")

        # Build performance report
        report = PerformanceReport(
            trades=self.risk_manager.trades,
            equity_curve=self.equity_curve,
            bar_times=self.bar_times,
            initial_capital=self.initial_capital,
            signals=self.signals,
        )
        return report
