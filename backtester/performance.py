"""
Performance analytics and reporting for backtests.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


class PerformanceReport:
    """
    Comprehensive performance analytics for a completed backtest.
    Mirrors TradingView's strategy report metrics.
    """

    def __init__(
        self,
        trades: list,
        equity_curve: list[float],
        bar_times: list,
        initial_capital: float,
        signals: list[dict],
    ):
        self.trades = trades
        self.equity_curve = np.array(equity_curve) if equity_curve else np.array([initial_capital])
        self.bar_times = bar_times
        self.initial_capital = initial_capital
        self.signals = signals

        self._compute_metrics()

    def _compute_metrics(self):
        """Compute all performance metrics."""
        trades = self.trades
        ec = self.equity_curve

        # --- Trade Metrics ---
        self.total_trades = len(trades)
        self.long_trades = sum(1 for t in trades if t.direction == 1)
        self.short_trades = sum(1 for t in trades if t.direction == -1)

        if self.total_trades == 0:
            self._set_zero_metrics()
            return

        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades * 100

        self.net_profit = np.sum(pnls)
        self.gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        self.gross_loss = np.sum(losses) if len(losses) > 0 else 0.0

        self.profit_factor = (
            abs(self.gross_profit / self.gross_loss)
            if self.gross_loss != 0
            else float("inf")
        )

        self.avg_trade = np.mean(pnls)
        self.avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        self.avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

        self.largest_win = np.max(wins) if len(wins) > 0 else 0.0
        self.largest_loss = np.min(losses) if len(losses) > 0 else 0.0

        self.avg_bars_held = np.mean([t.bars_held for t in trades])
        self.avg_bars_winners = (
            np.mean([t.bars_held for t in trades if t.pnl > 0])
            if self.winning_trades > 0
            else 0
        )
        self.avg_bars_losers = (
            np.mean([t.bars_held for t in trades if t.pnl <= 0])
            if self.losing_trades > 0
            else 0
        )

        # Consecutive wins/losses
        self.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        self.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        # Commission & slippage
        self.total_commission = sum(t.commission for t in trades)
        self.total_slippage = sum(t.slippage for t in trades)

        # --- Equity Curve Metrics ---
        self.final_equity = ec[-1] if len(ec) > 0 else self.initial_capital
        self.total_return_pct = (self.final_equity - self.initial_capital) / self.initial_capital * 100

        # Max drawdown
        peak = np.maximum.accumulate(ec)
        drawdowns = (peak - ec) / peak * 100
        self.max_drawdown_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        self.max_drawdown_abs = np.max(peak - ec) if len(ec) > 0 else 0.0

        # Sharpe ratio (annualized, assuming 1-min bars → 252 * 390 bars/year)
        if len(ec) > 1:
            returns = np.diff(ec) / ec[:-1]
            if np.std(returns) > 0:
                bars_per_year = 252 * 390  # 1-min bars
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
            else:
                self.sharpe_ratio = 0.0
        else:
            self.sharpe_ratio = 0.0

        # Sortino ratio
        if len(ec) > 1:
            returns = np.diff(ec) / ec[:-1]
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                bars_per_year = 252 * 390
                self.sortino_ratio = np.mean(returns) / np.std(downside) * np.sqrt(bars_per_year)
            else:
                self.sortino_ratio = 0.0
        else:
            self.sortino_ratio = 0.0

        # Calmar ratio
        if self.max_drawdown_pct > 0:
            self.calmar_ratio = self.total_return_pct / self.max_drawdown_pct
        else:
            self.calmar_ratio = 0.0

        # Expectancy
        if self.total_trades > 0:
            win_prob = self.winning_trades / self.total_trades
            loss_prob = self.losing_trades / self.total_trades
            self.expectancy = (win_prob * self.avg_win) + (loss_prob * self.avg_loss)
        else:
            self.expectancy = 0.0

        # Exit reason breakdown
        self.exit_reasons = {}
        for t in trades:
            self.exit_reasons[t.exit_reason] = self.exit_reasons.get(t.exit_reason, 0) + 1

        # Long vs Short breakdown
        long_pnls = [t.pnl for t in trades if t.direction == 1]
        short_pnls = [t.pnl for t in trades if t.direction == -1]
        self.long_net_profit = sum(long_pnls)
        self.short_net_profit = sum(short_pnls)
        self.long_win_rate = (
            sum(1 for p in long_pnls if p > 0) / len(long_pnls) * 100
            if long_pnls
            else 0
        )
        self.short_win_rate = (
            sum(1 for p in short_pnls if p > 0) / len(short_pnls) * 100
            if short_pnls
            else 0
        )

    def _set_zero_metrics(self):
        """Set all metrics to zero when there are no trades."""
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.net_profit = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.profit_factor = 0.0
        self.avg_trade = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.avg_bars_held = 0.0
        self.avg_bars_winners = 0.0
        self.avg_bars_losers = 0.0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.final_equity = self.initial_capital
        self.total_return_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.max_drawdown_abs = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.expectancy = 0.0
        self.exit_reasons = {}
        self.long_net_profit = 0.0
        self.short_net_profit = 0.0
        self.long_win_rate = 0.0
        self.short_win_rate = 0.0

    @staticmethod
    def _max_consecutive(pnls: np.ndarray, positive: bool) -> int:
        """Count max consecutive wins or losses."""
        max_streak = 0
        current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p <= 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def summary(self) -> str:
        """Generate a formatted performance summary (TradingView-style)."""
        lines = [
            "=" * 60,
            "  PHANTOM STRATEGY - BACKTEST REPORT",
            "=" * 60,
            "",
            "--- Overview ---",
            f"  Net Profit:           ${self.net_profit:>12,.2f}  ({self.total_return_pct:+.2f}%)",
            f"  Initial Capital:      ${self.initial_capital:>12,.2f}",
            f"  Final Equity:         ${self.final_equity:>12,.2f}",
            "",
            "--- Trade Statistics ---",
            f"  Total Trades:         {self.total_trades:>8d}",
            f"  Long Trades:          {self.long_trades:>8d}",
            f"  Short Trades:         {self.short_trades:>8d}",
            f"  Win Rate:             {self.win_rate:>8.1f}%",
            f"  Winning Trades:       {self.winning_trades:>8d}",
            f"  Losing Trades:        {self.losing_trades:>8d}",
            "",
            "--- Profit Analysis ---",
            f"  Gross Profit:         ${self.gross_profit:>12,.2f}",
            f"  Gross Loss:           ${self.gross_loss:>12,.2f}",
            f"  Profit Factor:        {self.profit_factor:>8.2f}",
            f"  Expectancy:           ${self.expectancy:>12,.2f}",
            f"  Avg Trade:            ${self.avg_trade:>12,.2f}",
            f"  Avg Win:              ${self.avg_win:>12,.2f}",
            f"  Avg Loss:             ${self.avg_loss:>12,.2f}",
            f"  Largest Win:          ${self.largest_win:>12,.2f}",
            f"  Largest Loss:         ${self.largest_loss:>12,.2f}",
            "",
            "--- Long vs Short ---",
            f"  Long Net Profit:      ${self.long_net_profit:>12,.2f}  (WR: {self.long_win_rate:.1f}%)",
            f"  Short Net Profit:     ${self.short_net_profit:>12,.2f}  (WR: {self.short_win_rate:.1f}%)",
            "",
            "--- Risk Metrics ---",
            f"  Max Drawdown:         ${self.max_drawdown_abs:>12,.2f}  ({self.max_drawdown_pct:.2f}%)",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>8.2f}",
            f"  Sortino Ratio:        {self.sortino_ratio:>8.2f}",
            f"  Calmar Ratio:         {self.calmar_ratio:>8.2f}",
            "",
            "--- Trade Duration ---",
            f"  Avg Bars Held:        {self.avg_bars_held:>8.1f}",
            f"  Avg Bars (Winners):   {self.avg_bars_winners:>8.1f}",
            f"  Avg Bars (Losers):    {self.avg_bars_losers:>8.1f}",
            f"  Max Consec. Wins:     {self.max_consecutive_wins:>8d}",
            f"  Max Consec. Losses:   {self.max_consecutive_losses:>8d}",
            "",
            "--- Costs ---",
            f"  Total Commission:     ${self.total_commission:>12,.2f}",
            f"  Total Slippage:       ${self.total_slippage:>12,.2f}",
            "",
            "--- Exit Reasons ---",
        ]

        for reason, count in sorted(self.exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / self.total_trades * 100 if self.total_trades > 0 else 0
            lines.append(f"  {reason:<24s} {count:>4d}  ({pct:.1f}%)")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "direction": "Long" if t.direction == 1 else "Short",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "commission": t.commission,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "exit_time": t.exit_time,
            })
        return pd.DataFrame(records)

    def equity_dataframe(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame."""
        return pd.DataFrame({
            "datetime": self.bar_times,
            "equity": self.equity_curve[:len(self.bar_times)],
        }).set_index("datetime")
