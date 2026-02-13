"""
TradingView-inspired chart visualization for backtests.
Uses Plotly for interactive candlestick charts with trade overlays.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class BacktestChart:
    """
    Generate TradingView-style interactive charts for backtest results.

    Features:
    - Candlestick price chart
    - Volume bars (colored by direction)
    - Trade entry/exit markers
    - Stop loss and take profit lines
    - Equity curve
    - CVD subplot (optional)
    """

    TV_DARK_BG = "#131722"
    TV_GRID = "#1e222d"
    TV_TEXT = "#d1d4dc"
    TV_GREEN = "#26a69a"
    TV_RED = "#ef5350"
    TV_BLUE = "#2196f3"
    TV_ORANGE = "#ff9800"
    TV_PURPLE = "#ab47bc"

    def __init__(self, data: pd.DataFrame, report=None, signals: Optional[list] = None):
        """
        Args:
            data: OHLCV DataFrame with datetime index
            report: PerformanceReport instance
            signals: List of signal dicts from the engine
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for visualization: pip install plotly")

        self.data = data
        self.report = report
        self.signals = signals or []
        self.trades = report.trades if report else []

    def plot_backtest(
        self,
        title: str = "PHANTOM Strategy Backtest",
        show_volume: bool = True,
        show_equity: bool = True,
        show_cvd: bool = False,
        cvd_values: Optional[np.ndarray] = None,
        bar_range: Optional[tuple] = None,
        height: int = 900,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create the full backtest chart.

        Args:
            title: Chart title
            show_volume: Show volume subplot
            show_equity: Show equity curve subplot
            show_cvd: Show CVD subplot
            cvd_values: CVD array (same length as data)
            bar_range: Tuple (start_idx, end_idx) to zoom into specific bars
            height: Chart height in pixels
            save_path: Path to save as HTML file
        """
        df = self.data
        if bar_range:
            df = df.iloc[bar_range[0]:bar_range[1]]

        # Determine subplot layout
        n_subplots = 1
        row_heights = [0.6]
        subplot_titles = ["Price"]

        if show_volume:
            n_subplots += 1
            row_heights.append(0.1)
            subplot_titles.append("Volume")
        if show_equity and self.report:
            n_subplots += 1
            row_heights.append(0.15)
            subplot_titles.append("Equity")
        if show_cvd and cvd_values is not None:
            n_subplots += 1
            row_heights.append(0.15)
            subplot_titles.append("CVD")

        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        # --- Candlestick Chart ---
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=self.TV_GREEN,
                decreasing_line_color=self.TV_RED,
                increasing_fillcolor=self.TV_GREEN,
                decreasing_fillcolor=self.TV_RED,
                name="Price",
            ),
            row=1,
            col=1,
        )

        # --- Trade Markers ---
        self._add_trade_markers(fig, df, bar_range)

        # --- Volume ---
        current_row = 2
        if show_volume:
            colors = [
                self.TV_GREEN if c >= o else self.TV_RED
                for c, o in zip(df["close"], df["open"])
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["volume"],
                    marker_color=colors,
                    opacity=0.5,
                    name="Volume",
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
            current_row += 1

        # --- Equity Curve ---
        if show_equity and self.report:
            ec = self.report.equity_curve
            times = self.report.bar_times

            if bar_range:
                start_idx = max(0, bar_range[0] - 1)
                end_idx = min(len(ec), bar_range[1] - 1)
                ec = ec[start_idx:end_idx]
                times = times[start_idx:end_idx]

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=ec,
                    mode="lines",
                    name="Equity",
                    line=dict(color=self.TV_BLUE, width=1.5),
                ),
                row=current_row,
                col=1,
            )
            current_row += 1

        # --- CVD ---
        if show_cvd and cvd_values is not None:
            cvd_data = cvd_values
            if bar_range:
                cvd_data = cvd_values[bar_range[0]:bar_range[1]]

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=cvd_data[:len(df)],
                    mode="lines",
                    name="CVD",
                    line=dict(color=self.TV_PURPLE, width=1.5),
                ),
                row=current_row,
                col=1,
            )

        # --- Styling (TradingView dark theme) ---
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=self.TV_TEXT, size=16),
                x=0.5,
            ),
            template="plotly_dark",
            paper_bgcolor=self.TV_DARK_BG,
            plot_bgcolor=self.TV_DARK_BG,
            font=dict(color=self.TV_TEXT, family="Trebuchet MS"),
            height=height,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(size=10),
            ),
            hovermode="x unified",
        )

        # Style all axes
        for i in range(1, n_subplots + 1):
            fig.update_xaxes(
                gridcolor=self.TV_GRID,
                showgrid=True,
                zeroline=False,
                row=i,
                col=1,
            )
            fig.update_yaxes(
                gridcolor=self.TV_GRID,
                showgrid=True,
                zeroline=False,
                row=i,
                col=1,
            )

        if save_path:
            fig.write_html(save_path)
            print(f"Chart saved to {save_path}")

        return fig

    def _add_trade_markers(self, fig: go.Figure, df: pd.DataFrame, bar_range: Optional[tuple]):
        """Add entry/exit markers and lines for trades."""
        if not self.trades:
            return

        timestamps = df.index
        offset = bar_range[0] if bar_range else 0

        # Collect entry/exit points
        long_entries_x, long_entries_y = [], []
        short_entries_x, short_entries_y = [], []
        win_exits_x, win_exits_y = [], []
        loss_exits_x, loss_exits_y = [], []

        for trade in self.trades:
            entry_idx = trade.entry_bar - offset
            exit_idx = trade.exit_bar - offset

            if 0 <= entry_idx < len(timestamps):
                if trade.direction == 1:
                    long_entries_x.append(timestamps[entry_idx])
                    long_entries_y.append(trade.entry_price)
                else:
                    short_entries_x.append(timestamps[entry_idx])
                    short_entries_y.append(trade.entry_price)

            if 0 <= exit_idx < len(timestamps):
                if trade.pnl > 0:
                    win_exits_x.append(timestamps[exit_idx])
                    win_exits_y.append(trade.exit_price)
                else:
                    loss_exits_x.append(timestamps[exit_idx])
                    loss_exits_y.append(trade.exit_price)

        # Long entries
        if long_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=long_entries_x,
                    y=long_entries_y,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=self.TV_GREEN,
                        line=dict(width=1, color="white"),
                    ),
                    name="Long Entry",
                    text=["LONG" for _ in long_entries_x],
                ),
                row=1,
                col=1,
            )

        # Short entries
        if short_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=short_entries_x,
                    y=short_entries_y,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=self.TV_RED,
                        line=dict(width=1, color="white"),
                    ),
                    name="Short Entry",
                    text=["SHORT" for _ in short_entries_x],
                ),
                row=1,
                col=1,
            )

        # Winning exits
        if win_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=win_exits_x,
                    y=win_exits_y,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=8,
                        color=self.TV_GREEN,
                        line=dict(width=1, color="white"),
                    ),
                    name="Win Exit",
                ),
                row=1,
                col=1,
            )

        # Losing exits
        if loss_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=loss_exits_x,
                    y=loss_exits_y,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=8,
                        color=self.TV_RED,
                        line=dict(width=1, color="white"),
                    ),
                    name="Loss Exit",
                ),
                row=1,
                col=1,
            )

    def plot_performance_dashboard(
        self,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a performance dashboard with:
        - Equity curve
        - Drawdown chart
        - P&L distribution
        - Win/Loss breakdown
        """
        if not self.report or self.report.total_trades == 0:
            print("No trades to display.")
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Equity Curve", "Drawdown", "P&L Distribution", "Cumulative P&L"],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        ec = self.report.equity_curve
        times = self.report.bar_times

        # --- Equity Curve ---
        fig.add_trace(
            go.Scatter(
                x=times, y=ec, mode="lines",
                name="Equity",
                line=dict(color=self.TV_BLUE, width=1.5),
            ),
            row=1, col=1,
        )

        # --- Drawdown ---
        peak = np.maximum.accumulate(ec)
        dd_pct = (peak - ec) / peak * 100
        fig.add_trace(
            go.Scatter(
                x=times, y=-dd_pct, mode="lines",
                fill="tozeroy",
                name="Drawdown %",
                line=dict(color=self.TV_RED, width=1),
                fillcolor="rgba(239, 83, 80, 0.3)",
            ),
            row=1, col=2,
        )

        # --- P&L Distribution ---
        pnls = [t.pnl for t in self.trades]
        colors = [self.TV_GREEN if p > 0 else self.TV_RED for p in pnls]
        fig.add_trace(
            go.Bar(
                x=list(range(len(pnls))),
                y=pnls,
                marker_color=colors,
                name="Trade P&L",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # --- Cumulative P&L ---
        cum_pnl = np.cumsum(pnls)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cum_pnl))),
                y=cum_pnl,
                mode="lines",
                name="Cumulative P&L",
                line=dict(color=self.TV_ORANGE, width=2),
            ),
            row=2, col=2,
        )

        # --- Styling ---
        fig.update_layout(
            title=dict(
                text="Performance Dashboard",
                font=dict(color=self.TV_TEXT, size=16),
                x=0.5,
            ),
            template="plotly_dark",
            paper_bgcolor=self.TV_DARK_BG,
            plot_bgcolor=self.TV_DARK_BG,
            font=dict(color=self.TV_TEXT, family="Trebuchet MS"),
            height=700,
            showlegend=False,
        )

        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(gridcolor=self.TV_GRID, row=row, col=col)
                fig.update_yaxes(gridcolor=self.TV_GRID, row=row, col=col)

        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")

        return fig
