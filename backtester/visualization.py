"""
TradingView-style chart visualization for backtests.
Uses TradingView Lightweight Charts (v5) for the main backtest chart
and Plotly for the performance dashboard.
"""

import json
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
    - Candlestick price chart (TradingView Lightweight Charts)
    - Volume bars (colored by direction)
    - Trade entry/exit markers
    - Equity curve
    - CVD subplot (optional)
    - Performance dashboard (Plotly)
    """

    TV_DARK_BG = "#121825"
    TV_GRID = "#f0f3fa00"  # transparent grid
    TV_TEXT = "#b2b5be"
    TV_CANDLE = "rgba(255, 235, 59, 0.75)"  # yellow candles
    TV_GREEN = "#26a69a"
    TV_RED = "#ef5350"
    TV_BLUE = "#2196f3"
    TV_ORANGE = "#ff9800"
    TV_PURPLE = "#ab47bc"

    def __init__(self, data: pd.DataFrame, report=None, signals: Optional[list] = None):
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
    ) -> str:
        """
        Create a backtest chart using TradingView Lightweight Charts v5.

        Returns the HTML string. Saves to file if save_path is provided.
        """
        df = self.data
        offset = 0
        if bar_range:
            offset = bar_range[0]
            df = df.iloc[bar_range[0]:bar_range[1]]

        # Deduplicate timestamps (multi-contract data may have overlapping bars).
        # Keep the row with highest volume for each timestamp.
        if df.index.duplicated().any():
            df = df.sort_values("volume", ascending=False)
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()

        # Build candlestick data as list of {time, open, high, low, close}
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })

        # Build volume data
        vol_data = []
        if show_volume:
            for ts, row in df.iterrows():
                color = self.TV_GREEN if row["close"] >= row["open"] else self.TV_RED
                vol_data.append({
                    "time": int(ts.timestamp()),
                    "value": int(row["volume"]),
                    "color": color + "80",  # 50% opacity
                })

        # Build trade markers
        markers = []
        timestamps = df.index
        for trade in self.trades:
            entry_idx = trade.entry_bar - offset
            exit_idx = trade.exit_bar - offset

            if 0 <= entry_idx < len(timestamps):
                entry_ts = int(timestamps[entry_idx].timestamp())
                if trade.direction == 1:
                    markers.append({
                        "time": entry_ts,
                        "position": "belowBar",
                        "color": self.TV_GREEN,
                        "shape": "arrowUp",
                        "text": "L",
                    })
                else:
                    markers.append({
                        "time": entry_ts,
                        "position": "aboveBar",
                        "color": self.TV_RED,
                        "shape": "arrowDown",
                        "text": "S",
                    })

            if 0 <= exit_idx < len(timestamps):
                exit_ts = int(timestamps[exit_idx].timestamp())
                if trade.pnl > 0:
                    markers.append({
                        "time": exit_ts,
                        "position": "aboveBar" if trade.direction == 1 else "belowBar",
                        "color": self.TV_GREEN,
                        "shape": "circle",
                        "text": f"+${trade.pnl:.0f}",
                    })
                else:
                    markers.append({
                        "time": exit_ts,
                        "position": "aboveBar" if trade.direction == 1 else "belowBar",
                        "color": self.TV_RED,
                        "shape": "circle",
                        "text": f"-${abs(trade.pnl):.0f}",
                    })

        # Sort markers by time (required by Lightweight Charts)
        markers.sort(key=lambda m: m["time"])

        # Build equity data (deduplicate timestamps, keep last value)
        equity_data = []
        if show_equity and self.report:
            ec = self.report.equity_curve
            times = self.report.bar_times
            if bar_range:
                start_idx = max(0, bar_range[0] - 1)
                end_idx = min(len(ec), bar_range[1] - 1)
                ec = ec[start_idx:end_idx]
                times = times[start_idx:end_idx]
            seen_times = set()
            eq_pairs = []
            for t, v in zip(times, ec):
                ts = int(t.timestamp())
                eq_pairs.append((ts, round(float(v), 2)))
            # Keep last occurrence per timestamp
            deduped_eq = {}
            for ts, v in eq_pairs:
                deduped_eq[ts] = v
            for ts in sorted(deduped_eq):
                equity_data.append({"time": ts, "value": deduped_eq[ts]})

        # Build CVD data
        cvd_data = []
        if show_cvd and cvd_values is not None:
            cvd_slice = cvd_values[bar_range[0]:bar_range[1]] if bar_range else cvd_values
            for i, ts in enumerate(df.index):
                if i < len(cvd_slice):
                    cvd_data.append({
                        "time": int(ts.timestamp()),
                        "value": round(float(cvd_slice[i]), 2),
                    })

        # Count chart panes
        n_panes = 1  # price always
        if show_volume:
            n_panes += 1
        if equity_data:
            n_panes += 1
        if cvd_data:
            n_panes += 1

        # Build results summary data
        results_data = {}
        if self.report and self.report.total_trades > 0:
            r = self.report
            results_data = {
                "net_profit": round(r.net_profit, 2),
                "total_return_pct": round(r.total_return_pct, 2),
                "total_trades": r.total_trades,
                "win_rate": round(r.win_rate, 1),
                "profit_factor": round(r.profit_factor, 2),
                "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                "max_drawdown_abs": round(r.max_drawdown_abs, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 2),
                "avg_trade": round(r.avg_trade, 2),
                "avg_win": round(r.avg_win, 2),
                "avg_loss": round(r.avg_loss, 2),
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "long_trades": r.long_trades,
                "short_trades": r.short_trades,
                "expectancy": round(r.expectancy, 2),
                "initial_capital": round(r.initial_capital, 2),
                "final_equity": round(r.final_equity, 2),
            }

        html = self._build_lightweight_html(
            title=title,
            candles_json=json.dumps(candles),
            vol_json=json.dumps(vol_data),
            markers_json=json.dumps(markers),
            equity_json=json.dumps(equity_data),
            cvd_json=json.dumps(cvd_data),
            results_json=json.dumps(results_data),
            height=height,
            show_volume=show_volume,
            show_equity=bool(equity_data),
            show_cvd=bool(cvd_data),
        )

        if save_path:
            with open(save_path, "w") as f:
                f.write(html)
            print(f"Chart saved to {save_path}")

        return html

    def _build_lightweight_html(
        self,
        title: str,
        candles_json: str,
        vol_json: str,
        markers_json: str,
        equity_json: str,
        cvd_json: str,
        results_json: str,
        height: int,
        show_volume: bool,
        show_equity: bool,
        show_cvd: bool,
    ) -> str:
        # Calculate pane heights
        sub_panes = []
        if show_volume:
            sub_panes.append(("volume", 15))
        if show_equity:
            sub_panes.append(("equity", 15))
        if show_cvd:
            sub_panes.append(("cvd", 15))

        sub_total = sum(p[1] for p in sub_panes)
        price_pct = 100 - sub_total

        pane_divs = "".join(
            '<div id="' + name + '-pane" class="chart-pane sub-pane" style="position:relative;">'
            '<div class="pane-label">' + name.upper() + '</div></div>'
            for name, _ in sub_panes
        )

        # Build optional JS sections
        vol_js = ""
        if show_volume:
            vol_js = (
                "const volChart = LightweightCharts.createChart("
                "document.getElementById('volume-pane'),"
                "{ ...commonOpts, height: document.getElementById('volume-pane').clientHeight }"
                ");\n"
                "const volSeries = volChart.addSeries(LightweightCharts.HistogramSeries, {"
                "priceFormat: { type: 'volume' },"
                "});\n"
                "volSeries.setData(volData);\n"
                "charts.push(volChart);\n"
            )

        eq_js = ""
        if show_equity:
            eq_js = (
                "const eqChart = LightweightCharts.createChart("
                "document.getElementById('equity-pane'),"
                "{ ...commonOpts, height: document.getElementById('equity-pane').clientHeight }"
                ");\n"
                "const eqSeries = eqChart.addSeries(LightweightCharts.LineSeries, {"
                "color: '" + self.TV_BLUE + "', lineWidth: 2,"
                "});\n"
                "eqSeries.setData(equityData);\n"
                "charts.push(eqChart);\n"
            )

        cvd_js = ""
        if show_cvd:
            cvd_js = (
                "const cvdChart = LightweightCharts.createChart("
                "document.getElementById('cvd-pane'),"
                "{ ...commonOpts, height: document.getElementById('cvd-pane').clientHeight }"
                ");\n"
                "const cvdSeries = cvdChart.addSeries(LightweightCharts.LineSeries, {"
                "color: '" + self.TV_PURPLE + "', lineWidth: 2,"
                "});\n"
                "cvdSeries.setData(cvdData);\n"
                "charts.push(cvdChart);\n"
            )

        resize_vol = "volChart.applyOptions({ width: document.getElementById('volume-pane').clientWidth, height: document.getElementById('volume-pane').clientHeight });" if show_volume else ""
        resize_eq = "eqChart.applyOptions({ width: document.getElementById('equity-pane').clientWidth, height: document.getElementById('equity-pane').clientHeight });" if show_equity else ""
        resize_cvd = "cvdChart.applyOptions({ width: document.getElementById('cvd-pane').clientWidth, height: document.getElementById('cvd-pane').clientHeight });" if show_cvd else ""

        BG = self.TV_DARK_BG
        GRID = self.TV_GRID
        TEXT = self.TV_TEXT
        GREEN = self.TV_GREEN
        RED = self.TV_RED
        CANDLE = self.TV_CANDLE
        SCALE_COLOR = "#b2b5be"

        return (
            '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
            '<title>' + title + '</title>\n'
            '<script src="https://unpkg.com/lightweight-charts@5.0.3/dist/lightweight-charts.standalone.production.js"></script>\n'
            '<style>\n'
            '  * { margin: 0; padding: 0; box-sizing: border-box; }\n'
            '  body { background: ' + BG + '; color: ' + TEXT + '; font-family: "Trebuchet MS", sans-serif; overflow: hidden; }\n'
            '  #header { padding: 8px 16px; font-size: 14px; color: ' + TEXT + '; background: ' + BG + '; border-bottom: 1px solid ' + GRID + '; }\n'
            '  #header h1 { font-size: 16px; font-weight: 600; display: inline; }\n'
            '  #header .stats { font-size: 12px; color: #787b86; margin-left: 16px; }\n'
            '  .chart-pane { width: 100%; border-bottom: 1px solid ' + GRID + '; }\n'
            '  #price-pane { height: ' + str(price_pct) + 'vh; }\n'
            '  .sub-pane { height: 15vh; }\n'
            '  .pane-label { position: absolute; top: 4px; left: 8px; font-size: 11px; color: #787b86; z-index: 10; pointer-events: none; }\n'
            '  #results-toggle { position: fixed; top: 8px; right: 16px; z-index: 100; background: #2a2e39; border: 1px solid #363a45; color: ' + TEXT + '; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: "Trebuchet MS", sans-serif; }\n'
            '  #results-toggle:hover { background: #363a45; }\n'
            '  #results-panel { position: fixed; top: 36px; right: 16px; z-index: 99; background: rgba(30,34,45,0.95); border: 1px solid #363a45; border-radius: 6px; padding: 12px 16px; font-size: 12px; color: ' + TEXT + '; min-width: 260px; max-height: 80vh; overflow-y: auto; display: none; }\n'
            '  #results-panel.visible { display: block; }\n'
            '  #results-panel h2 { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #fff; border-bottom: 1px solid #363a45; padding-bottom: 6px; }\n'
            '  #results-panel .section { margin-bottom: 10px; }\n'
            '  #results-panel .section-title { font-size: 11px; color: #787b86; text-transform: uppercase; margin-bottom: 4px; }\n'
            '  #results-panel .row { display: flex; justify-content: space-between; padding: 2px 0; }\n'
            '  #results-panel .label { color: #787b86; }\n'
            '  #results-panel .value { font-weight: 500; }\n'
            '  #results-panel .positive { color: ' + GREEN + '; }\n'
            '  #results-panel .negative { color: ' + RED + '; }\n'
            '</style>\n</head>\n<body>\n'
            '<div id="header">\n  <h1>' + title + '</h1>\n'
            '  <span class="stats" id="stats"></span>\n</div>\n'
            '<button id="results-toggle" onclick="document.getElementById(\'results-panel\').classList.toggle(\'visible\')">Results</button>\n'
            '<div id="results-panel"></div>\n'
            '<div id="price-pane" class="chart-pane" style="position:relative;">\n'
            '  <div class="pane-label">Price</div>\n</div>\n'
            + pane_divs +
            '\n<script>\n'
            '(function() {\n'
            '  const candleData = ' + candles_json + ';\n'
            '  const volData = ' + vol_json + ';\n'
            '  const markerData = ' + markers_json + ';\n'
            '  const equityData = ' + equity_json + ';\n'
            '  const cvdData = ' + cvd_json + ';\n'
            '  const results = ' + results_json + ';\n'
            '\n'
            '  // Populate results panel\n'
            '  if (results && results.total_trades) {\n'
            '    const fmt = (v) => v.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});\n'
            '    const cls = (v) => v >= 0 ? "positive" : "negative";\n'
            '    const sign = (v) => v >= 0 ? "+" : "";\n'
            '    document.getElementById("results-panel").innerHTML = \'<h2>Backtest Results</h2>\'\n'
            '      + \'<div class="section"><div class="section-title">Overview</div>\'\n'
            '      + \'<div class="row"><span class="label">Net Profit</span><span class="value \' + cls(results.net_profit) + \'">$\' + fmt(results.net_profit) + \' (\' + sign(results.total_return_pct) + results.total_return_pct.toFixed(2) + \'%)</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Initial Capital</span><span class="value">$\' + fmt(results.initial_capital) + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Final Equity</span><span class="value">$\' + fmt(results.final_equity) + \'</span></div>\'\n'
            '      + \'</div>\'\n'
            '      + \'<div class="section"><div class="section-title">Trades</div>\'\n'
            '      + \'<div class="row"><span class="label">Total Trades</span><span class="value">\' + results.total_trades + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Win Rate</span><span class="value">\' + results.win_rate.toFixed(1) + \'%</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Winners / Losers</span><span class="value \' + cls(results.winning_trades - results.losing_trades) + \'">\' + results.winning_trades + \' / \' + results.losing_trades + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Long / Short</span><span class="value">\' + results.long_trades + \' / \' + results.short_trades + \'</span></div>\'\n'
            '      + \'</div>\'\n'
            '      + \'<div class="section"><div class="section-title">Performance</div>\'\n'
            '      + \'<div class="row"><span class="label">Profit Factor</span><span class="value">\' + results.profit_factor.toFixed(2) + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Expectancy</span><span class="value \' + cls(results.expectancy) + \'">$\' + fmt(results.expectancy) + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Avg Trade</span><span class="value \' + cls(results.avg_trade) + \'">$\' + fmt(results.avg_trade) + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Avg Win</span><span class="value positive">$\' + fmt(results.avg_win) + \'</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Avg Loss</span><span class="value negative">$\' + fmt(results.avg_loss) + \'</span></div>\'\n'
            '      + \'</div>\'\n'
            '      + \'<div class="section"><div class="section-title">Risk</div>\'\n'
            '      + \'<div class="row"><span class="label">Max Drawdown</span><span class="value negative">$\' + fmt(results.max_drawdown_abs) + \' (\' + results.max_drawdown_pct.toFixed(2) + \'%)</span></div>\'\n'
            '      + \'<div class="row"><span class="label">Sharpe Ratio</span><span class="value">\' + results.sharpe_ratio.toFixed(2) + \'</span></div>\'\n'
            '      + \'</div>\';\n'
            '    document.getElementById("results-panel").classList.add("visible");\n'
            '  }\n'
            '\n'
            '  const BG = "' + BG + '";\n'
            '  const GRID = "' + GRID + '";\n'
            '  const TEXT = "' + TEXT + '";\n'
            '\n'
            '  const commonOpts = {\n'
            '    layout: { background: { color: BG }, textColor: TEXT, fontFamily: "Trebuchet MS" },\n'
            '    grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },\n'
            '    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },\n'
            '    timeScale: { timeVisible: true, secondsVisible: false, borderColor: "' + SCALE_COLOR + '" },\n'
            '    rightPriceScale: { borderColor: "' + SCALE_COLOR + '" },\n'
            '  };\n'
            '\n'
            '  // --- Price Chart ---\n'
            '  const priceChart = LightweightCharts.createChart(\n'
            '    document.getElementById("price-pane"),\n'
            '    { ...commonOpts, height: document.getElementById("price-pane").clientHeight }\n'
            '  );\n'
            '  const candleSeries = priceChart.addSeries(LightweightCharts.CandlestickSeries, {\n'
            '    upColor: "' + CANDLE + '",\n'
            '    downColor: "' + CANDLE + '",\n'
            '    borderVisible: true,\n'
            '    borderUpColor: "' + CANDLE + '",\n'
            '    borderDownColor: "' + CANDLE + '",\n'
            '    wickUpColor: "' + CANDLE + '",\n'
            '    wickDownColor: "' + CANDLE + '",\n'
            '  });\n'
            '  candleSeries.setData(candleData);\n'
            '\n'
            '  // Markers\n'
            '  if (markerData.length > 0) {\n'
            '    LightweightCharts.createSeriesMarkers(candleSeries, markerData);\n'
            '  }\n'
            '\n'
            '  // Stats in header\n'
            '  if (candleData.length > 0) {\n'
            '    const first = candleData[0];\n'
            '    const last = candleData[candleData.length - 1];\n'
            '    const d1 = new Date(first.time * 1000).toLocaleDateString();\n'
            '    const d2 = new Date(last.time * 1000).toLocaleDateString();\n'
            '    document.getElementById("stats").textContent =\n'
            '      candleData.length.toLocaleString() + " bars | " + d1 + " \\u2014 " + d2 +\n'
            '      " | " + markerData.filter(m => m.shape === "arrowUp" || m.shape === "arrowDown").length + " trades";\n'
            '  }\n'
            '\n'
            '  const charts = [priceChart];\n'
            '\n'
            + vol_js + eq_js + cvd_js +
            '\n'
            '  // Sync all charts time scales\n'
            '  function syncCharts(sourceChart) {\n'
            '    const timeRange = sourceChart.timeScale().getVisibleLogicalRange();\n'
            '    if (timeRange !== null) {\n'
            '      charts.forEach(c => {\n'
            '        if (c !== sourceChart) {\n'
            '          c.timeScale().setVisibleLogicalRange(timeRange);\n'
            '        }\n'
            '      });\n'
            '    }\n'
            '  }\n'
            '\n'
            '  charts.forEach(chart => {\n'
            '    chart.timeScale().subscribeVisibleLogicalRangeChange(() => syncCharts(chart));\n'
            '  });\n'
            '\n'
            '  // Sync crosshair\n'
            '  charts.forEach((chart, idx) => {\n'
            '    chart.subscribeCrosshairMove(param => {\n'
            '      if (!param || !param.time) return;\n'
            '      charts.forEach((other, oidx) => {\n'
            '        if (idx !== oidx) {\n'
            '          other.setCrosshairPosition(undefined, undefined, other.timeScale());\n'
            '        }\n'
            '      });\n'
            '    });\n'
            '  });\n'
            '\n'
            '  // Show last ~500 bars initially (about half a trading day) so candles are visible\n'
            '  var totalBars = candleData.length;\n'
            '  var visibleBars = Math.min(500, totalBars);\n'
            '  priceChart.timeScale().setVisibleLogicalRange({ from: totalBars - visibleBars, to: totalBars });\n'
            '\n'
            '  // Resize handler\n'
            '  window.addEventListener("resize", () => {\n'
            '    const pricePaneEl = document.getElementById("price-pane");\n'
            '    priceChart.applyOptions({ width: pricePaneEl.clientWidth, height: pricePaneEl.clientHeight });\n'
            '    ' + resize_vol + '\n'
            '    ' + resize_eq + '\n'
            '    ' + resize_cvd + '\n'
            '  });\n'
            '})();\n'
            '</script>\n</body>\n</html>'
        )

    def plot_performance_dashboard(
        self,
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """
        Create a performance dashboard with Plotly:
        - Equity curve
        - Drawdown chart
        - P&L distribution
        - Win/Loss breakdown
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for the performance dashboard: pip install plotly")

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
