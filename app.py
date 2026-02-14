"""
PHANTOM CVD Backtester - Streamlit Web App

Interactive web interface for running backtests with the PHANTOM CVD
Divergence Strategy on ES futures data.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.performance import PerformanceReport
from backtester.visualization import BacktestChart
from backtester.risk_manager import (
    RiskConfig,
    StopLossMode,
    TakeProfitMode,
    TrailingMode,
    PositionSizeMode,
)
from strategies.phantom_cvd import PhantomCVDStrategy

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="PHANTOM CVD Backtester",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ───────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1d29;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.8em; font-weight: 700; }
    .metric-label { font-size: 0.85em; color: #9ea3b0; }
    .profit { color: #26a69a; }
    .loss { color: #ef5350; }
    div[data-testid="stSidebar"] { background-color: #131722; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = "white"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def color_pnl(val):
    """Color positive/negative values."""
    if isinstance(val, (int, float)):
        return "#26a69a" if val >= 0 else "#ef5350"
    return "white"


@st.cache_data
def load_csv_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load and cache CSV data from uploaded file."""
    loader = DataLoader()
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        data = loader.load_csv(tmp_path)
    finally:
        os.unlink(tmp_path)
    return data


@st.cache_data
def load_databento_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load and cache Databento DBN data from uploaded file."""
    loader = DataLoader()
    import tempfile, os
    suffix = ".dbn.zst" if ".dbn.zst" in filename else ".zst"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        data = loader.load_databento(tmp_path)
    finally:
        os.unlink(tmp_path)
    return data


def run_backtest(data, strategy_params, risk_config, initial_capital):
    """Run backtest and return report + engine."""
    strategy = PhantomCVDStrategy(params=strategy_params)
    engine = BacktestEngine(initial_capital=initial_capital, risk_config=risk_config)
    report = engine.run(data, strategy, verbose=False)
    return report, engine, strategy


# ── Sidebar: Data Upload ─────────────────────────────────────────────

st.sidebar.title("👻 PHANTOM CVD")
st.sidebar.markdown("---")

st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload data (CSV or Databento .dbn.zst)",
    type=["csv", "zst"],
    help="Supports: standard OHLCV CSV, TradingView exports with CVD, Databento DBN files (.dbn.zst)"
)

use_sample = st.sidebar.checkbox("Use sample data instead", value=uploaded_file is None)

if use_sample:
    sample_bars = st.sidebar.slider("Sample bars", 1000, 20000, 5000, 1000)

# ── Sidebar: Strategy Parameters ─────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy")

with st.sidebar.expander("Swing Detection", expanded=True):
    pivot_left = st.slider("Pivot Left Bars", 3, 30, 10)
    pivot_right = st.slider("Pivot Right Bars", 1, 5, 1)
    max_level_age = st.slider("Max Level Age (bars)", 50, 500, 100, 25)
    max_levels = st.slider("Max Tracked Levels", 5, 50, 30)

with st.sidebar.expander("Sweep Detection", expanded=True):
    sweep_ticks = st.slider("Sweep Threshold (ticks)", 1, 10, 3)
    tick_size = st.number_input("Tick Size", value=0.25, step=0.05, format="%.2f")

with st.sidebar.expander("CVD Settings", expanded=False):
    cvd_smooth = st.slider("CVD Smoothing (EMA)", 1, 20, 1)
    cvd_anchor = st.slider("CVD Anchor (bars)", 0, 1000, 390, 10,
                           help="Reset CVD accumulation. 390 = 1 session")
    min_strength = st.slider("Min Divergence Strength", 0.0, 1.0, 0.5, 0.05)
    strength_period = st.slider("Strength Period", 10, 200, 30)

with st.sidebar.expander("Signal Filters", expanded=False):
    signal_cooldown = st.slider("Signal Cooldown (bars)", 1, 30, 3)
    use_momentum = st.checkbox("Use CVD Momentum Filter", value=False)
    momentum_bars = st.slider("Momentum Bars", 2, 20, 5) if use_momentum else 5
    momentum_tol = st.slider("Momentum Tolerance", 0.0, 0.1, 0.02, 0.005) if use_momentum else 0.02
    allow_longs = st.checkbox("Allow Longs", value=True)
    allow_shorts = st.checkbox("Allow Shorts", value=True)

# ── Sidebar: Risk Parameters ─────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Management")

with st.sidebar.expander("Position Sizing", expanded=True):
    pos_mode = st.selectbox("Sizing Mode", ["Risk Per Trade", "% Equity", "Fixed Contracts"])
    initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)

    if pos_mode == "Risk Per Trade":
        risk_pct = st.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25)
    elif pos_mode == "% Equity":
        equity_pct = st.slider("Equity %", 1.0, 100.0, 100.0, 1.0)
    else:
        fixed_contracts = st.number_input("Contracts", value=1, min_value=1)

with st.sidebar.expander("Stop Loss", expanded=True):
    use_sl = st.checkbox("Enable Stop Loss", value=True)
    sl_mode_str = st.selectbox("SL Mode", ["ATR", "Fixed Points", "Percentage", "Swept Level"])
    sl_mode_map = {"ATR": StopLossMode.ATR, "Fixed Points": StopLossMode.FIXED_POINTS,
                   "Percentage": StopLossMode.PERCENTAGE, "Swept Level": StopLossMode.SWEPT_LEVEL}

    if sl_mode_str == "ATR":
        sl_atr_mult = st.slider("SL ATR Multiplier", 0.5, 5.0, 3.0, 0.25)
        sl_atr_period = st.slider("ATR Period", 5, 30, 14)
    elif sl_mode_str == "Fixed Points":
        sl_fixed = st.number_input("SL Points", value=20.0, step=1.0)
    elif sl_mode_str == "Percentage":
        sl_pct = st.number_input("SL %", value=1.0, step=0.1)

with st.sidebar.expander("Take Profit", expanded=True):
    use_tp = st.checkbox("Enable Take Profit", value=True)
    tp_mode_str = st.selectbox("TP Mode", ["Risk:Reward", "ATR", "Fixed Points", "Percentage"])
    tp_mode_map = {"Risk:Reward": TakeProfitMode.RISK_REWARD, "ATR": TakeProfitMode.ATR,
                   "Fixed Points": TakeProfitMode.FIXED_POINTS, "Percentage": TakeProfitMode.PERCENTAGE}

    if tp_mode_str == "Risk:Reward":
        tp_rr = st.slider("R:R Ratio", 0.5, 5.0, 3.0, 0.25)
    elif tp_mode_str == "ATR":
        tp_atr_mult = st.number_input("TP ATR Mult", value=3.0, step=0.5)
    elif tp_mode_str == "Fixed Points":
        tp_fixed = st.number_input("TP Points", value=40.0, step=1.0)
    elif tp_mode_str == "Percentage":
        tp_pct = st.number_input("TP %", value=2.0, step=0.1)

with st.sidebar.expander("Trailing / Break Even", expanded=False):
    use_trailing = st.checkbox("Enable Trailing Stop", value=False)
    if use_trailing:
        trail_mode_str = st.selectbox("Trail Mode", ["ATR", "Fixed Points", "Percentage"])
        trail_atr_mult = st.slider("Trail ATR Mult", 0.5, 3.0, 1.5, 0.25)

    use_be = st.checkbox("Enable Break Even", value=False)
    if use_be:
        be_trigger = st.number_input("BE Trigger (points)", value=15.0, step=1.0)
        be_offset = st.number_input("BE Offset", value=1.0, step=0.5)

with st.sidebar.expander("Session & Limits", expanded=False):
    use_session = st.checkbox("Session Filter", value=False)
    if use_session:
        scol1, scol2 = st.columns(2)
        session_start_h = scol1.number_input("Start Hour", value=9, min_value=0, max_value=23)
        session_start_m = scol2.number_input("Start Min", value=30, min_value=0, max_value=59)
        scol3, scol4 = st.columns(2)
        session_end_h = scol3.number_input("End Hour", value=16, min_value=0, max_value=23)
        session_end_m = scol4.number_input("End Min", value=0, min_value=0, max_value=59)
        close_eod = st.checkbox("Close at Session End", value=False)

    use_max_hold = st.checkbox("Max Hold Time", value=False)
    if use_max_hold:
        max_hold_bars = st.slider("Max Bars", 10, 500, 50)

with st.sidebar.expander("Costs (ES Futures)", expanded=False):
    commission = st.number_input("Commission/Contract ($)", value=1.24, step=0.01, format="%.2f")
    slippage = st.number_input("Slippage (ticks)", value=1, min_value=0)
    point_value = st.number_input("Point Value ($)", value=50.0, step=5.0)

# ── Build Configs from Sidebar ────────────────────────────────────────

strategy_params = {
    "pivot_left": pivot_left,
    "pivot_right": pivot_right,
    "max_level_age": max_level_age,
    "max_levels": max_levels,
    "sweep_ticks": sweep_ticks,
    "tick_size": tick_size,
    "cvd_smooth": cvd_smooth,
    "cvd_anchor_bars": cvd_anchor,
    "min_strength": min_strength,
    "strength_period": strength_period,
    "signal_cooldown": signal_cooldown,
    "use_momentum_filter": use_momentum,
    "momentum_bars": momentum_bars,
    "momentum_tolerance": momentum_tol,
    "allow_longs": allow_longs,
    "allow_shorts": allow_shorts,
}

# Build RiskConfig
pos_mode_map = {
    "Risk Per Trade": PositionSizeMode.RISK_PER_TRADE,
    "% Equity": PositionSizeMode.PERCENT_EQUITY,
    "Fixed Contracts": PositionSizeMode.FIXED_CONTRACTS,
}

risk_config = RiskConfig(
    position_mode=pos_mode_map[pos_mode],
    risk_pct=risk_pct if pos_mode == "Risk Per Trade" else 1.0,
    equity_pct=equity_pct if pos_mode == "% Equity" else 100.0,
    fixed_contracts=fixed_contracts if pos_mode == "Fixed Contracts" else 1,
    use_stop_loss=use_sl,
    sl_mode=sl_mode_map[sl_mode_str],
    sl_atr_mult=sl_atr_mult if sl_mode_str == "ATR" else 2.0,
    sl_atr_period=sl_atr_period if sl_mode_str == "ATR" else 14,
    sl_fixed_points=sl_fixed if sl_mode_str == "Fixed Points" else 20.0,
    sl_percentage=sl_pct if sl_mode_str == "Percentage" else 1.0,
    use_take_profit=use_tp,
    tp_mode=tp_mode_map[tp_mode_str],
    tp_rr_ratio=tp_rr if tp_mode_str == "Risk:Reward" else 2.0,
    tp_atr_mult=tp_atr_mult if tp_mode_str == "ATR" else 3.0,
    tp_fixed_points=tp_fixed if tp_mode_str == "Fixed Points" else 40.0,
    tp_percentage=tp_pct if tp_mode_str == "Percentage" else 2.0,
    use_trailing=use_trailing,
    trail_mode=TrailingMode.ATR,
    trail_atr_mult=trail_atr_mult if use_trailing else 1.5,
    use_breakeven=use_be,
    be_trigger_points=be_trigger if use_be else 15.0,
    be_offset=be_offset if use_be else 1.0,
    use_session_filter=use_session if 'use_session' in dir() else False,
    session_start_hour=session_start_h if use_session else 9,
    session_start_minute=session_start_m if use_session else 30,
    session_end_hour=session_end_h if use_session else 16,
    session_end_minute=session_end_m if use_session else 0,
    close_at_session_end=close_eod if use_session else False,
    use_max_hold=use_max_hold,
    max_hold_bars=max_hold_bars if use_max_hold else 50,
    commission_per_contract=commission,
    slippage_ticks=slippage,
    tick_size=tick_size,
    point_value=point_value,
)

# ── Main Area ─────────────────────────────────────────────────────────

st.title("👻 PHANTOM CVD Backtester")

# Load data
data = None
if uploaded_file is not None and not use_sample:
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".zst") or fname.endswith(".dbn.zst"):
            data = load_databento_data(uploaded_file.getvalue(), uploaded_file.name)
            st.success(f"Loaded **{len(data):,}** bars from Databento file `{uploaded_file.name}`  "
                       f"({data.index[0]} → {data.index[-1]})")
        else:
            data = load_csv_data(uploaded_file.getvalue(), uploaded_file.name)
            st.success(f"Loaded **{len(data):,}** bars from `{uploaded_file.name}`  "
                       f"({data.index[0]} → {data.index[-1]})")
    except Exception as e:
        st.error(f"Error loading data: {e}")
elif use_sample:
    loader = DataLoader()
    data = loader.generate_sample_data(bars=sample_bars)
    st.info(f"Using **{sample_bars:,}** bars of synthetic ES data")

# Data preview
if data is not None:
    with st.expander("📊 Data Preview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bars", f"{len(data):,}")
        col2.metric("Price Range", f"{data['low'].min():.2f} - {data['high'].max():.2f}")
        col3.metric("Has Real CVD", "Yes" if "cvd" in data.columns and data["cvd"].notna().sum() > 0 else "No")
        col4.metric("Has Volume", "Yes" if data["volume"].nunique() > 1 else "Synthetic")
        st.dataframe(data.head(20), use_container_width=True)

# ── Run Button ────────────────────────────────────────────────────────

if data is not None:
    run_col1, run_col2 = st.columns([1, 4])
    run_clicked = run_col1.button("🚀 Run Backtest", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running backtest..."):
            t0 = time.time()
            report, engine, strategy = run_backtest(
                data, strategy_params, risk_config, float(initial_capital)
            )
            elapsed = time.time() - t0

        st.success(f"Backtest complete in {elapsed:.1f}s  |  {report.total_trades} trades generated")

        # Store in session state
        st.session_state["report"] = report
        st.session_state["engine"] = engine
        st.session_state["strategy"] = strategy
        st.session_state["data"] = data

# ── Results Display ───────────────────────────────────────────────────

if "report" in st.session_state:
    report = st.session_state["report"]
    engine = st.session_state["engine"]
    strategy = st.session_state["strategy"]
    bt_data = st.session_state["data"]

    st.markdown("---")

    # ── Key Metrics Row ───────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    pnl_color = "#26a69a" if report.net_profit >= 0 else "#ef5350"
    with m1:
        metric_card("Net Profit", f"${report.net_profit:,.2f}", pnl_color)
    with m2:
        metric_card("Win Rate", f"{report.win_rate:.1f}%",
                    "#26a69a" if report.win_rate >= 50 else "#ef5350")
    with m3:
        pf_color = "#26a69a" if report.profit_factor >= 1.0 else "#ef5350"
        metric_card("Profit Factor", f"{report.profit_factor:.2f}", pf_color)
    with m4:
        metric_card("Total Trades", str(report.total_trades), "#2196f3")
    with m5:
        metric_card("Max Drawdown", f"{report.max_drawdown_pct:.2f}%", "#ef5350")
    with m6:
        sharpe_color = "#26a69a" if report.sharpe_ratio > 0 else "#ef5350"
        metric_card("Sharpe Ratio", f"{report.sharpe_ratio:.2f}", sharpe_color)

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_chart, tab_dashboard, tab_trades, tab_report = st.tabs(
        ["📈 Chart", "📊 Dashboard", "📋 Trade Log", "📄 Full Report"]
    )

    # ── Chart Tab ─────────────────────────────────────────────────────
    with tab_chart:
        chart = BacktestChart(bt_data, report, engine.signals)
        fig = chart.plot_backtest(
            title="PHANTOM CVD Strategy",
            show_volume=True,
            show_equity=True,
            show_cvd=True,
            cvd_values=strategy.cvd,
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # ── Dashboard Tab ─────────────────────────────────────────────────
    with tab_dashboard:
        chart = BacktestChart(bt_data, report, engine.signals)
        dash_fig = chart.plot_performance_dashboard()
        st.plotly_chart(dash_fig, use_container_width=True)

        # Extra stats columns
        st.markdown("### Detailed Breakdown")
        dc1, dc2, dc3 = st.columns(3)

        with dc1:
            st.markdown("**Profit Analysis**")
            st.markdown(f"- Gross Profit: **${report.gross_profit:,.2f}**")
            st.markdown(f"- Gross Loss: **${report.gross_loss:,.2f}**")
            st.markdown(f"- Avg Win: **${report.avg_win:,.2f}**")
            st.markdown(f"- Avg Loss: **${report.avg_loss:,.2f}**")
            st.markdown(f"- Largest Win: **${report.largest_win:,.2f}**")
            st.markdown(f"- Largest Loss: **${report.largest_loss:,.2f}**")
            st.markdown(f"- Expectancy: **${report.expectancy:,.2f}**/trade")

        with dc2:
            st.markdown("**Long vs Short**")
            st.markdown(f"- Long Trades: **{report.long_trades}** (WR: {report.long_win_rate:.1f}%)")
            st.markdown(f"- Short Trades: **{report.short_trades}** (WR: {report.short_win_rate:.1f}%)")
            st.markdown(f"- Long P&L: **${report.long_net_profit:,.2f}**")
            st.markdown(f"- Short P&L: **${report.short_net_profit:,.2f}**")

        with dc3:
            st.markdown("**Risk Metrics**")
            st.markdown(f"- Max DD: **${report.max_drawdown_abs:,.2f}** ({report.max_drawdown_pct:.2f}%)")
            st.markdown(f"- Sharpe: **{report.sharpe_ratio:.2f}**")
            st.markdown(f"- Sortino: **{report.sortino_ratio:.2f}**")
            st.markdown(f"- Calmar: **{report.calmar_ratio:.2f}**")
            st.markdown(f"- Avg Bars Held: **{report.avg_bars_held:.1f}**")
            st.markdown(f"- Max Consec Wins: **{report.max_consecutive_wins}**")
            st.markdown(f"- Max Consec Losses: **{report.max_consecutive_losses}**")

    # ── Trade Log Tab ─────────────────────────────────────────────────
    with tab_trades:
        trades_df = report.to_dataframe()
        if not trades_df.empty:
            st.dataframe(
                trades_df.style.applymap(
                    lambda v: f"color: {'#26a69a' if v >= 0 else '#ef5350'}"
                    if isinstance(v, (int, float)) else "",
                    subset=["pnl", "pnl_pct"]
                ),
                use_container_width=True,
                height=400,
            )

            # Download button
            csv_buf = io.StringIO()
            trades_df.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download Trades CSV",
                csv_buf.getvalue(),
                "phantom_trades.csv",
                "text/csv",
            )
        else:
            st.warning("No trades generated with current parameters.")

    # ── Full Report Tab ───────────────────────────────────────────────
    with tab_report:
        st.code(report.summary(), language="text")

        # Exit reasons
        st.markdown("### Exit Reasons")
        if report.exit_reasons:
            reasons_df = pd.DataFrame(
                list(report.exit_reasons.items()),
                columns=["Reason", "Count"]
            )
            reasons_df["Pct"] = (reasons_df["Count"] / reasons_df["Count"].sum() * 100).round(1)
            st.dataframe(reasons_df, use_container_width=True, hide_index=True)

        # Costs
        st.markdown("### Costs")
        cc1, cc2 = st.columns(2)
        cc1.metric("Total Commission", f"${report.total_commission:,.2f}")
        cc2.metric("Total Slippage", f"${report.total_slippage:,.2f}")

else:
    if data is not None:
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#666;'>"
            "<h3>Configure parameters in the sidebar, then click 🚀 Run Backtest</h3>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#666;'>"
            "<h3>Upload a CSV file or enable sample data to get started</h3>"
            "</div>",
            unsafe_allow_html=True,
        )
