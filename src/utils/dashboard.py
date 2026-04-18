"""
dashboard.py
────────────
Streamlit live monitoring dashboard.

Run:
    streamlit run src/utils/dashboard.py -- --log-dir ./logs

Shows:
  • Live equity curve
  • Last 50 trades table
  • Win rate, profit factor, max drawdown, Sharpe
  • Current position & unrealized PnL
  • Feature importance (from PPO policy gradient norms)
"""

import os
import glob
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (read from CSV logs)
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_csvs(log_dir: str) -> dict:
    """Find the most recently modified trade + equity CSVs in log_dir."""
    trade_files = sorted(glob.glob(os.path.join(log_dir, "trades_*.csv")))
    equity_files = sorted(glob.glob(os.path.join(log_dir, "equity_*.csv")))

    return {
        "trades": trade_files[-1] if trade_files else None,
        "equity": equity_files[-1] if equity_files else None,
    }


@st.cache_data(ttl=10)  # refresh every 10 seconds
def load_trades(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_equity(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=equity_df["equity"],
        name="Equity (USDT)",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.05)",
    ))

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="USDT",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def drawdown_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=-equity_df["drawdown"],  # negate: drawdown shown as negative
        name="Drawdown %",
        line=dict(color="#ff4444", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 68, 68, 0.10)",
    ))

    fig.update_layout(
        title="Drawdown %",
        xaxis_title="Time",
        yaxis_title="Drawdown %",
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    pnls = trades_df["pnl_pct"].values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=40,
        name="PnL %",
        marker_color=["#00cc66" if p > 0 else "#ff4444" for p in pnls],
        opacity=0.8,
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title="PnL Distribution",
        xaxis_title="PnL %",
        yaxis_title="Count",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def trade_duration_chart(trades_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        trades_df.tail(200),
        x="duration_bars",
        y="pnl_pct",
        color="pnl_pct",
        color_continuous_scale=["#ff4444", "#888888", "#00cc66"],
        title="PnL vs Duration (last 200 trades)",
        template="plotly_dark",
        height=250,
        labels={"duration_bars": "Duration (5m bars)", "pnl_pct": "PnL %"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]

    total = len(trades_df)
    win_rate = len(wins) / total * 100 if total > 0 else 0

    avg_win = float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0
    avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0

    gross_profit = float(wins["pnl_usdt"].sum()) if len(wins) > 0 else 0
    gross_loss = abs(float(losses["pnl_usdt"].sum())) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    max_dd = float(equity_df["drawdown"].max()) if not equity_df.empty else 0

    returns = trades_df["pnl_pct"].values / 100.0
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(288 * 252) if len(returns) > 1 else 0

    return {
        "Total Trades": total,
        "Win Rate %": f"{win_rate:.1f}%",
        "Avg Win %": f"{avg_win:.3f}%",
        "Avg Loss %": f"{avg_loss:.3f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown %": f"{max_dd:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Total PnL (USDT)": f"{trades_df['pnl_usdt'].sum():.2f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard(log_dir: str = "./logs"):
    st.set_page_config(
        page_title="BTC PPO SMC Bot",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("₿ BTC PPO SMC Trading Bot — Live Monitor")
    st.caption(f"Log directory: `{log_dir}` | Auto-refreshes every 10s")

    # Auto-refresh
    st_autorefresh = st.empty()

    csvs = find_latest_csvs(log_dir)
    trades_df = load_trades(csvs["trades"])
    equity_df = load_equity(csvs["equity"])

    # ── Top metrics row ───────────────────────────────────────────────
    stats = compute_stats(trades_df, equity_df)

    if stats:
        cols = st.columns(len(stats))
        for i, (k, v) in enumerate(stats.items()):
            color = "normal"
            if "PnL" in k and float(v.replace("USDT", "").replace("%", "")) < 0:
                color = "inverse"
            cols[i].metric(k, v)
    else:
        st.info("No trades logged yet. Training may still be in the warm-up phase.")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        if not equity_df.empty:
            st.plotly_chart(equity_chart(equity_df), use_container_width=True)
            st.plotly_chart(drawdown_chart(equity_df), use_container_width=True)
        else:
            st.info("Waiting for equity data...")

    with col2:
        if not trades_df.empty:
            st.plotly_chart(pnl_distribution(trades_df), use_container_width=True)
            st.plotly_chart(trade_duration_chart(trades_df), use_container_width=True)
        else:
            st.info("Waiting for trade data...")

    st.divider()

    # ── Recent trades table ───────────────────────────────────────────
    st.subheader("Last 50 Trades")
    if not trades_df.empty:
        display_cols = [
            "timestamp", "side", "entry_price", "exit_price",
            "pnl_usdt", "pnl_pct", "duration_bars", "exit_reason"
        ]
        display_cols = [c for c in display_cols if c in trades_df.columns]
        last50 = trades_df.tail(50)[display_cols].copy()
        last50 = last50.sort_values("timestamp", ascending=False)

        # Colour-code PnL
        def style_pnl(val):
            if isinstance(val, (int, float)):
                color = "#00cc66" if val > 0 else "#ff4444" if val < 0 else ""
                return f"color: {color}"
            return ""

        st.dataframe(
            last50.style.applymap(style_pnl, subset=["pnl_usdt", "pnl_pct"]),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No trades yet.")

    # ── Raw log file paths ────────────────────────────────────────────
    with st.expander("Log file paths"):
        st.code(f"Trades:  {csvs['trades'] or 'not found'}")
        st.code(f"Equity:  {csvs['equity'] or 'not found'}")

    # Auto-refresh by re-running every 10s using streamlit's rerun
    time.sleep(10)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    args = parser.parse_args()

    run_dashboard(log_dir=args.log_dir)
