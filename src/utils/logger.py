"""
logger.py
──────────
Trade journal and equity curve tracker.
Writes to CSV and provides summary stats.
"""

import os
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Records every trade and generates stats.

    CSV columns:
    timestamp, episode, step, side, entry_price, exit_price,
    qty, leverage, pnl_usdt, pnl_pct, duration_bars,
    entry_reason (SMC tag), exit_reason, equity, drawdown
    """

    def __init__(self, log_dir: str = None):
        log_dir = Path(log_dir or os.getenv("LOG_PATH", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trade_csv = log_dir / f"trades_{ts}.csv"
        self.equity_csv = log_dir / f"equity_{ts}.csv"

        self._trade_headers = [
            "timestamp", "episode", "step", "side",
            "entry_price", "exit_price", "qty", "leverage",
            "pnl_usdt", "pnl_pct", "duration_bars",
            "entry_reason", "exit_reason", "equity", "drawdown",
        ]
        self._equity_headers = ["timestamp", "episode", "step", "equity", "drawdown"]

        self._init_csv(self.trade_csv, self._trade_headers)
        self._init_csv(self.equity_csv, self._equity_headers)

        self.trades: list = []
        self.equity_curve: list = []
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0

    # ─────────────────────────────────────────────────────────────────

    def log_trade(
        self,
        *,
        episode: int,
        step: int,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        leverage: int,
        pnl_usdt: float,
        pnl_pct: float,
        duration_bars: int,
        entry_reason: str = "",
        exit_reason: str = "",
        equity: float,
    ):
        drawdown = self._update_drawdown(equity)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "step": step,
            "side": side,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "qty": round(qty, 5),
            "leverage": leverage,
            "pnl_usdt": round(pnl_usdt, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "duration_bars": duration_bars,
            "entry_reason": entry_reason,
            "exit_reason": exit_reason,
            "equity": round(equity, 2),
            "drawdown": round(drawdown * 100, 4),
        }
        self.trades.append(row)
        self._write_csv_row(self.trade_csv, self._trade_headers, row)

    def log_equity(self, episode: int, step: int, equity: float):
        drawdown = self._update_drawdown(equity)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "step": step,
            "equity": round(equity, 2),
            "drawdown": round(drawdown * 100, 4),
        }
        self.equity_curve.append(row)
        # Don't write every step to CSV (too slow) — batch write every 100 steps
        if len(self.equity_curve) % 100 == 0:
            self._flush_equity()

    def _update_drawdown(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            self.current_drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
        return self.current_drawdown

    def _flush_equity(self):
        if not self.equity_curve:
            return
        to_write = self.equity_curve[-100:]
        for row in to_write:
            self._write_csv_row(self.equity_csv, self._equity_headers, row)

    # ─────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return summary performance stats from all logged trades."""
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)
        wins = df[df["pnl_pct"] > 0]
        losses = df[df["pnl_pct"] <= 0]

        total = len(df)
        win_rate = len(wins) / total if total > 0 else 0.0
        avg_win = float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
        profit_factor = (
            float(wins["pnl_usdt"].sum()) / abs(float(losses["pnl_usdt"].sum()))
            if losses["pnl_usdt"].sum() != 0 else np.inf
        )

        equity_df = pd.DataFrame(self.equity_curve)
        max_dd = float(equity_df["drawdown"].max()) if len(equity_df) > 0 else 0.0

        # Sharpe (annualised, assuming 5m bars)
        if len(df) > 1:
            returns = df["pnl_pct"].values / 100.0
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(288 * 252)
        else:
            sharpe = 0.0

        return {
            "total_trades": total,
            "win_rate": round(win_rate * 100, 2),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_pct": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 3),
            "total_pnl_usdt": round(float(df["pnl_usdt"].sum()), 2),
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "═" * 50)
        print("  TRADE STATISTICS")
        print("═" * 50)
        for k, v in stats.items():
            print(f"  {k:<25} {v}")
        print("═" * 50 + "\n")

    # ─────────────────────────────────────────────────────────────────

    def _init_csv(self, path: Path, headers: list):
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def _write_csv_row(self, path: Path, headers: list, row: dict):
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)
