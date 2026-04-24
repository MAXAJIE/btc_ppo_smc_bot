"""
logger.py
──────────
Trade journal and equity curve tracker.
Writes to CSV and provides summary stats.

Changes from original (backward-compatible additions only)
----------------------------------------------------------
1. __init__ accepts optional `run_id` kwarg — appended to filenames so
   concurrent runs (testnet, live, real) don't overwrite each other.
   If omitted the behaviour is identical to the original.

2. log_trade() accepts the simplified kwargs our callers pass:
     pnl_pct      – fraction (e.g. 0.012), converted to % internally
     hold_steps   – alias for duration_bars
     qty          – defaults to 0.0 if not supplied
     leverage     – defaults to 3 if not supplied
     pnl_usdt     – defaults to pnl_pct * 1000 estimate if not supplied
     entry_reason / exit_reason – default to ""

3. log_equity() accepts optional `reward` kwarg (silently ignored by the
   CSV since the original schema has no reward column, but callers won't crash).

4. summary() added as an alias for get_stats() so both names work.
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

    def __init__(self, log_dir: str = None, run_id: str = None):
        log_dir = Path(log_dir or os.getenv("LOG_PATH", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        # If run_id given, include it in filename so runs don't collide
        prefix  = f"{run_id}_" if run_id else ""
        self.trade_csv  = log_dir / f"{prefix}trades_{ts}.csv"
        self.equity_csv = log_dir / f"{prefix}equity_{ts}.csv"

        self._trade_headers = [
            "timestamp", "episode", "step", "side",
            "entry_price", "exit_price", "qty", "leverage",
            "pnl_usdt", "pnl_pct", "duration_bars",
            "entry_reason", "exit_reason", "equity", "drawdown",
        ]
        self._equity_headers = ["timestamp", "episode", "step", "equity", "drawdown"]

        self._init_csv(self.trade_csv,  self._trade_headers)
        self._init_csv(self.equity_csv, self._equity_headers)

        self.trades:        list  = []
        self.equity_curve:  list  = []
        self.peak_equity:   float = 0.0
        self.current_drawdown: float = 0.0

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────

    def log_trade(
        self,
        *,
        episode:       int,
        step:          int,
        side:          str,
        entry_price:   float,
        exit_price:    float,
        equity:        float,
        # ── original required fields with defaults for simplified callers ──
        qty:           float = 0.0,
        leverage:      int   = 3,
        pnl_usdt:      Optional[float] = None,
        # ── pnl_pct: accept fraction (0.012) OR percent (1.2) ──────────────
        pnl_pct:       float = 0.0,
        # ── aliases ─────────────────────────────────────────────────────────
        duration_bars: Optional[int] = None,
        hold_steps:    Optional[int] = None,   # alias for duration_bars
        entry_reason:  str = "",
        exit_reason:   str = "",
    ):
        # Resolve aliases
        dur = duration_bars if duration_bars is not None else (hold_steps or 0)

        # Normalise pnl_pct: if caller passed a fraction like 0.012, convert
        # to percent (1.2). Heuristic: if abs value < 1.0 treat as fraction.
        pnl_pct_pct = pnl_pct * 100.0 if abs(pnl_pct) < 1.0 else pnl_pct

        # Estimate pnl_usdt from pnl_pct if not supplied
        if pnl_usdt is None:
            pnl_usdt = equity * (pnl_pct_pct / 100.0)

        drawdown = self._update_drawdown(equity)

        row = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "episode":       episode,
            "step":          step,
            "side":          side,
            "entry_price":   round(entry_price,  2),
            "exit_price":    round(exit_price,   2),
            "qty":           round(qty,          5),
            "leverage":      leverage,
            "pnl_usdt":      round(pnl_usdt,     4),
            "pnl_pct":       round(pnl_pct_pct,  4),
            "duration_bars": dur,
            "entry_reason":  entry_reason,
            "exit_reason":   exit_reason,
            "equity":        round(equity,        2),
            "drawdown":      round(drawdown * 100, 4),
        }
        self.trades.append(row)
        self._write_csv_row(self.trade_csv, self._trade_headers, row)

    def log_equity(
        self,
        episode: int,
        step:    int,
        equity:  float,
        reward:  float = 0.0,   # accepted but not written (original schema has no reward col)
    ):
        drawdown = self._update_drawdown(equity)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode":   episode,
            "step":      step,
            "equity":    round(equity,        2),
            "drawdown":  round(drawdown * 100, 4),
        }
        self.equity_curve.append(row)
        # Batch-write every 100 steps (original behaviour)
        if len(self.equity_curve) % 100 == 0:
            self._flush_equity()

    def _update_drawdown(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            self.current_drawdown = max(
                0.0, (self.peak_equity - equity) / self.peak_equity
            )
        return self.current_drawdown

    def _flush_equity(self):
        if not self.equity_curve:
            return
        for row in self.equity_curve[-100:]:
            self._write_csv_row(self.equity_csv, self._equity_headers, row)

    # ─────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return summary performance stats from all logged trades."""
        if not self.trades:
            return {}

        df   = pd.DataFrame(self.trades)
        wins = df[df["pnl_pct"] > 0]
        losses = df[df["pnl_pct"] <= 0]

        total        = len(df)
        win_rate     = len(wins) / total if total > 0 else 0.0
        avg_win      = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0.0
        avg_loss     = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
        profit_factor = (
            float(wins["pnl_usdt"].sum()) / abs(float(losses["pnl_usdt"].sum()))
            if losses["pnl_usdt"].sum() != 0 else np.inf
        )

        equity_df = pd.DataFrame(self.equity_curve)
        max_dd    = float(equity_df["drawdown"].max()) if len(equity_df) > 0 else 0.0

        if len(df) > 1:
            returns = df["pnl_pct"].values / 100.0
            sharpe  = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(288 * 252)
        else:
            sharpe  = 0.0

        return {
            "total_trades":    total,
            "win_rate":        round(win_rate * 100, 2),
            "avg_win_pct":     round(avg_win,   4),
            "avg_loss_pct":    round(avg_loss,  4),
            "profit_factor":   round(profit_factor, 3),
            "max_drawdown_pct":round(max_dd,    4),
            "sharpe_ratio":    round(sharpe,    3),
            "total_pnl_usdt":  round(float(df["pnl_usdt"].sum()), 2),
        }

    def summary(self) -> dict:
        """Alias for get_stats() — used by main_live.py / main_real.py."""
        return self.get_stats()

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
                csv.DictWriter(f, fieldnames=headers).writeheader()

    def _write_csv_row(self, path: Path, headers: list, row: dict):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writerow(row)