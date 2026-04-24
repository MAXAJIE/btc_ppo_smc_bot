"""
multi_tf_features.py  –  90-dim observation builder
====================================================

Improvements from code review
------------------------------
1. Replace repeated padding in the 15-dim 5m block
   Old: obs[10:15] = log_ret, mom3, mom10, ema9_dist, ema21_dist (repeated)
   New: obs[10:15] = rsi_norm, macd_hist_norm, hour_sin, hour_cos, day_sin
   PPO sees diverse information instead of the same 5 values twice.

2. Time features (sin/cos encoding)
   Crypto has strong intraday cycles (US open, Asia session, weekends).
   Using sin/cos encoding preserves the cyclical property:
     - hour_sin / hour_cos  → 24-hour cycle
     - day_sin              → 7-day week cycle
   These replace the 5 redundant repeated columns in the 15-dim block
   so the obs_dim stays at 90 with no env changes needed.

3. RSI normalised to [-1, 1]  (RSI - 50) / 50
   MACD histogram normalised by ATR

Observation layout (90-dim, unchanged):
  [00:15]  OHLCV + momentum + RSI + MACD + time  5m  (15)  ← improved
  [15:25]  OHLCV + momentum                       15m (10)
  [25:35]  OHLCV + momentum                       1h  (10)
  [35:43]  OHLCV + momentum                       4h  (8)
  [43:51]  OHLCV + momentum                       1d  (8)
  [51:59]  SMC features                            5m  (8)
  [59:67]  SMC features                            1h  (8)
  [67:73]  S&R levels                              1h  (6)
  [73:79]  AMT/Vol profile                         4h  (6)
  [79:83]  GARCH + Kelly                               (4)
  [83:90]  Position state                              (7)
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OBS_DIM = 90


class MultiTFFeatureBuilder:
    """
    Pre-loads all timeframe DataFrames and exposes a single build() call.

    Parameters
    ----------
    tf_data : dict[str, pd.DataFrame]
        Output of DataLoader.load().
    """

    def __init__(self, tf_data: Dict[str, pd.DataFrame]):
        self._tf  = tf_data
        self._mom: Dict[str, pd.DataFrame] = {
            tf: _add_momentum(df) for tf, df in tf_data.items()
        }
        logger.info("MultiTFFeatureBuilder initialised — TFs: %s", list(tf_data.keys()))

    def build(
        self,
        ts:          pd.Timestamp,
        smc_5m:      np.ndarray,   # (8,)
        smc_1h:      np.ndarray,   # (8,)
        snr_1h:      np.ndarray,   # (6,)
        amt_4h:      np.ndarray,   # (6,)
        garch_kelly: np.ndarray,   # (4,)
        position:    np.ndarray,   # (7,)
    ) -> np.ndarray:

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # 5m block: 15-dim with RSI, MACD, time features in last 5 slots
        obs[0:15]  = _block_5m(self._mom["5m"], ts)
        obs[15:25] = _block(self._mom["15m"], ts, 10)
        obs[25:35] = _block(self._mom["1h"],  ts, 10)
        obs[35:43] = _block(self._mom["4h"],  ts, 8)
        obs[43:51] = _block(self._mom["1d"],  ts, 8)

        obs[51:59] = _pad(smc_5m,      8)
        obs[59:67] = _pad(smc_1h,      8)
        obs[67:73] = _pad(snr_1h,      6)
        obs[73:79] = _pad(amt_4h,      6)
        obs[79:83] = _pad(garch_kelly, 4)
        obs[83:90] = _pad(position,    7)

        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)


# ---------------------------------------------------------------------------
# Momentum + indicator computation
# ---------------------------------------------------------------------------

def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    out       = df.copy()
    c         = out["close"]
    roll_mean = c.rolling(20, min_periods=1).mean()
    roll_std  = c.rolling(20, min_periods=1).std().replace(0, 1e-8).fillna(1e-8)
    vol_min   = out["volume"].rolling(100, min_periods=1).min()
    vol_max   = out["volume"].rolling(100, min_periods=1).max()

    out["c_norm"]     = (c            - roll_mean) / roll_std
    out["o_norm"]     = (out["open"]  - roll_mean) / roll_std
    out["h_norm"]     = (out["high"]  - roll_mean) / roll_std
    out["l_norm"]     = (out["low"]   - roll_mean) / roll_std
    out["vol_norm"]   = (out["volume"]- vol_min)   / (vol_max - vol_min + 1e-8)
    out["log_ret"]    = np.log(c / c.shift(1)).fillna(0.0)
    out["mom3"]       = np.log(c / c.shift(3)).fillna(0.0)
    out["mom10"]      = np.log(c / c.shift(10)).fillna(0.0)
    out["ema9_dist"]  = (c - c.ewm(span=9,  adjust=False).mean()) / roll_std
    out["ema21_dist"] = (c - c.ewm(span=21, adjust=False).mean()) / roll_std

    # RSI(14), normalised to [-1, 1]  →  (RSI - 50) / 50
    out["rsi_norm"]   = (_rsi(c, 14) - 50.0) / 50.0

    # MACD histogram (12, 26, 9), normalised by rolling std of close
    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist    = macd_line - signal_line
    out["macd_norm"] = (macd_hist / (roll_std + 1e-8)).fillna(0.0)

    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Time features  (sin/cos encoding for cyclical properties)
# ---------------------------------------------------------------------------

def _time_features(ts: pd.Timestamp) -> np.ndarray:
    """
    Returns [hour_sin, hour_cos, day_sin] — 3 floats in [-1, 1].

    Using sin/cos preserves periodicity: 23:00 is close to 00:00, and
    Sunday is close to Monday — which a raw integer would not capture.
    """
    h = ts.hour + ts.minute / 60.0          # fractional hour [0, 24)
    d = ts.weekday()                         # 0=Mon … 6=Sun

    hour_sin = math.sin(2 * math.pi * h / 24.0)
    hour_cos = math.cos(2 * math.pi * h / 24.0)
    day_sin  = math.sin(2 * math.pi * d / 7.0)

    return np.array([hour_sin, hour_cos, day_sin], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature block builders
# ---------------------------------------------------------------------------

_COLS_10 = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
            "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]

_COLS_8  = ["c_norm", "o_norm", "h_norm", "l_norm",
            "log_ret", "mom3", "ema9_dist", "ema21_dist"]

# 5m 15-dim: first 10 = standard features, last 5 = rsi + macd + time
_COLS_5M_BASE = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
                 "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]


def _block_5m(df: pd.DataFrame, ts: pd.Timestamp) -> np.ndarray:
    """
    15-dim block for 5m TF.
    Slots [0:10]  = standard price/momentum features
    Slots [10:12] = rsi_norm, macd_norm  (diverse indicators)
    Slots [12:15] = hour_sin, hour_cos, day_sin  (time context)
    """
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return np.zeros(15, dtype=np.float32)

    row  = df.iloc[idx]
    base = np.array(
        [float(row.get(c, 0.0)) for c in _COLS_5M_BASE], dtype=np.float32
    )
    indicators = np.array(
        [float(row.get("rsi_norm", 0.0)),
         float(row.get("macd_norm", 0.0))],
        dtype=np.float32,
    )
    time_feats = _time_features(ts)        # [hour_sin, hour_cos, day_sin]

    out = np.concatenate([base, indicators, time_feats])  # shape (15,)
    return np.clip(np.nan_to_num(out, nan=0.0), -5.0, 5.0)


def _block(df: pd.DataFrame, ts: pd.Timestamp, n: int) -> np.ndarray:
    """Standard 10-dim or 8-dim block for higher TFs."""
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return np.zeros(n, dtype=np.float32)
    row  = df.iloc[idx]
    cols = _COLS_10 if n == 10 else _COLS_8
    vals = np.array([float(row.get(c, 0.0)) for c in cols], dtype=np.float32)
    return np.clip(np.nan_to_num(vals, nan=0.0), -5.0, 5.0)


def _pad(arr: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) < size:
        arr = np.concatenate([arr, np.zeros(size - len(arr), dtype=np.float32)])
    return np.clip(np.nan_to_num(arr[:size], nan=0.0), -5.0, 5.0)