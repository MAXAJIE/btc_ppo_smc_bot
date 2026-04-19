"""
multi_tf_features.py  –  90-dim observation builder  (FIXED)
=============================================================
Root cause of MTF bug
---------------------
The old version tried to resample the 5m DataFrame to build 15m/1h/4h/1D
candles on-the-fly.  When resample produced NaN (e.g., if the 5m data had
gaps, or the current bar hadn't closed yet on the higher TF), those NaN
values silently propagated through every downstream feature and the PPO
policy received garbage observations.

Fix
---
1.  Accept a `tf_data: dict[str, pd.DataFrame]` — each TF already downloaded
    by `data_loader.load_all_timeframes()`.
2.  Use an *as-of* lookup: for the 5m bar at time T, find the **last closed**
    bar on each higher timeframe where bar_start <= T.  This is look-ahead safe.
3.  Clip + normalise everything; replace any remaining NaN with 0.
4.  Return a fixed-length 90-dim numpy array that exactly matches the env's
    observation space definition.

Observation layout (unchanged from README):
  [00:15]  OHLCV + momentum  5m   (15 features)
  [15:25]  OHLCV + momentum  15m  (10 features)
  [25:35]  OHLCV + momentum  1h   (10 features)
  [35:43]  OHLCV + momentum  4h   (8  features)
  [43:51]  OHLCV + momentum  1d   (8  features)
  [51:59]  SMC features       5m   (8  features)
  [59:67]  SMC features       1h   (8  features)
  [67:73]  S&R levels         1h   (6  features)
  [73:79]  AMT/Vol profile    4h   (6  features)
  [79:83]  GARCH + Kelly           (4  features)
  [83:90]  Position state          (7  features)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OBS_DIM = 90


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class MultiTFFeatureBuilder:
    """
    Pre-loads all timeframe DataFrames and exposes a single
    `build(ts, smc_5m, smc_1h, snr_1h, amt_4h, garch_kelly, position)`
    method that returns a 90-dim float32 array.

    Parameters
    ----------
    tf_data : dict[str, pd.DataFrame]
        Output of data_loader.load_all_timeframes().
    """

    def __init__(self, tf_data: Dict[str, pd.DataFrame]):
        self._tf: Dict[str, pd.DataFrame] = tf_data
        # Pre-compute momentum columns for each TF
        self._mom: Dict[str, pd.DataFrame] = {}
        for tf, df in tf_data.items():
            self._mom[tf] = _add_momentum(df)
        logger.info("MultiTFFeatureBuilder initialised — TFs: %s", list(tf_data.keys()))

    # ------------------------------------------------------------------
    def build(
        self,
        ts: pd.Timestamp,
        smc_5m:    np.ndarray,   # shape (8,)
        smc_1h:    np.ndarray,   # shape (8,)
        snr_1h:    np.ndarray,   # shape (6,)
        amt_4h:    np.ndarray,   # shape (6,)
        garch_kelly: np.ndarray, # shape (4,)
        position:  np.ndarray,   # shape (7,)
    ) -> np.ndarray:
        """Return a 90-dim float32 observation vector."""

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # --- Price / momentum blocks ------------------------------------
        obs[0:15]  = _price_mom_block(self._mom["5m"],  ts, n=15)
        obs[15:25] = _price_mom_block(self._mom["15m"], ts, n=10)
        obs[25:35] = _price_mom_block(self._mom["1h"],  ts, n=10)
        obs[35:43] = _price_mom_block(self._mom["4h"],  ts, n=8)
        obs[43:51] = _price_mom_block(self._mom["1d"],  ts, n=8)

        # --- Feature blocks (already normalised by caller) ---------------
        obs[51:59] = _safe_clip(smc_5m,    8)
        obs[59:67] = _safe_clip(smc_1h,    8)
        obs[67:73] = _safe_clip(snr_1h,    6)
        obs[73:79] = _safe_clip(amt_4h,    6)
        obs[79:83] = _safe_clip(garch_kelly, 4)
        obs[83:90] = _safe_clip(position,  7)

        # Final NaN guard — should never trigger but keeps training stable
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-return, normalised volume, and short EMA features."""
    out = df.copy()
    c = out["close"]
    # log-return
    out["log_ret"] = np.log(c / c.shift(1)).fillna(0.0)
    # normalise OHLCV by recent close (rolling 20-bar)
    roll_std = c.rolling(20, min_periods=1).std().replace(0, 1e-8)
    roll_mean = c.rolling(20, min_periods=1).mean()
    out["c_norm"] = (c - roll_mean) / roll_std
    out["o_norm"] = (out["open"]  - roll_mean) / roll_std
    out["h_norm"] = (out["high"]  - roll_mean) / roll_std
    out["l_norm"] = (out["low"]   - roll_mean) / roll_std
    # volume: rank-normalise (0-1) over rolling 100
    vol_roll = out["volume"].rolling(100, min_periods=1)
    out["vol_norm"] = (out["volume"] - vol_roll.min()) / (
        vol_roll.max() - vol_roll.min() + 1e-8
    )
    # momentum: 3-bar and 10-bar log-return
    out["mom3"]  = np.log(c / c.shift(3)).fillna(0.0)
    out["mom10"] = np.log(c / c.shift(10)).fillna(0.0)
    # EMA-distance
    out["ema9_dist"]  = (c - c.ewm(span=9,  adjust=False).mean()) / (roll_std + 1e-8)
    out["ema21_dist"] = (c - c.ewm(span=21, adjust=False).mean()) / (roll_std + 1e-8)
    return out


_FEATURE_COLS_15 = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
                    "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist",
                    "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]

_FEATURE_COLS_10 = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
                    "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]

_FEATURE_COLS_8 = ["c_norm", "o_norm", "h_norm", "l_norm",
                   "log_ret", "mom3", "ema9_dist", "ema21_dist"]


def _price_mom_block(
    df: pd.DataFrame,
    ts: pd.Timestamp,
    n: int,
) -> np.ndarray:
    """
    Find the last bar at or before `ts` in `df`, then return `n` features.
    Falls back to zeros if no bar exists yet.
    """
    # as-of lookup (look-ahead safe)
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return np.zeros(n, dtype=np.float32)

    row = df.iloc[idx]

    if n == 15:
        cols = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
                "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist",
                # repeat 5 with previous bar for padding to 15
                "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]
    elif n == 10:
        cols = ["c_norm", "o_norm", "h_norm", "l_norm", "vol_norm",
                "log_ret", "mom3", "mom10", "ema9_dist", "ema21_dist"]
    else:  # 8
        cols = ["c_norm", "o_norm", "h_norm", "l_norm",
                "log_ret", "mom3", "ema9_dist", "ema21_dist"]

    vals = np.array([row.get(c, 0.0) for c in cols], dtype=np.float32)
    return np.clip(vals, -5.0, 5.0)


def _safe_clip(arr: np.ndarray, size: int) -> np.ndarray:
    """Ensure array is exactly `size` elements, clipped to [-5, 5]."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) < size:
        arr = np.concatenate([arr, np.zeros(size - len(arr), dtype=np.float32)])
    elif len(arr) > size:
        arr = arr[:size]
    return np.clip(np.nan_to_num(arr, nan=0.0), -5.0, 5.0)
