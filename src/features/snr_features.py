"""
snr_features.py  –  Support & Resistance level extractor  (FIXED)
==================================================================
Old bug: S&R levels returned raw price values, which are on completely
different scales than normalised features.  The policy received raw BTC
prices (~60 000) mixed with normalised values near 0 — this caused the
value network to diverge immediately.

Fix: return normalised distance from current price to each S&R level,
     expressed in ATR units (same normalisation as SMC features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNR_LOOKBACK = 200   # bars to look back for pivots
SWING_N      = 5     # bars each side for pivot detection
ATR_PERIOD   = 14
MAX_LEVELS   = 3     # number of support levels + resistance levels = 6 features


def compute_snr_features(df: pd.DataFrame, current_idx: int) -> np.ndarray:
    """
    Return 6 S&R features (float32):
      [0] dist_to_nearest_support_1   (ATR units, negative = price below S)
      [1] dist_to_nearest_support_2
      [2] dist_to_nearest_support_3
      [3] dist_to_nearest_resistance_1
      [4] dist_to_nearest_resistance_2
      [5] dist_to_nearest_resistance_3
    """
    end   = current_idx + 1
    start = max(0, end - SNR_LOOKBACK)
    sub   = df.iloc[start:end].copy()
    sub   = sub.reset_index(drop=True)

    atr = _atr(sub, ATR_PERIOD)
    if atr < 1e-8:
        return np.zeros(6, dtype=np.float32)

    close = sub["close"].iloc[-1]

    supports, resistances = _pivot_levels(sub, SWING_N)
    if not supports and not resistances:
        return np.zeros(6, dtype=np.float32)

    # Pick nearest MAX_LEVELS levels on each side
    s_below = sorted([s for s in supports    if s <= close], reverse=True)[:MAX_LEVELS]
    r_above = sorted([r for r in resistances if r >= close])[:MAX_LEVELS]

    # Pad to MAX_LEVELS with the outermost level repeated (better than 0)
    while len(s_below) < MAX_LEVELS:
        s_below.append(s_below[-1] if s_below else close - atr * 5)
    while len(r_above) < MAX_LEVELS:
        r_above.append(r_above[-1] if r_above else close + atr * 5)

    # Normalise: positive = price is above support (safe side)
    #            negative = price has broken below
    s_dists = [(close - s) / atr for s in s_below]
    r_dists = [(r - close) / atr for r in r_above]

    feats = np.array(s_dists + r_dists, dtype=np.float32)
    return np.clip(feats, -10.0, 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def _pivot_levels(df: pd.DataFrame, n: int):
    highs  = df["high"].values
    lows   = df["low"].values
    m      = len(highs)

    pivot_highs = []
    pivot_lows  = []

    for i in range(n, m - n):
        window_h = highs[max(0, i-n): i+n+1]
        window_l = lows [max(0, i-n): i+n+1]
        if highs[i] == max(window_h):
            pivot_highs.append(float(highs[i]))
        if lows[i]  == min(window_l):
            pivot_lows.append(float(lows[i]))

    # De-duplicate: merge levels within 0.2% of each other
    pivot_highs = _merge_levels(pivot_highs)
    pivot_lows  = _merge_levels(pivot_lows)

    return pivot_lows, pivot_highs   # (supports, resistances)


def _merge_levels(levels: list, pct: float = 0.002) -> list:
    if not levels:
        return []
    levels = sorted(set(levels))
    merged = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - merged[-1]) / (merged[-1] + 1e-8) > pct:
            merged.append(lvl)
    return merged
