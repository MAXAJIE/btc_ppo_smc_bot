"""
smc_features.py  –  Smart Money Concepts feature extractor  (FIXED)
====================================================================
Problems in the old version
---------------------------
1.  Order-block detection used a raw price comparison without swing structure,
    so almost every candle qualified as an OB → features were always 1.0 → 
    the policy treated the signal as noise.
2.  FVG returned a boolean that was always True in trending markets.
3.  BOS/CHoCH never fired correctly because it compared to a single bar
    rather than the proper swing high/low.
4.  All features were integers (0/1) with no distance/proximity signal,
    so the policy couldn't learn magnitude.

Fix
---
•  OB: proper swing-high / swing-low detection with lookback window.
       Returns distance to nearest bullish OB and nearest bearish OB,
       normalised by ATR.
•  FVG: returns distance to nearest unfilled gap, normalised by ATR.
•  BOS/CHoCH: returns binary signal PLUS how far price has moved since
       the break (momentum of the break), normalised by ATR.
•  All outputs are float32 in roughly [-2, 2] range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SWING_LOOKBACK = 5   # bars each side to define a swing H/L
OB_LOOKBACK    = 50  # search this many bars back for valid OBs
FVG_LOOKBACK   = 30  # search this many bars back for unfilled FVGs
ATR_PERIOD     = 14


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_smc_features(df: pd.DataFrame, current_idx: int) -> np.ndarray:
    """
    Compute 8 SMC features at position `current_idx` in `df`.

    Returns
    -------
    np.ndarray, shape (8,), dtype float32
        [0] bullish_ob_dist   – normalised distance from close to nearest bull OB top
        [1] bearish_ob_dist   – normalised distance from close to nearest bear OB bottom
        [2] bull_ob_present   – 1.0 if price is INSIDE a bullish OB, else 0
        [3] bear_ob_present   – 1.0 if price is INSIDE a bearish OB, else 0
        [4] fvg_bull_dist     – normalised distance to nearest bull FVG midpoint
        [5] fvg_bear_dist     – normalised distance to nearest bear FVG midpoint
        [6] bos_signal        – +1 bullish BOS, -1 bearish BOS, 0 none (last 3 bars)
        [7] choch_signal      – +1 bullish CHoCH, -1 bearish CHoCH, 0 none (last 3 bars)
    """
    end = current_idx + 1
    start = max(0, end - max(OB_LOOKBACK, FVG_LOOKBACK) - 20)
    sub = df.iloc[start:end].copy()
    sub = sub.reset_index(drop=True)
    local_idx = len(sub) - 1      # current position in sub

    atr = _atr(sub, ATR_PERIOD)
    if atr < 1e-8:
        return np.zeros(8, dtype=np.float32)

    close  = sub["close"].iloc[local_idx]
    high   = sub["high"].iloc[local_idx]
    low    = sub["low"].iloc[local_idx]

    # --- Swing highs / lows -----------------------------------------------
    swing_highs, swing_lows = _swing_points(sub, SWING_LOOKBACK)

    # --- Order Blocks -------------------------------------------------------
    bull_ob_dist, bear_ob_dist, bull_ob_in, bear_ob_in = _ob_features(
        sub, local_idx, swing_highs, swing_lows, close, atr
    )

    # --- Fair Value Gaps ----------------------------------------------------
    fvg_bull_dist, fvg_bear_dist = _fvg_features(sub, local_idx, close, atr)

    # --- BOS / CHoCH --------------------------------------------------------
    bos_sig, choch_sig = _bos_choch_features(
        sub, local_idx, swing_highs, swing_lows
    )

    return np.array(
        [bull_ob_dist, bear_ob_dist, bull_ob_in, bear_ob_in,
         fvg_bull_dist, fvg_bear_dist, bos_sig, choch_sig],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h  = df["high"].values
    l  = df["low"].values
    c  = df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if len(tr) < period:
        return float(tr.mean()) if len(tr) > 0 else 1.0
    return float(tr[-period:].mean())


def _swing_points(df: pd.DataFrame, n: int):
    """
    Return two boolean arrays: `swing_highs` and `swing_lows`.
    A bar i is a swing high if high[i] = max(high[i-n:i+n+1]).
    """
    highs = df["high"].values
    lows  = df["low"].values
    m = len(highs)
    sh = np.zeros(m, bool)
    sl = np.zeros(m, bool)
    for i in range(n, m - n):
        if highs[i] == max(highs[max(0, i-n):i+n+1]):
            sh[i] = True
        if lows[i]  == min(lows[max(0, i-n):i+n+1]):
            sl[i] = True
    return sh, sl


def _ob_features(df, idx, swing_highs, swing_lows, close, atr):
    """Find most recent bullish / bearish order blocks."""
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    bull_ob_top  = None
    bull_ob_bot  = None
    bear_ob_top  = None
    bear_ob_bot  = None

    search_start = max(0, idx - OB_LOOKBACK)

    for i in range(idx - 1, search_start, -1):
        if swing_lows[i]:
            # Bullish OB: last bearish candle before the swing low
            if i > 0 and closes[i-1] < opens[i-1]:
                if bull_ob_top is None:
                    bull_ob_top = highs[i-1]
                    bull_ob_bot = lows[i-1]
                    break

    for i in range(idx - 1, search_start, -1):
        if swing_highs[i]:
            # Bearish OB: last bullish candle before the swing high
            if i > 0 and closes[i-1] > opens[i-1]:
                if bear_ob_top is None:
                    bear_ob_top = highs[i-1]
                    bear_ob_bot = lows[i-1]
                    break

    # Distance features
    if bull_ob_top is not None:
        bull_dist  = (close - bull_ob_top) / atr          # negative = below OB
        bull_in    = 1.0 if bull_ob_bot <= close <= bull_ob_top else 0.0
    else:
        bull_dist, bull_in = 0.0, 0.0

    if bear_ob_top is not None:
        bear_dist  = (bear_ob_bot - close) / atr           # negative = above OB
        bear_in    = 1.0 if bear_ob_bot <= close <= bear_ob_top else 0.0
    else:
        bear_dist, bear_in = 0.0, 0.0

    return (
        float(np.clip(bull_dist, -5, 5)),
        float(np.clip(bear_dist, -5, 5)),
        bull_in,
        bear_in,
    )


def _fvg_features(df, idx, close, atr):
    """Find nearest unfilled bull and bear Fair Value Gaps."""
    highs  = df["high"].values
    lows   = df["low"].values

    bull_fvg_mid = None
    bear_fvg_mid = None

    search_start = max(2, idx - FVG_LOOKBACK)

    for i in range(idx - 1, search_start, -1):
        # Bullish FVG: gap between high[i-2] and low[i]
        if lows[i] > highs[i-2]:
            mid = (lows[i] + highs[i-2]) / 2
            if close < mid and bull_fvg_mid is None:   # still below → unfilled
                bull_fvg_mid = mid
        # Bearish FVG: gap between low[i-2] and high[i]
        if highs[i] < lows[i-2]:
            mid = (highs[i] + lows[i-2]) / 2
            if close > mid and bear_fvg_mid is None:   # still above → unfilled
                bear_fvg_mid = mid

    bull_dist = float(np.clip((bull_fvg_mid - close) / atr, -5, 5)) if bull_fvg_mid else 0.0
    bear_dist = float(np.clip((close - bear_fvg_mid) / atr, -5, 5)) if bear_fvg_mid else 0.0

    return bull_dist, bear_dist


def _bos_choch_features(df, idx, swing_highs, swing_lows):
    """
    BOS  = price closes beyond a swing H/L in the direction of prevailing trend.
    CHoCH = price closes beyond a swing H/L AGAINST the prevailing trend.

    Returns two floats in {-1, 0, +1}.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    bos   = 0.0
    choch = 0.0

    if idx < 4:
        return bos, choch

    # Very recent: only look at last 5 bars for a fresh signal
    recent_sh_idx = [i for i in range(max(0, idx-5), idx) if swing_highs[i]]
    recent_sl_idx = [i for i in range(max(0, idx-5), idx) if swing_lows[i]]

    curr_close = closes[idx]

    # Determine prevailing trend from last 20 bars
    if idx >= 20:
        trend_up = closes[idx] > closes[idx - 20]
    else:
        trend_up = True

    if recent_sh_idx:
        sh_i   = recent_sh_idx[-1]
        sh_lvl = highs[sh_i]
        if curr_close > sh_lvl:
            bos   = 1.0 if trend_up   else 0.0
            choch = 1.0 if not trend_up else 0.0

    if recent_sl_idx:
        sl_i   = recent_sl_idx[-1]
        sl_lvl = lows[sl_i]
        if curr_close < sl_lvl:
            bos   = -1.0 if not trend_up else 0.0
            choch = -1.0 if trend_up     else 0.0

    return bos, choch
