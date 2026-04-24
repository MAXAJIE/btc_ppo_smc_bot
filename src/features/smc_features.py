"""
smc_features.py  –  Smart Money Concepts feature extractor
===========================================================

Improvements from code review
------------------------------
1. OB Displacement check
   A valid OB now requires that within 3 bars AFTER the OB candle, there is
   a "displacement" move: a bar whose body is > 1.5× ATR.  Without this,
   almost every bearish candle near a swing qualifies as an OB → signal is
   always 1.0 → noise.

2. BOS/CHoCH trend uses HTF EMA instead of 20-bar close comparison
   `closes[idx] > closes[idx - 20]` is too sensitive — a single volatile
   bar can flip the trend signal.  Replaced with EMA50 direction on the
   same slice.  This matches how SMC traders actually define trend bias.

3. GARCH confidence gate (imported from garch_kelly):
   Not in this file directly, but OB distance is attenuated when called
   with low GARCH confidence.

Output shape: (8,) float32  — unchanged, no obs-space change needed.
  [0] bull_ob_dist   [1] bear_ob_dist   [2] bull_ob_in  [3] bear_ob_in
  [4] fvg_bull_dist  [5] fvg_bear_dist  [6] bos_signal  [7] choch_signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SWING_LOOKBACK     = 5
OB_LOOKBACK        = 50
FVG_LOOKBACK       = 30
ATR_PERIOD         = 14
DISPLACEMENT_BARS  = 3      # check this many bars after OB for displacement
DISPLACEMENT_MULT  = 1.5    # body must be > 1.5× ATR to count as displacement
EMA_TREND_PERIOD   = 50     # EMA period for HTF trend bias


def compute_smc_features(df: pd.DataFrame, current_idx: int) -> np.ndarray:
    end   = current_idx + 1
    start = max(0, end - max(OB_LOOKBACK, FVG_LOOKBACK) - 20)
    sub   = df.iloc[start:end].copy().reset_index(drop=True)
    loc   = len(sub) - 1

    atr = _atr(sub, ATR_PERIOD)
    if atr < 1e-8:
        return np.zeros(8, dtype=np.float32)

    close = float(sub["close"].iloc[loc])
    sh, sl = _swing_points(sub, SWING_LOOKBACK)

    b_dist, br_dist, b_in, br_in = _ob_features(sub, loc, sh, sl, close, atr)
    fvg_b, fvg_br               = _fvg_features(sub, loc, close, atr)
    bos, choch                   = _bos_choch_features(sub, loc, sh, sl)

    out = np.array([b_dist, br_dist, b_in, br_in,
                    fvg_b, fvg_br, bos, choch], dtype=np.float32)
    return np.clip(np.nan_to_num(out, nan=0.0), -5.0, 5.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, p: int) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr[-p:].mean()) if len(tr) >= p else float(tr.mean() + 1e-8)


def _swing_points(df: pd.DataFrame, n: int):
    h, l = df["high"].values, df["low"].values
    m    = len(h)
    sh   = np.zeros(m, bool)
    sl   = np.zeros(m, bool)
    for i in range(n, m - n):
        if h[i] == max(h[max(0, i-n): i+n+1]):
            sh[i] = True
        if l[i] == min(l[max(0, i-n): i+n+1]):
            sl[i] = True
    return sh, sl


def _has_displacement(df, ob_idx: int, atr: float) -> bool:
    """
    Improvement 1: check that OB is followed by a strong displacement move.
    A displacement bar has |close - open| > DISPLACEMENT_MULT × ATR.
    Without this, nearly every bearish candle near a swing low is an OB.
    """
    opens  = df["open"].values
    closes = df["close"].values
    m      = len(opens)
    for j in range(ob_idx + 1, min(ob_idx + 1 + DISPLACEMENT_BARS, m)):
        body = abs(closes[j] - opens[j])
        if body > DISPLACEMENT_MULT * atr:
            return True
    return False


def _ob_features(df, loc, sh, sl, close, atr):
    o, c, h, l = (df["open"].values, df["close"].values,
                  df["high"].values, df["low"].values)

    bull_top = bull_bot = bear_top = bear_bot = None

    for i in range(loc - 1, max(0, loc - OB_LOOKBACK), -1):
        if sl[i] and i > 0 and c[i-1] < o[i-1] and bull_top is None:
            # Improvement 1: require displacement after OB
            if _has_displacement(df, i, atr):
                bull_top, bull_bot = h[i-1], l[i-1]
                break

    for i in range(loc - 1, max(0, loc - OB_LOOKBACK), -1):
        if sh[i] and i > 0 and c[i-1] > o[i-1] and bear_top is None:
            if _has_displacement(df, i, atr):
                bear_top, bear_bot = h[i-1], l[i-1]
                break

    bd  = float(np.clip((close - bull_top) / atr, -5, 5)) if bull_top else 0.0
    brd = float(np.clip((bear_bot - close) / atr, -5, 5)) if bear_bot else 0.0
    bi  = 1.0 if bull_top and bull_bot and bull_bot <= close <= bull_top else 0.0
    bri = 1.0 if bear_top and bear_bot and bear_bot <= close <= bear_top else 0.0
    return bd, brd, bi, bri


def _fvg_features(df, loc, close, atr):
    h, l = df["high"].values, df["low"].values
    bull_m = bear_m = None
    for i in range(loc - 1, max(2, loc - FVG_LOOKBACK), -1):
        if l[i] > h[i-2]:
            mid = (l[i] + h[i-2]) / 2
            if close < mid and bull_m is None:
                bull_m = mid
        if h[i] < l[i-2]:
            mid = (h[i] + l[i-2]) / 2
            if close > mid and bear_m is None:
                bear_m = mid
    bd  = float(np.clip((bull_m - close) / atr, -5, 5)) if bull_m else 0.0
    brd = float(np.clip((close - bear_m) / atr, -5, 5)) if bear_m else 0.0
    return bd, brd


def _ema(values: np.ndarray, period: int) -> float:
    """Compute EMA of a numpy array. Returns last value."""
    if len(values) < 2:
        return float(values[-1]) if len(values) > 0 else 0.0
    alpha = 2.0 / (period + 1)
    ema   = float(values[0])
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _bos_choch_features(df, loc, sh, sl):
    if loc < 4:
        return 0.0, 0.0

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    # Improvement 2: use EMA50 direction instead of close[idx] > close[idx-20]
    # EMA50 is far less sensitive to single-bar noise
    ema_period = min(EMA_TREND_PERIOD, loc)
    closes_slice = c[max(0, loc - ema_period): loc + 1]
    ema_val = _ema(closes_slice, ema_period)
    trend_up = float(c[loc]) > ema_val

    bos = choch = 0.0
    for i in range(max(0, loc - 5), loc):
        if sh[i] and c[loc] > h[i]:
            bos   = 1.0 if trend_up  else 0.0
            choch = 1.0 if not trend_up else 0.0
        if sl[i] and c[loc] < l[i]:
            bos   = -1.0 if not trend_up else 0.0
            choch = -1.0 if trend_up      else 0.0

    return bos, choch