"""
amt_features.py  –  Volume Profile / AMT feature extractor  (FIXED)
====================================================================
Old bug: POC / VAH / VAL were returned as raw price values → same
         scale-mismatch problem as S&R.  Additionally the volume profile
         was computed on the full dataset instead of a rolling window,
         so it reflected the entire 2-year history not the recent 4h context.

Fix:
  • Rolling 100-bar volume profile (≈ 8 hours on 5m) on the 4h slice.
  • Return ATR-normalised distances from close to POC, VAH, VAL.
  • Extra features: Value Area width (spread signal) and volume rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_BARS = 100   # bars in the rolling volume profile window
PRICE_BINS   = 50    # price buckets for the volume profile
VALUE_AREA_PCT = 0.70  # 70% value area


def compute_amt_features(df_4h: pd.DataFrame, current_idx: int) -> np.ndarray:
    """
    Compute 6 AMT / Volume-Profile features for the current 4h bar.

    Returns
    -------
    np.ndarray, shape (6,), dtype float32
        [0] poc_dist     – (close - POC) / ATR
        [1] vah_dist     – (VAH  - close) / ATR
        [2] val_dist     – (close - VAL) / ATR
        [3] va_width     – (VAH - VAL) / ATR  (spread / compression signal)
        [4] in_value_area – 1.0 if VAL <= close <= VAH else 0.0
        [5] vol_rank      – rank of current bar's volume (0..1) vs window
    """
    end   = current_idx + 1
    start = max(0, end - PROFILE_BARS)
    sub   = df_4h.iloc[start:end]

    if len(sub) < 5:
        return np.zeros(6, dtype=np.float32)

    atr = _atr(sub)
    if atr < 1e-8:
        return np.zeros(6, dtype=np.float32)

    close = float(sub["close"].iloc[-1])

    poc, vah, val = _volume_profile(sub)

    poc_dist = (close - poc) / atr
    vah_dist = (vah  - close) / atr
    val_dist = (close - val)  / atr
    va_width = (vah - val)    / atr
    in_va    = 1.0 if val <= close <= vah else 0.0

    # Volume rank of current bar
    vol_series = sub["volume"].values
    cur_vol    = vol_series[-1]
    vol_rank   = float(np.mean(vol_series < cur_vol))  # percentile rank

    feats = np.array([poc_dist, vah_dist, val_dist, va_width, in_va, vol_rank],
                     dtype=np.float32)
    return np.clip(feats, -10.0, 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h  = df["high"].values
    l  = df["low"].values
    c  = df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def _volume_profile(df: pd.DataFrame):
    """Compute POC, VAH, VAL over the given window using a price-bucketed profile."""
    lo   = df["low"].min()
    hi   = df["high"].max()
    if hi - lo < 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid

    bins  = np.linspace(lo, hi, PRICE_BINS + 1)
    vols  = np.zeros(PRICE_BINS, dtype=float)

    for _, row in df.iterrows():
        # Distribute bar's volume evenly across the price range it covers
        bar_lo, bar_hi, vol = row["low"], row["high"], row["volume"]
        for b in range(PRICE_BINS):
            overlap = min(bar_hi, bins[b+1]) - max(bar_lo, bins[b])
            if overlap > 0:
                span = bar_hi - bar_lo + 1e-8
                vols[b] += vol * (overlap / span)

    total_vol = vols.sum()
    poc_idx   = int(np.argmax(vols))
    poc       = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)

    # Value area: expand from POC until 70% of volume is enclosed
    va_vol   = vols[poc_idx]
    lo_idx   = poc_idx
    hi_idx   = poc_idx

    while va_vol < VALUE_AREA_PCT * total_vol:
        can_expand_lo = lo_idx > 0
        can_expand_hi = hi_idx < PRICE_BINS - 1
        if not can_expand_lo and not can_expand_hi:
            break
        add_lo = vols[lo_idx - 1] if can_expand_lo else -1
        add_hi = vols[hi_idx + 1] if can_expand_hi else -1
        if add_lo >= add_hi:
            lo_idx -= 1; va_vol += vols[lo_idx]
        else:
            hi_idx += 1; va_vol += vols[hi_idx]

    val = float(bins[lo_idx])
    vah = float(bins[hi_idx + 1])

    return poc, vah, val
