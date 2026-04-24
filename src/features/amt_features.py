"""
amt_features.py  –  Volume Profile / AMT feature extractor
===========================================================
Rolling 100-bar volume profile on the 4h slice.
Returns ATR-normalised distances — same scale as SMC and SNR features.

Output: (6,) float32
  [0] poc_dist      (close - POC) / ATR
  [1] vah_dist      (VAH  - close) / ATR
  [2] val_dist      (close - VAL) / ATR
  [3] va_width      (VAH - VAL)   / ATR   ← compression/expansion signal
  [4] in_value_area 1.0 if VAL ≤ close ≤ VAH
  [5] vol_rank      percentile rank of current bar's volume [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_BARS   = 100
PRICE_BINS     = 50
VALUE_AREA_PCT = 0.70
ATR_PERIOD     = 14


def compute_amt_features(df_4h: pd.DataFrame, current_idx: int) -> np.ndarray:
    if current_idx < 0 or len(df_4h) == 0:
        return np.zeros(6, dtype=np.float32)

    end   = current_idx + 1
    start = max(0, end - PROFILE_BARS)
    sub   = df_4h.iloc[start:end]

    if len(sub) < 3:
        return np.zeros(6, dtype=np.float32)

    atr = _atr(sub, ATR_PERIOD)
    if atr < 1e-8:
        return np.zeros(6, dtype=np.float32)

    close = float(sub["close"].iloc[-1])
    poc, vah, val = _vol_profile(sub)

    feats = np.array([
        (close - poc) / atr,
        (vah  - close) / atr,
        (close - val)  / atr,
        (vah   - val)  / atr,
        1.0 if val <= close <= vah else 0.0,
        float(np.mean(sub["volume"].values < sub["volume"].iloc[-1])),
    ], dtype=np.float32)
    return np.clip(np.nan_to_num(feats, nan=0.0), -10.0, 10.0)


def _atr(df: pd.DataFrame, p: int) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr[-p:].mean()) if len(tr) >= p else float(tr.mean() + 1e-8)


def _vol_profile(df: pd.DataFrame):
    lo = df["low"].min()
    hi = df["high"].max()
    if hi - lo < 1e-8:
        m = (hi + lo) / 2
        return m, m, m

    bins = np.linspace(lo, hi, PRICE_BINS + 1)
    vols = np.zeros(PRICE_BINS, dtype=float)

    for _, row in df.iterrows():
        bl, bh, bv = row["low"], row["high"], row["volume"]
        span = bh - bl + 1e-8
        for b in range(PRICE_BINS):
            ov = min(bh, bins[b+1]) - max(bl, bins[b])
            if ov > 0:
                vols[b] += bv * (ov / span)

    total = vols.sum()
    pi    = int(np.argmax(vols))
    poc   = float((bins[pi] + bins[pi+1]) / 2)

    va = vols[pi]; li = pi; hi_i = pi
    while va < VALUE_AREA_PCT * total:
        cl = li > 0
        ch = hi_i < PRICE_BINS - 1
        if not cl and not ch:
            break
        al = vols[li-1]   if cl else -1
        ah = vols[hi_i+1] if ch else -1
        if al >= ah:
            li   -= 1; va += vols[li]
        else:
            hi_i += 1; va += vols[hi_i]

    return poc, float(bins[hi_i+1]), float(bins[li])