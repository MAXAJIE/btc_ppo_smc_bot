"""
snr_features.py  –  Support & Resistance level extractor
=========================================================
S&R levels expressed as ATR-normalised distances so the policy network
receives a scale-invariant signal regardless of whether BTC is at $20k or $100k.

Output: (6,) float32
  [0..2] distance to 3 nearest support levels   (positive = price above S)
  [3..5] distance to 3 nearest resistance levels (positive = R above price)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNR_LOOKBACK = 200
SWING_N      = 5
ATR_PERIOD   = 14
MAX_LEVELS   = 3


def compute_snr_features(df: pd.DataFrame, current_idx: int) -> np.ndarray:
    if current_idx < 0 or len(df) == 0:
        return np.zeros(6, dtype=np.float32)

    end   = current_idx + 1
    start = max(0, end - SNR_LOOKBACK)
    sub   = df.iloc[start:end].copy().reset_index(drop=True)

    atr = _atr(sub, ATR_PERIOD)
    if atr < 1e-8:
        return np.zeros(6, dtype=np.float32)

    close = float(sub["close"].iloc[-1])
    supps, ress = _pivots(sub, SWING_N)

    if not supps and not ress:
        return np.zeros(6, dtype=np.float32)

    s_near = sorted([s for s in supps if s <= close], reverse=True)[:MAX_LEVELS]
    r_near = sorted([r for r in ress  if r >= close])[:MAX_LEVELS]

    # Pad by repeating the outermost level
    while len(s_near) < MAX_LEVELS:
        s_near.append(s_near[-1] if s_near else close - atr * 5)
    while len(r_near) < MAX_LEVELS:
        r_near.append(r_near[-1] if r_near else close + atr * 5)

    s_dists = [np.clip((close - s) / atr, -10, 10) for s in s_near]
    r_dists = [np.clip((r - close) / atr, -10, 10) for r in r_near]

    return np.array(s_dists + r_dists, dtype=np.float32)


def _atr(df: pd.DataFrame, p: int) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr[-p:].mean()) if len(tr) >= p else float(tr.mean() + 1e-8)


def _pivots(df: pd.DataFrame, n: int):
    h, l = df["high"].values, df["low"].values
    m    = len(h)
    ph, pl = [], []
    for i in range(n, m - n):
        if h[i] == max(h[max(0, i-n): i+n+1]):
            ph.append(float(h[i]))
        if l[i] == min(l[max(0, i-n): i+n+1]):
            pl.append(float(l[i]))
    return _merge(pl), _merge(ph)


def _merge(levels: list, pct: float = 0.002) -> list:
    if not levels:
        return []
    levels = sorted(set(levels))
    out    = [levels[0]]
    for lv in levels[1:]:
        if abs(lv - out[-1]) / (out[-1] + 1e-8) > pct:
            out.append(lv)
    return out