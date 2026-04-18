"""
snr_features.py
───────────────
Support & Resistance features via fractal pivot points.

For each timeframe (5m, 1h, 4h):
  1. r1_dist   – normalised distance to nearest resistance level
  2. r2_dist   – 2nd resistance
  3. r3_dist   – 3rd resistance
  4. s1_dist   – normalised distance to nearest support level
  5. s2_dist   – 2nd support
  6. s3_dist   – 3rd support

Total: 6 features × 3 timeframes = 18 features
"""

import numpy as np
import pandas as pd
import yaml
import os


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = _load_cfg()["features"]["snr"]
    return _CFG


# ─────────────────────────────────────────────────────────────────────────────

def find_pivot_levels(df: pd.DataFrame) -> dict:
    """
    Find pivot-based support and resistance levels.

    Uses the Williams Fractal method: a bar is a swing high if its
    high is the highest in a window of (left + right) bars, and
    a swing low if its low is the lowest.

    Returns
    -------
    dict with 'resistance': list[float], 'support': list[float]
    both sorted by distance to current price (nearest first).
    """
    cfg = _cfg()
    left = cfg["pivot_left"]
    right = cfg["pivot_right"]
    n_levels = cfg["n_levels"]

    if df is None or len(df) < left + right + 1:
        current = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
        return {"resistance": [current] * n_levels, "support": [current] * n_levels}

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    resistance_levels = []
    support_levels = []

    # Only look at confirmed pivots (exclude last `right` bars — no lookahead)
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]

        if highs[i] == np.max(window_h):
            resistance_levels.append(float(highs[i]))

        if lows[i] == np.min(window_l):
            support_levels.append(float(lows[i]))

    current_price = float(closes[-1])

    # Filter: only levels that are actual resistance (above price) or support (below price)
    res = sorted([lvl for lvl in resistance_levels if lvl >= current_price])
    sup = sorted([lvl for lvl in support_levels if lvl <= current_price], reverse=True)

    # Cluster nearby levels (within 0.3%) to avoid duplicates
    res = _cluster_levels(res, threshold=0.003)
    sup = _cluster_levels(sup, threshold=0.003)

    # Pad with current price if fewer than n_levels found
    while len(res) < n_levels:
        res.append(current_price * (1 + 0.01 * (len(res) + 1)))
    while len(sup) < n_levels:
        sup.append(current_price * (1 - 0.01 * (len(sup) + 1)))

    return {
        "resistance": res[:n_levels],
        "support": sup[:n_levels],
    }


def extract_snr_features(df: pd.DataFrame, current_price: float) -> np.ndarray:
    """
    Extract 6 SNR features from an OHLCV DataFrame.

    Returns np.ndarray of shape (6,).
    All distances are normalised to [-1, 1]:
      • resistance: positive (price is below the level)
      • support: negative (price is above the level)
    """
    if df is None or len(df) < 30:
        return np.zeros(6, dtype=np.float32)

    try:
        levels = find_pivot_levels(df)
        res = levels["resistance"]
        sup = levels["support"]

        cfg = _cfg()
        n = cfg["n_levels"]

        # Resistance distances: positive, normalised by price range
        r_dists = [_norm_dist(r, current_price) for r in res[:n]]
        # Support distances: negative, normalised
        s_dists = [-_norm_dist(s, current_price) for s in sup[:n]]

        # Pad if necessary
        while len(r_dists) < n:
            r_dists.append(0.0)
        while len(s_dists) < n:
            s_dists.append(0.0)

        return np.array(r_dists[:n] + s_dists[:n], dtype=np.float32)

    except Exception:
        return np.zeros(6, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def _norm_dist(level: float, price: float, scale: float = 20.0) -> float:
    """
    Absolute normalised distance, clipped to [0, 1].
    5% away → 1.0 when scale=20.
    """
    if price == 0:
        return 0.0
    return float(np.clip(abs(level - price) / price * scale, 0.0, 1.0))


def _cluster_levels(levels: list, threshold: float = 0.003) -> list:
    """Merge levels that are within `threshold` fraction of each other."""
    if not levels:
        return []

    clustered = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - clustered[-1]) / max(clustered[-1], 1e-10) > threshold:
            clustered.append(lvl)

    return clustered
