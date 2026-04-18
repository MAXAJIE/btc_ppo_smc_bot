"""
amt_features.py
───────────────
Auction Market Theory (AMT) / Volume Profile features.

We build a simple pandas-based volume profile histogram without
relying on py-market-profile (which has sparse maintenance).

Features per timeframe (1h, 4h, 1D):
  1. poc_dist       – normalised distance from price to POC
  2. vah_dist       – normalised distance from price to VAH
  3. val_dist       – normalised distance from price to VAL
  4. in_value_area  – 1.0 if price is inside Value Area, else 0.0
  5. poc_volume_pct – POC volume as % of total (liquidity concentration)
  6. imbalance      – buy/sell volume imbalance in profile window

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
        _CFG = _load_cfg()["features"]["amt"]
    return _CFG


# ─────────────────────────────────────────────────────────────────────────────

def build_volume_profile(df: pd.DataFrame, n_bins: int = None, value_area_pct: float = None) -> dict:
    """
    Build a volume profile from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    n_bins : int
        Number of price buckets in the histogram.
    value_area_pct : float
        Fraction of total volume that defines the Value Area (e.g. 0.70).

    Returns
    -------
    dict with keys: poc_price, vah, val, total_volume, poc_volume,
                    buy_volume, sell_volume
    """
    cfg = _cfg()
    n_bins = n_bins or cfg["n_bins"]
    va_pct = value_area_pct or cfg["value_area_pct"]

    if df is None or len(df) < 5:
        return _empty_profile(df["close"].iloc[-1] if df is not None and len(df) > 0 else 0)

    lo = df["low"].min()
    hi = df["high"].max()

    if hi == lo:
        return _empty_profile(df["close"].iloc[-1])

    # Distribute volume uniformly across each candle's price range
    bins = np.linspace(lo, hi, n_bins + 1)
    price_levels = (bins[:-1] + bins[1:]) / 2.0
    vol_hist = np.zeros(n_bins)

    for _, row in df.iterrows():
        # Find which bins the candle spans
        low_bin = np.searchsorted(bins, row["low"], side="left")
        high_bin = np.searchsorted(bins, row["high"], side="right")
        low_bin = max(0, low_bin - 1)
        high_bin = min(n_bins - 1, high_bin)
        span = high_bin - low_bin + 1
        if span > 0:
            vol_hist[low_bin:high_bin + 1] += row["volume"] / span

    # POC = bin with highest volume
    poc_idx = int(np.argmax(vol_hist))
    poc_price = float(price_levels[poc_idx])

    # Value Area: start at POC, expand to adjacent bins until VA_PCT of volume included
    total_vol = float(vol_hist.sum())
    target_vol = total_vol * va_pct

    vah_idx, val_idx = poc_idx, poc_idx
    accumulated = float(vol_hist[poc_idx])

    while accumulated < target_vol:
        up_vol = float(vol_hist[vah_idx + 1]) if vah_idx + 1 < n_bins else 0.0
        dn_vol = float(vol_hist[val_idx - 1]) if val_idx - 1 >= 0 else 0.0

        if up_vol == 0 and dn_vol == 0:
            break
        if up_vol >= dn_vol:
            vah_idx = min(vah_idx + 1, n_bins - 1)
            accumulated += up_vol
        else:
            val_idx = max(val_idx - 1, 0)
            accumulated += dn_vol

    vah = float(price_levels[vah_idx])
    val = float(price_levels[val_idx])

    # Rough buy/sell imbalance: candles where close > open are "buy" volume
    buy_vol = float(df.loc[df["close"] >= df["open"], "volume"].sum())
    sell_vol = float(df["volume"].sum() - buy_vol)

    return {
        "poc_price": poc_price,
        "vah": vah,
        "val": val,
        "total_volume": total_vol,
        "poc_volume": float(vol_hist[poc_idx]),
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
    }


def extract_amt_features(df: pd.DataFrame, current_price: float) -> np.ndarray:
    """
    Extract 6 AMT features from an OHLCV DataFrame.

    Returns np.ndarray of shape (6,), values in [-1, 1].
    """
    if df is None or len(df) < 10:
        return np.zeros(6, dtype=np.float32)

    try:
        profile = build_volume_profile(df)

        poc_price = profile["poc_price"]
        vah = profile["vah"]
        val = profile["val"]
        total_vol = profile["total_volume"]
        poc_vol = profile["poc_volume"]
        buy_vol = profile["buy_volume"]
        sell_vol = profile["sell_volume"]

        # Normalised distances (signed: positive = price above level)
        poc_dist = _signed_dist(current_price, poc_price)
        vah_dist = _signed_dist(current_price, vah)
        val_dist = _signed_dist(current_price, val)

        # Inside Value Area flag
        in_va = 1.0 if val <= current_price <= vah else 0.0

        # POC volume concentration (how strong is the POC?)
        poc_pct = float(poc_vol / total_vol) if total_vol > 0 else 0.0
        poc_pct_norm = float(np.clip(poc_pct * 5.0 - 1.0, -1.0, 1.0))  # 20% → neutral

        # Buy/sell imbalance [-1 (all sell) … +1 (all buy)]
        total = buy_vol + sell_vol
        imbalance = float((buy_vol - sell_vol) / total) if total > 0 else 0.0

        return np.array([
            poc_dist,
            vah_dist,
            val_dist,
            in_va,
            poc_pct_norm,
            imbalance,
        ], dtype=np.float32)

    except Exception:
        return np.zeros(6, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def _signed_dist(price: float, level: float, scale: float = 20.0) -> float:
    """
    Signed normalised distance: (price - level) / level, clipped to [-1, 1].
    scale=20 means ±5% price deviation maps to ±1.0.
    """
    if level == 0:
        return 0.0
    return float(np.clip((price - level) / level * scale, -1.0, 1.0))


def _empty_profile(price: float) -> dict:
    return {
        "poc_price": price,
        "vah": price,
        "val": price,
        "total_volume": 0.0,
        "poc_volume": 0.0,
        "buy_volume": 0.0,
        "sell_volume": 0.0,
    }
