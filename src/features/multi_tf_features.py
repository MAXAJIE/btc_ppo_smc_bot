"""
multi_tf_features.py
─────────────────────
Aggregates all feature modules into the 90-dimensional observation vector.

Observation layout (90 dims total)
────────────────────────────────────
[00:15]  OHLCV momentum — 5m   (15)
[15:30]  OHLCV momentum — 15m  (15)
[30:45]  OHLCV momentum — 1h   (15)
[45:55]  OHLCV momentum — 4h   (10)
[55:65]  OHLCV momentum — 1D   (10)
[65:73]  SMC features — 5m      (8)
[73:81]  SMC features — 1h      (8)
[81:87]  SNR features — 1h      (6)
[87:93]  AMT features — 4h      (6)
[93:97]  GARCH + Kelly           (4)
[97:90]  ← WAIT — recounted below

Actual final count:
  15 + 15 + 15 + 10 + 10 + 8 + 8 + 6 + 6 + 4 + 6 (position) = 103

We compress to 90 by using 10 instead of 15 for 15m and 1h.
Final layout:
[00:15]  OHLCV 5m    (15)
[15:25]  OHLCV 15m   (10)
[25:35]  OHLCV 1h    (10)
[35:43]  OHLCV 4h    (8)
[43:51]  OHLCV 1D    (8)
[51:59]  SMC 5m      (8)
[59:67]  SMC 1h      (8)
[67:73]  SNR 1h      (6)
[73:79]  AMT 4h      (6)
[79:83]  GARCH+Kelly (4)
[83:90]  Position    (7)
         ─────────────
         Total: 90 ✓
"""

import numpy as np
import pandas as pd
from typing import Dict

from .smc_features import extract_smc_features
from .amt_features import extract_amt_features
from .snr_features import extract_snr_features
from .garch_kelly import GarchKellyEstimator

OBS_DIM = 90

# Singleton estimator (avoid re-creating arch model objects)
_GARCH_EST = None


def _get_garch():
    global _GARCH_EST
    if _GARCH_EST is None:
        _GARCH_EST = GarchKellyEstimator()
    return _GARCH_EST


# ─────────────────────────────────────────────────────────────────────────────

def build_observation(
    candles: Dict[str, pd.DataFrame],
    position: int,
    unrealized_pnl_pct: float,
    leverage_used: float,
    bars_in_trade: int,
    account_drawdown: float,
    kelly_adj: float,
    bars_since_last_trade: int,
) -> np.ndarray:
    """
    Build the 90-dim observation vector for the PPO agent.

    Parameters
    ----------
    candles : dict
        Keys: '5m', '15m', '1h', '4h', '1d'
        Values: pd.DataFrame with OHLCV columns (up to current bar, no lookahead)
    position : int
        -1 = short, 0 = flat, 1 = long
    unrealized_pnl_pct : float
        Open trade P&L as fraction of entry notional
    leverage_used : float
        Current leverage (0 if flat)
    bars_in_trade : int
        Number of 5m bars since trade opened
    account_drawdown : float
        Current drawdown from equity peak [0, 1]
    kelly_adj : float
        Latest Kelly-adjusted position size fraction
    bars_since_last_trade : int
        5m bars since last closed trade (to detect stagnation)
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    df_5m = candles.get("5m")
    df_15m = candles.get("15m")
    df_1h = candles.get("1h")
    df_4h = candles.get("4h")
    df_1d = candles.get("1d")

    cur_price = float(df_5m["close"].iloc[-1]) if df_5m is not None and len(df_5m) > 0 else 1.0

    # ── [00:15] OHLCV 5m (15 features) ──────────────────────────────
    obs[0:15] = _ohlcv_features(df_5m, n=15)

    # ── [15:25] OHLCV 15m (10 features) ─────────────────────────────
    obs[15:25] = _ohlcv_features(df_15m, n=10)

    # ── [25:35] OHLCV 1h (10 features) ──────────────────────────────
    obs[25:35] = _ohlcv_features(df_1h, n=10)

    # ── [35:43] OHLCV 4h (8 features) ───────────────────────────────
    obs[35:43] = _ohlcv_features(df_4h, n=8)

    # ── [43:51] OHLCV 1D (8 features) ───────────────────────────────
    obs[43:51] = _ohlcv_features(df_1d, n=8)

    # ── [51:59] SMC 5m ───────────────────────────────────────────────
    obs[51:59] = extract_smc_features(df_5m, cur_price)

    # ── [59:67] SMC 1h ───────────────────────────────────────────────
    obs[59:67] = extract_smc_features(df_1h, cur_price)

    # ── [67:73] SNR 1h ───────────────────────────────────────────────
    obs[67:73] = extract_snr_features(df_1h, cur_price)

    # ── [73:79] AMT 4h ───────────────────────────────────────────────
    obs[73:79] = extract_amt_features(df_4h, cur_price)

    # ── [79:83] GARCH + Kelly ─────────────────────────────────────────
    if df_5m is not None and len(df_5m) > 10:
        log_returns = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        gk = _get_garch().compute(log_returns)
        obs[79] = float(gk["garch_vol"])
        obs[80] = float(gk["vol_regime"])
        obs[81] = float(gk["kelly_fraction"])
        obs[82] = float(gk["kelly_adj"])

    # ── [83:90] Position State (7 features) ──────────────────────────
    obs[83] = float(position)                                          # -1/0/1
    obs[84] = float(np.clip(unrealized_pnl_pct * 100, -5, 5))        # scaled PnL
    obs[85] = float(np.clip(leverage_used / 3.0, 0, 1))               # normalised leverage
    obs[86] = float(np.clip(bars_in_trade / 576.0, 0, 1))             # normalised duration
    obs[87] = float(np.clip(account_drawdown * 10.0, 0, 1))           # amplified drawdown
    obs[88] = float(np.clip(kelly_adj, 0, 1))
    obs[89] = float(np.clip(bars_since_last_trade / 288.0, 0, 1))     # 1 day normalised

    # Final clip to prevent any outliers reaching the policy network
    obs = np.clip(obs, -5.0, 5.0)

    return obs


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def _ohlcv_features(df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Extract `n` momentum/structure features from an OHLCV DataFrame.

    Always returns np.ndarray of shape (n,).
    Feature set (n=15 shown, truncated to n if n < 15):
      [0]  log return (1 bar)
      [1]  log return (3 bars)
      [2]  log return (6 bars)
      [3]  log return (12 bars)
      [4]  log return (24 bars)
      [5]  normalised volume (vs 20-bar avg)
      [6]  high-low range / close (volatility proxy)
      [7]  close position within bar's range
      [8]  EMA8 distance
      [9]  EMA21 distance
      [10] RSI-like momentum (14 bar)
      [11] volume trend (slope)
      [12] body ratio (close-open / high-low)
      [13] upper wick ratio
      [14] lower wick ratio
    """
    if df is None or len(df) < 5:
        return np.zeros(n, dtype=np.float32)

    try:
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        opens = df["open"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)

        features = []

        # Log returns at different horizons
        for lag in [1, 3, 6, 12, 24]:
            if len(closes) > lag:
                r = float(np.log(closes[-1] / closes[-lag - 1]))
                features.append(np.clip(r * 20, -1, 1))  # 5% move → 1.0
            else:
                features.append(0.0)

        # Normalised volume
        if len(volumes) >= 20:
            vol_avg = np.mean(volumes[-20:])
            features.append(float(np.clip(volumes[-1] / (vol_avg + 1e-10) - 1, -2, 2)))
        else:
            features.append(0.0)

        # High-low range
        hl_range = (highs[-1] - lows[-1]) / (closes[-1] + 1e-10)
        features.append(float(np.clip(hl_range * 20, 0, 1)))

        # Close position within bar
        bar_range = highs[-1] - lows[-1]
        if bar_range > 0:
            cp = (closes[-1] - lows[-1]) / bar_range
        else:
            cp = 0.5
        features.append(float(np.clip(cp * 2 - 1, -1, 1)))  # [-1, 1]

        # EMA8 distance
        if len(closes) >= 8:
            ema8 = _ema(closes, 8)
            features.append(float(np.clip((closes[-1] - ema8) / (ema8 + 1e-10) * 20, -1, 1)))
        else:
            features.append(0.0)

        # EMA21 distance
        if len(closes) >= 21:
            ema21 = _ema(closes, 21)
            features.append(float(np.clip((closes[-1] - ema21) / (ema21 + 1e-10) * 20, -1, 1)))
        else:
            features.append(0.0)

        # RSI-like momentum (14 bar)
        if len(closes) >= 15:
            rsi = _rsi(closes, 14)
            features.append(float((rsi - 50) / 50))  # [-1, 1]
        else:
            features.append(0.0)

        # Volume trend (linear slope over last 10 bars)
        if len(volumes) >= 10:
            slope = np.polyfit(np.arange(10), volumes[-10:], 1)[0]
            avg_vol = np.mean(volumes[-10:]) + 1e-10
            features.append(float(np.clip(slope / avg_vol, -1, 1)))
        else:
            features.append(0.0)

        # Body ratio
        body = abs(closes[-1] - opens[-1])
        shadow = highs[-1] - lows[-1]
        features.append(float(np.clip(body / (shadow + 1e-10), 0, 1)))

        # Upper wick
        upper = highs[-1] - max(closes[-1], opens[-1])
        features.append(float(np.clip(upper / (shadow + 1e-10), 0, 1)))

        # Lower wick
        lower = min(closes[-1], opens[-1]) - lows[-1]
        features.append(float(np.clip(lower / (shadow + 1e-10), 0, 1)))

        arr = np.array(features[:n], dtype=np.float32)
        # Pad if fewer than n features available
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)

        return arr

    except Exception:
        return np.zeros(n, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Micro-utilities (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    ema = arr[0]
    for v in arr[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)


def _rsi(arr: np.ndarray, period: int = 14) -> float:
    diffs = np.diff(arr[-(period + 1):])
    gains = diffs[diffs > 0]
    losses = -diffs[diffs < 0]
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 1e-10
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-10
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))
