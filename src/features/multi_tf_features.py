"""
multi_tf_features.py
─────────────────────
Aggregates all feature modules into the 90-dimensional observation vector.

Observation layout (90 dims total)
────────────────────────────────────
[00:15]  OHLCV momentum — 5m   (15)
[15:25]  OHLCV momentum — 15m  (10)
[25:35]  OHLCV momentum — 1h   (10)
[35:43]  OHLCV momentum — 4h   (8)
[43:51]  OHLCV momentum — 1D   (8)
[51:59]  SMC features  — 5m    (8)
[59:67]  SMC features  — 1h    (8)
[67:73]  SNR levels    — 1h    (6)
[73:79]  AMT/Vol Prof  — 4h    (6)
[79:83]  GARCH+Kelly           (4)
[83:90]  Position state        (7)
         ─────────────
         Total: 90 ✓

Stationarity guarantees  [C]
─────────────────────────────
Every feature entering the observation vector is either:
  1. A log-return (first difference of log-price)      → I(0) by construction
  2. A ratio relative to a rolling window              → approximately I(0)
  3. A z-score normalised with a rolling window mean+std → zero-mean, unit-var
  4. A bounded ratio (wick/body/range) in [0,1] or [-1,1] → already stationary
  5. A distance to a level, expressed as % of price    → approximately I(0)

Non-stationary inputs (raw price levels, raw volumes) are NEVER passed to
the policy network.  This is enforced in _ohlcv_features() below.
"""

import numpy as np
import pandas as pd
from typing import Dict

from .smc_features import extract_smc_features
from .amt_features import extract_amt_features
from .snr_features import extract_snr_features
from .garch_kelly import GarchKellyEstimator

OBS_DIM = 90

# Singleton GARCH estimator (avoid re-creating arch model objects every step)
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

    All features are stationary — see module docstring for guarantees.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    df_5m  = candles.get("5m")
    df_15m = candles.get("15m")
    df_1h  = candles.get("1h")
    df_4h  = candles.get("4h")
    df_1d  = candles.get("1d")

    cur_price = (
        float(df_5m["close"].iloc[-1])
        if df_5m is not None and len(df_5m) > 0
        else 1.0
    )

    # ── [00:15] OHLCV 5m (15 stationary features) ────────────────────
    obs[0:15]  = _ohlcv_features(df_5m, n=15)

    # ── [15:25] OHLCV 15m (10 stationary features) ───────────────────
    obs[15:25] = _ohlcv_features(df_15m, n=10)

    # ── [25:35] OHLCV 1h (10 stationary features) ────────────────────
    obs[25:35] = _ohlcv_features(df_1h, n=10)

    # ── [35:43] OHLCV 4h (8 stationary features) ─────────────────────
    obs[35:43] = _ohlcv_features(df_4h, n=8)

    # ── [43:51] OHLCV 1D (8 stationary features) ─────────────────────
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

    # ── [83:90] Position state (7 features) ──────────────────────────
    obs[83] = float(position)
    obs[84] = float(np.clip(unrealized_pnl_pct * 100, -5, 5))
    obs[85] = float(np.clip(leverage_used / 3.0, 0, 1))
    obs[86] = float(np.clip(bars_in_trade / 576.0, 0, 1))
    obs[87] = float(np.clip(account_drawdown * 10.0, 0, 1))
    obs[88] = float(np.clip(kelly_adj, 0, 1))
    obs[89] = float(np.clip(bars_since_last_trade / 288.0, 0, 1))

    # Hard clip — no outlier escapes to the policy network
    obs = np.clip(obs, -5.0, 5.0)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV feature extractor — stationarity enforced at each feature
# ─────────────────────────────────────────────────────────────────────────────

def _ohlcv_features(df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Extract `n` stationary features from an OHLCV DataFrame.

    Feature set (n=15 shown; truncated/padded to n as needed):
    ─────────────────────────────────────────────────────────
    [0]  log-return 1 bar           — I(0) by definition
    [1]  log-return 3 bars          — I(0)
    [2]  log-return 6 bars          — I(0)
    [3]  log-return 12 bars         — I(0)
    [4]  log-return 24 bars         — I(0)
    [5]  volume z-score (50-bar)    — [C] z-scored → zero-mean, stable σ
    [6]  high-low range / close     — price-normalised ratio → stationary
    [7]  close position in bar      — bounded [−1, 1] → stationary
    [8]  EMA8 distance z-score      — [C] z-scored 20-bar rolling window
    [9]  EMA21 distance z-score     — [C] z-scored 30-bar rolling window
    [10] RSI (0-centred, /50)       — bounded, stationary
    [11] EMA8 distance velocity     — [C] 1-bar Δ(EMA8_dist) → stationary
    [12] body ratio                 — bounded [0, 1] → stationary
    [13] upper wick ratio           — bounded [0, 1] → stationary
    [14] lower wick ratio           — bounded [0, 1] → stationary

    All clipped to [−1, 1] before returning.
    """
    if df is None or len(df) < 5:
        return np.zeros(n, dtype=np.float32)

    try:
        closes  = df["close"].values.astype(np.float64)
        highs   = df["high"].values.astype(np.float64)
        lows    = df["low"].values.astype(np.float64)
        opens   = df["open"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)

        features = []

        # ── [0-4] Log-returns at 5 horizons — inherently I(0) ────────
        for lag in [1, 3, 6, 12, 24]:
            if len(closes) > lag:
                r = float(np.log(closes[-1] / closes[-lag - 1]))
                features.append(float(np.clip(r * 20.0, -1.0, 1.0)))  # 5% → 1.0
            else:
                features.append(0.0)

        # ── [5] Volume z-score (50-bar rolling) ───────────────────────
        # Raw volume is non-stationary (grows with market cap).
        # z-score relative to a 50-bar rolling window is stationary.
        features.append(_zscore_last(volumes, window=50))

        # ── [6] High-low range / close — price-normalised ─────────────
        hl_range = (highs[-1] - lows[-1]) / (closes[-1] + 1e-10)
        features.append(float(np.clip(hl_range * 20.0, 0.0, 1.0)))

        # ── [7] Close position within bar ─────────────────────────────
        bar_range = highs[-1] - lows[-1]
        cp = (closes[-1] - lows[-1]) / bar_range if bar_range > 0 else 0.5
        features.append(float(np.clip(cp * 2.0 - 1.0, -1.0, 1.0)))

        # ── [8] EMA8 distance z-score ─────────────────────────────────
        # Raw (close - EMA) / EMA drifts slowly in trending markets.
        # Rolling z-score over 20 bars removes the drift.
        if len(closes) >= 8:
            ema8_dists = _rolling_ema_dists(closes, period=8, lookback=25)
            features.append(_zscore_last_from_series(ema8_dists))
        else:
            features.append(0.0)

        # ── [9] EMA21 distance z-score ────────────────────────────────
        if len(closes) >= 21:
            ema21_dists = _rolling_ema_dists(closes, period=21, lookback=35)
            features.append(_zscore_last_from_series(ema21_dists))
        else:
            features.append(0.0)

        # ── [10] RSI (centred) — bounded, stationary ──────────────────
        if len(closes) >= 15:
            rsi = _rsi(closes, 14)
            features.append(float((rsi - 50.0) / 50.0))
        else:
            features.append(0.0)

        # ── [11] EMA8 distance velocity — Δ(dist) per bar ─────────────
        # Rate-of-change of the EMA distance is strictly stationary:
        # it measures how fast price is moving relative to the trend.
        if len(closes) >= 10:
            ema8_dists = _rolling_ema_dists(closes, period=8, lookback=10)
            if len(ema8_dists) >= 2:
                velocity = float(ema8_dists[-1] - ema8_dists[-2])
                features.append(float(np.clip(velocity * 50.0, -1.0, 1.0)))
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        # ── [12-14] Candle structure ratios — bounded by construction ──
        body   = abs(closes[-1] - opens[-1])
        shadow = highs[-1] - lows[-1] + 1e-10
        upper  = highs[-1] - max(closes[-1], opens[-1])
        lower  = min(closes[-1], opens[-1]) - lows[-1]

        features.append(float(np.clip(body   / shadow, 0.0, 1.0)))
        features.append(float(np.clip(upper  / shadow, 0.0, 1.0)))
        features.append(float(np.clip(lower  / shadow, 0.0, 1.0)))

        # Assemble, truncate/pad to exactly n
        arr = np.array(features[:n], dtype=np.float32)
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)

        return arr

    except Exception:
        return np.zeros(n, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity helpers [C]
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_last(arr: np.ndarray, window: int = 50) -> float:
    """
    Z-score of the last value relative to a rolling window.

    z = (x_t - mean(x_{t-window:t})) / (std(x_{t-window:t}) + ε)

    Clips result to [-3, 3] then rescales to [-1, 1] so extreme
    outliers don't dominate the observation.
    """
    if len(arr) < 3:
        return 0.0
    w = min(window, len(arr))
    window_vals = arr[-w:]
    mu  = float(np.mean(window_vals))
    std = float(np.std(window_vals)) + 1e-8
    z   = (arr[-1] - mu) / std
    # Map ±3σ → ±1 (outside that is extreme noise, not signal)
    return float(np.clip(z / 3.0, -1.0, 1.0))


def _zscore_last_from_series(series: np.ndarray) -> float:
    """Z-score of last element from a pre-computed series array."""
    return _zscore_last(series, window=len(series))


def _rolling_ema_dists(closes: np.ndarray, period: int, lookback: int) -> np.ndarray:
    """
    Compute (close - EMA_period) / EMA_period for each bar in the
    last `lookback` bars.  Returns an array of price-relative distances.

    These distances are approximately stationary in a range-bound
    market; the z-score wrapper handles residual drift in trends.
    """
    n_bars = min(lookback, len(closes))
    dists  = np.zeros(n_bars, dtype=np.float64)

    for i in range(n_bars):
        end_idx = len(closes) - n_bars + i + 1
        sub     = closes[:end_idx]
        if len(sub) < period:
            continue
        ema = _ema(sub, period)
        if ema > 0:
            dists[i] = (sub[-1] - ema) / ema

    return dists


# ─────────────────────────────────────────────────────────────────────────────
# Micro-utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> float:
    """Exponential moving average of the last element of arr."""
    alpha = 2.0 / (period + 1.0)
    ema   = arr[0]
    for v in arr[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return float(ema)


def _rsi(arr: np.ndarray, period: int = 14) -> float:
    """Wilder RSI on the last `period+1` values of arr."""
    diffs    = np.diff(arr[-(period + 1):])
    gains    = diffs[diffs > 0]
    losses   = -diffs[diffs < 0]
    avg_gain = float(np.mean(gains))  if len(gains)  > 0 else 1e-10
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-10
    rs       = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))
