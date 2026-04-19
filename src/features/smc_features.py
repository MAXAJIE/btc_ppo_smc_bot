"""
smc_features.py
───────────────
Smart Money Concepts features using the `smartmoneyconcepts` library.

Features extracted per timeframe:
  1. ob_bull_dist   – normalised distance to nearest bullish order block
  2. ob_bear_dist   – normalised distance to nearest bearish order block
  3. fvg_bull_act   – 1 if an unmitigated bullish FVG is present
  4. fvg_bear_act   – 1 if an unmitigated bearish FVG is present
  5. bos_bull_cnt   – count of bullish BOS in last 20 bars (normalised)
  6. choch_cnt      – count of CHoCH in last 20 bars (normalised)
  7. swing_high_dist – distance to nearest swing high
  8. swing_low_dist  – distance to nearest swing low

All 8 features pass through RunningZScore normalisation [2]:
  • Continuous distances (1,2,7,8): z-score → stable distribution regardless
    of regime volatility or market structure density
  • Binary/count features (3,4,5,6): passed through as-is, already [0,1]

Total: 8 features × 2 timeframes (5m, 1h) = 16 features

Changes vs v2  [2]
──────────────────
Added RunningZScore class per feature × per timeframe.
Continuous OB distances and swing distances are z-scored using a 200-sample
rolling window.  This ensures:
  - Zero mean (feature distribution is centred regardless of regime)
  - Approximately unit variance (PPO optimizer sees consistent gradient scale)
  - No raw price levels ever reach the policy network
  - Graceful warm-up: returns raw values until 20+ samples collected
"""

import numpy as np
import pandas as pd
import yaml
import os
from collections import deque

try:
    import smartmoneyconcepts as smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("[WARN] smartmoneyconcepts not installed — SMC features will be zeros")


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = _load_cfg()["features"]["smc"]
    return _CFG


# ─────────────────────────────────────────────────────────────────────────────
# [2] Running z-score normaliser
# ─────────────────────────────────────────────────────────────────────────────

class RunningZScore:
    """
    Online z-score normaliser with a rolling window.

    Maintains a deque of the last `window` observed values.
    On each call to `normalise(x)`, returns:

        z = clip((x - μ) / (σ + ε), -clip_val, clip_val) / clip_val

    This maps the distribution to approximately [-1, 1] with:
      • μ = rolling mean of the window
      • σ = rolling std of the window
      • clip_val = 3.0  (3σ outliers → ±1.0)

    Warm-up behaviour:
      - Until `min_samples` values have been seen, returns the raw value
        passed through the same ±1 clip.  This avoids garbage z-scores
        from a near-empty window on episode start.

    Thread safety:
      Each timeframe × feature pair gets its own RunningZScore instance
      (stored in _NORMALIZERS dict), so no locking is needed.
    """

    def __init__(self, window: int = 200, min_samples: int = 20, clip_val: float = 3.0):
        self.window     = window
        self.min_samples = min_samples
        self.clip_val   = clip_val
        self._buf: deque = deque(maxlen=window)

    def normalise(self, x: float) -> float:
        """Update buffer with x, return z-scored value."""
        self._buf.append(float(x))

        if len(self._buf) < self.min_samples:
            # Warm-up: return raw value clipped to [-1, 1]
            return float(np.clip(x, -1.0, 1.0))

        arr = np.array(self._buf, dtype=np.float64)
        mu  = arr.mean()
        std = arr.std() + 1e-8
        z   = (x - mu) / std
        return float(np.clip(z / self.clip_val, -1.0, 1.0))

    def reset(self):
        """Clear buffer — call on env reset to avoid inter-episode leakage."""
        self._buf.clear()


# ── Per-(timeframe, feature_index) normaliser registry ───────────────────────
# Key: (timeframe_label: str, feature_idx: int)
# Indices 0,1,6,7 are continuous distances → z-scored
# Indices 2,3,4,5 are binary/count → pass-through (no normalisation)
_CONTINUOUS_IDXS = {0, 1, 6, 7}   # ob_bull_dist, ob_bear_dist, sh_dist, sl_dist

_NORMALIZERS: dict[tuple, RunningZScore] = {}


def _get_norm(tf_label: str, feat_idx: int) -> RunningZScore:
    key = (tf_label, feat_idx)
    if key not in _NORMALIZERS:
        _NORMALIZERS[key] = RunningZScore(window=200, min_samples=20)
    return _NORMALIZERS[key]


def reset_normalizers():
    """
    Clear all normaliser buffers.
    Call this on environment reset to prevent information bleeding
    between episodes (particularly important in offline training
    where episodes are randomly sampled from different time periods).
    """
    for norm in _NORMALIZERS.values():
        norm.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_smc_features(
    df: pd.DataFrame,
    current_price: float,
    tf_label: str = "5m",
) -> np.ndarray:
    """
    Extract 8 SMC features, continuous ones z-score normalised.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with columns ['open','high','low','close','volume'].
        Must have at least 60 rows.
    current_price : float
        Most recent close price (for distance normalisation).
    tf_label : str
        Label used to key the per-timeframe normaliser registry
        (e.g. "5m", "1h").  Different timeframes maintain separate
        rolling statistics.

    Returns
    -------
    np.ndarray of shape (8,), values in [-1, 1].
    """
    if not SMC_AVAILABLE or len(df) < 60:
        return np.zeros(8, dtype=np.float32)

    try:
        cfg = _cfg()
        swing_len = cfg["swing_length"]

        swing_hl = smc.swing_highs_lows(df, swing_length=swing_len)
        ob_df    = smc.ob(df, swing_hl, close_mitigation=cfg["ob_mitigation"])
        fvg_df   = smc.fvg(df, join_consecutive=cfg["fvg_join_consecutive"])
        bos_df   = smc.bos_choch(df, swing_hl)

        raw = np.array([
            _nearest_ob_dist(ob_df,  current_price, direction=1),   # 0
            _nearest_ob_dist(ob_df,  current_price, direction=-1),  # 1
            _fvg_active(fvg_df,      direction=1),                  # 2
            _fvg_active(fvg_df,      direction=-1),                 # 3
            *_bos_choch_counts(bos_df, lookback=20),                # 4, 5
            _swing_dist(swing_hl, df["high"], current_price, kind=1),   # 6
            _swing_dist(swing_hl, df["low"],  current_price, kind=-1),  # 7
        ], dtype=np.float64)

        # Apply z-score to continuous distance features only
        out = np.empty(8, dtype=np.float32)
        for i, v in enumerate(raw):
            if i in _CONTINUOUS_IDXS:
                out[i] = float(_get_norm(tf_label, i).normalise(v))
            else:
                # Binary / count features: already [0,1], just clip
                out[i] = float(np.clip(v, 0.0, 1.0))

        return out

    except Exception:
        return np.zeros(8, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Distance / count helpers — produce price-relative values in [0, 1]
# (these feed into RunningZScore, so they don't need to be ±1 themselves)
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_ob_dist(ob_df: pd.DataFrame, price: float, direction: int) -> float:
    """
    % distance to nearest active order block, mapped to [0, 1].
    0 = no OB found; 1 = OB exactly at current price.
    Inverted so closer OB → larger value.
    """
    try:
        if ob_df is None or ob_df.empty:
            return 0.0
        mask = ob_df["OB"] == direction
        active = ob_df[mask].dropna()
        if active.empty:
            return 0.0
        col = "Top" if direction == 1 else "Bottom"
        if col not in active.columns:
            return 0.0
        levels = active[col].dropna().values
        if len(levels) == 0:
            return 0.0
        dists = np.abs(levels - price) / price
        min_dist = float(np.min(dists))
        return float(np.clip(1.0 / (1.0 + min_dist * 20.0), 0.0, 1.0))
    except Exception:
        return 0.0


def _fvg_active(fvg_df: pd.DataFrame, direction: int) -> float:
    """1.0 if an unmitigated FVG exists in the given direction, else 0.0."""
    try:
        if fvg_df is None or fvg_df.empty:
            return 0.0
        mask = (fvg_df["FVG"] == direction) & (fvg_df["MitigatedIndex"].isna())
        return 1.0 if mask.any() else 0.0
    except Exception:
        return 0.0


def _bos_choch_counts(bos_df: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Count of bullish BOS and CHoCH events in the last `lookback` bars.
    Returns (bos_bull_norm, choch_norm), each in [0, 1].
    """
    try:
        if bos_df is None or bos_df.empty:
            return 0.0, 0.0
        recent = bos_df.iloc[-lookback:]
        bos_bull = int((recent["BOS"]   == 1).sum()) if "BOS"   in recent.columns else 0
        choch    = int((recent["CHOCH"] != 0).sum()) if "CHOCH" in recent.columns else 0
        return float(bos_bull / lookback), float(choch / lookback)
    except Exception:
        return 0.0, 0.0


def _swing_dist(swing_hl: pd.DataFrame, price_series: pd.Series,
                current_price: float, kind: int) -> float:
    """
    % distance to nearest swing high (kind=1) or swing low (kind=-1),
    mapped to [0, 1] — higher = swing is closer.
    """
    try:
        if swing_hl is None or swing_hl.empty:
            return 0.0
        idx = swing_hl[swing_hl["HighLow"] == kind].index
        if len(idx) == 0:
            return 0.0
        prices = price_series.loc[idx].dropna().values
        if len(prices) == 0:
            return 0.0
        dists = np.abs(prices - current_price) / current_price
        return float(np.clip(1.0 / (1.0 + float(np.min(dists)) * 20.0), 0.0, 1.0))
    except Exception:
        return 0.0
