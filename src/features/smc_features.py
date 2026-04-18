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
  7. swing_high_dist – normalised distance to nearest swing high
  8. swing_low_dist  – normalised distance to nearest swing low

Total: 8 features × 2 timeframes (5m, 1h) = 16 features
"""

import numpy as np
import pandas as pd
import yaml
import os

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

def extract_smc_features(df: pd.DataFrame, current_price: float) -> np.ndarray:
    """
    Extract 8 SMC features from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with columns ['open','high','low','close','volume']
        Must have at least 100 rows.
    current_price : float
        Most recent close price (for distance normalisation).

    Returns
    -------
    np.ndarray of shape (8,), all values in [-1, 1]
    """
    if not SMC_AVAILABLE or len(df) < 60:
        return np.zeros(8, dtype=np.float32)

    try:
        cfg = _cfg()
        swing_len = cfg["swing_length"]

        # ── Step 1: Swing Highs / Lows ───────────────────────────────
        swing_hl = smc.swing_highs_lows(df, swing_length=swing_len)

        # ── Step 2: Order Blocks ──────────────────────────────────────
        ob_df = smc.ob(df, swing_hl, close_mitigation=cfg["ob_mitigation"])

        # ── Step 3: Fair Value Gaps ───────────────────────────────────
        fvg_df = smc.fvg(df, join_consecutive=cfg["fvg_join_consecutive"])

        # ── Step 4: BOS / CHOCH ───────────────────────────────────────
        bos_df = smc.bos_choch(df, swing_hl)

        # ─────────────────────────────────────────────────────────────
        # Extract distances / counts
        # ─────────────────────────────────────────────────────────────

        ob_bull_dist = _nearest_ob_dist(ob_df, current_price, direction=1)
        ob_bear_dist = _nearest_ob_dist(ob_df, current_price, direction=-1)

        fvg_bull_act = _fvg_active(fvg_df, direction=1)
        fvg_bear_act = _fvg_active(fvg_df, direction=-1)

        bos_bull_cnt, choch_cnt = _bos_choch_counts(bos_df, lookback=20)

        sh_dist = _swing_dist(swing_hl, df["high"], current_price, kind=1)
        sl_dist = _swing_dist(swing_hl, df["low"], current_price, kind=-1)

        return np.array([
            ob_bull_dist,
            ob_bear_dist,
            fvg_bull_act,
            fvg_bear_act,
            bos_bull_cnt,
            choch_cnt,
            sh_dist,
            sl_dist,
        ], dtype=np.float32)

    except Exception as e:
        # Graceful degradation — never crash the env
        return np.zeros(8, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_ob_dist(ob_df: pd.DataFrame, price: float, direction: int) -> float:
    """
    Returns normalised distance to nearest active order block in the given direction.
    direction: 1 = bullish OB (below price), -1 = bearish OB (above price)

    Returns value in [0, 1] where 0 = no OB found, near-0 = OB very close.
    Inverted: closer OB → larger feature value (more relevant).
    """
    try:
        if ob_df is None or ob_df.empty:
            return 0.0

        # OB column: 1 = bullish, -1 = bearish
        mask = ob_df["OB"] == direction
        active_obs = ob_df[mask].dropna()

        if active_obs.empty:
            return 0.0

        # For bullish OBs: use 'Top' column (upper boundary)
        # For bearish OBs: use 'Bottom' column (lower boundary)
        col = "Top" if direction == 1 else "Bottom"
        if col not in active_obs.columns:
            return 0.0

        levels = active_obs[col].dropna().values
        if len(levels) == 0:
            return 0.0

        dists = np.abs(levels - price) / price
        min_dist = float(np.min(dists))

        # Invert and clip: dist=0 → 1.0, dist=5% → ~0.5, dist>20% → ~0
        return float(np.clip(1.0 / (1.0 + min_dist * 20.0), 0.0, 1.0))

    except Exception:
        return 0.0


def _fvg_active(fvg_df: pd.DataFrame, direction: int) -> float:
    """1.0 if an unmitigated FVG exists in the given direction, else 0.0."""
    try:
        if fvg_df is None or fvg_df.empty:
            return 0.0
        # FVG column: 1 = bullish, -1 = bearish; NaN if not mitigated
        mask = (fvg_df["FVG"] == direction) & (fvg_df["MitigatedIndex"].isna())
        return 1.0 if mask.any() else 0.0
    except Exception:
        return 0.0


def _bos_choch_counts(bos_df: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Count bullish BOS and CHoCH events in the last `lookback` bars.
    Returns (bos_bull_norm, choch_norm) normalised to [0, 1].
    """
    try:
        if bos_df is None or bos_df.empty:
            return 0.0, 0.0

        recent = bos_df.iloc[-lookback:]

        bos_bull = 0
        choch = 0

        if "BOS" in recent.columns:
            bos_bull = int((recent["BOS"] == 1).sum())

        if "CHOCH" in recent.columns:
            choch = int((recent["CHOCH"] != 0).sum())

        max_count = float(lookback)
        return float(bos_bull / max_count), float(choch / max_count)

    except Exception:
        return 0.0, 0.0


def _swing_dist(swing_hl: pd.DataFrame, price_series: pd.Series, current_price: float, kind: int) -> float:
    """
    Normalised distance to nearest swing high (kind=1) or swing low (kind=-1).
    Returns value in [0, 1] — higher = swing is close.
    """
    try:
        if swing_hl is None or swing_hl.empty:
            return 0.0

        mask = swing_hl["HighLow"] == kind
        swing_idx = swing_hl[mask].index

        if len(swing_idx) == 0:
            return 0.0

        swing_prices = price_series.loc[swing_idx].dropna().values
        if len(swing_prices) == 0:
            return 0.0

        dists = np.abs(swing_prices - current_price) / current_price
        min_dist = float(np.min(dists))
        return float(np.clip(1.0 / (1.0 + min_dist * 20.0), 0.0, 1.0))

    except Exception:
        return 0.0
