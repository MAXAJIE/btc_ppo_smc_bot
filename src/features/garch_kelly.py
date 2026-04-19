"""
garch_kelly.py  –  GARCH(1,1) + fractional Kelly  (FIXED)
==========================================================
Old bug: the function returned raw GARCH parameters (omega/alpha/beta)
         which are extremely small numbers (1e-6 range) and a raw Kelly
         fraction.  These were placed directly into the observation vector
         without normalisation, effectively contributing ~0 signal while
         adding noise.

Fix: return a 4-dim float32 array of normalised, interpretable quantities:
  [0] garch_vol_norm   – forecast daily volatility as % of price, clipped [0, 1]
  [1] vol_ratio        – current 5m realised vol / 20-bar rolling vol (regime signal)
  [2] kelly_fraction   – quarter-Kelly position size [0, 1]
  [3] kelly_confidence – confidence in the kelly estimate (0 if insufficient data)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_FIT_LEN    = 200    # minimum bars to fit GARCH (was 500 — too slow on 1h slices)
KELLY_CAP      = 0.25   # quarter-Kelly cap
FALLBACK_VOL   = 0.02   # 2% daily vol fallback when GARCH fails


def compute_garch_kelly(close_series: pd.Series) -> np.ndarray:
    """
    Parameters
    ----------
    close_series : pd.Series
        A recent window of closing prices (5m bars recommended, last 500).

    Returns
    -------
    np.ndarray, shape (4,), dtype float32
    """
    if len(close_series) < 10:
        return np.zeros(4, dtype=np.float32)

    log_rets = np.log(close_series / close_series.shift(1)).dropna().values

    garch_vol, confidence = _fit_garch(log_rets)

    # Convert per-bar vol to daily (assuming 5m bars: 288 bars/day)
    daily_vol = garch_vol * np.sqrt(288)

    # Current realised vol vs rolling vol
    current_vol = float(log_rets[-5:].std())   if len(log_rets) >= 5  else garch_vol
    roll_vol    = float(log_rets[-20:].std())   if len(log_rets) >= 20 else garch_vol
    vol_ratio   = float(np.clip(current_vol / (roll_vol + 1e-8), 0.0, 5.0))

    # Kelly fraction: simple heuristic (win_rate and win/loss ratio unknown here;
    # use vol-adjusted approach: smaller position when vol is high)
    base_kelly  = KELLY_CAP
    vol_adj     = float(np.clip(FALLBACK_VOL / (daily_vol + 1e-8), 0.1, 1.0))
    kelly       = float(np.clip(base_kelly * vol_adj, 0.05, KELLY_CAP))

    result = np.array(
        [
            float(np.clip(daily_vol, 0.0, 1.0)),   # normalised daily vol
            vol_ratio,                              # vol regime signal
            kelly,                                  # position size suggestion
            confidence,                             # fit quality
        ],
        dtype=np.float32,
    )
    return result


# ---------------------------------------------------------------------------
# GARCH(1,1) fitting (try arch library, fall back to rolling std)
# ---------------------------------------------------------------------------

def _fit_garch(log_rets: np.ndarray):
    """
    Returns (forecast_vol_per_bar, confidence).
    confidence = 1.0 if GARCH fitted, 0.5 if rolling std fallback.
    """
    if len(log_rets) >= MIN_FIT_LEN:
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am  = arch_model(log_rets * 100, vol="Garch", p=1, q=1,
                                 dist="normal", rescale=False)
                res = am.fit(disp="off", show_warning=False)
            forecast     = res.forecast(horizon=1, reindex=False)
            garch_var    = forecast.variance.values[-1, 0]
            garch_vol    = float(np.sqrt(max(garch_var, 0.0))) / 100.0
            return garch_vol, 1.0
        except Exception as exc:
            logger.debug("GARCH fit failed: %s", exc)

    # Fallback: 20-bar rolling std
    roll_vol = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else FALLBACK_VOL
    return roll_vol, 0.5
