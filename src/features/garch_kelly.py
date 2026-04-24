"""
garch_kelly.py  –  GARCH(1,1) + fractional Kelly
=================================================

Improvement from code review
------------------------------
Kelly halved when GARCH confidence < 1.0 (i.e. GARCH fit failed and we
fell back to rolling-std).  Rolling-std underestimates vol during regime
shifts, causing the model to oversize positions exactly when it's most
uncertain.  Halving kelly on low confidence is a simple but effective
risk gate.

Output: (4,) float32
  [0] daily_vol_norm   clipped [0, 1]
  [1] vol_ratio        current / rolling vol  [0, 5]
  [2] kelly_fraction   [0.025, 0.25]  (halved when confidence < 1.0)
  [3] fit_confidence   1.0 GARCH | 0.5 rolling-std fallback
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_FIT_LEN  = 200
KELLY_CAP    = 0.25
KELLY_MIN    = 0.025    # floor when halved: 0.05 / 2
FALLBACK_VOL = 0.02


def compute_garch_kelly(close_series: pd.Series) -> np.ndarray:
    if len(close_series) < 10:
        return np.array([FALLBACK_VOL, 1.0, KELLY_CAP, 0.0], dtype=np.float32)

    log_rets = np.log(
        close_series.values / np.roll(close_series.values, 1)
    )[1:]

    garch_vol, confidence = _fit_garch(log_rets)

    daily_vol = float(garch_vol * np.sqrt(288))   # 288 × 5m bars per day

    cur_vol  = float(np.std(log_rets[-5:]))  if len(log_rets) >= 5  else garch_vol
    roll_vol = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else garch_vol
    vol_ratio = float(np.clip(cur_vol / (roll_vol + 1e-8), 0.0, 5.0))

    vol_adj = float(np.clip(FALLBACK_VOL / (daily_vol + 1e-8), 0.1, 1.0))
    kelly   = float(np.clip(KELLY_CAP * vol_adj, 0.05, KELLY_CAP))

    # Improvement: halve kelly when GARCH fit failed (confidence == 0.5)
    # Rolling-std underestimates vol in regime shifts → don't oversize
    if confidence < 1.0:
        kelly = max(kelly * 0.5, KELLY_MIN)
        logger.debug(
            "GARCH confidence=%.1f — kelly halved to %.4f", confidence, kelly
        )

    return np.array(
        [np.clip(daily_vol, 0.0, 1.0), vol_ratio, kelly, confidence],
        dtype=np.float32,
    )


def _fit_garch(log_rets: np.ndarray) -> Tuple[float, float]:
    if len(log_rets) >= MIN_FIT_LEN:
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am  = arch_model(log_rets * 100, vol="Garch", p=1, q=1,
                                 dist="normal", rescale=False)
                res = am.fit(disp="off", show_warning=False)
            fc       = res.forecast(horizon=1, reindex=False)
            garch_var = fc.variance.values[-1, 0]
            vol       = float(np.sqrt(max(garch_var, 0.0))) / 100.0
            return vol, 1.0
        except Exception as e:
            logger.debug("GARCH fit failed: %s", e)

    roll = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else FALLBACK_VOL
    return roll, 0.5