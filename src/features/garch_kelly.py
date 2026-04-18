"""
garch_kelly.py
──────────────
GARCH(1,1) volatility forecast + fractional Kelly position sizing.

GARCH gives us a 1-step-ahead volatility forecast.  We convert this
into a Kelly fraction: f* = edge / variance, then apply fractional
Kelly (quarter Kelly by default) for risk management.

The features returned are:
  • garch_vol          – annualised 1-step-ahead σ (normalised)
  • vol_regime         – 0=low, 0.5=medium, 1.0=high
  • kelly_fraction     – raw Kelly f* (clipped to [0,1])
  • kelly_adj          – fractional Kelly after applying cfg scale
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
import yaml
import os

warnings.filterwarnings("ignore")


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────

class GarchKellyEstimator:
    """
    Rolling GARCH(1,1) estimator with Kelly sizing.

    Usage
    ─────
        est = GarchKellyEstimator()
        features = est.compute(returns_series)
        # returns dict with 4 float values
    """

    def __init__(self):
        cfg = _load_cfg()
        self.g_cfg = cfg["features"]["garch"]
        self.r_cfg = cfg["risk"]

        self.window = self.g_cfg["returns_window"]      # 200 bars
        self.p = self.g_cfg["p"]                        # 1
        self.q = self.g_cfg["q"]                        # 1
        self.thresholds = self.g_cfg["vol_regime_thresholds"]  # [0.01, 0.025]
        self.kelly_scale = self.r_cfg["kelly_fraction"]  # 0.25 (quarter Kelly)
        self.min_k = self.r_cfg["min_kelly_size"]
        self.max_k = self.r_cfg["max_kelly_size"]

        # Cache last fit to avoid re-fitting every step
        self._last_vol: float = 0.01
        self._fit_counter: int = 0
        self._refit_every: int = 12  # refit every 12 steps (~1h on 5m tf)

    # ─────────────────────────────────────────────────────────────────

    def compute(self, returns: pd.Series) -> dict:
        """
        Compute GARCH + Kelly features from a rolling window of log-returns.

        Parameters
        ----------
        returns : pd.Series
            Log-returns of the close price, at least `window` bars long.

        Returns
        -------
        dict with keys: garch_vol, vol_regime, kelly_fraction, kelly_adj
        """
        if len(returns) < self.window:
            return self._fallback()

        r = returns.iloc[-self.window:].dropna()

        # Scale returns to percentage (arch library expects ~1-5 range, not 0.001)
        r_scaled = r * 100.0

        self._fit_counter += 1
        if self._fit_counter % self._refit_every == 0:
            self._last_vol = self._fit_garch(r_scaled)

        vol = self._last_vol  # annualised daily vol (%)

        # Vol regime
        regime = self._classify_regime(vol / 100.0)  # convert back to fraction

        # Kelly fraction: f* = (μ/σ²) approximation
        # We use a simple empirical edge estimate from the window
        mu = float(r.mean())
        sigma2 = float(r.var())
        kelly_raw = self._kelly(mu, sigma2)
        kelly_adj = float(np.clip(kelly_raw * self.kelly_scale, self.min_k, self.max_k))

        return {
            "garch_vol": float(np.clip(vol / 100.0, 0.0, 0.5)),   # daily vol fraction
            "vol_regime": regime,
            "kelly_fraction": float(np.clip(kelly_raw, 0.0, 1.0)),
            "kelly_adj": kelly_adj,
        }

    # ─────────────────────────────────────────────────────────────────

    def _fit_garch(self, r_scaled: pd.Series) -> float:
        """Fit GARCH(1,1) and return 1-step-ahead annualised vol (percentage)."""
        try:
            am = arch_model(r_scaled, vol="Garch", p=self.p, q=self.q, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecasts = res.forecast(horizon=1, reindex=False)
            var_1step = float(forecasts.variance.iloc[-1, 0])
            vol_daily_pct = float(np.sqrt(var_1step))
            # annualise: ×√(288) for 5m candles (288 per day)
            vol_annual_pct = vol_daily_pct * np.sqrt(288)
            return float(np.clip(vol_annual_pct, 0.01, 500.0))
        except Exception:
            return self._last_vol if self._last_vol > 0 else 1.0

    def _classify_regime(self, daily_vol: float) -> float:
        """Map daily vol fraction → 0.0 (low), 0.5 (medium), 1.0 (high)."""
        lo, hi = self.thresholds
        if daily_vol < lo:
            return 0.0
        elif daily_vol < hi:
            return 0.5
        else:
            return 1.0

    def _kelly(self, mu: float, sigma2: float) -> float:
        """
        Continuous Kelly fraction: f* = μ / σ²

        Negative mu → flat (no trade), so clip to 0.
        """
        if sigma2 <= 1e-10:
            return 0.0
        k = mu / sigma2
        return float(np.clip(k, 0.0, 5.0))  # clip before fractional scaling

    def _fallback(self) -> dict:
        return {
            "garch_vol": 0.02,
            "vol_regime": 0.5,
            "kelly_fraction": 0.1,
            "kelly_adj": 0.025,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper for position sizing (used by executor)
# ─────────────────────────────────────────────────────────────────────────────

def kelly_position_size(
    balance: float,
    price: float,
    kelly_adj: float,
    leverage: float,
    size_fraction: float = 1.0,  # 1.0 = full Kelly, 0.5 = half Kelly
) -> float:
    """
    Convert Kelly fraction → actual BTC quantity.

    Returns quantity in BTC (rounded to 3dp, Binance min step).
    """
    max_lev = leverage
    notional = balance * kelly_adj * size_fraction * max_lev
    qty = notional / price
    return float(round(qty, 3))
