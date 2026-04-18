"""
tests/test_features.py
───────────────────────
Unit tests for all feature extraction modules.
Uses synthetic OHLCV data — no network, no Binance API required.

Run:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic OHLCV data
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, start_price: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic-looking synthetic OHLCV DataFrame.
    Uses a random walk with mean reversion so it doesn't drift too far.
    """
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        mu = 0.0002 * (start_price - prices[-1]) / start_price  # mean reversion
        shock = rng.normal(0, 0.005)
        prices.append(prices[-1] * (1 + mu + shock))

    prices = np.array(prices)

    # Build OHLCV from close prices
    opens = prices * (1 + rng.uniform(-0.003, 0.003, n))
    highs = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.005, n))
    lows  = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.005, n))
    volumes = rng.lognormal(10, 1, n)  # log-normal volumes

    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": volumes,
    }, index=idx)


@pytest.fixture
def df_5m():
    return make_ohlcv(500)


@pytest.fixture
def df_1h():
    return make_ohlcv(200)


@pytest.fixture
def df_4h():
    return make_ohlcv(100)


# ─────────────────────────────────────────────────────────────────────────────
# AMT features
# ─────────────────────────────────────────────────────────────────────────────

class TestAMTFeatures:

    def test_output_shape(self, df_5m):
        from src.features.amt_features import extract_amt_features
        price = float(df_5m["close"].iloc[-1])
        feat = extract_amt_features(df_5m, price)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_output_dtype(self, df_5m):
        from src.features.amt_features import extract_amt_features
        feat = extract_amt_features(df_5m, float(df_5m["close"].iloc[-1]))
        assert feat.dtype == np.float32

    def test_all_finite(self, df_5m):
        from src.features.amt_features import extract_amt_features
        feat = extract_amt_features(df_5m, float(df_5m["close"].iloc[-1]))
        assert np.all(np.isfinite(feat)), f"Non-finite values in AMT features: {feat}"

    def test_values_in_range(self, df_5m):
        from src.features.amt_features import extract_amt_features
        feat = extract_amt_features(df_5m, float(df_5m["close"].iloc[-1]))
        assert np.all(feat >= -1.0) and np.all(feat <= 1.0), f"Out of range: {feat}"

    def test_empty_df_returns_zeros(self):
        from src.features.amt_features import extract_amt_features
        feat = extract_amt_features(pd.DataFrame(), 50000.0)
        assert feat.shape == (6,)
        assert np.all(feat == 0.0)

    def test_poc_dist_sign(self, df_5m):
        """POC distance should be signed: positive if price > POC, negative if below."""
        from src.features.amt_features import extract_amt_features, build_volume_profile
        profile = build_volume_profile(df_5m)
        poc = profile["poc_price"]
        price_above = poc * 1.02
        price_below = poc * 0.98
        feat_above = extract_amt_features(df_5m, price_above)
        feat_below = extract_amt_features(df_5m, price_below)
        assert feat_above[0] > 0, "price above POC should give positive poc_dist"
        assert feat_below[0] < 0, "price below POC should give negative poc_dist"

    def test_value_profile_build(self, df_5m):
        from src.features.amt_features import build_volume_profile
        profile = build_volume_profile(df_5m)
        assert profile["poc_price"] > 0
        assert profile["vah"] >= profile["poc_price"] >= profile["val"]
        assert profile["total_volume"] > 0

    def test_in_value_area_flag(self, df_5m):
        from src.features.amt_features import extract_amt_features, build_volume_profile
        profile = build_volume_profile(df_5m)
        price_inside = (profile["vah"] + profile["val"]) / 2.0
        feat = extract_amt_features(df_5m, price_inside)
        assert feat[3] == 1.0, "Price inside VA should give in_va=1.0"


# ─────────────────────────────────────────────────────────────────────────────
# SNR features
# ─────────────────────────────────────────────────────────────────────────────

class TestSNRFeatures:

    def test_output_shape(self, df_1h):
        from src.features.snr_features import extract_snr_features
        feat = extract_snr_features(df_1h, float(df_1h["close"].iloc[-1]))
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_output_dtype(self, df_1h):
        from src.features.snr_features import extract_snr_features
        feat = extract_snr_features(df_1h, float(df_1h["close"].iloc[-1]))
        assert feat.dtype == np.float32

    def test_all_finite(self, df_1h):
        from src.features.snr_features import extract_snr_features
        feat = extract_snr_features(df_1h, float(df_1h["close"].iloc[-1]))
        assert np.all(np.isfinite(feat))

    def test_values_in_range(self, df_1h):
        from src.features.snr_features import extract_snr_features
        feat = extract_snr_features(df_1h, float(df_1h["close"].iloc[-1]))
        assert np.all(feat >= -1.0) and np.all(feat <= 1.0)

    def test_empty_df_returns_zeros(self):
        from src.features.snr_features import extract_snr_features
        feat = extract_snr_features(None, 50000.0)
        assert np.all(feat == 0.0)

    def test_resistance_positive_support_negative(self, df_1h):
        """First 3 features are resistance (positive), last 3 are support (negative)."""
        from src.features.snr_features import extract_snr_features
        price = float(df_1h["close"].iloc[-1])
        feat = extract_snr_features(df_1h, price)
        # Resistance = positive distance
        assert all(f >= 0 for f in feat[:3]), f"Resistance should be >= 0: {feat[:3]}"
        # Support = negative distance
        assert all(f <= 0 for f in feat[3:]), f"Support should be <= 0: {feat[3:]}"

    def test_pivot_levels_found(self, df_1h):
        from src.features.snr_features import find_pivot_levels
        levels = find_pivot_levels(df_1h)
        assert "resistance" in levels
        assert "support" in levels
        assert len(levels["resistance"]) == 3
        assert len(levels["support"]) == 3

    def test_resistance_above_support(self, df_1h):
        from src.features.snr_features import find_pivot_levels
        price = float(df_1h["close"].iloc[-1])
        levels = find_pivot_levels(df_1h)
        for r in levels["resistance"]:
            assert r >= price * 0.98, f"Resistance {r:.0f} should be near or above price {price:.0f}"
        for s in levels["support"]:
            assert s <= price * 1.02, f"Support {s:.0f} should be near or below price {price:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# GARCH + Kelly
# ─────────────────────────────────────────────────────────────────────────────

class TestGarchKelly:

    def test_output_keys(self, df_5m):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        log_ret = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        result = est.compute(log_ret)
        assert set(result.keys()) == {"garch_vol", "vol_regime", "kelly_fraction", "kelly_adj"}

    def test_all_values_finite(self, df_5m):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        log_ret = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        result = est.compute(log_ret)
        for k, v in result.items():
            assert np.isfinite(v), f"Non-finite value for {k}: {v}"

    def test_garch_vol_positive(self, df_5m):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        log_ret = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        result = est.compute(log_ret)
        assert result["garch_vol"] > 0

    def test_vol_regime_valid_values(self, df_5m):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        log_ret = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        result = est.compute(log_ret)
        assert result["vol_regime"] in (0.0, 0.5, 1.0)

    def test_kelly_adj_clipped(self, df_5m):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        log_ret = np.log(df_5m["close"] / df_5m["close"].shift(1)).dropna()
        result = est.compute(log_ret)
        assert 0.0 <= result["kelly_adj"] <= 1.0

    def test_fallback_on_short_series(self):
        from src.features.garch_kelly import GarchKellyEstimator
        est = GarchKellyEstimator()
        short = pd.Series([0.001, -0.002, 0.003])  # too short
        result = est.compute(short)
        assert all(np.isfinite(v) for v in result.values())

    def test_kelly_position_size_reasonable(self):
        from src.features.garch_kelly import kelly_position_size
        qty = kelly_position_size(
            balance=10000.0, price=50000.0, kelly_adj=0.1,
            leverage=3.0, size_fraction=1.0
        )
        assert qty > 0
        assert qty < 100  # sanity: shouldn't be more than 100 BTC


# ─────────────────────────────────────────────────────────────────────────────
# Multi-TF feature builder
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTFFeatures:

    def _make_candles(self):
        return {
            "5m":  make_ohlcv(500),
            "15m": make_ohlcv(200),
            "1h":  make_ohlcv(150),
            "4h":  make_ohlcv(80),
            "1d":  make_ohlcv(50),
        }

    def test_output_shape(self):
        from src.features.multi_tf_features import build_observation, OBS_DIM
        candles = self._make_candles()
        obs = build_observation(
            candles=candles, position=0, unrealized_pnl_pct=0.0,
            leverage_used=0.0, bars_in_trade=0, account_drawdown=0.0,
            kelly_adj=0.1, bars_since_last_trade=0,
        )
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_output_dtype(self):
        from src.features.multi_tf_features import build_observation
        candles = self._make_candles()
        obs = build_observation(
            candles=candles, position=0, unrealized_pnl_pct=0.0,
            leverage_used=0.0, bars_in_trade=0, account_drawdown=0.0,
            kelly_adj=0.1, bars_since_last_trade=0,
        )
        assert obs.dtype == np.float32

    def test_all_finite(self):
        from src.features.multi_tf_features import build_observation
        candles = self._make_candles()
        obs = build_observation(
            candles=candles, position=0, unrealized_pnl_pct=0.0,
            leverage_used=0.0, bars_in_trade=0, account_drawdown=0.0,
            kelly_adj=0.1, bars_since_last_trade=0,
        )
        assert np.all(np.isfinite(obs)), f"Non-finite in obs at indices: {np.where(~np.isfinite(obs))}"

    def test_clipped_to_range(self):
        from src.features.multi_tf_features import build_observation
        candles = self._make_candles()
        obs = build_observation(
            candles=candles, position=1, unrealized_pnl_pct=0.50,  # extreme
            leverage_used=3.0, bars_in_trade=1000, account_drawdown=0.99,
            kelly_adj=0.9, bars_since_last_trade=9999,
        )
        assert np.all(obs >= -5.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 5.0), f"Max value: {obs.max()}"

    def test_position_encoded_correctly(self):
        from src.features.multi_tf_features import build_observation, OBS_DIM
        candles = self._make_candles()

        def _get_pos(p):
            obs = build_observation(
                candles=candles, position=p, unrealized_pnl_pct=0.0,
                leverage_used=0.0, bars_in_trade=0, account_drawdown=0.0,
                kelly_adj=0.1, bars_since_last_trade=0,
            )
            return obs[83]  # position feature index

        assert _get_pos(-1) == -1.0
        assert _get_pos(0)  ==  0.0
        assert _get_pos(1)  ==  1.0

    def test_empty_candles_no_crash(self):
        from src.features.multi_tf_features import build_observation, OBS_DIM
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        candles = {k: empty.copy() for k in ["5m", "15m", "1h", "4h", "1d"]}
        obs = build_observation(
            candles=candles, position=0, unrealized_pnl_pct=0.0,
            leverage_used=0.0, bars_in_trade=0, account_drawdown=0.0,
            kelly_adj=0.1, bars_since_last_trade=0,
        )
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
