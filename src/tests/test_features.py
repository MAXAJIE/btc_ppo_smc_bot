"""
tests/test_features.py
───────────────────────
Unit tests for all feature extraction modules.
Uses synthetic OHLCV data — no network, no Binance API required.

FIXES:
  - Imports corrected to match actual function names in feature modules:
      compute_amt_features  (not extract_amt_features / build_volume_profile)
      compute_snr_features  (not extract_snr_features / find_pivot_levels)
      compute_garch_kelly   (not GarchKellyEstimator)
      MultiTFFeatureBuilder.build()  (not build_observation / _ohlcv_features)
  - Tests rewritten to match actual APIs.

Run:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic OHLCV data
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, start_price: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        mu = 0.0002 * (start_price - prices[-1]) / start_price
        shock = rng.normal(0, 0.005)
        prices.append(prices[-1] * (1 + mu + shock))

    prices = np.array(prices)
    opens  = prices * (1 + rng.uniform(-0.003, 0.003, n))
    highs  = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.005, n))
    lows   = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.005, n))
    volumes = rng.lognormal(10, 1, n)

    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": prices, "volume": volumes,
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
# AMT features  (compute_amt_features)
# ─────────────────────────────────────────────────────────────────────────────

class TestAMTFeatures:

    def test_output_shape(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_output_dtype(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert feat.dtype == np.float32

    def test_all_finite(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert np.all(np.isfinite(feat)), f"Non-finite values in AMT features: {feat}"

    def test_short_df_returns_zeros(self):
        from src.features.amt_features import compute_amt_features
        tiny = make_ohlcv(3)
        feat = compute_amt_features(tiny, 2)
        assert feat.shape == (6,)
        assert np.all(feat == 0.0)

    def test_values_within_clip_range(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert np.all(feat >= -10.0) and np.all(feat <= 10.0)

    def test_in_value_area_is_binary(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert feat[4] in (0.0, 1.0), f"in_value_area should be 0 or 1, got {feat[4]}"

    def test_vol_rank_between_0_and_1(self, df_4h):
        from src.features.amt_features import compute_amt_features
        feat = compute_amt_features(df_4h, len(df_4h) - 1)
        assert 0.0 <= feat[5] <= 1.0, f"vol_rank should be [0,1], got {feat[5]}"


# ─────────────────────────────────────────────────────────────────────────────
# SNR features  (compute_snr_features)
# ─────────────────────────────────────────────────────────────────────────────

class TestSNRFeatures:

    def test_output_shape(self, df_1h):
        from src.features.snr_features import compute_snr_features
        feat = compute_snr_features(df_1h, len(df_1h) - 1)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_output_dtype(self, df_1h):
        from src.features.snr_features import compute_snr_features
        feat = compute_snr_features(df_1h, len(df_1h) - 1)
        assert feat.dtype == np.float32

    def test_all_finite(self, df_1h):
        from src.features.snr_features import compute_snr_features
        feat = compute_snr_features(df_1h, len(df_1h) - 1)
        assert np.all(np.isfinite(feat))

    def test_short_df_returns_zeros(self):
        from src.features.snr_features import compute_snr_features
        tiny = make_ohlcv(3)
        feat = compute_snr_features(tiny, 2)
        assert feat.shape == (6,)
        assert np.all(feat == 0.0)

    def test_values_within_clip_range(self, df_1h):
        from src.features.snr_features import compute_snr_features
        feat = compute_snr_features(df_1h, len(df_1h) - 1)
        assert np.all(feat >= -10.0) and np.all(feat <= 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# GARCH + Kelly  (compute_garch_kelly)
# ─────────────────────────────────────────────────────────────────────────────

class TestGarchKelly:

    def test_output_shape(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    def test_all_values_finite(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert np.all(np.isfinite(result)), f"Non-finite: {result}"

    def test_output_dtype(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert result.dtype == np.float32

    def test_garch_vol_non_negative(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert result[0] >= 0.0, f"garch_vol should be >= 0, got {result[0]}"

    def test_kelly_fraction_bounded(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert 0.0 <= result[2] <= 1.0, f"kelly fraction out of range: {result[2]}"

    def test_fallback_on_short_series(self):
        from src.features.garch_kelly import compute_garch_kelly
        short = pd.Series([50000.0, 50100.0, 49900.0])  # too short
        result = compute_garch_kelly(short)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_confidence_between_0_and_1(self, df_5m):
        from src.features.garch_kelly import compute_garch_kelly
        result = compute_garch_kelly(df_5m["close"])
        assert 0.0 <= result[3] <= 1.0, f"confidence out of range: {result[3]}"


# ─────────────────────────────────────────────────────────────────────────────
# SMC features  (compute_smc_features)
# ─────────────────────────────────────────────────────────────────────────────

class TestSMCFeatures:

    def test_output_shape(self, df_5m):
        from src.features.smc_features import compute_smc_features
        feat = compute_smc_features(df_5m, len(df_5m) - 1)
        assert feat.shape == (8,), f"Expected (8,), got {feat.shape}"

    def test_output_dtype(self, df_5m):
        from src.features.smc_features import compute_smc_features
        feat = compute_smc_features(df_5m, len(df_5m) - 1)
        assert feat.dtype == np.float32

    def test_all_finite(self, df_5m):
        from src.features.smc_features import compute_smc_features
        feat = compute_smc_features(df_5m, len(df_5m) - 1)
        assert np.all(np.isfinite(feat)), f"Non-finite: {feat}"

    def test_short_df_returns_zeros(self):
        from src.features.smc_features import compute_smc_features
        tiny = make_ohlcv(3)
        feat = compute_smc_features(tiny, 2)
        assert feat.shape == (8,)

    def test_ob_present_features_binary(self, df_5m):
        from src.features.smc_features import compute_smc_features
        feat = compute_smc_features(df_5m, len(df_5m) - 1)
        # features [2] and [3] are bull_ob_present and bear_ob_present (binary)
        assert feat[2] in (0.0, 1.0), f"bull_ob_present should be 0 or 1: {feat[2]}"
        assert feat[3] in (0.0, 1.0), f"bear_ob_present should be 0 or 1: {feat[3]}"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-TF feature builder  (MultiTFFeatureBuilder)
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
        from src.features.multi_tf_features import MultiTFFeatureBuilder
        candles = self._make_candles()
        builder = MultiTFFeatureBuilder(candles)
        ts = candles["5m"].index[-1]
        obs = builder.build(
            ts=ts,
            smc_5m=np.zeros(8, np.float32),
            smc_1h=np.zeros(8, np.float32),
            snr_1h=np.zeros(6, np.float32),
            amt_4h=np.zeros(6, np.float32),
            garch_kelly=np.zeros(4, np.float32),
            position=np.zeros(7, np.float32),
        )
        assert obs.shape == (90,), f"Expected (90,), got {obs.shape}"

    def test_output_dtype(self):
        from src.features.multi_tf_features import MultiTFFeatureBuilder
        candles = self._make_candles()
        builder = MultiTFFeatureBuilder(candles)
        ts = candles["5m"].index[-1]
        obs = builder.build(
            ts=ts,
            smc_5m=np.zeros(8, np.float32),
            smc_1h=np.zeros(8, np.float32),
            snr_1h=np.zeros(6, np.float32),
            amt_4h=np.zeros(6, np.float32),
            garch_kelly=np.zeros(4, np.float32),
            position=np.zeros(7, np.float32),
        )
        assert obs.dtype == np.float32

    def test_all_finite(self):
        from src.features.multi_tf_features import MultiTFFeatureBuilder
        candles = self._make_candles()
        builder = MultiTFFeatureBuilder(candles)
        ts = candles["5m"].index[-1]
        obs = builder.build(
            ts=ts,
            smc_5m=np.zeros(8, np.float32),
            smc_1h=np.zeros(8, np.float32),
            snr_1h=np.zeros(6, np.float32),
            amt_4h=np.zeros(6, np.float32),
            garch_kelly=np.zeros(4, np.float32),
            position=np.zeros(7, np.float32),
        )
        assert np.all(np.isfinite(obs))

    def test_clipped_to_range(self):
        from src.features.multi_tf_features import MultiTFFeatureBuilder
        candles = self._make_candles()
        builder = MultiTFFeatureBuilder(candles)
        ts = candles["5m"].index[-1]
        # Pass extreme values in feature arrays
        obs = builder.build(
            ts=ts,
            smc_5m=np.full(8, 999.0, np.float32),
            smc_1h=np.full(8, -999.0, np.float32),
            snr_1h=np.full(6, 999.0, np.float32),
            amt_4h=np.full(6, -999.0, np.float32),
            garch_kelly=np.full(4, 999.0, np.float32),
            position=np.full(7, -999.0, np.float32),
        )
        assert np.all(obs >= -10.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 10.0), f"Max value: {obs.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
