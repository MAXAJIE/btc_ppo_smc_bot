"""
data_loader.py – Fixed MTF historical downloader
====================================================
Fixes:
  - DataLoader now supports both the new dict-based API (load_all_timeframes)
    AND the legacy method names used by main_train.py / train_lightning.py:
      loader.load_base_df()
      loader.n_candles
      loader.download_history(years=N)
      loader.get_multi_tf_candles(end_idx, lookback_5m)
      loader.get_episode_start_indices(episode_len, warmup)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL   = "BTC/USDT"
EXCHANGE = "binanceusdm"
DATA_DIR = Path("./data")

TIMEFRAMES: Dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

CANDLE_LIMITS: Dict[str, int] = {
    "5m":  210_240,
    "15m":  70_080,
    "1h":   17_520,
    "4h":    4_380,
    "1d":    1_095,
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Public module-level API
# ---------------------------------------------------------------------------

def load_all_timeframes(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for every required timeframe."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}

    for tf_key, tf_ccxt in TIMEFRAMES.items():
        cache_path = DATA_DIR / f"btcusdt_{tf_key}.parquet"

        if not force_refresh and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info("Loaded %s from cache (%d rows)", tf_key, len(df))
        else:
            df = _download_timeframe(tf_ccxt, CANDLE_LIMITS[tf_key])
            df.to_parquet(cache_path)
            logger.info("Downloaded & cached %s (%d rows)", tf_key, len(df))

        result[tf_key] = df

    return result


def _download_timeframe(timeframe: str, target_rows: int) -> pd.DataFrame:
    """Fetch candles for a timeframe from Binance USDM futures."""
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt is required: pip install ccxt")

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()

    all_candles: list = []
    batch_size = 1500
    ms_per_candle = _tf_to_ms(timeframe)
    since = exchange.milliseconds() - target_rows * ms_per_candle

    fetched = 0
    while fetched < target_rows:
        try:
            batch = exchange.fetch_ohlcv(
                SYMBOL, timeframe=timeframe,
                since=since, limit=batch_size
            )
        except ccxt.NetworkError as exc:
            logger.warning("Network error fetching %s: %s — retrying in 5s", timeframe, exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error fetching %s: %s", timeframe, exc)
            break

        if not batch:
            break

        all_candles.extend(batch)
        fetched += len(batch)
        since = batch[-1][0] + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)

        if len(batch) < batch_size:
            break

    if not all_candles:
        raise RuntimeError(f"No data fetched for timeframe {timeframe}")

    df = pd.DataFrame(all_candles, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp").astype(float).dropna()
    return df


def _tf_to_ms(timeframe: str) -> int:
    unit_map = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * unit_map[unit]


def build_aligned_dataset(all_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = all_tf["5m"].copy()
    for tf in ["15m", "1h", "4h", "1d"]:
        htf = all_tf[tf].copy()
        htf.columns = [f"{tf}_{c}" for c in htf.columns]
        base = pd.merge_asof(
            base.reset_index(),
            htf.reset_index().rename(columns={"timestamp": f"ts_{tf}"}),
            left_on="timestamp",
            right_on=f"ts_{tf}",
            direction="backward",
        ).set_index("timestamp").drop(columns=[f"ts_{tf}"], errors="ignore")
    return base


# ---------------------------------------------------------------------------
# DataLoader class — supports both legacy and new API
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Unified DataLoader.

    New API (used by BinanceEnv):
        loader = DataLoader(data_dir="./data")
        tf_data = loader.load()          # dict[str, pd.DataFrame]
        df_5m   = loader.get("5m")

    Legacy API (used by main_train.py / train_lightning.py):
        loader.load_base_df()            # loads / downloads all TFs
        loader.n_candles                 # number of 5m bars
        loader.download_history(years=2) # download and cache
        loader.get_multi_tf_candles(end_idx, lookback_5m=500)
        loader.get_episode_start_indices(episode_len, warmup)
    """

    def __init__(
        self,
        data_dir: str = "./data",
        years: int = 2,
        symbol: str = SYMBOL,
        force_refresh: bool = False,
    ):
        global DATA_DIR
        DATA_DIR = Path(data_dir)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self._scale = years / 2.0
        self._force = force_refresh
        self._tf_data: Dict[str, pd.DataFrame] = {}
        self._aligned: Optional[pd.DataFrame] = None

    # ── New API ────────────────────────────────────────────────────────

    def load(self) -> Dict[str, pd.DataFrame]:
        """Download / load all timeframes. Returns dict[str, pd.DataFrame]."""
        original_limits = dict(CANDLE_LIMITS)
        for k in CANDLE_LIMITS:
            CANDLE_LIMITS[k] = int(original_limits[k] * self._scale)
        try:
            self._tf_data = load_all_timeframes(force_refresh=self._force)
        finally:
            CANDLE_LIMITS.update(original_limits)
        return self._tf_data

    def get(self, timeframe: str) -> pd.DataFrame:
        if not self._tf_data:
            self.load()
        if timeframe not in self._tf_data:
            raise KeyError(f"Timeframe '{timeframe}' not available. Available: {list(self._tf_data.keys())}")
        return self._tf_data[timeframe]

    @property
    def aligned(self) -> pd.DataFrame:
        if not self._tf_data:
            self.load()
        if self._aligned is None:
            self._aligned = build_aligned_dataset(self._tf_data)
        return self._aligned

    @property
    def tf_data(self) -> Dict[str, pd.DataFrame]:
        if not self._tf_data:
            self.load()
        return self._tf_data

    def __getitem__(self, timeframe: str) -> pd.DataFrame:
        return self.get(timeframe)

    # ── Legacy API ─────────────────────────────────────────────────────

    def load_base_df(self) -> pd.DataFrame:
        """
        Legacy method used by main_train.py / train_lightning.py.
        Loads all timeframes from cache (raises FileNotFoundError if missing).
        """
        # Check if 5m cache exists
        cache_5m = DATA_DIR / "btcusdt_5m.parquet"
        # Also check old naming convention
        cache_5m_alt = DATA_DIR / "BTCUSDT_5m.parquet"

        if not cache_5m.exists() and not cache_5m_alt.exists():
            raise FileNotFoundError(
                f"No cached data found in {DATA_DIR}. "
                "Call loader.download_history() first."
            )

        self._tf_data = load_all_timeframes(force_refresh=False)
        return self._tf_data.get("5m", pd.DataFrame())

    def download_history(self, years: float = 2) -> pd.DataFrame:
        """
        Legacy method: download and cache all timeframes.
        Returns the 5m DataFrame.
        """
        original_limits = dict(CANDLE_LIMITS)
        scale = years / 2.0
        for k in CANDLE_LIMITS:
            CANDLE_LIMITS[k] = max(1, int(original_limits[k] * scale))
        try:
            self._tf_data = load_all_timeframes(force_refresh=True)
        finally:
            CANDLE_LIMITS.update(original_limits)
        return self._tf_data.get("5m", pd.DataFrame())

    @property
    def n_candles(self) -> int:
        """Number of 5m bars loaded."""
        if not self._tf_data:
            return 0
        return len(self._tf_data.get("5m", pd.DataFrame()))

    @property
    def _base_df(self) -> pd.DataFrame:
        """Direct access to 5m DataFrame (used internally by legacy methods)."""
        if not self._tf_data:
            self.load_base_df()
        return self._tf_data.get("5m", pd.DataFrame())

    def get_multi_tf_candles(
        self,
        end_idx: int,
        lookback_5m: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a dict of DataFrames for each timeframe ending at end_idx
        on the 5m index (resampled from 5m base).
        """
        if not self._tf_data:
            self.load_base_df()

        base = self._tf_data["5m"]
        start = max(0, end_idx - lookback_5m)
        df_5m = base.iloc[start: end_idx + 1].copy()

        def resample(rule: str) -> pd.DataFrame:
            return df_5m.resample(rule).agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna()

        return {
            "5m":  df_5m,
            "15m": resample("15min"),
            "1h":  resample("1h"),
            "4h":  resample("4h"),
            "1d":  resample("1D"),
        }

    def get_episode_start_indices(
        self,
        episode_len: int = 4320,
        warmup: int = 500,
    ) -> list:
        """Return valid episode start indices for offline training."""
        n = self.n_candles
        return list(range(warmup, n - episode_len, episode_len // 2))
