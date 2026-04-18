"""
data_loader.py
──────────────
Downloads historical OHLCV data from Binance via ccxt.
Stores locally as compressed Parquet files to avoid re-downloading.
Supports multi-timeframe resampling from the base 5m data.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
import yaml

logger = logging.getLogger(__name__)


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and caches BTCUSDT OHLCV data.

    Usage
    ─────
        loader = DataLoader()
        candles = loader.get_multi_tf_candles(end_idx=1000)
        # returns dict: '5m' -> DataFrame, '15m' -> DataFrame, etc.
    """

    TF_MINUTES = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    def __init__(self, data_dir: str = None):
        cfg = _load_cfg()
        self.symbol = cfg["symbol"]
        self.data_dir = Path(data_dir or os.getenv("DATA_PATH", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._base_df: pd.DataFrame = None  # 5m master DataFrame

    # ─────────────────────────────────────────────────────────────────
    # Download & cache
    # ─────────────────────────────────────────────────────────────────

    def download_history(self, years: int = 2, force: bool = False) -> pd.DataFrame:
        """
        Download `years` of 5m OHLCV data from Binance.
        Caches to {data_dir}/BTCUSDT_5m.parquet.
        Returns the DataFrame.
        """
        cache_path = self.data_dir / "BTCUSDT_5m.parquet"

        if cache_path.exists() and not force:
            logger.info(f"Loading cached 5m data from {cache_path}")
            df = pd.read_parquet(cache_path)
            self._base_df = df
            return df

        logger.info(f"Downloading {years}y of BTCUSDT 5m OHLCV data via ccxt...")

        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        since_dt = datetime.now(timezone.utc) - timedelta(days=365 * years)
        since_ms = int(since_dt.timestamp() * 1000)

        all_ohlcv = []
        while True:
            batch = exchange.fetch_ohlcv(
                self.symbol,
                timeframe="5m",
                since=since_ms,
                limit=1000,
            )
            if not batch:
                break

            all_ohlcv.extend(batch)
            since_ms = batch[-1][0] + 1

            if batch[-1][0] >= int(datetime.now(timezone.utc).timestamp() * 1000):
                break

            logger.info(
                f"  Downloaded up to {datetime.utcfromtimestamp(batch[-1][0]/1000)} "
                f"({len(all_ohlcv):,} rows)"
            )

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)

        df.to_parquet(cache_path, compression="snappy")
        logger.info(f"Saved {len(df):,} rows to {cache_path}")

        self._base_df = df
        return df

    # ─────────────────────────────────────────────────────────────────
    # Multi-timeframe slice (zero lookahead)
    # ─────────────────────────────────────────────────────────────────

    def get_multi_tf_candles(
        self,
        end_idx: int,
        lookback_5m: int = 500,
    ) -> dict:
        """
        Return a dict of OHLCV DataFrames for each timeframe, all aligned
        to the same wall-clock endpoint.  NO lookahead — only data up to
        and including `end_idx` in the base 5m DataFrame is used.

        Parameters
        ----------
        end_idx : int
            Current step index in the 5m base DataFrame (0-based).
        lookback_5m : int
            How many 5m bars to include in the 5m slice.
            Higher TFs are resampled from this window.
        """
        if self._base_df is None:
            raise RuntimeError("Call download_history() or load_base_df() first")

        start_idx = max(0, end_idx - lookback_5m)
        df_5m = self._base_df.iloc[start_idx: end_idx + 1].copy()

        return {
            "5m": df_5m,
            "15m": self._resample(df_5m, "15min"),
            "1h": self._resample(df_5m, "1h"),
            "4h": self._resample(df_5m, "4h"),
            "1d": self._resample(df_5m, "1D"),
        }

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample 5m OHLCV to a higher timeframe."""
        return df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    # ─────────────────────────────────────────────────────────────────

    def load_base_df(self) -> pd.DataFrame:
        """Load from cache (must have been downloaded first)."""
        cache_path = self.data_dir / "BTCUSDT_5m.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached data at {cache_path}. Run download_history() first."
            )
        self._base_df = pd.read_parquet(cache_path)
        return self._base_df

    @property
    def n_candles(self) -> int:
        if self._base_df is None:
            return 0
        return len(self._base_df)

    def get_episode_start_indices(self, episode_len: int = 4320, warmup: int = 500) -> list:
        """
        Return all valid episode start indices (shuffled for offline training).
        Each episode needs warmup + episode_len bars available after it.
        """
        n = self.n_candles
        starts = list(range(warmup, n - episode_len, episode_len // 2))  # 50% overlap
        return starts
