"""
data_loader.py
==============
Bugs fixed
----------
BUG 1 (always re-fetching):
    train_lightning._ensure_data() checked for "BTCUSDT_5m.parquet" (uppercase)
    but the downloader saved "btcusdt_5m.parquet" (lowercase).
    Cache check always failed → always re-downloaded.

    Fix: ALL code now uses _cache_path() which always produces lowercase names.
         load_base_df() also auto-migrates any old uppercase files.

BUG 2 (missing legacy methods):
    main_train.py calls load_base_df(), download_history(), n_candles,
    get_multi_tf_candles(), get_episode_start_indices() on the DataLoader.
    These were missing → AttributeError on startup.

    Fix: full legacy API implemented on the DataLoader class.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SYMBOL   = "BTC/USDT:USDT"
EXCHANGE = "binanceusdm"

TIMEFRAMES: Dict[str, str] = {
    "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
}

CANDLE_LIMITS: Dict[str, int] = {
    "5m": 210_240, "15m": 70_080, "1h": 17_520, "4h": 4_380, "1d": 1_095,
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _cache_path(data_dir: Path, tf_key: str) -> Path:
    """Single source of truth: always lowercase filename."""
    return data_dir / f"btcusdt_{tf_key}.parquet"


# ---------------------------------------------------------------------------
# Module-level functional API
# ---------------------------------------------------------------------------

def load_all_timeframes(
    force_refresh: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    _dir = Path(data_dir) if data_dir else Path("./data")
    _dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}

    for tf_key, tf_ccxt in TIMEFRAMES.items():
        cache = _cache_path(_dir, tf_key)

        # Auto-migrate old uppercase files (one-time)
        old = _dir / f"BTCUSDT_{tf_key}.parquet"
        if old.exists() and not cache.exists():
            old.rename(cache)
            logger.info("Migrated %s → %s", old.name, cache.name)

        if not force_refresh and cache.exists():
            df = pd.read_parquet(cache)
            logger.info("Cache hit  %s (%d rows)", tf_key, len(df))
        else:
            df = _download(tf_ccxt, CANDLE_LIMITS[tf_key])
            df.to_parquet(cache)
            logger.info("Downloaded %s (%d rows)", tf_key, len(df))

        result[tf_key] = df
    return result


def build_aligned_dataset(all_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = all_tf["5m"].copy()
    for tf in ["15m", "1h", "4h", "1d"]:
        htf = all_tf[tf].copy()
        htf.columns = [f"{tf}_{c}" for c in htf.columns]
        base = pd.merge_asof(
            base.reset_index(),
            htf.reset_index().rename(columns={"timestamp": f"ts_{tf}"}),
            left_on="timestamp", right_on=f"ts_{tf}", direction="backward",
        ).set_index("timestamp").drop(columns=[f"ts_{tf}"], errors="ignore")
    return base


# ---------------------------------------------------------------------------
# DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    New API:
        dl = DataLoader(data_dir="./data", years=2)
        tf_data = dl.load()        # dict[str, pd.DataFrame]
        df_5m   = dl["5m"]
        aligned = dl.aligned

    Legacy API (used by main_train.py / train_lightning.py):
        dl.load_base_df()
        dl.download_history(years=2)
        dl.n_candles
        dl.get_multi_tf_candles(end_idx, lookback_5m=500)
        dl.get_episode_start_indices(episode_len, warmup)
        dl.tf_data
    """

    def __init__(
        self,
        data_dir: str = "./data",
        years: int = 2,
        symbol: str = SYMBOL,
        force_refresh: bool = False,
    ):
        self._dir   = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._scale = years / 2.0
        self._force = force_refresh
        self._tf_data: Dict[str, pd.DataFrame] = {}
        self._aligned: Optional[pd.DataFrame]  = None

    # ── New API ─────────────────────────────────────────────────────────────

    def load(self) -> Dict[str, pd.DataFrame]:
        orig = dict(CANDLE_LIMITS)
        for k in CANDLE_LIMITS:
            CANDLE_LIMITS[k] = max(1, int(orig[k] * self._scale))
        try:
            self._tf_data = load_all_timeframes(
                force_refresh=self._force, data_dir=self._dir
            )
        finally:
            CANDLE_LIMITS.update(orig)
        return self._tf_data

    def get(self, timeframe: str) -> pd.DataFrame:
        if not self._tf_data:
            self.load()
        if timeframe not in self._tf_data:
            raise KeyError(f"'{timeframe}' not available. Got: {list(self._tf_data)}")
        return self._tf_data[timeframe]

    def __getitem__(self, tf: str) -> pd.DataFrame:
        return self.get(tf)

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

    # ── Legacy API ───────────────────────────────────────────────────────────

    def load_base_df(self) -> pd.DataFrame:
        """
        Load all TFs from existing cache only — no download.
        Raises FileNotFoundError if cache missing (caller should call
        download_history() first).
        """
        cache_5m = _cache_path(self._dir, "5m")

        # Auto-migrate old uppercase files
        old_5m = self._dir / "BTCUSDT_5m.parquet"
        if old_5m.exists() and not cache_5m.exists():
            old_5m.rename(cache_5m)
            logger.info("Migrated BTCUSDT_5m.parquet → btcusdt_5m.parquet")

        if not cache_5m.exists():
            raise FileNotFoundError(
                f"No cached data at {cache_5m}. "
                "Run loader.download_history() first."
            )
        self._tf_data = load_all_timeframes(force_refresh=False, data_dir=self._dir)
        return self._tf_data["5m"]

    def download_history(self, years: float = 2) -> pd.DataFrame:
        """Force-download all TFs, save to cache, return 5m DataFrame."""
        orig  = dict(CANDLE_LIMITS)
        scale = years / 2.0
        for k in CANDLE_LIMITS:
            CANDLE_LIMITS[k] = max(1, int(orig[k] * scale))
        try:
            self._tf_data = load_all_timeframes(force_refresh=True, data_dir=self._dir)
        finally:
            CANDLE_LIMITS.update(orig)
        return self._tf_data["5m"]

    @property
    def n_candles(self) -> int:
        if not self._tf_data:
            return 0
        return len(self._tf_data.get("5m", pd.DataFrame()))

    def get_multi_tf_candles(
        self, end_idx: int, lookback_5m: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Per-TF slices aligned to the 5m window ending at end_idx."""
        if not self._tf_data:
            self.load_base_df()
        base     = self._tf_data["5m"]
        start    = max(0, end_idx - lookback_5m)
        slice_5m = base.iloc[start: end_idx + 1].copy()
        ts_s, ts_e = slice_5m.index[0], slice_5m.index[-1]

        result = {"5m": slice_5m}
        for tf in ["15m", "1h", "4h", "1d"]:
            df = self._tf_data.get(tf, pd.DataFrame())
            result[tf] = df.loc[(df.index >= ts_s) & (df.index <= ts_e)].copy() \
                         if not df.empty else pd.DataFrame()
        return result

    def get_episode_start_indices(
        self, episode_len: int = 4320, warmup: int = 500
    ) -> List[int]:
        n = self.n_candles
        return list(range(warmup, n - episode_len, episode_len // 2))


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download(timeframe: str, target_rows: int) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        raise ImportError("pip install ccxt")

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()

    ms_per     = _tf_ms(timeframe)
    since      = exchange.milliseconds() - target_rows * ms_per
    batch_size = 1500
    all_rows: list = []

    logger.info("Downloading %s (target %d bars) …", timeframe, target_rows)

    while len(all_rows) < target_rows:
        try:
            batch = exchange.fetch_ohlcv(
                SYMBOL, timeframe=timeframe, since=since, limit=batch_size
            )
        except Exception as e:
            logger.warning("Fetch error (%s): %s — retrying in 5s", timeframe, e)
            time.sleep(5)
            continue

        if not batch:
            break

        all_rows.extend(batch)
        since = batch[-1][0] + ms_per
        time.sleep(exchange.rateLimit / 1000)

        if len(all_rows) >= target_rows:
            break

    if not all_rows:
        raise RuntimeError(f"No data fetched for {timeframe}")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates("timestamp")
          .sort_values("timestamp")
          .set_index("timestamp")
          .astype(float)
          .dropna()
    )
    return df.iloc[-target_rows:]


def _tf_ms(tf: str) -> int:
    return int(tf[:-1]) * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[tf[-1]]