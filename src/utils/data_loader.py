"""
data_loader.py  –  Fixed MTF historical downloader
====================================================
Bug fixed:
  - Previously only downloaded 5m data; all higher timeframes were silently
    missing, causing multi_tf_features.py to fill them with zeros/NaN.
  - Now downloads EACH timeframe independently via ccxt and caches them
    in separate parquet files.
  - Returns a dict[str, pd.DataFrame] keyed by timeframe string.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL   = "BTC/USDT"
EXCHANGE = "binanceusdm"          # USDT-M futures
DATA_DIR = Path("./data")

# Timeframes we ACTUALLY need — each downloaded independently
TIMEFRAMES: Dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# How many candles to fetch per timeframe (≈ 2 years of 5m = 210 240)
CANDLE_LIMITS: Dict[str, int] = {
    "5m":  210_240,
    "15m":  70_080,
    "1h":   17_520,
    "4h":    4_380,
    "1d":    1_095,  # 3 years
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_timeframes(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for every required timeframe.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are timeframe strings ("5m", "15m", "1h", "4h", "1d").
        DataFrames are indexed by UTC datetime, columns = open/high/low/close/volume.
    """
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_timeframe(timeframe: str, target_rows: int) -> pd.DataFrame:
    """Fetch `target_rows` candles for `timeframe` from Binance USDM futures."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()

    all_candles: list = []
    # ccxt limit per call is 1500 for Binance
    batch_size = 1500
    since: int | None = None

    # Work out the earliest 'since' timestamp
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
        since = batch[-1][0] + ms_per_candle  # next candle after last

        # Don't hammer the exchange
        time.sleep(exchange.rateLimit / 1000)

        if len(batch) < batch_size:
            # Reached the present
            break

    if not all_candles:
        raise RuntimeError(f"No data fetched for timeframe {timeframe}")

    df = pd.DataFrame(all_candles, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df.astype(float)

    # Sanity: drop rows with any NaN in OHLCV
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        logger.warning("Dropped %d NaN rows in %s", before - len(df), timeframe)

    return df


def _tf_to_ms(timeframe: str) -> int:
    """Convert a ccxt timeframe string to milliseconds."""
    unit_map = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * unit_map[unit]


# ---------------------------------------------------------------------------
# Convenience: get aligned 5m data with MTF context columns (for env)
# ---------------------------------------------------------------------------

def build_aligned_dataset(all_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Take the 5m base DataFrame and merge in the LAST KNOWN bar from each
    higher timeframe, aligned by timestamp (as-of merge so no look-ahead).

    This produces a flat DataFrame on the 5m index that the environment
    can iterate over step-by-step.
    """
    base = all_tf["5m"].copy()

    for tf in ["15m", "1h", "4h", "1d"]:
        htf = all_tf[tf].copy()
        htf.columns = [f"{tf}_{c}" for c in htf.columns]
        # pd.merge_asof requires sorted indices — already sorted
        base = pd.merge_asof(
            base.reset_index(),
            htf.reset_index().rename(columns={"timestamp": f"ts_{tf}"}),
            left_on="timestamp",
            right_on=f"ts_{tf}",
            direction="backward",
        ).set_index("timestamp").drop(columns=[f"ts_{tf}"], errors="ignore")

    return base
