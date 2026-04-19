"""
websocket_feed.py
──────────────────
Real-time 5m candle WebSocket feed using python-binance.

FIX: All binance imports are now lazy (inside functions/class methods)
so the module can be imported without python-binance installed.
"""

import time
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BUFFER_SIZES = {
    "1m":  600,
    "5m":  500,
    "15m": 300,
    "1h":  200,
    "4h":  100,
    "1d":  60,
}

RESAMPLE_RULES = {
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}


class WebSocketCandleFeed:
    """
    Streams 1m futures klines via Binance WebSocket and maintains
    rolling OHLCV DataFrames for 5m / 15m / 1h / 4h / 1d.
    """

    def __init__(self, symbol: str = "BTCUSDT", api_key: str = "", api_secret: str = ""):
        self.symbol = symbol.lower()
        self._api_key = api_key
        self._api_secret = api_secret
        self._lock = threading.Lock()
        self._raw_1m: deque = deque(maxlen=BUFFER_SIZES["1m"])
        self._open_candle: Optional[dict] = None
        self._client = None
        self._twm = None
        self._running = False
        self._cached: Dict[str, pd.DataFrame] = {}

    def start(self, warmup_candles: int = 60):
        if self._running:
            return
        # Lazy import
        try:
            from binance.client import Client
            from binance import ThreadedWebsocketManager
        except ImportError:
            raise ImportError("python-binance is required: pip install python-binance")

        logger.info(f"Starting WebSocket feed for {self.symbol.upper()} futures...")
        self._client = Client(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=True,
        )
        self._client.FUTURES_URL = "https://testnet.binancefuture.com"
        self._prefill_rest(warmup_candles)
        self._twm = ThreadedWebsocketManager(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=True,
        )
        self._twm.start()
        self._twm.start_kline_futures_socket(
            callback=self._handle_kline,
            symbol=self.symbol.upper(),
            interval="1m",
        )
        self._running = True
        logger.info("WebSocket feed started.")

    def stop(self):
        if self._twm:
            self._twm.stop()
        self._running = False
        logger.info("WebSocket feed stopped.")

    def get_candles(self) -> Dict[str, pd.DataFrame]:
        with self._lock:
            if not self._cached:
                return self._empty_candles()
            return dict(self._cached)

    def get_current_price(self) -> float:
        with self._lock:
            if self._open_candle:
                return float(self._open_candle.get("close", 0.0))
            if self._raw_1m:
                return float(self._raw_1m[-1]["close"])
        return 0.0

    def is_ready(self, min_5m_candles: int = 10) -> bool:
        df = self._cached.get("5m")
        return df is not None and len(df) >= min_5m_candles

    def _handle_kline(self, msg: dict):
        try:
            if msg.get("e") != "kline":
                return
            k = msg["k"]
            candle = {
                "timestamp": pd.Timestamp(k["t"], unit="ms", tz="UTC"),
                "open":   float(k["o"]),
                "high":   float(k["h"]),
                "low":    float(k["l"]),
                "close":  float(k["c"]),
                "volume": float(k["v"]),
            }
            is_closed = bool(k["x"])
            with self._lock:
                if is_closed:
                    self._raw_1m.append(candle)
                    self._open_candle = None
                    self._rebuild_cache()
                else:
                    self._open_candle = candle
        except Exception as e:
            logger.warning(f"WebSocket handler error: {e}")

    def _prefill_rest(self, n_candles: int):
        try:
            raw = self._client.futures_klines(
                symbol=self.symbol.upper(),
                interval="1m",
                limit=min(n_candles, 1000),
            )
            for row in raw[:-1]:
                self._raw_1m.append({
                    "timestamp": pd.Timestamp(row[0], unit="ms", tz="UTC"),
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })
            self._rebuild_cache()
        except Exception as e:
            logger.warning(f"REST prefill failed: {e}")

    def _rebuild_cache(self):
        if not self._raw_1m:
            return
        try:
            df_1m = pd.DataFrame(list(self._raw_1m))
            df_1m.set_index("timestamp", inplace=True)
            df_1m = df_1m.astype(float)
            new_cache = {}
            for tf, rule in RESAMPLE_RULES.items():
                resampled = df_1m.resample(rule).agg({
                    "open": "first", "high": "max",
                    "low": "min", "close": "last", "volume": "sum",
                }).dropna()
                new_cache[tf] = resampled
            self._cached = new_cache
        except Exception as e:
            logger.warning(f"_rebuild_cache error: {e}")

    def _empty_candles(self) -> Dict[str, pd.DataFrame]:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return {tf: empty.copy() for tf in RESAMPLE_RULES}


class RESTCandleFeed:
    """Slower fallback that fetches candles via REST on every call."""

    def __init__(self, client, symbol: str = "BTCUSDT"):
        self.client = client
        self.symbol = symbol
        self._last_fetch: float = 0.0
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_candles(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        now = time.time()
        if not force and now - self._last_fetch < 55:
            return self._cache

        tf_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        result = {}
        for key, interval in tf_map.items():
            try:
                raw = self.client.futures_klines(
                    symbol=self.symbol, interval=interval, limit=500,
                )
                df = pd.DataFrame(raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "qv", "n", "tbv", "tqv", "ig"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                result[key] = df.iloc[:-1]
            except Exception as e:
                logger.warning(f"RESTCandleFeed error for {key}: {e}")
                result[key] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        self._cache = result
        self._last_fetch = now
        return result

    def get_current_price(self) -> float:
        try:
            ticker = self.client.futures_mark_price(symbol=self.symbol)
            return float(ticker["markPrice"])
        except Exception:
            return 0.0
