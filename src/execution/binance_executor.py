"""
binance_executor.py
────────────────────
Handles all real Binance Futures Testnet order execution.

Uses python-binance for REST calls.

Testnet base URL: https://testnet.binancefuture.com
"""

import os
import time
import logging
from typing import Optional, Dict, Tuple

import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class BinanceFuturesExecutor:
    """
    Wraps Binance Futures Testnet REST API.

    All methods return (success: bool, data: dict).

    Important testnet notes:
    • Testnet URL must be injected into the client.
    • Funding rates are simulated but roughly realistic.
    • Use isolated margin mode + leverage ≤ 3.
    """

    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(self, symbol: str = "BTCUSDT", max_leverage: int = 3):
        self.symbol = symbol
        self.max_leverage = max_leverage

        api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")

        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set in .environment"
            )

        # python-binance Client with testnet=True for futures
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
        # Override futures base URL to point to testnet
        self.client.FUTURES_URL = self.FUTURES_TESTNET_URL

        self._init_account()

    # ─────────────────────────────────────────────────────────────────

    def _init_account(self):
        """Set leverage and margin mode on startup."""
        try:
            # Set to isolated margin
            self.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType="ISOLATED"
            )
        except BinanceAPIException as e:
            if "No need to change margin type" not in str(e):
                logger.warning(f"Could not set isolated margin: {e}")

        try:
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.max_leverage
            )
            logger.info(f"Leverage set to {self.max_leverage}x on {self.symbol}")
        except BinanceAPIException as e:
            logger.warning(f"Could not set leverage: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Account information
    # ─────────────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Returns available USDT balance (float)."""
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["availableBalance"])
        except BinanceAPIException as e:
            logger.error(f"get_account_balance error: {e}")
        return 0.0

    def get_position(self) -> Dict:
        """
        Returns current position info for self.symbol.

        Returns dict:
          side: 'LONG' | 'SHORT' | 'NONE'
          qty: float (positive)
          entry_price: float
          unrealized_pnl: float
          leverage: int
        """
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                amt = float(pos["positionAmt"])
                if abs(amt) > 1e-6:
                    side = "LONG" if amt > 0 else "SHORT"
                    return {
                        "side": side,
                        "qty": abs(amt),
                        "entry_price": float(pos["entryPrice"]),
                        "unrealized_pnl": float(pos["unRealizedProfit"]),
                        "leverage": int(pos["leverage"]),
                    }
        except BinanceAPIException as e:
            logger.error(f"get_position error: {e}")

        return {"side": "NONE", "qty": 0.0, "entry_price": 0.0,
                "unrealized_pnl": 0.0, "leverage": self.max_leverage}

    def get_current_price(self) -> float:
        """Returns the latest mark price."""
        try:
            ticker = self.client.futures_mark_price(symbol=self.symbol)
            return float(ticker["markPrice"])
        except BinanceAPIException as e:
            logger.error(f"get_current_price error: {e}")
            return 0.0

    def get_funding_rate(self) -> float:
        """Returns the current funding rate (e.g. 0.0001 = 0.01%)."""
        try:
            funding = self.client.futures_funding_rate(symbol=self.symbol, limit=1)
            if funding:
                return float(funding[-1]["fundingRate"])
        except BinanceAPIException as e:
            logger.error(f"get_funding_rate error: {e}")
        return 0.0001  # default assumption

    # ─────────────────────────────────────────────────────────────────
    # Order execution
    # ─────────────────────────────────────────────────────────────────

    def open_long(self, qty: float) -> Tuple[bool, dict]:
        """Open a new long position. qty = BTC amount (e.g. 0.001)."""
        return self._place_market_order("BUY", qty)

    def open_short(self, qty: float) -> Tuple[bool, dict]:
        """Open a new short position."""
        return self._place_market_order("SELL", qty)

    def close_position(self, position: dict) -> Tuple[bool, dict]:
        """Close the entire current position."""
        if position["side"] == "NONE" or position["qty"] == 0:
            return True, {"msg": "no position to close"}

        close_side = "SELL" if position["side"] == "LONG" else "BUY"
        return self._place_market_order(close_side, position["qty"], reduce_only=True)

    def reduce_position(self, position: dict, pct: float = 0.5) -> Tuple[bool, dict]:
        """Close `pct` fraction of the current position."""
        if position["side"] == "NONE":
            return True, {"msg": "no position"}

        qty = round(position["qty"] * pct, 3)
        if qty < 0.001:
            return False, {"msg": "quantity too small"}

        close_side = "SELL" if position["side"] == "LONG" else "BUY"
        return self._place_market_order(close_side, qty, reduce_only=True)

    def _place_market_order(
        self,
        side: str,
        qty: float,
        reduce_only: bool = False,
        max_retries: int = 3,
    ) -> Tuple[bool, dict]:
        """
        Place a market order with retry logic.

        Returns (success, order_dict)
        """
        qty = round(qty, 3)  # Binance min step for BTCUSDT = 0.001

        if qty < 0.001:
            return False, {"error": f"qty {qty} below minimum 0.001"}

        for attempt in range(max_retries):
            try:
                params = dict(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=qty,
                )
                if reduce_only:
                    params["reduceOnly"] = "true"

                order = self.client.futures_create_order(**params)

                fill_price = self._get_fill_price(order)
                logger.info(
                    f"ORDER {side} {qty} BTC @ ~{fill_price:.2f} | "
                    f"orderId={order['orderId']}"
                )
                return True, {
                    "orderId": order["orderId"],
                    "fill_price": fill_price,
                    "qty": qty,
                    "side": side,
                }

            except BinanceAPIException as e:
                logger.warning(f"Order attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))

        return False, {"error": "max retries exceeded"}

    def _get_fill_price(self, order: dict) -> float:
        """Parse average fill price from order response."""
        try:
            fills = order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                return total_cost / total_qty if total_qty > 0 else 0.0
            return float(order.get("avgPrice", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────
    # Stop Loss / Take Profit (set as orders, not just environment-side logic)
    # ─────────────────────────────────────────────────────────────────

    def set_stop_loss(self, side: str, stop_price: float, qty: float) -> Tuple[bool, dict]:
        """
        Place a STOP_MARKET order as a hard stop loss.
        side: 'LONG' or 'SHORT' (the current position side)
        """
        close_side = "SELL" if side == "LONG" else "BUY"
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=round(stop_price, 2),
                closePosition=True,
                timeInForce="GTE_GTC",
            )
            return True, order
        except BinanceAPIException as e:
            logger.warning(f"SL order failed: {e}")
            return False, {"error": str(e)}

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            return True
        except BinanceAPIException as e:
            logger.warning(f"Cancel all orders failed: {e}")
            return False
