"""
binance_executor.py
===================
Testnet order execution via python-binance.

Bugs fixed
----------
1. MARKET order slippage risk
   All orders used type="MARKET" with no fill-price check.
   Fix: _get_fill_price() now compares actual fill against the mark price at
   order submission time. If slippage exceeds MAX_SLIPPAGE_PCT the position
   is immediately closed and the trade is logged as rejected.

2. timeInForce="GTE_GTC" invalid value
   Binance Futures only accepts "GTC", "IOC", "GTX", "FOK" for timeInForce.
   "GTE_GTC" is not a valid value and causes an API error on STOP_MARKET orders.
   Fix: timeInForce is removed entirely from STOP_MARKET / TAKE_PROFIT_MARKET
   orders (Binance does not require it for those types). For LIMIT orders it
   is set to "GTC".

3. BinanceAPIException on startup in restricted regions (HTTP 451)
   client.ping() in __init__ raises immediately in US-hosted servers
   (Lightning.ai, some Thunder Compute nodes).
   Fix:
     a. __init__ accepts an optional `proxies` dict that is applied to the
        underlying requests.Session before ping().
     b. ping() failure is caught and logged as a warning — the executor
        continues in DEGRADED mode (paper-trade only, no real orders).
     c. An IN_RESTRICTED_REGION flag is set so callers can check status.

4. Minimum order quantity is static (0.001 BTC)
   At BTC = $65,000 that's $65 minimum notional — far above a small testnet
   balance even with 3× leverage.
   Fix: min_qty is computed dynamically from Binance's exchangeInfo
   (LOT_SIZE filter). If that call also fails (restricted region), falls back
   to computing MIN_NOTIONAL / current_price so the minimum is expressed
   in dollars not BTC quantity.
"""

from __future__ import annotations

import logging

from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL           = "BTCUSDT"
LEVERAGE         = 3
MAX_SLIPPAGE_PCT = 0.003    # 0.3% — realistic for BTC; tighten for illiquid pairs
MIN_NOTIONAL_USD = 5.0      # hard floor — don't open positions below $5 notional
PING_TIMEOUT_S   = 5        # seconds before ping is considered failed


class BinanceExecutor:
    """
    Manages order execution on Binance USDM Futures testnet.

    Parameters
    ----------
    api_key    : str
    api_secret : str
    testnet    : bool  — always True for training; set False only for live production
    proxies    : dict  — e.g. {"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890"}
                         Needed when running on US-region servers (Lightning.ai,
                         some Thunder Compute nodes) to bypass geo-restriction.
    """

    def __init__(
        self,
        api_key:    str,
        api_secret: str,
        testnet:    bool = True,
        proxies:    Optional[Dict[str, str]] = None,
    ):
        from binance.client import Client
        from binance.exceptions import BinanceAPIException

        self.testnet             = testnet
        self.in_restricted_region = False
        self._min_qty: Optional[float] = None   # populated lazily

        self.client = Client(
            api_key    = api_key,
            api_secret = api_secret,
            testnet    = testnet,
        )

        # ── Fix 3a: apply proxy before any network call ──────────────────────
        if proxies:
            self.client.session.proxies.update(proxies)
            logger.info("Proxy configured: %s", proxies)

        # ── Fix 3b: catch 451 / connection errors on ping ────────────────────
        try:
            self.client.futures_ping()
            logger.info(
                "Binance %s connection OK.",
                "testnet" if testnet else "LIVE",
            )
        except BinanceAPIException as e:
            if e.status_code in (451, 403):
                logger.warning(
                    "Binance API returned HTTP %d — likely a geo-restriction "
                    "(restricted region / US server).\n"
                    "The executor will run in DEGRADED mode: positions are "
                    "simulated locally, no real orders will be placed.\n"
                    "To fix: pass proxies={'https': 'http://127.0.0.1:PORT'} "
                    "or run from a non-restricted server.",
                    e.status_code,
                )
                self.in_restricted_region = True
            else:
                logger.warning(
                    "Binance ping failed (code=%s, msg=%s). "
                    "Continuing in degraded mode.",
                    e.status_code, e.message,
                )
                self.in_restricted_region = True
        except Exception as e:
            logger.warning(
                "Binance connection error: %s. Continuing in degraded mode.", e
            )
            self.in_restricted_region = True

        # Set leverage (best-effort — skip if restricted)
        if not self.in_restricted_region:
            self._set_leverage()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def execute(
        self,
        action:        int,
        current_price: float,
        position:      int,      # -1 short / 0 flat / 1 long
        entry_price:   float,
    ) -> Tuple[int, float, float]:
        """
        Execute a trading action.

        Returns
        -------
        (new_position, new_entry_price, realised_pnl_pct)
        """
        # Action map — mirrors BinanceEnv
        # 0=HOLD 1=LONG_FULL 2=LONG_HALF 3=SHORT_FULL 4=SHORT_HALF
        # 5=CLOSE 6=REDUCE_HALF

        if action == 0:
            return position, entry_price, 0.0

        if action in (1, 2):
            if position == 1:
                return position, entry_price, 0.0   # already long
            pnl = self._close_position(position, entry_price, current_price)
            size_mult = 1.0 if action == 1 else 0.5
            ok, fill = self._open_position("BUY", current_price, size_mult)
            if ok:
                return 1, fill, pnl
            return 0, 0.0, pnl

        if action in (3, 4):
            if position == -1:
                return position, entry_price, 0.0   # already short
            pnl = self._close_position(position, entry_price, current_price)
            size_mult = 1.0 if action == 3 else 0.5
            ok, fill = self._open_position("SELL", current_price, size_mult)
            if ok:
                return -1, fill, pnl
            return 0, 0.0, pnl

        if action == 5:
            pnl = self._close_position(position, entry_price, current_price)
            return 0, 0.0, pnl

        if action == 6:
            if position == 0:
                return 0, 0.0, 0.0
            pnl = self._close_half(position, entry_price, current_price)
            new_entry = (entry_price + current_price) / 2
            return position, new_entry, pnl

        return position, entry_price, 0.0

    def get_equity(self) -> float:
        """Return account balance in USDT. Returns 0.0 on failure."""
        if self.in_restricted_region:
            return 0.0
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
        except Exception as e:
            logger.warning("get_equity failed: %s", e)
        return 0.0

    def close_all(self) -> None:
        """Emergency close of all open positions."""
        if self.in_restricted_region:
            logger.warning("close_all skipped — restricted region / degraded mode.")
            return
        try:
            positions = self.client.futures_position_information(symbol=SYMBOL)
            for pos in positions:
                qty = float(pos["positionAmt"])
                if abs(qty) > 0:
                    side = "SELL" if qty > 0 else "BUY"
                    self.client.futures_create_order(
                        symbol   = SYMBOL,
                        side     = side,
                        type     = "MARKET",
                        quantity = abs(qty),
                        reduceOnly = True,
                    )
                    logger.info("Emergency closed %.4f BTC (%s)", abs(qty), side)
        except Exception as e:
            logger.error("close_all failed: %s", e)

    # -----------------------------------------------------------------------
    # Stop-loss & Take-profit
    # -----------------------------------------------------------------------

    def set_stop_loss(self, side: str, stop_price: float) -> bool:
        """
        Place a STOP_MARKET order.

        Fix 2: removed timeInForce entirely.
        Binance Futures STOP_MARKET does not require timeInForce; passing
        "GTE_GTC" (the old value) causes APIError: -1104 (invalid parameter).
        """
        if self.in_restricted_region:
            return False
        try:
            close_side = "SELL" if side == "BUY" else "BUY"
            self.client.futures_create_order(
                symbol     = SYMBOL,
                side       = close_side,
                type       = "STOP_MARKET",
                stopPrice  = round(stop_price, 2),
                closePosition = True,   # closes entire position at stop
                # timeInForce NOT passed — not valid for STOP_MARKET
            )
            logger.info("Stop-loss set at %.2f", stop_price)
            return True
        except Exception as e:
            logger.warning("set_stop_loss failed: %s", e)
            return False

    def set_take_profit(self, side: str, tp_price: float) -> bool:
        """
        Place a TAKE_PROFIT_MARKET order.
        Same fix as set_stop_loss — no timeInForce.
        """
        if self.in_restricted_region:
            return False
        try:
            close_side = "SELL" if side == "BUY" else "BUY"
            self.client.futures_create_order(
                symbol        = SYMBOL,
                side          = close_side,
                type          = "TAKE_PROFIT_MARKET",
                stopPrice     = round(tp_price, 2),
                closePosition = True,
                # timeInForce NOT passed
            )
            logger.info("Take-profit set at %.2f", tp_price)
            return True
        except Exception as e:
            logger.warning("set_take_profit failed: %s", e)
            return False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _set_leverage(self) -> None:
        try:
            self.client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
            logger.info("Leverage set to %d×", LEVERAGE)
        except Exception as e:
            logger.warning("set_leverage failed: %s", e)

    def _get_min_qty(self, current_price: float) -> float:
        """
        Fix 4: dynamic minimum quantity.

        Priority:
          1. Binance exchangeInfo LOT_SIZE filter  (exact exchange minimum)
          2. Fallback: MIN_NOTIONAL_USD / current_price
             At $65k BTC, MIN_NOTIONAL=$5 → min_qty = 0.000077 BTC
             (much lower than the old hardcoded 0.001 = $65)
        """
        if self._min_qty is not None:
            return self._min_qty

        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == SYMBOL:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            self._min_qty = float(f["minQty"])
                            logger.debug(
                                "LOT_SIZE minQty from exchangeInfo: %s",
                                self._min_qty,
                            )
                            return self._min_qty
        except Exception as e:
            logger.debug("exchangeInfo fetch failed: %s — using notional fallback", e)

        # Fallback: derive from notional floor
        fallback = MIN_NOTIONAL_USD / max(current_price, 1.0)
        # Round to 3 decimal places (Binance standard for BTCUSDT)
        fallback = round(fallback, 3)
        logger.debug("Min qty fallback (notional): %.6f BTC", fallback)
        self._min_qty = fallback
        return fallback

    def _calc_quantity(
        self, current_price: float, size_mult: float = 1.0
    ) -> float:
        """
        Calculate order quantity from account balance + leverage.

        qty = (balance × LEVERAGE × kelly_fraction × size_mult) / price
        Clamps to exchange minimum.
        """
        equity = self.get_equity()
        if equity <= 0:
            return 0.0

        # Use 25% Kelly (quarter Kelly) of leveraged balance
        notional = equity * LEVERAGE * 0.25 * size_mult
        qty      = notional / current_price
        qty      = round(qty, 3)

        min_qty = self._get_min_qty(current_price)

        if qty < min_qty:
            logger.warning(
                "Calculated qty %.6f BTC < min %.6f BTC "
                "(equity=%.2f USDT, price=%.2f). "
                "Order skipped. Add more testnet funds or reduce min notional.",
                qty, min_qty, equity, current_price,
            )
            return 0.0

        # Also check absolute notional floor
        if qty * current_price < MIN_NOTIONAL_USD:
            logger.warning(
                "Notional %.2f USDT < floor %.2f USDT. Order skipped.",
                qty * current_price, MIN_NOTIONAL_USD,
            )
            return 0.0

        return qty

    def _get_fill_price(self, order: dict) -> float:
        """Extract average fill price from an order response."""
        try:
            avg = float(order.get("avgPrice", 0) or order.get("price", 0))
            if avg > 0:
                return avg
            # Fallback: query the order
            filled = self.client.futures_get_order(
                symbol=SYMBOL, orderId=order["orderId"]
            )
            return float(filled.get("avgPrice", 0))
        except Exception:
            return 0.0

    def _check_slippage(
        self, side: str, mark_price: float, fill_price: float
    ) -> bool:
        """
        Fix 1: slippage guard.

        Returns True if slippage is acceptable, False if it exceeds threshold.
        For a BUY:  fill > mark * (1 + MAX_SLIPPAGE_PCT) is bad
        For a SELL: fill < mark * (1 - MAX_SLIPPAGE_PCT) is bad
        """
        if fill_price <= 0 or mark_price <= 0:
            return True   # can't check — allow through

        if side == "BUY":
            slippage = (fill_price - mark_price) / mark_price
        else:
            slippage = (mark_price - fill_price) / mark_price

        if slippage > MAX_SLIPPAGE_PCT:
            logger.warning(
                "Excessive slippage %.4f%% on %s order "
                "(mark=%.2f fill=%.2f, threshold=%.4f%%). "
                "Position will be closed immediately.",
                slippage * 100, side, mark_price, fill_price,
                MAX_SLIPPAGE_PCT * 100,
            )
            return False
        return True

    def _open_position(
        self, side: str, current_price: float, size_mult: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Place a MARKET order to open a position.
        Returns (success, fill_price).
        """
        if self.in_restricted_region:
            # Degraded mode — simulate fill at current price
            logger.debug("Degraded mode: simulated %s fill at %.2f", side, current_price)
            return True, current_price

        qty = self._calc_quantity(current_price, size_mult)
        if qty <= 0:
            return False, 0.0

        try:
            order = self.client.futures_create_order(
                symbol   = SYMBOL,
                side     = side,
                type     = "MARKET",
                quantity = qty,
            )
            fill_price = self._get_fill_price(order)

            # ── Fix 1: slippage check ──────────────────────────────────────
            if not self._check_slippage(side, current_price, fill_price):
                # Immediately close the badly-filled position
                close_side = "SELL" if side == "BUY" else "BUY"
                self.client.futures_create_order(
                    symbol     = SYMBOL,
                    side       = close_side,
                    type       = "MARKET",
                    quantity   = qty,
                    reduceOnly = True,
                )
                return False, 0.0

            logger.info(
                "Opened %s %.4f BTC @ %.2f (mark=%.2f)",
                side, qty, fill_price, current_price,
            )
            return True, fill_price if fill_price > 0 else current_price

        except Exception as e:
            logger.error("_open_position failed (%s): %s", side, e)
            return False, 0.0

    def _close_position(
        self, position: int, entry_price: float, current_price: float
    ) -> float:
        """Close entire position. Returns realised PnL as a fraction."""
        if position == 0:
            return 0.0

        pnl = (current_price - entry_price) / entry_price * position

        if self.in_restricted_region:
            logger.debug(
                "Degraded mode: simulated close at %.2f (pnl=%.4f%%)",
                current_price, pnl * 100,
            )
            return pnl

        side = "SELL" if position == 1 else "BUY"
        try:
            self.client.futures_create_order(
                symbol     = SYMBOL,
                side       = side,
                type       = "MARKET",
                reduceOnly = True,
                # quantity not needed when reduceOnly + closePosition implied
                quantity   = self._get_open_qty(),
            )
            logger.info(
                "Closed %s position @ %.2f (pnl=%.4f%%)",
                "LONG" if position == 1 else "SHORT",
                current_price, pnl * 100,
            )
        except Exception as e:
            logger.error("_close_position failed: %s", e)

        return pnl

    def _close_half(
        self, position: int, entry_price: float, current_price: float
    ) -> float:
        """Close half the position. Returns 50% of unrealised PnL."""
        pnl = (current_price - entry_price) / entry_price * position * 0.5

        if self.in_restricted_region:
            return pnl

        side = "SELL" if position == 1 else "BUY"
        qty  = self._get_open_qty()
        half = round(qty / 2, 3)

        min_qty = self._get_min_qty(current_price)
        if half < min_qty:
            logger.warning(
                "Half qty %.6f < min %.6f — skipping REDUCE_HALF.", half, min_qty
            )
            return 0.0

        try:
            self.client.futures_create_order(
                symbol     = SYMBOL,
                side       = side,
                type       = "MARKET",
                quantity   = half,
                reduceOnly = True,
            )
            logger.info("Reduced position by %.4f BTC @ %.2f", half, current_price)
        except Exception as e:
            logger.error("_close_half failed: %s", e)

        return pnl

    def _get_open_qty(self) -> float:
        """Return the absolute open quantity for SYMBOL."""
        try:
            positions = self.client.futures_position_information(symbol=SYMBOL)
            for pos in positions:
                qty = abs(float(pos["positionAmt"]))
                if qty > 0:
                    return qty
        except Exception as e:
            logger.warning("_get_open_qty failed: %s", e)
        return 0.0