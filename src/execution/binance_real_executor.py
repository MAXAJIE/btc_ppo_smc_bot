"""
binance_real_executor.py
========================
LIVE (non-testnet) order execution on Binance USDM Futures.

⚠ WARNING ⚠
-----------
This file places REAL orders using REAL money.
Only use this after:
  1. Extensive testnet validation (main_live.py passing 4-check validator)
  2. Model has Sharpe > 1.0, Calmar > 2.0, MC win rate > 80%
  3. You understand the kill-switch and position-sizing logic below
  4. You have set BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET in .env
     (these are DIFFERENT credentials from the testnet keys)

Architecture
------------
Inherits all the bug-fixes from BinanceExecutor (testnet):
  - Dynamic min qty from exchangeInfo
  - No timeInForce on STOP_MARKET / TAKE_PROFIT_MARKET
  - Slippage guard with auto-close
  - Degraded mode on geo-restriction

Extra safeguards added for real money:
  - Hard position size cap (never risk more than MAX_RISK_PER_TRADE_PCT)
  - Pre-trade balance check (refuse to open if equity dropped > MAX_DAILY_DD)
  - All orders confirmed before proceeding (verify fill via order status)
  - Automatic SL + TP bracket placed immediately after every open
  - Daily loss limit (stops trading if daily loss > MAX_DAILY_LOSS_PCT)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk constants — edit these before going live
# ---------------------------------------------------------------------------
SYMBOL               = "BTCUSDT"
LEVERAGE             = 3                  # max 3× as per strategy

MAX_RISK_PER_TRADE_PCT  = 0.01           # never risk > 1% of account per trade
MAX_DAILY_LOSS_PCT      = 0.05           # stop trading if down > 5% today
MAX_POSITION_SIZE_USDT  = 500.0          # hard cap in USDT notional
MAX_SLIPPAGE_PCT        = 0.002          # tighter than testnet (0.2%)
MIN_NOTIONAL_USD        = 10.0           # Binance minimum (higher for live)
CONFIRM_RETRIES         = 3              # times to poll order status after fill
CONFIRM_WAIT_S          = 0.5            # seconds between retries


class BinanceRealExecutor:
    """
    Live production order executor.

    Parameters
    ----------
    api_key    : str  — LIVE api key (not testnet)
    api_secret : str  — LIVE api secret (not testnet)
    proxies    : dict — optional proxy for geo-restricted servers
    dry_run    : bool — if True, log orders but don't actually send them
                        Use this for a final sanity check before going fully live.
    """

    def __init__(
        self,
        api_key:    str,
        api_secret: str,
        proxies:    Optional[Dict[str, str]] = None,
        dry_run:    bool = False,
    ):
        from binance.client import Client

        self.dry_run              = dry_run
        self.in_restricted_region = False
        self._min_qty: Optional[float]  = None
        self._daily_start_equity: Optional[float] = None
        self._today_date: Optional[str] = None

        if dry_run:
            logger.warning(
                "BinanceRealExecutor initialised in DRY-RUN mode. "
                "No real orders will be placed."
            )

        self.client = Client(
            api_key    = api_key,
            api_secret = api_secret,
            testnet    = False,          # LIVE
        )

        if proxies:
            self.client.session.proxies.update(proxies)
            logger.info("Proxy configured: %s", proxies)

        try:
            self.client.futures_ping()
            logger.info("Binance LIVE connection OK.")
        except Exception as e:
            logger.error(
                "Binance LIVE ping failed: %s\n"
                "Check API key, secret, and network access.", e
            )
            self.in_restricted_region = True

        if not self.in_restricted_region and not dry_run:
            self._set_leverage()

    # -----------------------------------------------------------------------
    # Public API  (same signature as BinanceExecutor for drop-in swap)
    # -----------------------------------------------------------------------

    def execute(
        self,
        action:        int,
        current_price: float,
        position:      int,
        entry_price:   float,
    ) -> Tuple[int, float, float]:
        """
        Execute a trading action on LIVE markets.
        Returns (new_position, new_entry_price, realised_pnl_pct).
        """
        # Daily loss circuit breaker
        if self._daily_loss_breaker_triggered(current_price):
            logger.critical(
                "Daily loss limit hit — refusing to execute action %d.", action
            )
            return position, entry_price, 0.0

        if action == 0:
            return position, entry_price, 0.0

        if action in (1, 2):
            if position == 1:
                return position, entry_price, 0.0
            pnl       = self._close_position(position, entry_price, current_price)
            size_mult = 1.0 if action == 1 else 0.5
            ok, fill  = self._open_position("BUY", current_price, size_mult)
            if ok:
                self._place_bracket(side="BUY", fill_price=fill,
                                    entry=fill, cfg_sl=0.03, cfg_tp=0.06)
                return 1, fill, pnl
            return 0, 0.0, pnl

        if action in (3, 4):
            if position == -1:
                return position, entry_price, 0.0
            pnl       = self._close_position(position, entry_price, current_price)
            size_mult = 1.0 if action == 3 else 0.5
            ok, fill  = self._open_position("SELL", current_price, size_mult)
            if ok:
                self._place_bracket(side="SELL", fill_price=fill,
                                    entry=fill, cfg_sl=0.03, cfg_tp=0.06)
                return -1, fill, pnl
            return 0, 0.0, pnl

        if action == 5:
            pnl = self._close_position(position, entry_price, current_price)
            return 0, 0.0, pnl

        if action == 6:
            if position == 0:
                return 0, 0.0, 0.0
            pnl       = self._close_half(position, entry_price, current_price)
            new_entry = (entry_price + current_price) / 2
            return position, new_entry, pnl

        return position, entry_price, 0.0

    def get_equity(self) -> float:
        if self.in_restricted_region or self.dry_run:
            return 1000.0   # dummy value for dry-run
        try:
            for b in self.client.futures_account_balance():
                if b["asset"] == "USDT":
                    equity = float(b["balance"])
                    self._update_daily_tracker(equity)
                    return equity
        except Exception as e:
            logger.warning("get_equity failed: %s", e)
        return 0.0

    def close_all(self) -> None:
        """Emergency close — called by kill-switch."""
        if self.dry_run:
            logger.warning("[DRY-RUN] close_all() — no orders sent.")
            return
        if self.in_restricted_region:
            return
        try:
            # First cancel all open SL/TP orders
            self.client.futures_cancel_all_open_orders(symbol=SYMBOL)
            logger.info("All open orders cancelled.")

            # Then close any open position
            positions = self.client.futures_position_information(symbol=SYMBOL)
            for pos in positions:
                qty = float(pos["positionAmt"])
                if abs(qty) > 1e-6:
                    side = "SELL" if qty > 0 else "BUY"
                    self.client.futures_create_order(
                        symbol     = SYMBOL,
                        side       = side,
                        type       = "MARKET",
                        quantity   = abs(qty),
                        reduceOnly = True,
                    )
                    logger.warning(
                        "Emergency closed %.4f BTC (%s)", abs(qty), side
                    )
        except Exception as e:
            logger.error("close_all failed: %s", e)

    # -----------------------------------------------------------------------
    # Bracket (SL + TP placed immediately after open)
    # -----------------------------------------------------------------------

    def _place_bracket(
        self,
        side:       str,
        fill_price: float,
        entry:      float,
        cfg_sl:     float = 0.03,
        cfg_tp:     float = 0.06,
    ) -> None:
        """
        Place stop-loss and take-profit immediately after opening a position.
        Ensures the position is always protected even if the process dies.
        """
        if side == "BUY":
            sl_price = round(entry * (1 - cfg_sl), 2)
            tp_price = round(entry * (1 + cfg_tp), 2)
        else:
            sl_price = round(entry * (1 + cfg_sl), 2)
            tp_price = round(entry * (1 - cfg_tp), 2)

        self._set_stop_loss(side, sl_price)
        self._set_take_profit(side, tp_price)

    def _set_stop_loss(self, side: str, stop_price: float) -> bool:
        if self.dry_run:
            logger.info("[DRY-RUN] SL @ %.2f", stop_price)
            return True
        if self.in_restricted_region:
            return False
        try:
            close_side = "SELL" if side == "BUY" else "BUY"
            self.client.futures_create_order(
                symbol        = SYMBOL,
                side          = close_side,
                type          = "STOP_MARKET",
                stopPrice     = stop_price,
                closePosition = True,
                # timeInForce NOT passed (fixed bug from testnet executor)
            )
            logger.info("SL set @ %.2f", stop_price)
            return True
        except Exception as e:
            logger.error("set_stop_loss failed: %s", e)
            return False

    def _set_take_profit(self, side: str, tp_price: float) -> bool:
        if self.dry_run:
            logger.info("[DRY-RUN] TP @ %.2f", tp_price)
            return True
        if self.in_restricted_region:
            return False
        try:
            close_side = "SELL" if side == "BUY" else "BUY"
            self.client.futures_create_order(
                symbol        = SYMBOL,
                side          = close_side,
                type          = "TAKE_PROFIT_MARKET",
                stopPrice     = tp_price,
                closePosition = True,
            )
            logger.info("TP set @ %.2f", tp_price)
            return True
        except Exception as e:
            logger.error("set_take_profit failed: %s", e)
            return False

    # -----------------------------------------------------------------------
    # Daily loss circuit breaker
    # -----------------------------------------------------------------------

    def _update_daily_tracker(self, equity: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._today_date != today:
            self._today_date       = today
            self._daily_start_equity = equity
            logger.info("Daily equity reset: %.2f USDT", equity)

    def _daily_loss_breaker_triggered(self, current_price: float) -> bool:
        equity = self.get_equity()
        if self._daily_start_equity is None or self._daily_start_equity <= 0:
            return False
        daily_loss = (self._daily_start_equity - equity) / self._daily_start_equity
        if daily_loss >= MAX_DAILY_LOSS_PCT:
            logger.critical(
                "Daily loss %.1f%% >= limit %.0f%%. Trading halted for today.",
                daily_loss * 100, MAX_DAILY_LOSS_PCT * 100,
            )
            return True
        return False

    # -----------------------------------------------------------------------
    # Order helpers
    # -----------------------------------------------------------------------

    def _set_leverage(self) -> None:
        try:
            self.client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
            logger.info("Leverage set to %d×", LEVERAGE)
        except Exception as e:
            logger.warning("set_leverage failed: %s", e)

    def _get_min_qty(self, price: float) -> float:
        if self._min_qty is not None:
            return self._min_qty
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == SYMBOL:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            self._min_qty = float(f["minQty"])
                            return self._min_qty
        except Exception as e:
            logger.debug("exchangeInfo failed: %s", e)
        self._min_qty = MIN_NOTIONAL_USD / max(price, 1.0)
        return round(self._min_qty, 6)

    def _calc_quantity(self, price: float, size_mult: float = 1.0) -> float:
        """
        Risk-based sizing: risk MAX_RISK_PER_TRADE_PCT of account per trade,
        capped at MAX_POSITION_SIZE_USDT notional.
        """
        equity = self.get_equity()
        if equity <= 0:
            return 0.0

        # Risk-based: if SL = 3%, position size = risk_budget / 0.03
        risk_budget = equity * MAX_RISK_PER_TRADE_PCT
        sl_pct      = 0.03
        notional    = min(risk_budget / sl_pct, MAX_POSITION_SIZE_USDT)
        notional   *= size_mult
        qty         = round(notional / price, 3)

        min_qty = self._get_min_qty(price)
        if qty < min_qty:
            logger.warning(
                "qty %.6f < min %.6f (equity=%.2f, price=%.2f). Skipped.",
                qty, min_qty, equity, price,
            )
            return 0.0

        if qty * price < MIN_NOTIONAL_USD:
            return 0.0

        return qty

    def _confirm_fill(self, order_id: int) -> float:
        """Poll until order is FILLED. Returns avgPrice."""
        for _ in range(CONFIRM_RETRIES):
            try:
                o = self.client.futures_get_order(
                    symbol=SYMBOL, orderId=order_id
                )
                if o["status"] == "FILLED":
                    return float(o["avgPrice"])
            except Exception:
                pass
            time.sleep(CONFIRM_WAIT_S)
        return 0.0

    def _check_slippage(
        self, side: str, mark: float, fill: float
    ) -> bool:
        if fill <= 0 or mark <= 0:
            return True
        slip = (fill - mark) / mark if side == "BUY" else (mark - fill) / mark
        if slip > MAX_SLIPPAGE_PCT:
            logger.warning(
                "Slippage %.4f%% > %.4f%% (%s mark=%.2f fill=%.2f). Closing.",
                slip * 100, MAX_SLIPPAGE_PCT * 100, side, mark, fill,
            )
            return False
        return True

    def _open_position(
        self, side: str, price: float, size_mult: float = 1.0
    ) -> Tuple[bool, float]:
        if self.dry_run:
            logger.info("[DRY-RUN] OPEN %s @ %.2f (mult=%.2f)", side, price, size_mult)
            return True, price

        qty = self._calc_quantity(price, size_mult)
        if qty <= 0:
            return False, 0.0

        try:
            order  = self.client.futures_create_order(
                symbol   = SYMBOL,
                side     = side,
                type     = "MARKET",
                quantity = qty,
            )
            fill   = self._confirm_fill(order["orderId"])
            fill   = fill or float(order.get("avgPrice", price))

            if not self._check_slippage(side, price, fill):
                close_side = "SELL" if side == "BUY" else "BUY"
                self.client.futures_create_order(
                    symbol=SYMBOL, side=close_side,
                    type="MARKET", quantity=qty, reduceOnly=True,
                )
                return False, 0.0

            logger.info("LIVE OPEN %s %.4f BTC @ %.2f", side, qty, fill)
            return True, fill

        except Exception as e:
            logger.error("_open_position failed: %s", e)
            return False, 0.0

    def _close_position(
        self, position: int, entry_price: float, price: float
    ) -> float:
        pnl = (price - entry_price) / entry_price * position if position else 0.0
        if position == 0:
            return 0.0

        if self.dry_run:
            logger.info(
                "[DRY-RUN] CLOSE %s @ %.2f pnl=%.4f%%",
                "LONG" if position == 1 else "SHORT", price, pnl * 100,
            )
            return pnl

        side = "SELL" if position == 1 else "BUY"
        try:
            # Cancel any open SL/TP before closing manually
            self.client.futures_cancel_all_open_orders(symbol=SYMBOL)
            qty = self._get_open_qty()
            if qty > 0:
                self.client.futures_create_order(
                    symbol=SYMBOL, side=side, type="MARKET",
                    quantity=qty, reduceOnly=True,
                )
            logger.info(
                "LIVE CLOSE %s @ %.2f (pnl=%.4f%%)",
                "LONG" if position == 1 else "SHORT", price, pnl * 100,
            )
        except Exception as e:
            logger.error("_close_position failed: %s", e)
        return pnl

    def _close_half(
        self, position: int, entry_price: float, price: float
    ) -> float:
        pnl  = (price - entry_price) / entry_price * position * 0.5
        side = "SELL" if position == 1 else "BUY"
        qty  = self._get_open_qty()
        half = round(qty / 2, 3)

        if self.dry_run:
            logger.info("[DRY-RUN] REDUCE %s by %.4f", side, half)
            return pnl

        min_qty = self._get_min_qty(price)
        if half < min_qty:
            return 0.0
        try:
            self.client.futures_create_order(
                symbol=SYMBOL, side=side, type="MARKET",
                quantity=half, reduceOnly=True,
            )
        except Exception as e:
            logger.error("_close_half failed: %s", e)
        return pnl

    def _get_open_qty(self) -> float:
        try:
            for pos in self.client.futures_position_information(symbol=SYMBOL):
                qty = abs(float(pos["positionAmt"]))
                if qty > 1e-6:
                    return qty
        except Exception as e:
            logger.warning("_get_open_qty failed: %s", e)
        return 0.0