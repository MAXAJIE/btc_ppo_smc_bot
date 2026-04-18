"""
binance_testnet_env.py
───────────────────────
The core Gymnasium environment.

Two modes:
  • "offline"  – steps through historical 5m candles (fast, used for PPO training)
  • "live"     – waits for real 5m candle closes on testnet (used for fine-tuning)

Episode length: 4,320 steps (15 days of 5m candles)

Action space:  Discrete(7)
  0 = HOLD
  1 = LONG_FULL    (Kelly-sized long)
  2 = LONG_HALF    (half-Kelly long)
  3 = SHORT_FULL   (Kelly-sized short)
  4 = SHORT_HALF   (half-Kelly short)
  5 = CLOSE        (close current position)
  6 = REDUCE_50    (close 50% of current position)

Observation: np.float32 array of shape (90,)  — see multi_tf_features.py
"""

import time
import logging
import random
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import yaml
import os

from ..features.multi_tf_features import build_observation, OBS_DIM
from ..features.garch_kelly import GarchKellyEstimator, kelly_position_size
from ..utils.reward import compute_step_reward, cost_penalty, funding_cost, killswitch_penalty
from ..utils.websocket_feed import WebSocketCandleFeed, RESTCandleFeed
from src.utils.logger import TradeLogger


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────

class BTCFuturesEnv(gym.Env):
    """
    BTCUSDT USDT-M Futures trading environment.

    Parameters
    ----------
    data_loader : DataLoader
        Loaded historical data (offline mode only).
    mode : str
        "offline" or "live"
    executor : BinanceFuturesExecutor | None
        Required for live mode.
    trade_logger : TradeLogger | None
        Optional, will create one internally if not provided.
    episode_idx : int | None
        Force a specific episode start index (for deterministic eval).
    """

    metadata = {"render_modes": []}

    # Action indices
    HOLD       = 0
    LONG_FULL  = 1
    LONG_HALF  = 2
    SHORT_FULL = 3
    SHORT_HALF = 4
    CLOSE      = 5
    REDUCE_50  = 6

    def __init__(
        self,
        data_loader=None,
        mode: str = "offline",
        executor=None,
        trade_logger: Optional[TradeLogger] = None,
        episode_idx: Optional[int] = None,
    ):
        super().__init__()

        self.cfg = _load_cfg()
        self.mode = mode
        self.data_loader = data_loader
        self.executor = executor
        self.trade_logger = trade_logger or TradeLogger()
        self._forced_episode_idx = episode_idx

        # Config shortcuts
        ep_cfg = self.cfg["episode"]
        self.episode_len = ep_cfg["candles_per_episode"]    # 4320
        self.warmup = ep_cfg["warmup_candles"]              # 500
        self.max_bars_in_trade = ep_cfg["max_bars_in_trade"]  # 576

        risk = self.cfg["risk"]
        self.initial_balance = risk["initial_balance"]       # 10000 USDT
        self.max_leverage = self.cfg["max_leverage"]         # 3
        self.sl_pct = risk["stop_loss_pct"]                  # 0.03
        self.tp_pct = risk["take_profit_pct"]                # 0.06
        self.max_drawdown_kill = risk["max_drawdown_kill"]   # 0.15

        act_cfg = self.cfg["actions"]
        self.size_fracs = {
            self.LONG_FULL:  act_cfg["long_full_size"],
            self.LONG_HALF:  act_cfg["long_half_size"],
            self.SHORT_FULL: act_cfg["short_full_size"],
            self.SHORT_HALF: act_cfg["short_half_size"],
        }

        # Gym spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Episode state (reset in reset())
        self._garch = GarchKellyEstimator()
        self._episode_counter = 0
        self._episode_start_indices = []

        # Live mode candle feed (started lazily on first reset in live mode)
        self._ws_feed: "WebSocketCandleFeed | None" = None

        # Initialise state attributes
        self._reset_state()

    # ─────────────────────────────────────────────────────────────────
    # Gym interface
    # ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._episode_counter += 1
        self._reset_state()

        if self.mode == "offline":
            self._cur_idx = self._pick_episode_start()
        else:
            # Live mode: episode length is still bounded by config
            self._cur_idx = 0
            self._episode_end_idx = self.episode_len  # use step_count comparison in live mode
            # Pre-populate live candles via REST so first obs doesn't fail
            if self.executor is not None:
                try:
                    self._update_live_candles()
                except Exception as e:
                    logger.warning(f"Initial live candle fetch failed: {e}")

        obs = self._build_obs()
        info = {"episode": self._episode_counter, "start_idx": self._cur_idx}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step (one 5m bar close).

        Returns: obs, reward, terminated, truncated, info
        """
        terminated = False
        truncated = False
        trade_closed = False
        realized_pnl_pct = 0.0
        tx_cost = 0.0
        fund_fee = 0.0

        prev_price = self._current_price()

        # ── Advance one bar ──────────────────────────────────────────
        if self.mode == "offline":
            self._cur_idx += 1
            if self._cur_idx >= self._episode_end_idx:
                truncated = True
        else:
            self._wait_for_next_5m_close()
            if self._step_count >= self.episode_len:
                truncated = True

        cur_price = self._current_price()
        self._step_count += 1
        self._global_step += 1

        # ── Advance per-trade and idle counters ───────────────────────
        if self._position != 0:
            self._bars_in_trade += 1
            self._bars_since_last_trade = 0
        else:
            self._bars_since_last_trade += 1

        # ── Hard stop-loss / take-profit check ───────────────────────
        if self._position != 0:
            sl_hit, tp_hit = self._check_sl_tp(cur_price)
            if sl_hit or tp_hit:
                reason = "SL" if sl_hit else "TP"
                realized_pnl_pct = self._close_position(cur_price, reason)
                trade_closed = True
                tx_cost = cost_penalty(self._notional, self._balance)

        # ── Max bars in trade check ───────────────────────────────────
        if self._position != 0 and self._bars_in_trade >= self.max_bars_in_trade:
            realized_pnl_pct = self._close_position(cur_price, "max_duration")
            trade_closed = True
            tx_cost = cost_penalty(self._notional, self._balance)

        # ── Process agent action ─────────────────────────────────────
        if not trade_closed:
            realized_pnl_pct, tx_cost, trade_closed = self._process_action(
                action, cur_price
            )

        # ── Funding rate (every 480 steps = 8h) ──────────────────────
        if self._step_count % 480 == 0 and self._position != 0:
            fund_fee = funding_cost(self._notional, self._funding_rate, self._balance)
            # Note: balance deduction happens via reward signal only — not doubled here

        # ── Update unrealised PnL ─────────────────────────────────────
        unrealized_pnl_pct = self._unrealized_pnl_pct(cur_price)

        # ── Update equity ─────────────────────────────────────────────
        self._equity = self._balance + (
            self._unrealized_pnl_usdt(cur_price) if self._position != 0 else 0.0
        )
        drawdown = max(0.0, (self._peak_equity - self._equity) / max(self._peak_equity, 1e-10))
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # ── Log equity periodically ───────────────────────────────────
        if self._step_count % 50 == 0:
            self.trade_logger.log_equity(
                self._episode_counter, self._step_count, self._equity
            )

        # ── Kill-switch ───────────────────────────────────────────────
        kill_triggered = False
        if drawdown >= self.max_drawdown_kill:
            logger.warning(
                f"Kill-switch triggered: drawdown={drawdown:.1%} >= {self.max_drawdown_kill:.0%} "
                f"| penalty = -{self.cfg['reward'].get('kill_penalty_scale', 50.0) * (1 + drawdown):.1f}"
            )
            if self._position != 0:
                realized_pnl_pct = self._close_position(cur_price, "kill_switch")
                trade_closed = True
                tx_cost += cost_penalty(self._notional, self._balance)
            kill_triggered = True
            terminated = True

        # ── Reward ───────────────────────────────────────────────────
        reward = compute_step_reward(
            position=self._position,
            unrealized_pnl_pct=unrealized_pnl_pct,
            bars_in_trade=self._bars_in_trade,
            current_drawdown=drawdown,
            trade_closed=trade_closed,
            realized_pnl_pct=realized_pnl_pct,
            transaction_cost=tx_cost,
            funding_fee=fund_fee,
            kill_triggered=kill_triggered,    # [A] punitive penalty
        )

        # ── Observation ──────────────────────────────────────────────
        obs = self._build_obs()

        info = {
            "step": self._step_count,
            "price": cur_price,
            "equity": self._equity,
            "balance": self._balance,
            "drawdown": drawdown,
            "position": self._position,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "trade_closed": trade_closed,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass  # Use dashboard.py for visualisation

    # ─────────────────────────────────────────────────────────────────
    # Action processing
    # ─────────────────────────────────────────────────────────────────

    def _process_action(
        self, action: int, price: float
    ) -> Tuple[float, float, bool]:
        """
        Execute the chosen action.
        Returns (realized_pnl_pct, tx_cost, trade_closed).
        """
        realized_pnl_pct = 0.0
        tx_cost = 0.0
        trade_closed = False

        if action == self.HOLD:
            return 0.0, 0.0, False

        if action == self.CLOSE:
            if self._position != 0:
                realized_pnl_pct = self._close_position(price, "agent_close")
                tx_cost = cost_penalty(self._notional, self._balance)
                trade_closed = True
            return realized_pnl_pct, tx_cost, trade_closed

        if action == self.REDUCE_50:
            if self._position != 0:
                realized_pnl_pct = self._reduce_position(price, 0.5)
                tx_cost = cost_penalty(self._notional * 0.5, self._balance)
                trade_closed = False  # still partially open
            return realized_pnl_pct, tx_cost, trade_closed

        # Opening / flipping positions
        new_side = 1 if action in (self.LONG_FULL, self.LONG_HALF) else -1
        size_frac = self.size_fracs.get(action, 1.0)

        # Close existing opposite position first
        if self._position != 0 and self._position != new_side:
            realized_pnl_pct = self._close_position(price, "flip")
            tx_cost += cost_penalty(self._notional, self._balance)
            trade_closed = True

        # Don't open if already in the same direction
        if self._position == new_side:
            return realized_pnl_pct, tx_cost, trade_closed

        # Open new position
        self._open_position(price, new_side, size_frac)
        tx_cost += cost_penalty(self._notional, self._balance)

        # Execute real order in live mode
        if self.mode == "live" and self.executor is not None:
            qty = self._qty
            if new_side == 1:
                self.executor.open_long(qty)
            else:
                self.executor.open_short(qty)

        return realized_pnl_pct, tx_cost, trade_closed

    # ─────────────────────────────────────────────────────────────────
    # Position management
    # ─────────────────────────────────────────────────────────────────

    def _open_position(self, price: float, side: int, size_frac: float):
        """Open a new position using Kelly-adjusted sizing."""
        log_returns = self._get_log_returns()
        gk_features = self._garch.compute(log_returns)
        kelly_adj = gk_features["kelly_adj"] * size_frac

        leverage = min(self.max_leverage, max(1, round(1.0 / max(gk_features["garch_vol"], 0.01))))
        qty = kelly_position_size(self._balance, price, kelly_adj, leverage, size_frac)

        qty = max(qty, 0.001)  # minimum BTC qty

        self._position = side
        self._entry_price = price
        self._qty = qty
        self._leverage = leverage
        self._notional = qty * price
        self._bars_in_trade = 0
        self._sl_price = price * (1 - self.sl_pct * side)
        self._tp_price = price * (1 + self.tp_pct * side)
        self._last_kelly_adj = kelly_adj
        self._entry_reason = f"kelly={kelly_adj:.3f} lev={leverage}"

        logger.debug(
            f"OPEN {'LONG' if side == 1 else 'SHORT'} @ {price:.2f} "
            f"qty={qty:.4f} notional={self._notional:.0f} lev={leverage}"
        )

    def _close_position(self, price: float, reason: str) -> float:
        """
        Close entire position. Returns realized_pnl_pct.
        Updates self._balance.
        """
        if self._position == 0:
            return 0.0

        raw_pnl_pct = (price - self._entry_price) / self._entry_price * self._position
        pnl_usdt = raw_pnl_pct * self._notional * self._leverage

        # Deduct commission (already included in reward via cost_penalty)
        commission = self._notional * (self.cfg["reward"]["commission_rate"] +
                                        self.cfg["reward"]["slippage_rate"])
        net_pnl_usdt = pnl_usdt - commission

        self._balance += net_pnl_usdt
        realized_pnl_pct = net_pnl_usdt / (self._notional + 1e-10)

        # Log the trade
        self.trade_logger.log_trade(
            episode=self._episode_counter,
            step=self._step_count,
            side="LONG" if self._position == 1 else "SHORT",
            entry_price=self._entry_price,
            exit_price=price,
            qty=self._qty,
            leverage=self._leverage,
            pnl_usdt=round(net_pnl_usdt, 4),
            pnl_pct=realized_pnl_pct,
            duration_bars=self._bars_in_trade,
            entry_reason=self._entry_reason,
            exit_reason=reason,
            equity=self._balance,
        )

        logger.debug(
            f"CLOSE @ {price:.2f} | PnL={net_pnl_usdt:.2f} USDT "
            f"({realized_pnl_pct*100:.2f}%) | reason={reason}"
        )

        # Reset position state
        self._position = 0
        self._entry_price = 0.0
        self._qty = 0.0
        self._notional = 0.0
        self._leverage = self.max_leverage
        self._bars_in_trade = 0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._bars_since_last_trade = 0

        # Execute real close in live mode
        if self.mode == "live" and self.executor is not None:
            pos_info = self.executor.get_position()
            self.executor.close_position(pos_info)
            self.executor.cancel_all_orders()

        return realized_pnl_pct

    def _reduce_position(self, price: float, pct: float) -> float:
        """Reduce position by `pct`. Returns partial pnl_pct."""
        if self._position == 0:
            return 0.0

        closed_notional = self._notional * pct
        raw_pnl_pct = (price - self._entry_price) / self._entry_price * self._position
        partial_pnl = raw_pnl_pct * closed_notional * self._leverage

        commission = closed_notional * (self.cfg["reward"]["commission_rate"] +
                                         self.cfg["reward"]["slippage_rate"])
        net_partial = partial_pnl - commission

        self._balance += net_partial
        self._qty *= (1 - pct)
        self._notional *= (1 - pct)

        if self.mode == "live" and self.executor is not None:
            pos_info = self.executor.get_position()
            self.executor.reduce_position(pos_info, pct)

        return net_partial / (closed_notional + 1e-10)

    def _check_sl_tp(self, price: float) -> Tuple[bool, bool]:
        """Check if current price hits stop-loss or take-profit."""
        if self._position == 0:
            return False, False

        if self._position == 1:   # long
            sl_hit = price <= self._sl_price
            tp_hit = price >= self._tp_price
        else:                      # short
            sl_hit = price >= self._sl_price
            tp_hit = price <= self._tp_price

        return sl_hit, tp_hit

    # ─────────────────────────────────────────────────────────────────
    # Observation & helpers
    # ─────────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        candles = self._get_candles()
        log_returns = self._get_log_returns()
        gk = self._garch.compute(log_returns)

        return build_observation(
            candles=candles,
            position=self._position,
            unrealized_pnl_pct=self._unrealized_pnl_pct(self._current_price()),
            leverage_used=float(self._leverage) if self._position != 0 else 0.0,
            bars_in_trade=self._bars_in_trade,
            account_drawdown=max(
                0.0,
                (self._peak_equity - self._equity) / max(self._peak_equity, 1e-10)
            ),
            kelly_adj=gk["kelly_adj"],
            bars_since_last_trade=self._bars_since_last_trade,
        )

    def _get_candles(self) -> Dict[str, pd.DataFrame]:
        """Return multi-TF candle dict aligned to current step."""
        if self.mode == "offline":
            return self.data_loader.get_multi_tf_candles(
                end_idx=self._cur_idx,
                lookback_5m=500,
            )
        else:
            # Live: return cached WebSocket/REST data, with empty-DF fallback
            _EMPTY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return {
                "5m":  self._live_candles.get("5m",  _EMPTY),
                "15m": self._live_candles.get("15m", _EMPTY),
                "1h":  self._live_candles.get("1h",  _EMPTY),
                "4h":  self._live_candles.get("4h",  _EMPTY),
                "1d":  self._live_candles.get("1d",  _EMPTY),
            }

    def _current_price(self) -> float:
        if self.mode == "offline":
            if self.data_loader is None:
                return 50000.0
            base = self.data_loader._base_df
            if base is None or self._cur_idx >= len(base):
                return 50000.0
            return float(base["close"].iloc[self._cur_idx])
        else:
            return self.executor.get_current_price() if self.executor else 50000.0

    def _unrealized_pnl_pct(self, price: float) -> float:
        if self._position == 0 or self._entry_price == 0:
            return 0.0
        return (price - self._entry_price) / self._entry_price * self._position

    def _unrealized_pnl_usdt(self, price: float) -> float:
        return self._unrealized_pnl_pct(price) * self._notional * self._leverage

    def _get_log_returns(self) -> pd.Series:
        """Get log returns for GARCH estimation."""
        if self.mode == "offline" and self.data_loader is not None:
            start = max(0, self._cur_idx - 300)
            closes = self.data_loader._base_df["close"].iloc[start:self._cur_idx + 1]
            return np.log(closes / closes.shift(1)).dropna()
        return pd.Series(np.random.normal(0, 0.001, 200))

    def _pick_episode_start(self) -> int:
        """Pick a random valid episode start index from historical data."""
        if self._forced_episode_idx is not None:
            return self._forced_episode_idx

        if not self._episode_start_indices and self.data_loader is not None:
            self._episode_start_indices = self.data_loader.get_episode_start_indices(
                episode_len=self.episode_len,
                warmup=self.warmup,
            )

        if self._episode_start_indices:
            start = random.choice(self._episode_start_indices)
        else:
            start = self.warmup

        self._episode_end_idx = start + self.episode_len
        return start

    def _reset_state(self):
        """Reset all mutable episode state."""
        self._cur_idx = 0
        self._episode_end_idx = 0
        self._step_count = 0
        self._global_step = getattr(self, "_global_step", 0)

        # Account
        self._balance = self.initial_balance
        self._equity = self.initial_balance
        self._peak_equity = self.initial_balance

        # Position
        self._position = 0
        self._entry_price = 0.0
        self._qty = 0.0
        self._notional = 0.0
        self._leverage = self.max_leverage
        self._bars_in_trade = 0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._last_kelly_adj = 0.1
        self._entry_reason = ""
        self._bars_since_last_trade = 0
        self._funding_rate = 0.0001  # default

        # Live mode cache
        self._live_candles = {}

    def _wait_for_next_5m_close(self):
        """Block until the next 5m candle closes (live mode only)."""
        import time as _time
        now = _time.time()
        seconds_in_5m = 300
        next_close = (int(now / seconds_in_5m) + 1) * seconds_in_5m
        wait = next_close - now
        if wait > 0:
            _time.sleep(wait + 0.5)  # +0.5s buffer for exchange latency

        # Refresh funding rate
        if self.executor:
            self._funding_rate = self.executor.get_funding_rate()

        # Refresh live candles (user should implement WebSocket feed here)
        # For now, a REST fallback:
        if self.executor:
            self._update_live_candles()

    def _update_live_candles(self):
        """
        Update live candle cache.
        Uses WebSocket feed if available, falls back to REST.
        """
        # Start WebSocket feed on first call
        if self._ws_feed is None and self.executor is not None:
            import os
            self._ws_feed = WebSocketCandleFeed(
                symbol=self.cfg["symbol"],
                api_key=os.getenv("BINANCE_TESTNET_API_KEY", ""),
                api_secret=os.getenv("BINANCE_TESTNET_API_SECRET", ""),
            )
            self._ws_feed.start(warmup_candles=300)
            # Wait briefly for initial data
            for _ in range(10):
                if self._ws_feed.is_ready():
                    break
                import time as _t; _t.sleep(1.0)

        if self._ws_feed is not None and self._ws_feed.is_ready():
            candles = self._ws_feed.get_candles()
            if candles:
                self._live_candles = candles
                return

        # Fallback: REST polling
        if self.executor is not None:
            try:
                rest_feed = RESTCandleFeed(self.executor.client, symbol=self.cfg["symbol"])
                self._live_candles = rest_feed.get_candles()
            except Exception as e:
                logger.warning(f"_update_live_candles REST fallback failed: {e}")

    def close(self):
        """Clean up resources (called by SB3 on env teardown)."""
        if self._ws_feed is not None:
            self._ws_feed.stop()
            self._ws_feed = None
        super().close()
