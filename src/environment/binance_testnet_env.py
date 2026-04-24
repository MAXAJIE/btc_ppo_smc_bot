"""
binance_testnet_env.py
======================
Gymnasium environment for offline PPO training and live testnet fine-tuning.

Compatible with stable-baselines3 >= 2.0.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, Tuple

__all__ = ["BinanceEnv"]

import gymnasium as gym
import numpy as np
import pandas as pd

from src.features.amt_features      import compute_amt_features
from src.features.garch_kelly       import compute_garch_kelly
from src.features.multi_tf_features import MultiTFFeatureBuilder, OBS_DIM
from src.features.smc_features      import compute_smc_features
from src.features.snr_features      import compute_snr_features
from src.utils.reward               import RewardState, compute_reward

logger = logging.getLogger(__name__)

# Action indices
HOLD        = 0
LONG_FULL   = 1
LONG_HALF   = 2
SHORT_FULL  = 3
SHORT_HALF  = 4
CLOSE       = 5
REDUCE_HALF = 6

EPISODE_STEPS = 4320    # 15 days of 5m bars


class BinanceEnv(gym.Env):
    """
    Parameters
    ----------
    tf_data : dict[str, pd.DataFrame]
        All timeframes from DataLoader.load().
    config : dict
        Full config dict (loaded from config.yaml).
    episode_steps : int
        Override episode length (useful for fast tests).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tf_data:       Dict[str, pd.DataFrame],
        config:        Dict[str, Any],
        episode_steps: Optional[int] = None,
    ):
        super().__init__()
        self._cfg   = config
        self._ecfg  = config.get("environment", {})
        self._tf    = tf_data
        self._steps = episode_steps or self._ecfg.get("episode_steps", EPISODE_STEPS)

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(7)

        self._builder  = MultiTFFeatureBuilder(tf_data)
        self._base     = tf_data["5m"]
        self._n        = len(self._base)
        self._rstate   = RewardState()

        # Episode state (initialised in reset)
        self._pos          = 0
        self._entry_price  = 0.0
        self._equity       = 1.0
        self._prev_unreal  = 0.0
        self._real_pnl     = 0.0
        self._hold_steps   = 0
        self._ep_step      = 0
        self._start_idx    = 0
        self._cur_idx      = 0
        self._kelly        = 0.25
        self._episode      = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode += 1
        max_start = self._n - self._steps - 1

        if max_start < 0:
            # Live mode: dataset shorter than episode_steps.
            # Pin to bar 0 and let the step loop clamp the index.
            max_start = 0

        self._start_idx    = random.randint(0, max_start) if max_start > 0 else 0
        self._cur_idx      = self._start_idx
        self._pos          = 0
        self._entry_price  = 0.0
        self._equity       = 1.0
        self._prev_unreal  = 0.0
        self._real_pnl     = 0.0
        self._hold_steps   = 0
        self._ep_step      = 0
        self._kelly        = 0.25
        self._rstate       = RewardState()
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        prev_pos  = self._pos
        close     = float(self._base["close"].iloc[self._cur_idx])
        ts        = self._base.index[self._cur_idx]

        self._apply_action(action, close)

        unreal    = self._unrealised(close)
        real      = self._real_pnl
        self._real_pnl = 0.0

        # Update equity from unrealised delta
        if self._pos != 0:
            self._equity *= 1.0 + (unreal - self._prev_unreal)
        self._prev_unreal = unreal if self._pos != 0 else 0.0

        # SMC / SNR context for reward entry bonus
        smc5   = compute_smc_features(self._base, self._cur_idx)
        snr1h  = compute_snr_features(
            self._tf["1h"], self._tf_idx("1h", ts)
        )

        reward = compute_reward(
            action=action,
            prev_position=prev_pos,
            new_position=self._pos,
            realised_pnl_pct=real,
            unrealised_pct=unreal,
            equity=self._equity,
            state=self._rstate,
            bull_ob_dist=float(smc5[0]),
            bear_ob_dist=float(smc5[1]),
            snr_support_1=float(snr1h[0]),
            snr_resist_1=float(snr1h[3]),
        )

        self._cur_idx += 1
        self._ep_step += 1

        kill_dd = self._ecfg.get("kill_switch_drawdown", 0.15)
        done    = (self._ep_step >= self._steps) or (self._equity < 1.0 - kill_dd)

        info = {
            "equity":    self._equity,
            "position":  self._pos,
            "step":      self._ep_step,
            "episode":   self._episode,
        }
        return self._obs(), reward, done, False, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Aliases expected by main_live.py
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Public alias for _obs() used by the live loop."""
        return self._obs()

    @property
    def _base_df(self) -> pd.DataFrame:
        """Alias: main_live.py accesses env._base_df; internally it's self._base."""
        return self._base

    @_base_df.setter
    def _base_df(self, df: pd.DataFrame) -> None:
        """Allow main_live.py to hot-swap the base DataFrame with fresh live data."""
        self._base = df
        self._n    = len(df)
        # Rebuild momentum columns for the new data
        self._builder = MultiTFFeatureBuilder(self._tf)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        # Clamp to last valid bar — prevents IndexError when live loop
        # manually sets _cur_idx to len-1 and step() increments past end.
        safe_idx = min(self._cur_idx, self._n - 1)
        if safe_idx < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        ts    = self._base.index[safe_idx]
        close = self._base["close"].iloc[safe_idx]

        smc5  = compute_smc_features(self._base, safe_idx)
        smc1h = compute_smc_features(self._tf["1h"], self._tf_idx("1h", ts))
        snr1h = compute_snr_features(self._tf["1h"], self._tf_idx("1h", ts))
        amt4h = compute_amt_features(self._tf["4h"], self._tf_idx("4h", ts))

        gk = compute_garch_kelly(
            self._base["close"].iloc[max(0, safe_idx - 500): safe_idx + 1]
        )
        self._kelly = float(gk[2])

        pos_vec = self._pos_vec(close)

        return self._builder.build(
            ts=ts, smc_5m=smc5, smc_1h=smc1h,
            snr_1h=snr1h, amt_4h=amt4h,
            garch_kelly=gk, position=pos_vec,
        )

    def _tf_idx(self, tf: str, ts: pd.Timestamp) -> int:
        arr = self._tf[tf].index
        return max(0, int(arr.searchsorted(ts, side="right")) - 1)

    def _apply_action(self, action: int, close: float) -> None:
        sl = self._ecfg.get("stop_loss_pct",  0.03)
        tp = self._ecfg.get("take_profit_pct", 0.06)
        mh = self._ecfg.get("max_trade_hold_steps", 576)

        # Forced close
        if self._pos != 0:
            self._hold_steps += 1
            p = self._unrealised(close)
            if p <= -sl or p >= tp or self._hold_steps >= mh:
                self._close(close)
                return

        if action == HOLD:
            pass
        elif action in (LONG_FULL, LONG_HALF):
            if self._pos != 1:
                self._close(close)
                size = self._kelly if action == LONG_FULL else self._kelly * 0.5
                self._open(1, close, size)
        elif action in (SHORT_FULL, SHORT_HALF):
            if self._pos != -1:
                self._close(close)
                size = self._kelly if action == SHORT_FULL else self._kelly * 0.5
                self._open(-1, close, size)
        elif action == CLOSE:
            self._close(close)
        elif action == REDUCE_HALF:
            if self._pos != 0:
                self._real_pnl += self._unrealised(close) * 0.5
                self._entry_price = (self._entry_price + close) / 2

    def _open(self, direction: int, price: float, size: float) -> None:
        self._pos         = direction
        self._entry_price = price
        self._hold_steps  = 0

    def _close(self, price: float) -> None:
        if self._pos == 0:
            return
        pnl = self._unrealised(price)
        self._real_pnl   += pnl
        self._equity     *= (1.0 + pnl)
        self._pos         = 0
        self._entry_price = 0.0
        self._hold_steps  = 0
        self._prev_unreal = 0.0

    def _unrealised(self, close: float) -> float:
        if self._pos == 0 or self._entry_price == 0:
            return 0.0
        return float((close - self._entry_price) / self._entry_price * self._pos)

    def _pos_vec(self, close) -> np.ndarray:
        pnl = self._unrealised(float(close))
        return np.array([
            float(self._pos),
            float(self._pos ==  1),
            float(self._pos == -1),
            float(self._pos ==  0),
            float(np.clip(pnl * 10, -5, 5)),
            float(np.clip(self._hold_steps / 100, 0, 5)),
            float(np.clip(self._equity - 1.0, -1, 1)),
        ], dtype=np.float32)