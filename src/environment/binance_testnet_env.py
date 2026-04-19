"""
binance_testnet_env.py  –  Gymnasium environment  (FIXED)
==========================================================
Key changes
-----------
1.  Uses multi-TF dict from data_loader (not a single 5m DataFrame).
2.  Wires smc_features, snr_features, amt_features correctly at each step,
    passing the right per-TF slice to each feature extractor.
3.  Passes SMC/SNR context into the reward function (entry quality bonus).
4.  Observation is normalised and NaN-guarded before being returned.
5.  Episode reset picks a random start point so every episode sees a
    different slice of history — critical for generalisation.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from src.features.amt_features      import compute_amt_features
from src.features.garch_kelly       import compute_garch_kelly
from src.features.multi_tf_features import MultiTFFeatureBuilder
from src.features.smc_features      import compute_smc_features
from src.features.snr_features      import compute_snr_features
from src.utils.reward               import RewardState, compute_reward

logger = logging.getLogger(__name__)

OBS_DIM      = 90
ACTION_DIM   = 7
EPISODE_STEPS = 4320      # 15 days of 5m bars


class BinanceEnv(gym.Env):
    """
    Offline backtesting environment for PPO training.

    Parameters
    ----------
    tf_data : dict[str, pd.DataFrame]
        Output of data_loader.load_all_timeframes().
    config : dict
        Loaded from config.yaml.
    """

    metadata = {"render_modes": []}

    def __init__(self, tf_data: Dict[str, pd.DataFrame], config: dict):
        super().__init__()
        self.cfg = config
        self.tf  = tf_data

        # Continuous obs space — values roughly in [-5, 5]
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

        # Feature builder (holds pre-computed momentum columns)
        self.feat_builder = MultiTFFeatureBuilder(tf_data)

        # Base 5m index (used to step through time)
        self._base_df  = tf_data["5m"]
        self._n_bars   = len(self._base_df)

        # State variables (reset per episode)
        self._reset_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Random start: pick anywhere that leaves a full episode ahead
        max_start = self._n_bars - EPISODE_STEPS - 1
        if max_start <= 0:
            raise RuntimeError("Dataset too short for one episode.")
        self._start_idx = random.randint(0, max_start)
        self._cur_idx   = self._start_idx

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        ts    = self._base_df.index[self._cur_idx]
        close = float(self._base_df["close"].iloc[self._cur_idx])

        prev_position = self._position

        # Apply action
        self._apply_action(action, close)

        # Compute unrealised PnL
        unrealised_pct = self._unrealised_pnl_pct(close)

        # Compute realised PnL (non-zero only on close action)
        realised_pct = self._realised_pnl_pct
        self._realised_pnl_pct = 0.0   # consume

        # Update equity
        self._equity *= (1 + unrealised_pct - self._prev_unrealised)
        self._prev_unrealised = unrealised_pct if self._position != 0 else 0.0

        # Get SMC/SNR context for reward entry-quality bonus
        smc_5m_feats = compute_smc_features(self._base_df, self._cur_idx)
        snr_1h_feats = compute_snr_features(self.tf["1h"], self._get_tf_idx("1h", ts))

        # Reward
        reward = compute_reward(
            action=action,
            prev_position=prev_position,
            new_position=self._position,
            realised_pnl_pct=realised_pct,
            unrealised_pct=unrealised_pct,
            equity=self._equity,
            state=self._reward_state,
            bull_ob_dist=float(smc_5m_feats[0]),
            bear_ob_dist=float(smc_5m_feats[1]),
            snr_support_1=float(snr_1h_feats[0]),
            snr_resist_1=float(snr_1h_feats[3]),
        )

        self._cur_idx += 1
        self._step_count += 1

        done   = self._step_count >= EPISODE_STEPS
        done  |= self._equity < self.cfg["environment"]["kill_switch_drawdown"]

        obs = self._get_obs()
        info = {
            "equity":   self._equity,
            "position": self._position,
            "step":     self._step_count,
        }
        return obs, reward, done, False, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._position          = 0          # 0=flat, 1=long, -1=short
        self._entry_price       = 0.0
        self._equity            = 1.0        # normalised to 1.0 = 100%
        self._prev_unrealised   = 0.0
        self._realised_pnl_pct  = 0.0
        self._step_count        = 0
        self._trade_hold_steps  = 0
        self._kelly_size        = 0.25       # default quarter-Kelly
        self._start_idx         = 0
        self._cur_idx           = 0
        self._reward_state      = RewardState(peak_equity=1.0)

    def _get_obs(self) -> np.ndarray:
        if self._cur_idx >= self._n_bars:
            return np.zeros(OBS_DIM, dtype=np.float32)

        ts = self._base_df.index[self._cur_idx]

        # SMC features
        smc_5m = compute_smc_features(self._base_df, self._cur_idx)
        smc_1h = compute_smc_features(
            self.tf["1h"], self._get_tf_idx("1h", ts)
        )

        # S&R
        snr_1h = compute_snr_features(
            self.tf["1h"], self._get_tf_idx("1h", ts)
        )

        # AMT / Volume profile
        amt_4h = compute_amt_features(
            self.tf["4h"], self._get_tf_idx("4h", ts)
        )

        # GARCH + Kelly
        garch_kelly = compute_garch_kelly(
            self._base_df["close"].iloc[max(0, self._cur_idx-500): self._cur_idx+1]
        )
        self._kelly_size = float(garch_kelly[3])  # store for position sizing

        # Position state vector (7 dims)
        position_state = self._position_state_vector(
            self._base_df["close"].iloc[self._cur_idx]
        )

        obs = self.feat_builder.build(
            ts=ts,
            smc_5m=smc_5m,
            smc_1h=smc_1h,
            snr_1h=snr_1h,
            amt_4h=amt_4h,
            garch_kelly=garch_kelly,
            position=position_state,
        )
        return obs

    def _get_tf_idx(self, tf: str, ts: pd.Timestamp) -> int:
        """Return the index in the HTF DataFrame of the last closed bar <= ts."""
        arr = self.tf[tf].index
        idx = arr.searchsorted(ts, side="right") - 1
        return max(0, idx)

    def _apply_action(self, action: int, close: float):
        cfg_env = self.cfg["environment"]
        sl  = cfg_env["stop_loss_pct"]
        tp  = cfg_env["take_profit_pct"]

        # Check forced close conditions first
        if self._position != 0:
            self._trade_hold_steps += 1
            pnl = self._unrealised_pnl_pct(close)

            force_close = (
                pnl <= -sl
                or pnl >= tp
                or self._trade_hold_steps >= cfg_env["max_trade_hold_steps"]
            )
            if force_close:
                self._close_position(close)
                return

        # Action mapping
        # 0 = HOLD
        # 1 = LONG  full Kelly
        # 2 = LONG  half Kelly
        # 3 = SHORT full Kelly
        # 4 = SHORT half Kelly
        # 5 = CLOSE position
        # 6 = REDUCE 50%

        if action == 0:
            pass

        elif action in (1, 2):
            if self._position != 1:
                self._close_position(close)
                size = self._kelly_size if action == 1 else self._kelly_size * 0.5
                self._open_position(1, close, size)

        elif action in (3, 4):
            if self._position != -1:
                self._close_position(close)
                size = self._kelly_size if action == 3 else self._kelly_size * 0.5
                self._open_position(-1, close, size)

        elif action == 5:
            self._close_position(close)

        elif action == 6:
            # Reduce by 50%: close half by booking 50% realised PnL
            if self._position != 0:
                pnl = self._unrealised_pnl_pct(close)
                self._realised_pnl_pct += pnl * 0.5
                # Adjust entry price to midpoint (approximate)
                self._entry_price = (self._entry_price + close) / 2

    def _open_position(self, direction: int, price: float, size: float):
        self._position         = direction
        self._entry_price      = price
        self._trade_hold_steps = 0

    def _close_position(self, price: float):
        if self._position == 0:
            return
        pnl = self._unrealised_pnl_pct(price)
        self._realised_pnl_pct += pnl
        self._equity           *= (1 + pnl)
        self._position          = 0
        self._entry_price       = 0.0
        self._trade_hold_steps  = 0
        self._prev_unrealised   = 0.0

    def _unrealised_pnl_pct(self, close: float) -> float:
        if self._position == 0 or self._entry_price == 0:
            return 0.0
        raw = (close - self._entry_price) / self._entry_price
        return float(raw * self._position)

    def _position_state_vector(self, close) -> np.ndarray:
        """7-dim position state (same layout as README)."""
        close = float(close)
        pnl   = self._unrealised_pnl_pct(close)
        return np.array([
            float(self._position),                     # direction: -1/0/1
            float(self._position == 1),                # is_long
            float(self._position == -1),               # is_short
            float(self._position == 0),                # is_flat
            np.clip(pnl * 10, -5, 5),                 # unrealised PnL scaled
            np.clip(self._trade_hold_steps / 100, 0, 5),  # hold duration
            np.clip(self._equity - 1.0, -1, 1),       # equity delta from start
        ], dtype=np.float32)
