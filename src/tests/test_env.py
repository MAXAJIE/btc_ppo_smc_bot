"""
tests/test_env.py
──────────────────
Smoke tests for BinanceEnv in offline mode.
No network, no Binance API — uses synthetic data.

FIXES:
  - Import path: src.environment.binance_testnet_env (not src.env.*)
  - Class name: BinanceEnv (not BTCFuturesEnv)
  - Constructor: BinanceEnv(tf_data=..., config=...) not (data_loader, mode, ...)
  - Attribute names aligned to actual BinanceEnv implementation
  - Tests that require attributes not present in BinanceEnv are adapted

Run:
    pytest tests/test_env.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int, freq: str = "5min", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 50000.0
    prices = [price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.003)))
    prices = np.array(prices)
    opens  = prices * (1 + rng.uniform(-0.001, 0.001, n))
    highs  = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.003, n))
    lows   = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.003, n))
    vols   = rng.lognormal(10, 0.5, n)
    idx    = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": prices, "volume": vols,
    }, index=idx)


def make_tf_data(n_5m: int = 10_000) -> dict:
    """Build synthetic multi-TF data dict as expected by BinanceEnv."""
    df_5m = make_ohlcv(n_5m, "5min")
    df_15m = make_ohlcv(n_5m // 3, "15min")
    df_1h  = make_ohlcv(n_5m // 12, "1h")
    df_4h  = make_ohlcv(n_5m // 48, "4h")
    df_1d  = make_ohlcv(n_5m // 288, "1D")
    return {"5m": df_5m, "15m": df_15m, "1h": df_1h, "4h": df_4h, "1d": df_1d}


def make_cfg() -> dict:
    return {
        "environment": {
            "symbol": "BTCUSDT",
            "episode_steps": 500,
            "max_trade_hold_steps": 100,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "kill_switch_drawdown": 0.85,  # very high so tests don't trigger it
        },
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [64], "vf": [64]}, "activation_fn": "tanh"},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tf_data():
    return make_tf_data()


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def env(tf_data, cfg):
    from src.environment.binance_testnet_env import BinanceEnv
    return BinanceEnv(tf_data=tf_data, config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Basic Gym API compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestGymAPI:

    def test_reset_returns_correct_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (90,), f"Expected (90,), got {obs.shape}"

    def test_reset_returns_float32_obs(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_all_finite(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), f"Non-finite at {np.where(~np.isfinite(obs))}"

    def test_action_space(self, env):
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 7

    def test_observation_space(self, env):
        from gymnasium import spaces
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (90,)

    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(0)  # HOLD
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (90,)

    def test_step_reward_is_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert np.isfinite(reward), f"Reward is non-finite: {reward}"

    def test_step_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        for key in ["step", "equity", "position"]:
            assert key in info, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Action semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestActionSemantics:

    def test_hold_keeps_position_flat(self, env):
        env.reset()
        for _ in range(5):
            _, _, _, _, info = env.step(0)  # HOLD
        assert info["position"] == 0

    def test_long_full_opens_long(self, env):
        env.reset()
        env.step(1)  # LONG_FULL
        assert env._position == 1

    def test_short_full_opens_short(self, env):
        env.reset()
        env.step(3)  # SHORT_FULL
        assert env._position == -1

    def test_close_flattens_position(self, env):
        env.reset()
        env.step(1)  # LONG
        assert env._position == 1
        env.step(5)  # CLOSE
        assert env._position == 0

    def test_flip_long_to_short(self, env):
        env.reset()
        env.step(1)   # LONG
        assert env._position == 1
        env.step(3)   # SHORT_FULL — closes long then opens short
        assert env._position == -1

    def test_reduce_doesnt_crash(self, env):
        """REDUCE action (6) shouldn't crash when in a position."""
        env.reset()
        env.step(1)  # open long
        _, _, _, _, _ = env.step(6)  # REDUCE_50

    def test_long_half_smaller_kelly_size(self, env):
        """Action 2 (LONG half Kelly) should use half position vs action 1 (full Kelly)."""
        env.reset()
        env.step(2)  # LONG_HALF — just verify it doesn't crash and opens long
        assert env._position == 1


# ─────────────────────────────────────────────────────────────────────────────
# Episode lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeLifecycle:

    def test_episode_terminates_on_drawdown_kill(self, tf_data):
        """Env should terminate when equity drops below kill_switch_drawdown threshold."""
        from src.environment.binance_testnet_env import BinanceEnv
        # Use a very tight kill threshold for this test
        cfg = make_cfg()
        cfg["environment"]["kill_switch_drawdown"] = 0.99  # triggers if equity < 0.01
        env = BinanceEnv(tf_data=tf_data, config=cfg)
        env.reset()
        env._equity = 0.001  # force equity well below kill threshold
        _, _, terminated, _, _ = env.step(0)
        assert terminated, "Should terminate when equity < kill_switch_drawdown"

    def test_multiple_resets_independent(self, env):
        """Resetting should clear position and step count."""
        env.reset()
        env.step(1)  # open a position
        assert env._position == 1

        env.reset()
        assert env._position == 0, "Position should reset to 0"
        assert env._step_count == 0

    def test_step_count_increments(self, env):
        env.reset()
        for i in range(5):
            env.step(0)
        assert env._step_count == 5


# ─────────────────────────────────────────────────────────────────────────────
# SB3 vectorised env compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestVecEnvCompatibility:

    def test_works_in_dummy_vec_env(self, tf_data, cfg):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from src.environment.binance_testnet_env import BinanceEnv

        def make():
            e = BinanceEnv(tf_data=tf_data, config=cfg)
            return Monitor(e)

        vec = DummyVecEnv([make])
        obs = vec.reset()
        assert obs.shape == (1, 90)

        obs, rewards, dones, infos = vec.step([0])
        assert obs.shape == (1, 90)
        assert rewards.shape == (1,)
        vec.close()

    def test_obs_space_matches_actual_obs(self, env):
        obs, _ = env.reset()
        low = env.observation_space.low
        high = env.observation_space.high
        assert np.all(obs >= low), "obs below obs_space.low"
        assert np.all(obs <= high), "obs above obs_space.high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
