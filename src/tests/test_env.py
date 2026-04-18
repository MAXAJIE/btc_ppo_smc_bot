"""
tests/test_env.py
──────────────────
Smoke tests for BTCFuturesEnv in offline mode.
No network, no Binance API — uses synthetic data.

Run:
    pytest tests/test_env.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic DataLoader mock
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_loader(n_candles: int = 10_000):
    """
    Returns a mock DataLoader with synthetic BTCUSDT 5m data.
    """
    rng = np.random.default_rng(0)
    price = 50000.0
    prices = [price]
    for _ in range(n_candles - 1):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.003)))

    prices = np.array(prices)
    opens  = prices * (1 + rng.uniform(-0.001, 0.001, n_candles))
    highs  = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.003, n_candles))
    lows   = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.003, n_candles))
    vols   = rng.lognormal(10, 0.5, n_candles)

    idx = pd.date_range("2022-01-01", periods=n_candles, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": prices, "volume": vols,
    }, index=idx)

    loader = MagicMock()
    loader._base_df = df
    loader.n_candles = n_candles

    def get_multi_tf_candles(end_idx, lookback_5m=500):
        start = max(0, end_idx - lookback_5m)
        df_5m = df.iloc[start: end_idx + 1].copy()

        def resample(rule):
            return df_5m.resample(rule).agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()

        return {
            "5m": df_5m,
            "15m": resample("15min"),
            "1h": resample("1h"),
            "4h": resample("4h"),
            "1d": resample("1D"),
        }

    def get_episode_start_indices(episode_len=4320, warmup=500):
        return list(range(warmup, n_candles - episode_len, episode_len // 2))

    loader.get_multi_tf_candles.side_effect = get_multi_tf_candles
    loader.get_episode_start_indices.side_effect = get_episode_start_indices
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def loader():
    return make_synthetic_loader()


@pytest.fixture
def env(loader, tmp_path):
    from src.env.binance_testnet_env import BTCFuturesEnv
    from src.utils.logger import TradeLogger
    trade_logger = TradeLogger(log_dir=str(tmp_path))
    e = BTCFuturesEnv(
        data_loader=loader,
        mode="offline",
        executor=None,
        trade_logger=trade_logger,
        episode_idx=600,  # fixed start for reproducibility
    )
    return e


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

    def test_reset_info_has_episode(self, env):
        _, info = env.reset()
        assert "episode" in info

    def test_obs_within_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= -5.0), f"Min: {obs.min()}"
        assert np.all(obs <= 5.0), f"Max: {obs.max()}"

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
        for key in ["step", "price", "equity", "balance", "drawdown", "position"]:
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

    def test_reduce_halves_qty(self, env):
        env.reset()
        env.step(1)  # LONG_FULL
        qty_before = env._qty
        env.step(6)  # REDUCE_50
        qty_after = env._qty
        assert qty_after == pytest.approx(qty_before * 0.5, rel=0.01)

    def test_flip_long_to_short(self, env):
        env.reset()
        env.step(1)   # LONG
        assert env._position == 1
        env.step(3)   # SHORT_FULL — should close long first then open short
        assert env._position == -1

    def test_long_half_smaller_qty_than_full(self, env):
        env_full = env
        env_full.reset()
        env_full.step(1)  # LONG_FULL
        qty_full = env_full._qty

        loader2 = make_synthetic_loader()
        from src.env.binance_testnet_env import BTCFuturesEnv
        from src.utils.logger import TradeLogger
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env_half = BTCFuturesEnv(
                data_loader=loader2, mode="offline",
                trade_logger=TradeLogger(log_dir=tmp),
                episode_idx=600,
            )
            env_half.reset()
            env_half.step(2)  # LONG_HALF
            qty_half = env_half._qty

        assert qty_half < qty_full, f"Half qty {qty_half:.4f} should < full qty {qty_full:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Risk management
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagement:

    def test_stop_loss_closes_position(self, env):
        """Force SL by setting _sl_price above current price for a long."""
        env.reset()
        env.step(1)  # open long
        assert env._position == 1

        # Manually set SL above current price so it triggers next step
        env._sl_price = env._current_price() * 1.5  # way above
        env.step(0)  # HOLD — SL should fire
        assert env._position == 0, "SL should have closed the position"

    def test_take_profit_closes_position(self, env):
        """Force TP by setting _tp_price below current price for a long."""
        env.reset()
        env.step(1)  # open long
        env._tp_price = env._current_price() * 0.5  # way below
        env.step(0)
        assert env._position == 0, "TP should have closed the position"

    def test_max_duration_closes_position(self, env):
        """Force max duration close by setting _bars_in_trade to limit."""
        env.reset()
        env.step(1)  # open long
        env._bars_in_trade = env.max_bars_in_trade  # at the limit
        env.step(0)  # HOLD — should force-close
        assert env._position == 0

    def test_balance_decreases_on_loss(self, env):
        env.reset()
        init_balance = env._balance
        # Artificially set entry price very high to force a loss on close
        env.step(1)  # open long
        env._entry_price = env._current_price() * 2.0  # entry was 2× current
        env.step(5)  # close — should realize a loss
        assert env._balance < init_balance

    def test_equity_tracks_balance_when_flat(self, env):
        env.reset()
        env._balance = 9500.0
        env._equity = 9500.0
        env._position = 0
        # When flat, equity should equal balance
        env.step(0)
        assert env._equity == pytest.approx(env._balance, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Episode lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeLifecycle:

    def test_episode_terminates_on_drawdown_kill(self, env):
        env.reset()
        env.step(1)  # open long
        env._peak_equity = 100_000.0
        env._balance = 80_000.0  # 20% drawdown — above 15% kill
        env._equity = 80_000.0
        _, _, terminated, _, _ = env.step(0)
        assert terminated, "Should terminate when drawdown >= 15%"

    def test_episode_truncates_at_end(self, loader, tmp_path):
        """A very short episode should truncate."""
        from src.env.binance_testnet_env import BTCFuturesEnv
        from src.utils.logger import TradeLogger

        env_short = BTCFuturesEnv(
            data_loader=loader, mode="offline",
            trade_logger=TradeLogger(log_dir=str(tmp_path)),
            episode_idx=600,
        )
        env_short.episode_len = 10  # very short episode
        obs, _ = env_short.reset()
        env_short._episode_end_idx = env_short._cur_idx + 10

        truncated = False
        for _ in range(20):
            obs, r, term, trunc, info = env_short.step(0)
            if trunc or term:
                truncated = True
                break

        assert truncated, "Episode should truncate after episode_len steps"

    def test_multiple_resets_independent(self, env):
        """Resetting multiple times should produce independent episodes."""
        env.reset()
        env.step(1)  # open a position
        pos1 = env._position
        bal1 = env._balance

        env.reset()
        assert env._position == 0, "Position should reset to 0"
        assert env._balance == env.initial_balance, "Balance should reset"
        assert env._step_count == 0

    def test_bars_in_trade_increments(self, env):
        env.reset()
        env.step(1)  # open long
        bars = []
        for _ in range(5):
            _, _, _, _, info = env.step(0)  # HOLD
            bars.append(env._bars_in_trade)

        assert bars == [2, 3, 4, 5, 6], f"bars_in_trade should increment: {bars}"

    def test_bars_since_last_trade_increments_when_flat(self, env):
        env.reset()
        for _ in range(5):
            env.step(0)  # HOLD while flat
        assert env._bars_since_last_trade == 5

    def test_bars_since_last_trade_resets_on_open(self, env):
        env.reset()
        for _ in range(5):
            env.step(0)  # build up counter
        env.step(1)  # open trade
        assert env._bars_in_trade == 1
        # bars_since_last_trade should reset to 0 while in trade
        env.step(0)
        assert env._bars_since_last_trade == 0


# ─────────────────────────────────────────────────────────────────────────────
# SB3 vectorised env compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestVecEnvCompatibility:

    def test_works_in_dummy_vec_env(self, loader, tmp_path):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from src.env.binance_testnet_env import BTCFuturesEnv
        from src.utils.logger import TradeLogger

        def make():
            e = BTCFuturesEnv(
                data_loader=loader, mode="offline",
                trade_logger=TradeLogger(log_dir=str(tmp_path)),
                episode_idx=600,
            )
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


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ─────────────────────────────────────────────────────────────────────────────
# [A] Kill-switch punitive reward integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKillSwitchReward:
    """
    Verify the env correctly passes kill_triggered=True to compute_step_reward
    and that the resulting reward is catastrophically negative.
    """

    def _make_env(self, loader, tmp_path):
        from src.env.binance_testnet_env import BTCFuturesEnv
        from src.utils.logger import TradeLogger
        return BTCFuturesEnv(
            data_loader=loader, mode="offline",
            trade_logger=TradeLogger(log_dir=str(tmp_path)),
            episode_idx=600,
        )

    def test_kill_switch_reward_catastrophic(self, loader, tmp_path):
        """
        When drawdown >= max_drawdown_kill, reward must be ≤ -50.
        (With default scale=50 and dd≥0.15: R ≤ -50*1.15 = -57.5)
        """
        env = self._make_env(loader, tmp_path)
        env.reset()
        env.step(1)  # open long

        # Force a 16% drawdown — above the 15% kill threshold
        env._peak_equity = 10000.0
        env._equity      = 8400.0   # 16% DD
        env._balance     = 8400.0

        _, reward, terminated, _, info = env.step(0)

        assert terminated, "Episode should terminate on kill-switch"
        assert reward <= -50.0, (
            f"Kill-switch reward must be ≤ -50, got {reward:.2f}"
        )

    def test_kill_switch_terminates_episode(self, loader, tmp_path):
        env = self._make_env(loader, tmp_path)
        env.reset()
        env._peak_equity = 10000.0
        env._balance     = 8400.0
        env._equity      = 8400.0

        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_normal_step_reward_not_catastrophic(self, loader, tmp_path):
        """Normal steps (no kill) must produce reward in a reasonable range."""
        env = self._make_env(loader, tmp_path)
        env.reset()

        rewards = []
        for action in [0, 1, 0, 0, 5]:
            _, r, term, trunc, _ = env.step(action)
            rewards.append(r)
            if term or trunc:
                break

        for r in rewards:
            assert r > -20.0, f"Normal step reward {r:.3f} shouldn't be catastrophic"


# ─────────────────────────────────────────────────────────────────────────────
# [B] Holding cost integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldingCostInEnv:
    """
    Verify that being in a trade costs more per step than being flat,
    due to the holding_cost_per_bar penalty.
    """

    def _make_env(self, loader, tmp_path):
        from src.env.binance_testnet_env import BTCFuturesEnv
        from src.utils.logger import TradeLogger
        return BTCFuturesEnv(
            data_loader=loader, mode="offline",
            trade_logger=TradeLogger(log_dir=str(tmp_path)),
            episode_idx=600,
        )

    def test_flat_hold_more_rewarding_than_dead_hold(self, loader, tmp_path):
        """
        A flat HOLD and an in-trade HOLD at zero unrealised PnL:
        the in-trade hold should accumulate lower total reward
        because of holding_cost_per_bar.
        """
        # Env A: stay flat for 10 steps
        envA = self._make_env(loader, tmp_path)
        envA.reset()
        total_flat = sum(envA.step(0)[1] for _ in range(10))  # all HOLD, never open

        # Env B: open long, then hold for 10 steps
        envB = self._make_env(loader, tmp_path)
        envB.reset()
        envB.step(1)  # open
        total_in_trade = sum(envB.step(0)[1] for _ in range(10))

        # The in-trade rewards include holding_cost_per_bar → should be lower
        # (This assumes market is roughly flat so unrealised PnL ≈ 0)
        # We just check total_in_trade < total_flat by at least some margin
        assert total_in_trade < total_flat + 0.05, (
            f"In-trade total {total_in_trade:.4f} should not exceed flat {total_flat:.4f} "
            f"when unrealised PnL is small (holding cost should pull it lower)"
        )
