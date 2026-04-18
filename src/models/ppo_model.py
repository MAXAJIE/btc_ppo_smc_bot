"""
ppo_model.py
─────────────
Thin wrapper around Stable-Baselines3 PPO.
Handles model creation, loading, saving, and evaluation.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

logger = logging.getLogger(__name__)


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────

def build_ppo(env, cfg=None) -> PPO:
    """
    Create a new PPO model with config-driven hyperparameters.

    Parameters
    ----------
    env : gym.Env or VecEnv
        The training environment.
    cfg : dict | None
        If None, loads from config.yaml.

    Returns
    -------
    PPO instance, not yet trained.
    """
    if cfg is None:
        cfg = _load_cfg()

    ppo_cfg = cfg["ppo"]
    offline_cfg = cfg["offline"]

    # Map activation function string → torch class
    act_fns = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}
    act_fn = act_fns.get(ppo_cfg["policy_kwargs"]["activation_fn"], torch.nn.Tanh)

    policy_kwargs = {
        "net_arch": ppo_cfg["policy_kwargs"]["net_arch"],
        "activation_fn": act_fn,
    }

    model = PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=offline_cfg["tb_log_dir"],
        verbose=1,
    )

    logger.info(
        f"PPO created | policy={ppo_cfg['policy']} "
        f"net_arch={ppo_cfg['policy_kwargs']['net_arch']} "
        f"lr={ppo_cfg['learning_rate']}"
    )
    return model


def load_ppo(path: str, env) -> PPO:
    """Load a saved PPO model and attach a new env."""
    model = PPO.load(path, env=env, device="auto")
    logger.info(f"PPO loaded from {path}")
    return model


def save_ppo(model: PPO, path: str):
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info(f"PPO saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def make_callbacks(model_dir: str, eval_env=None, save_freq: int = 100_000) -> list:
    """Build standard callback list for training."""
    callbacks = []

    # Checkpoint every `save_freq` steps
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="ppo_btc",
        save_vecnormalize=False,
    )
    callbacks.append(ckpt_cb)

    # Eval callback (if eval env provided)
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best"),
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=save_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

    callbacks.append(EpisodeStatsCallback())
    return callbacks


class EpisodeStatsCallback(BaseCallback):
    """Logs episode returns to tensorboard every episode end."""

    def __init__(self):
        super().__init__()
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self._episode_rewards.append(ep_rew)
                self._episode_lengths.append(ep_len)
                self.logger.record("episode/reward", ep_rew)
                self.logger.record("episode/length", ep_len)
                if len(self._episode_rewards) % 10 == 0:
                    last10 = self._episode_rewards[-10:]
                    logger.info(
                        f"  [Ep {len(self._episode_rewards)}] "
                        f"avg_rew={np.mean(last10):.3f} "
                        f"avg_len={np.mean(self._episode_lengths[-10:]):.0f}"
                    )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: PPO, env, n_episodes: int = 10) -> dict:
    """
    Run deterministic evaluation and return metrics.
    """
    rewards = []
    final_equities = []
    max_drawdowns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_drawdown = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
            ep_drawdown = max(ep_drawdown, info.get("drawdown", 0.0))

        rewards.append(ep_reward)
        final_equities.append(info.get("equity", 0.0))
        max_drawdowns.append(ep_drawdown)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_final_equity": float(np.mean(final_equities)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "n_episodes": n_episodes,
    }
