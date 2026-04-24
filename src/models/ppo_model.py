"""
ppo_model.py
─────────────
Thin wrapper around Stable-Baselines3 PPO.

Bugs fixed vs original
-----------------------
1. build_ppo() read cfg["ppo"]["policy_kwargs"]["activation_fn"] and
   cfg["offline"]["tb_log_dir"] — both were missing from config.yaml.
   Config now has these keys; build_ppo() also has safe .get() fallbacks
   so it never crashes on a missing key.

2. update_lr() used `model.lr_schedule = lambda _: new_lr`.
   SB3 inlines the schedule into the optimiser on every update call, but
   the plain lambda is not picklable — model.save() would crash.
   Fixed: use stable_baselines3.common.utils.constant_fn (SB3's own
   picklable constant schedule).

3. load_ppo() required `env` as a positional argument.
   When called during evaluation without a live env it would crash.
   Fixed: env is now Optional[...] = None.

4. Added evaluate_policy() alias (some callers use that name).
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
import yaml

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_ppo(env, cfg: Optional[dict] = None, learning_rate: Optional[float] = None) -> PPO:
    """
    Create a new PPO model with config-driven hyperparameters.

    Parameters
    ----------
    env           : gym.Env or VecEnv
    cfg           : full config dict; if None loads from config.yaml
    learning_rate : override config value when provided

    Returns
    -------
    PPO instance (not yet trained)
    """
    if cfg is None:
        cfg = _load_cfg()

    ppo_cfg     = cfg["ppo"]
    offline_cfg = cfg["offline"]

    # ── policy_kwargs ────────────────────────────────────────────────────────
    # Support both nested (cfg["ppo"]["policy_kwargs"]["activation_fn"])
    # and flat (cfg["ppo"]["activation_fn"]) layout for backward compat.
    pk_raw   = ppo_cfg.get("policy_kwargs", {})
    act_str  = pk_raw.get("activation_fn",
               ppo_cfg.get("activation_fn", "tanh"))
    net_arch = pk_raw.get("net_arch",
               ppo_cfg.get("net_arch",
               {"pi": [256, 256, 128], "vf": [256, 256, 128]}))

    act_fns = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}
    act_fn  = act_fns.get(act_str, torch.nn.Tanh)

    policy_kwargs = {"net_arch": net_arch, "activation_fn": act_fn}

    # ── Learning rate ─────────────────────────────────────────────────────────
    lr = (
        learning_rate
        if learning_rate is not None
        else float(ppo_cfg.get("initial_lr", ppo_cfg.get("learning_rate", 1e-4)))
    )

    # ── TensorBoard log dir ───────────────────────────────────────────────────
    tb_log = offline_cfg.get("tb_log_dir", "./logs/tb")
    Path(tb_log).mkdir(parents=True, exist_ok=True)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # 在 build_ppo() 之前包装环境
    vec_env = DummyVecEnv([lambda: env])
    # 归一化观测值和奖励，这对于 PPO 寻找 3:1 RR 的稀疏奖励至关重要
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy        = ppo_cfg.get("policy", "MlpPolicy"),
        env           = vec_env,
        learning_rate = lr,
        n_steps       = int(ppo_cfg.get("n_steps",      4096)),
        batch_size    = int(ppo_cfg.get("batch_size",    256)),
        n_epochs      = int(ppo_cfg.get("n_epochs",      5)),
        gamma         = float(ppo_cfg.get("gamma",       0.999)),
        gae_lambda    = float(ppo_cfg.get("gae_lambda",  0.97)),
        clip_range    = float(ppo_cfg.get("clip_range",  0.15)),
        ent_coef      = float(ppo_cfg.get("ent_coef",    0.01)),
        vf_coef       = float(ppo_cfg.get("vf_coef",     0.7)),
        max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs = policy_kwargs,
        tensorboard_log = tb_log,
        verbose       = 1,
    )

    logger.info(
        "PPO created | policy=%s  net_arch=%s  lr=%.2e",
        ppo_cfg.get("policy", "MlpPolicy"), net_arch, lr,
    )
    return model


# ---------------------------------------------------------------------------
# Load / Save / LR update
# ---------------------------------------------------------------------------

def load_ppo(path: str, env=None) -> PPO:
    """
    Load a saved PPO model.

    Parameters
    ----------
    path : str – path to .zip checkpoint (with or without extension)
    env  : optional; attach a new env after loading
    """
    path = str(path)
    if not path.endswith(".zip") and not Path(path).exists():
        path = path + ".zip"

    model = PPO.load(path, env=env, device="auto")
    logger.info("PPO loaded from %s  (num_timesteps=%d)", path, model.num_timesteps)
    return model


def save_ppo(model: PPO, path: str) -> str:
    """
    Save model to disk. Returns the final .zip path.
    SB3 appends .zip automatically — we normalise the path here.
    """
    # Strip .zip so SB3 doesn't double-append
    save_path = str(path)
    if save_path.endswith(".zip"):
        save_path = save_path[:-4]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    final = save_path + ".zip"
    logger.info("PPO saved to %s  (num_timesteps=%d)", final, model.num_timesteps)
    return final


def update_lr(model: PPO, new_lr: float) -> None:
    """
    Hot-swap learning rate on a loaded model (used when --resume + fine-tuning).

    Fix: use constant_fn() instead of a plain lambda.
    SB3 pickles the lr_schedule when saving — a plain `lambda _: x`
    is NOT picklable and causes model.save() to crash.
    constant_fn() returns SB3's own picklable callable.
    """
    from stable_baselines3.common.utils import constant_fn

    model.learning_rate = new_lr
    model.lr_schedule   = constant_fn(float(new_lr))

    # Also patch the optimiser directly so the change takes effect immediately
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = new_lr

    logger.info("Learning rate updated → %.2e", new_lr)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(
    model_dir:  str,
    eval_env=None,
    save_freq:  int = 50_000,
) -> list:
    """Build standard SB3 callback list for offline training."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []

    callbacks.append(
        CheckpointCallback(
            save_freq       = save_freq,
            save_path       = model_dir,
            name_prefix     = "ppo_btc",
            save_vecnormalize = False,
            verbose         = 1,
        )
    )

    if eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path = os.path.join(model_dir, "best"),
                log_path             = os.path.join(model_dir, "eval_logs"),
                eval_freq            = save_freq,
                n_eval_episodes      = 5,
                deterministic        = True,
                render               = False,
                verbose              = 1,
            )
        )

    callbacks.append(EpisodeStatsCallback())
    return callbacks


class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode reward to TensorBoard via SB3's logger."""

    def __init__(self):
        super().__init__()
        self._ep_rewards: list = []
        self._ep_lengths: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                self._ep_rewards.append(r)
                self._ep_lengths.append(l)
                self.logger.record("episode/reward", r)
                self.logger.record("episode/length", l)
                if len(self._ep_rewards) % 10 == 0:
                    logger.info(
                        "[Ep %d] avg_rew=%.3f  avg_len=%.0f",
                        len(self._ep_rewards),
                        np.mean(self._ep_rewards[-10:]),
                        np.mean(self._ep_lengths[-10:]),
                    )
        return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: PPO, env, n_episodes: int = 10) -> dict:
    """
    Run deterministic evaluation and return metrics dict.
    Called by main_train.py as evaluate_model(...).
    """
    rewards      = []
    equities     = []
    max_dds      = []

    for ep in range(n_episodes):
        obs, _   = env.reset()
        done     = False
        ep_rew   = 0.0
        ep_dd    = 0.0
        info     = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done    = terminated or truncated
            ep_rew += float(reward)
            ep_dd   = max(ep_dd, float(info.get("drawdown", 0.0)))

        rewards.append(ep_rew)
        equities.append(float(info.get("equity", 1.0)))
        max_dds.append(ep_dd)
        logger.info(
            "Eval ep %d/%d — reward=%.3f  equity=%.4f  max_dd=%.3f",
            ep + 1, n_episodes, ep_rew, equities[-1], ep_dd,
        )

    return {
        "mean_reward":       float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        "mean_final_equity": float(np.mean(equities)),
        "mean_max_drawdown": float(np.mean(max_dds)),
        "n_episodes":        n_episodes,
    }


# Alias — some callers use evaluate_policy
evaluate_policy = evaluate_model