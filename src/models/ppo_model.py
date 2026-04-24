"""
ppo_model.py
─────────────
Thin wrapper around Stable-Baselines3 PPO.

Bugs fixed & preserved:
1. build_ppo() config safe access (.get() fallbacks).
2. update_lr() picklable constant_fn (fixes save crash).
3. load_ppo() Optional env support.
4. evaluate_policy() alias.
5. NEW: Fixed SubprocVecEnv nesting ValueError.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, unwrap_vec_normalize

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
    智能处理环境包装，防止 SubprocVecEnv 重复包装导致的 ValueError。
    """
    if cfg is None:
        cfg = _load_cfg()

    ppo_cfg     = cfg["ppo"]
    offline_cfg = cfg["offline"]

    # ── 1. policy_kwargs ────────────────────────────────────────────────────────
    pk_raw   = ppo_cfg.get("policy_kwargs", {})
    act_str  = pk_raw.get("activation_fn", ppo_cfg.get("activation_fn", "tanh"))
    net_arch = pk_raw.get("net_arch", ppo_cfg.get("net_arch", {"pi": [256, 256, 128], "vf": [256, 256, 128]}))

    act_fns = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}
    act_fn  = act_fns.get(act_str, torch.nn.Tanh)

    policy_kwargs = {"net_arch": net_arch, "activation_fn": act_fn}

    # ── 2. Learning rate ─────────────────────────────────────────────────────────
    lr = (
        learning_rate
        if learning_rate is not None
        else float(ppo_cfg.get("initial_lr", ppo_cfg.get("learning_rate", 1e-4)))
    )

    # ── 3. TensorBoard log dir ───────────────────────────────────────────────────
    tb_log = offline_cfg.get("tb_log_dir", "./logs/tb")
    Path(tb_log).mkdir(parents=True, exist_ok=True)

    # ── 4. 环境包装修复 (解决 SubprocVecEnv 报错) ──────────────────────────────────
    # 检查是否已经是向量化环境 (VecEnv)
    if isinstance(env, VecEnv):
        vec_env = env
        logger.info("Using existing VecEnv (Subproc/Dummy).")
    else:
        # 如果是原始的 Gymnasium 环境，才进行 DummyVecEnv 包装
        vec_env = DummyVecEnv([lambda: env])
        logger.info("Wrapped single environment in DummyVecEnv.")

    # ── 5. 归一化处理 (对 SMC 复杂奖励至关重要) ──────────────────────────────────
    # 检查是否已经存在 VecNormalize 包装，避免双重归一化
    if unwrap_vec_normalize(vec_env) is None:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=float(ppo_cfg.get("reward_clip", 15.0)) # 与 reward.py 保持同步
        )
        logger.info("Applied VecNormalize (norm_obs=True, norm_reward=True)")

    # ── 6. 实例化 PPO ──────────────────────────────────────────────────────────
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
# Load / Save / LR update (保留所有修复逻辑)
# ---------------------------------------------------------------------------

def load_ppo(path: str, env=None) -> PPO:
    path = str(path)
    if not path.endswith(".zip") and not Path(path).exists():
        path = path + ".zip"

    model = PPO.load(path, env=env, device="auto")
    logger.info("PPO loaded from %s  (num_timesteps=%d)", path, model.num_timesteps)
    return model


def save_ppo(model: PPO, path: str) -> str:
    save_path = str(path)
    if save_path.endswith(".zip"):
        save_path = save_path[:-4]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    final = save_path + ".zip"
    logger.info("PPO saved to %s  (num_timesteps=%d)", final, model.num_timesteps)
    return final


def update_lr(model: PPO, new_lr: float) -> None:
    from stable_baselines3.common.utils import constant_fn
    model.learning_rate = new_lr
    model.lr_schedule   = constant_fn(float(new_lr))
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = new_lr
    logger.info("Learning rate updated → %.2e", new_lr)


# ---------------------------------------------------------------------------
# Callbacks (保留原逻辑)
# ---------------------------------------------------------------------------

def make_callbacks(
        model_dir: str,
        training_env: VecNormalize,  # 传入训练环境
        eval_env=None,
        save_freq: int = 50_000,
) -> list:
    """Build standard SB3 callback list with synchronized VecNormalize."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []

    # 确保评估环境也使用相同的归一化统计
    if eval_env is not None:
        # 如果 eval_env 还不是 VecNormalize，包装它
        from stable_baselines3.common.vec_env import VecNormalize


        # 关键：allow_ns=True 允许在评估时不更新统计数据（只使用训练集的均值/方差）
        eval_vec_env = VecNormalize(
            eval_env,
            training_env=training_env,  # 共享统计数据！
            norm_obs=True,
            norm_reward=False,  # 评估不需要归一化奖励
            clip_obs=10.0
        )
        # 禁用评估时的统计数据更新，保持一致性
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False

        callbacks.append(
            EvalCallback(
                eval_vec_env,  # 使用包装后的环境
                best_model_save_path=os.path.join(model_dir, "best"),
                log_path=os.path.join(model_dir, "eval_logs"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
                render=False,
                verbose=1,
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
        # 1. 安全地从 locals 获取 infos 列表
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            # 2. 必须检查 info 是否为字典，且包含 "episode" 键
            # SB3 的底层实现中，只有当 episode 结束时，Monitor 包装器才会插入这个键
            if isinstance(info, dict) and "episode" in info:
                ep_info = info["episode"]

                # 3. 再次检查 ep_info 是否为字典（防止某些自定义包装器返回非标准数据）
                if isinstance(ep_info, dict):
                    r = ep_info.get("r", 0.0)
                    l = ep_info.get("l", 0)

                    self._ep_rewards.append(r)
                    self._ep_lengths.append(l)

                    # 写入 TensorBoard
                    self.logger.record("episode/reward", r)
                    self.logger.record("episode/length", l)

                    # 每 10 个 Episode 打印一次均值
                    if len(self._ep_rewards) % 10 == 0:
                        logger.info(
                            "[Ep %d] avg_rew=%.3f  avg_len=%.0f",
                            len(self._ep_rewards),
                            np.mean(self._ep_rewards[-10:]),
                            np.mean(self._ep_lengths[-10:]),
                        )
        return True


# ---------------------------------------------------------------------------
# Evaluation (保留原逻辑)
# ---------------------------------------------------------------------------

def evaluate_model(model: PPO, env, n_episodes: int = 10) -> dict:
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

evaluate_policy = evaluate_model