"""
main_train.py
─────────────
Offline warm-up: trains PPO on 2+ years of historical BTCUSDT 5m data.

Run locally:
    python -m src.main_train

Run on Modal:
    modal run train_modal.py

Run on Lightning.ai:
    Open a Studio terminal → python src/main_train.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import yaml
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environment.binance_testnet_env import BinanceEnv
from src.utils.data_loader import DataLoader
from src.utils.logger import TradeLogger
from src.models.ppo_model import build_ppo, save_ppo, evaluate_model, make_callbacks


def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def make_env(tf_data: dict, cfg: dict, seed: int = 0):
    """Factory function for vectorised env creation."""
    def _init():
        env = BinanceEnv(tf_data=tf_data, config=cfg)
        env = Monitor(env)
        return env
    return _init


def main(
    model_save_dir: str = None,
    data_dir: str = None,
    total_timesteps: int = None,
    n_envs: int = 1,
    pretrained_path: str = None,
    learning_rate: float = None,
):
    cfg = load_cfg()
    offline_cfg = cfg["offline"]

    model_save_dir = model_save_dir or os.getenv("MODEL_SAVE_PATH", "./models")
    data_dir = data_dir or os.getenv("DATA_PATH", "./data")
    total_timesteps = total_timesteps or offline_cfg["total_timesteps"]

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(parents=True, exist_ok=True)

    # ── 1. Load / download historical data ───────────────────────────
    logger.info("Loading historical data...")
    loader = DataLoader(data_dir=data_dir, years=offline_cfg.get("historical_years", 2))

    try:
        loader.load_base_df()
        logger.info(f"Loaded cached data: {loader.n_candles:,} 5m bars")
    except FileNotFoundError:
        logger.info("Downloading historical data (first run)...")
        loader.download_history(years=offline_cfg.get("historical_years", 2))

    if loader.n_candles < 5000:
        raise RuntimeError(
            f"Too few candles ({loader.n_candles}). "
            "Download more data with download_history()."
        )

    tf_data = loader.tf_data

    # ── 2. Build environments ─────────────────────────────────────────
    if n_envs == 1:
        env = DummyVecEnv([make_env(tf_data, cfg)])
    else:
        env = SubprocVecEnv([make_env(tf_data, cfg, seed=i) for i in range(n_envs)])

    eval_env = DummyVecEnv([make_env(tf_data, cfg)])

    # ── 3. Build or load model ────────────────────────────────────────
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained model from {pretrained_path}")
        from src.models.ppo_model import load_ppo, update_lr
        model = load_ppo(pretrained_path, env=env)
        effective_lr = learning_rate or cfg["ppo"].get("fine_tune_lr", 1e-4)
        update_lr(model, effective_lr)
    else:
        logger.info("Building new PPO model...")
        if learning_rate is None and pretrained_path:
            learning_rate = cfg["ppo"].get("fine_tune_lr", 1e-4)
        model = build_ppo(env, cfg=cfg, learning_rate=learning_rate)

    # ── 4. Callbacks ─────────────────────────────────────────────────
    callbacks = make_callbacks(
        model_dir=model_save_dir,
        eval_env=eval_env,
        save_freq=offline_cfg["save_every"],
    )

    # ── 5. Train ─────────────────────────────────────────────────────
    logger.info(f"Starting offline training: {total_timesteps:,} timesteps")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=pretrained_path is None,
    )

    # ── 6. Save final model ───────────────────────────────────────────
    final_path = os.path.join(model_save_dir, "ppo_btc_final")
    save_ppo(model, final_path)
    logger.info(f"Training complete. Model saved to {final_path}")

    # ── 7. Evaluate ───────────────────────────────────────────────────
    logger.info("Running post-training evaluation...")
    eval_env_single = make_env(tf_data, cfg)()
    results = evaluate_model(model, eval_env_single, n_episodes=offline_cfg.get("eval_episodes", 5))
    logger.info(f"Eval results: {results}")

    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline PPO training")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    main(
        model_save_dir=args.model_dir,
        data_dir=args.data_dir,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        pretrained_path=args.pretrained,
        learning_rate=args.lr,
    )
