"""
main_live.py
────────────
Live testnet fine-tuning loop.

FIXES:
  - BTCFuturesEnv → BinanceEnv (correct class name)
  - BinanceEnv called with correct signature (tf_data, config)
  - DataLoader usage fixed (load_base_df / tf_data)
  - Walk-forward validation uses correct BinanceEnv API
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta, timezone
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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# FIXED: use correct class name BinanceEnv (not BTCFuturesEnv)
from src.environment.binance_testnet_env import BinanceEnv
from src.execution.binance_executor import BinanceExecutor
from src.utils.data_loader import DataLoader
from src.utils.logger import TradeLogger
from src.models.ppo_model import load_ppo, save_ppo, evaluate_model


def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_SHUTDOWN = False


def _handle_signal(sig, frame):
    global _SHUTDOWN
    logger.warning(f"Signal {sig} received — will shutdown after current episode.")
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main(
    pretrained_model_path: str,
    model_save_dir: str = "./models",
    max_runtime_hours: float = None,
    walk_forward_days: int = None,
):
    global _SHUTDOWN

    cfg = load_cfg()
    live_cfg = cfg["live"]
    risk_cfg = cfg["risk"]

    max_runtime = max_runtime_hours or live_cfg["max_runtime_hours"]
    wf_days = walk_forward_days or live_cfg["walk_forward_days"]
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    deadline = start_time + timedelta(hours=max_runtime)
    next_walk_forward = start_time + timedelta(days=wf_days)

    # ── 1. Connect to Binance Testnet ────────────────────────────────
    logger.info("Connecting to Binance Futures Testnet...")
    executor = BinanceExecutor(
        symbol=cfg["SYMBOL"],
        max_leverage=cfg["Leverage"],
    )

    balance = executor.get_account_balance()
    logger.info(f"Testnet balance: {balance:.2f} USDT")

    if balance < 100:
        raise RuntimeError(
            "Testnet balance too low. Fund your testnet account at "
            "https://testnet.binancefuture.com"
        )

    # ── 2. Historical data loader ──────────────────────────────────────
    logger.info("Loading historical data for walk-forward validation...")
    loader = DataLoader()
    try:
        loader.load_base_df()
    except FileNotFoundError:
        logger.info("Downloading 6 months of data for validation...")
        loader.download_history(years=0.5)

    tf_data = loader.tf_data

    # ── 3. Build live environment ──────────────────────────────────────
    # FIXED: use BinanceEnv with correct signature
    live_env = BinanceEnv(tf_data=tf_data, config=cfg)
    live_env_monitored = Monitor(live_env)
    vec_env = DummyVecEnv([lambda: live_env_monitored])

    # ── 4. Load pre-trained model ─────────────────────────────────────
    logger.info(f"Loading model from {pretrained_model_path}")
    if not Path(pretrained_model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {pretrained_model_path}\n"
            "Run main_train.py first to generate a model."
        )

    model = load_ppo(pretrained_model_path, env=vec_env)
    logger.info("Model loaded. Starting live fine-tuning loop.")

    # ── 5. Live fine-tuning loop ──────────────────────────────────────
    episode = 0
    peak_equity = balance

    while not _SHUTDOWN:
        now = datetime.now(timezone.utc)

        if now >= deadline:
            logger.info(f"Max runtime of {max_runtime}h reached. Shutting down.")
            break

        episode += 1
        logger.info(f"\n{'═'*60}\n  EPISODE {episode}\n{'═'*60}")

        episode_reward, episode_steps = _run_live_episode(model, live_env, cfg)
        logger.info(f"Episode {episode} done | reward={episode_reward:.3f} | steps={episode_steps}")

        logger.info("Running PPO update on collected rollout...")
        model.learn(
            total_timesteps=episode_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        ckpt_path = model_save_dir / f"ppo_live_ep{episode}"
        save_ppo(model, str(ckpt_path))

        current_balance = executor.get_account_balance()
        if current_balance > peak_equity:
            peak_equity = current_balance

        drawdown = (peak_equity - current_balance) / max(peak_equity, 1e-10)
        if drawdown >= risk_cfg["max_drawdown_kill"]:
            logger.critical(
                f"KILL-SWITCH: Account drawdown {drawdown:.1%} >= "
                f"{risk_cfg['max_drawdown_kill']:.0%}. Halting."
            )
            _emergency_close(executor)
            break

        if now >= next_walk_forward:
            logger.info("Running walk-forward validation...")
            _walk_forward_validation(model, loader, cfg)
            next_walk_forward = now + timedelta(days=wf_days)

        if _SHUTDOWN:
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    logger.info("Shutting down live trading loop.")
    position = executor.get_position()
    if position["side"] != "NONE":
        executor.close_position(position)
        executor.cancel_all_orders()

    final_path = model_save_dir / "ppo_live_final"
    save_ppo(model, str(final_path))
    logger.info(f"Final model saved to {final_path}")


def _run_live_episode(model: PPO, env: BinanceEnv, cfg: dict):
    episode_len = cfg["episode"]["candles_per_episode"]
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(episode_len):
        if _SHUTDOWN:
            break

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1

        if steps % 100 == 0:
            logger.info(
                f"  step={steps} | equity={info.get('equity', 0):.4f} | "
                f"pos={'L' if info.get('position') == 1 else 'S' if info.get('position') == -1 else '-'}"
            )

        if terminated or truncated:
            break

    return total_reward, steps


def _walk_forward_validation(model: PPO, loader: DataLoader, cfg: dict):
    try:
        tf_data = loader.tf_data
        eval_env = BinanceEnv(tf_data=tf_data, config=cfg)
        results = evaluate_model(model, eval_env, n_episodes=3)
        logger.info(f"Walk-forward validation results: {results}")
    except Exception as e:
        logger.warning(f"Walk-forward validation failed: {e}")


def _emergency_close(executor: BinanceExecutor):
    try:
        pos = executor.get_position()
        if pos["side"] != "NONE":
            executor.close_position(pos)
        executor.cancel_all_orders()
        logger.info("Emergency close executed.")
    except Exception as e:
        logger.error(f"Emergency close failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live testnet fine-tuning")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--runtime-hours", type=float, default=None)
    parser.add_argument("--walk-forward-days", type=int, default=None)
    args = parser.parse_args()

    main(
        pretrained_model_path=args.model,
        model_save_dir=args.model_dir,
        max_runtime_hours=args.runtime_hours,
        walk_forward_days=args.walk_forward_days,
    )
