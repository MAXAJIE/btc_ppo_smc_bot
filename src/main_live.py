"""
main_live.py  (COMPATIBILITY PATCHED — FEATURES PRESERVED)
"""

import os
import sys
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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.binance_testnet_env import BinanceEnv
from src.execution.binance_executor import BinanceExecutor
from src.utils.data_loader import DataLoader
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

    # ── FIX 1: Correct executor init ────────────────────────────────
    logger.info("Connecting to Binance Futures Testnet...")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET")

    executor = BinanceExecutor(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True,
    )

    # FIX 2: method name
    balance = executor.get_equity()
    logger.info(f"Testnet balance: {balance:.2f} USDT")

    if balance < 100:
        raise RuntimeError("Testnet balance too low.")

    # ── Data ───────────────────────────────────────────────────────
    logger.info("Loading historical data for walk-forward validation...")
    loader = DataLoader()

    try:
        loader.load_base_df()
    except FileNotFoundError:
        logger.info("Downloading 6 months of data for validation...")
        loader.download_history(years=0.5)

    tf_data = loader.tf_data

    # ── Env ────────────────────────────────────────────────────────
    live_env = BinanceEnv(tf_data=tf_data, config=cfg)
    live_env_monitored = Monitor(live_env)
    vec_env = DummyVecEnv([lambda: live_env_monitored])

    # ── Model ──────────────────────────────────────────────────────
    logger.info(f"Loading model from {pretrained_model_path}")
    if not Path(pretrained_model_path).exists():
        raise FileNotFoundError("Model not found")

    model = load_ppo(pretrained_model_path, env=vec_env)
    logger.info("Model loaded. Starting live fine-tuning loop.")

    # ── NEW: local position tracking (executor has no getter) ───────
    position = 0
    entry_price = 0.0

    episode = 0
    peak_equity = balance

    while not _SHUTDOWN:
        now = datetime.now(timezone.utc)

        if now >= deadline:
            logger.info(f"Max runtime reached.")
            break

        episode += 1
        logger.info(f"\n{'═'*60}\n  EPISODE {episode}\n{'═'*60}")

        episode_reward, episode_steps, position, entry_price = _run_live_episode(
            model, executor, cfg, position, entry_price
        )

        logger.info(f"Episode {episode} done | reward={episode_reward:.3f}")

        # PPO update
        model.learn(
            total_timesteps=episode_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        save_ppo(model, str(model_save_dir / f"ppo_live_ep{episode}"))

        # ── FIX 3: equity
        current_balance = executor.get_equity()

        if current_balance > peak_equity:
            peak_equity = current_balance

        drawdown = (peak_equity - current_balance) / max(peak_equity, 1e-10)

        if drawdown >= risk_cfg["max_drawdown_kill"]:
            logger.critical(f"KILL-SWITCH triggered.")
            executor.close_all()
            break

        if now >= next_walk_forward:
            logger.info("Running walk-forward validation...")
            _walk_forward_validation(model, loader, cfg)
            next_walk_forward = now + timedelta(days=wf_days)

    # ── Cleanup ─────────────────────────────────────────────────────
    logger.info("Shutting down live trading loop.")
    executor.close_all()

    save_ppo(model, str(model_save_dir / "ppo_live_final"))


# ─────────────────────────────────────────────
# EPISODE (patched only where needed)
# ─────────────────────────────────────────────
def _run_live_episode(model, env, executor, cfg, position, entry_price):
    episode_len = cfg["episode"]["candles_per_episode"]

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(episode_len):
        if _SHUTDOWN:
            break

        action, _ = model.predict(obs, deterministic=False)

        # ── FIX 4: connect executor ─────────────────────────
        current_price = env.current_price

        position, entry_price, pnl = executor.execute(
            action=int(action),
            current_price=current_price,
            position=position,
            entry_price=entry_price,
        )

        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward + pnl
        steps += 1

        if steps % 100 == 0:
            logger.info(
                f"  step={steps} | equity={executor.get_equity():.2f} | "
                f"pos={'L' if position == 1 else 'S' if position == -1 else '-'}"
            )

        if terminated or truncated:
            break

    return total_reward, steps, position, entry_price


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
        pos = executor.positions()
        if pos["side"] != "NONE":
            executor._close_position(pos)
        executor.close_all()
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
