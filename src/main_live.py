"""
main_live.py
────────────
Live testnet fine-tuning loop.

Loads a pre-trained PPO model, connects to Binance Futures Testnet,
and continues training on real live data — episode by episode.

Episode = 4,320 steps (15 days of 5m candles, real-time pace).
PPO update fires after every episode.
Walk-forward validation every 7 days.

Run:
    python -m src.main_live --model ./models/ppo_btc_final.zip
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

from src.env.binance_testnet_env import BTCFuturesEnv
from src.execution.binance_executor import BinanceFuturesExecutor
from src.utils.data_loader import DataLoader
from src.utils.logger import TradeLogger
from src.models.ppo_model import load_ppo, save_ppo, evaluate_model


def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown handler
# ─────────────────────────────────────────────────────────────────────────────

_SHUTDOWN = False


def _handle_signal(sig, frame):
    global _SHUTDOWN
    logger.warning(f"Signal {sig} received — will shutdown after current episode.")
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─────────────────────────────────────────────────────────────────────────────

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
    executor = BinanceFuturesExecutor(
        symbol=cfg["symbol"],
        max_leverage=cfg["max_leverage"],
    )

    balance = executor.get_account_balance()
    logger.info(f"Testnet balance: {balance:.2f} USDT")

    if balance < 100:
        raise RuntimeError(
            "Testnet balance too low. Fund your testnet account at "
            "https://testnet.binancefuture.com"
        )

    # ── 2. Historical data loader (for walk-forward validation) ───────
    logger.info("Loading historical data for walk-forward validation...")
    loader = DataLoader()
    try:
        loader.load_base_df()
    except FileNotFoundError:
        logger.info("Downloading 6 months of data for validation...")
        loader.download_history(years=0.5)

    # ── 3. Build live environment ─────────────────────────────────────
    trade_logger = TradeLogger()

    live_env = BTCFuturesEnv(
        data_loader=None,          # no historical data in live mode
        mode="live",
        executor=executor,
        trade_logger=trade_logger,
    )
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
    total_steps_trained = 0
    peak_equity = balance

    while not _SHUTDOWN:
        now = datetime.now(timezone.utc)

        # Check runtime deadline
        if now >= deadline:
            logger.info(f"Max runtime of {max_runtime}h reached. Shutting down.")
            break

        episode += 1
        logger.info(
            f"\n{'═'*60}\n"
            f"  EPISODE {episode} | "
            f"Runtime: {(now - start_time).total_seconds() / 3600:.1f}h\n"
            f"  Balance: {executor.get_account_balance():.2f} USDT\n"
            f"{'═'*60}"
        )

        # ── 5a. Run one live episode (blocks for ~15 days real-time) ──
        # In practice for continuous fine-tuning you will want to use
        # shorter virtual episodes or trigger updates more frequently.
        # We set update_every_episodes=1 so PPO learns after each episode.

        episode_reward, episode_steps = _run_live_episode(model, live_env, cfg)
        total_steps_trained += episode_steps

        logger.info(
            f"Episode {episode} done | "
            f"reward={episode_reward:.3f} | "
            f"steps={episode_steps}"
        )

        # ── 5b. PPO update (online fine-tuning) ───────────────────────
        logger.info("Running PPO update on collected rollout...")
        model.learn(
            total_timesteps=episode_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # ── 5c. Save checkpoint ───────────────────────────────────────
        ckpt_path = model_save_dir / f"ppo_live_ep{episode}"
        save_ppo(model, str(ckpt_path))

        # ── 5d. Kill-switch check ─────────────────────────────────────
        current_balance = executor.get_account_balance()
        if current_balance > peak_equity:
            peak_equity = current_balance

        drawdown = (peak_equity - current_balance) / max(peak_equity, 1e-10)
        if drawdown >= risk_cfg["max_drawdown_kill"]:
            logger.critical(
                f"KILL-SWITCH: Account drawdown {drawdown:.1%} >= "
                f"{risk_cfg['max_drawdown_kill']:.0%}. "
                "Closing all positions and halting."
            )
            _emergency_close(executor)
            break

        # ── 5e. Walk-forward validation ───────────────────────────────
        if now >= next_walk_forward:
            logger.info("Running walk-forward validation on fresh historical data...")
            _walk_forward_validation(model, loader, cfg)
            next_walk_forward = now + timedelta(days=wf_days)

        # ── 5f. Shutdown check ────────────────────────────────────────
        if _SHUTDOWN:
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    logger.info("Shutting down live trading loop.")
    position = executor.get_position()
    if position["side"] != "NONE":
        logger.info(f"Closing open position: {position}")
        executor.close_position(position)
        executor.cancel_all_orders()

    final_path = model_save_dir / "ppo_live_final"
    save_ppo(model, str(final_path))
    logger.info(f"Final model saved to {final_path}")
    trade_logger.print_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_live_episode(model: PPO, env: BTCFuturesEnv, cfg: dict):
    """
    Run a single live episode using the current model (inference only).
    Collects (obs, action, reward) for the subsequent PPO update.

    Note: PPO.learn() internally collects rollouts. For true online
    fine-tuning we run model.predict() to generate actions, then call
    env.step() manually, collecting data for the next learn() call.

    For simplicity we use SB3's built-in collect_rollouts via
    model.learn(total_timesteps=episode_steps).
    """
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

        # Log key metrics every 100 steps
        if steps % 100 == 0:
            logger.info(
                f"  step={steps} | price={info.get('price', 0):.0f} | "
                f"equity={info.get('equity', 0):.2f} | "
                f"drawdown={info.get('drawdown', 0):.1%} | "
                f"pos={'L' if info.get('position') == 1 else 'S' if info.get('position') == -1 else '-'}"
            )

        if terminated or truncated:
            break

    return total_reward, steps


# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_validation(model: PPO, loader: DataLoader, cfg: dict):
    """
    Evaluate the model on the most recent historical data not used in training.
    Logs performance metrics.
    """
    from src.env.binance_testnet_env import BTCFuturesEnv
    from src.utils.logger import TradeLogger

    try:
        eval_logger = TradeLogger()
        eval_env = BTCFuturesEnv(
            data_loader=loader,
            mode="offline",
            trade_logger=eval_logger,
            episode_idx=loader.n_candles - cfg["episode"]["candles_per_episode"] - 100,
        )

        results = evaluate_model(model, eval_env, n_episodes=3)
        logger.info(f"Walk-forward validation results: {results}")

        if results["mean_max_drawdown"] > 0.20:
            logger.warning(
                f"Validation drawdown {results['mean_max_drawdown']:.1%} is high. "
                "Consider reducing position sizes."
            )
    except Exception as e:
        logger.warning(f"Walk-forward validation failed: {e}")


def _emergency_close(executor: BinanceFuturesExecutor):
    """Close all positions and cancel all orders immediately."""
    try:
        pos = executor.get_position()
        if pos["side"] != "NONE":
            executor.close_position(pos)
        executor.cancel_all_orders()
        logger.info("Emergency close executed.")
    except Exception as e:
        logger.error(f"Emergency close failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live testnet fine-tuning")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to pretrained .zip model (from main_train.py)"
    )
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument(
        "--runtime-hours",
        type=float,
        default=None,
        help="Max runtime in hours (default from config)"
    )
    parser.add_argument(
        "--walk-forward-days",
        type=int,
        default=None,
        help="Validate every N days (default from config)"
    )
    args = parser.parse_args()

    main(
        pretrained_model_path=args.model,
        model_save_dir="./models",
        max_runtime_hours=args.runtime_hours,
        walk_forward_days=args.walk_forward_days,
    )
