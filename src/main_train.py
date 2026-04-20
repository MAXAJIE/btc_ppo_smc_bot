"""
main_train.py
=============
Offline warm-up training.

Resume bug fix
--------------
Old code:
    is_resuming = pretrained_path and Path(pretrained_path).exists()
    if is_resuming:
        model = load_ppo(...)
    else:
        model = build_ppo(...)   # ← silently built fresh if path was wrong

If pretrained_path was provided but the file didn't exist (wrong path,
relative path resolved against wrong CWD, missing .zip extension, etc.)
the code silently ignored it and trained from scratch.

Fix:
    If pretrained_path is supplied → the file MUST exist → raise FileNotFoundError
    with the resolved absolute path so the user knows exactly what's wrong.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.binance_testnet_env import BinanceEnv
from src.utils.data_loader       import DataLoader
from src.utils.logger            import TradeLogger
from src.models.ppo_model        import (
    build_ppo,
    load_ppo,
    save_ppo,
    update_lr,
    make_callbacks,
    evaluate_model,
)


def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _resolve_checkpoint(path: str) -> Path:
    """
    Resolve a checkpoint path to an absolute Path, trying several variants:
      1. As given
      2. With .zip appended
      3. Under MODEL_SAVE_PATH env var
    Raises FileNotFoundError with a clear message if none exist.
    """
    candidates = [
        Path(path),
        Path(path + ".zip") if not path.endswith(".zip") else None,
        Path(os.getenv("MODEL_SAVE_PATH", "./models")) / Path(path).name,
        Path(os.getenv("MODEL_SAVE_PATH", "./models")) / (Path(path).name + ".zip"),
    ]
    for p in candidates:
        if p and p.exists():
            logger.info("Checkpoint resolved: %s", p.resolve())
            return p

    raise FileNotFoundError(
        f"\n\nCheckpoint not found: '{path}'\n"
        f"Tried:\n"
        + "\n".join(f"  {p.resolve()}" for p in candidates if p)
        + "\n\nCheck the path and try again. "
          "List available checkpoints with:\n"
          f"  ls {os.getenv('MODEL_SAVE_PATH', './models')}/ppo_btc_*.zip"
    )


def make_env(tf_data: dict, cfg: dict, seed: int = 0):
    def _init():
        return Monitor(BinanceEnv(tf_data=tf_data, config=cfg))
    return _init


def main(
    model_save_dir:  Optional[str]   = None,
    data_dir:        Optional[str]   = None,
    total_timesteps: Optional[int]   = None,
    n_envs:          int             = 1,
    pretrained_path: Optional[str]   = None,
    learning_rate:   Optional[float] = None,
) -> str:
    cfg         = _load_cfg()
    offline_cfg = cfg["offline"]

    model_save_dir  = model_save_dir  or os.getenv("MODEL_SAVE_PATH", "./models")
    data_dir        = data_dir        or os.getenv("DATA_PATH",       "./data")
    total_timesteps = total_timesteps or offline_cfg["total_timesteps"]

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("Loading historical data from %s …", data_dir)
    loader = DataLoader(
        data_dir=data_dir,
        years=offline_cfg.get("historical_years", 2),
    )
    try:
        loader.load_base_df()
        logger.info("Using cached data: %d 5m bars", loader.n_candles)
    except FileNotFoundError:
        logger.info("No cache — downloading …")
        loader.download_history(years=offline_cfg.get("historical_years", 2))
        logger.info("Downloaded %d 5m bars", loader.n_candles)

    if loader.n_candles < 5000:
        raise RuntimeError(f"Too few candles ({loader.n_candles}).")

    tf_data = loader.tf_data

    # ── 2. Environments ───────────────────────────────────────────────────────
    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(tf_data, cfg)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(tf_data, cfg, seed=i) for i in range(n_envs)]
        )
    eval_vec = DummyVecEnv([make_env(tf_data, cfg)])

    # ── 3. Model ──────────────────────────────────────────────────────────────
    if pretrained_path:
        # ── FIXED: resolve path with clear error if not found ─────────────────
        ckpt = _resolve_checkpoint(pretrained_path)
        logger.info("Resuming from checkpoint: %s", ckpt)
        model = load_ppo(str(ckpt), env=vec_env)

        # Apply fine-tune LR
        effective_lr = learning_rate or float(
            cfg["ppo"].get("fine_tune_lr",
            offline_cfg.get("fine_tune_lr", 1e-4))
        )
        update_lr(model, effective_lr)

        already_done = model.num_timesteps
        logger.info(
            "✓ Checkpoint loaded — steps done: %d | target: %d | remaining: %d",
            already_done, total_timesteps,
            max(0, total_timesteps - already_done),
        )
    else:
        logger.info("Building fresh PPO model …")
        model        = build_ppo(vec_env, cfg=cfg, learning_rate=learning_rate)
        already_done = 0

    # ── 4. Remaining steps ────────────────────────────────────────────────────
    remaining = max(0, total_timesteps - already_done)

    if remaining == 0:
        logger.warning(
            "Checkpoint already at %d steps (target=%d). "
            "Pass a higher --timesteps value to keep training.",
            already_done, total_timesteps,
        )
        return save_ppo(model, os.path.join(model_save_dir, "ppo_btc_final"))

    logger.info(
        "Training %d steps (checkpoint=%d, target=%d) …",
        remaining, already_done, total_timesteps,
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = make_callbacks(
        model_dir=model_save_dir,
        eval_env=eval_vec,
        save_freq=offline_cfg.get("save_every", 50_000),
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = remaining,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = True,   # count from 0 for THIS run's `remaining` budget
    )

    # ── 7. Save ───────────────────────────────────────────────────────────────
    final_path = save_ppo(model, os.path.join(model_save_dir, "ppo_btc_final"))
    logger.info("Training complete → %s", final_path)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Post-training evaluation …")
    results = evaluate_model(
        model, make_env(tf_data, cfg)(),
        n_episodes=offline_cfg.get("eval_episodes", 5),
    )
    logger.info("Eval: %s", results)

    return final_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  default="./models")
    p.add_argument("--data-dir",   default="./data")
    p.add_argument("--timesteps",  type=int,   default=None)
    p.add_argument("--n-envs",     type=int,   default=1)
    p.add_argument("--resume",     default=None)
    p.add_argument("--lr",         type=float, default=None)
    args = p.parse_args()
    main(
        model_save_dir  = args.model_dir,
        data_dir        = args.data_dir,
        total_timesteps = args.timesteps,
        n_envs          = args.n_envs,
        pretrained_path = args.resume,
        learning_rate   = args.lr,
    )