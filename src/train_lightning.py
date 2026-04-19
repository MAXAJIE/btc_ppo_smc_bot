"""
train_lightning.py
──────────────────
Lightning.ai training entrypoint.

FIXES:
  - loader.load_base_df() / loader.n_candles / loader.download_history()
    now work via the fixed DataLoader.
  - Cache path check uses correct filename (btcusdt_5m.parquet).
"""

import os
import sys
import argparse
import logging
import glob
from pathlib import Path

IN_LIGHTNING_STUDIO = Path("/teamspace").exists()

if IN_LIGHTNING_STUDIO:
    PERSISTENT_ROOT = Path("/teamspace/studios/this_studio")
    MODEL_DIR = str(PERSISTENT_ROOT / "models")
    DATA_DIR = str(PERSISTENT_ROOT / "data")
    LOG_DIR = str(PERSISTENT_ROOT / "logs")
    print(f"[Lightning.ai] Detected Studio environment. Saving to {PERSISTENT_ROOT}")
else:
    MODEL_DIR = "./models"
    DATA_DIR = "./data"
    LOG_DIR = "./logs"
    print("[Local] Running locally. Saving to ./models, ./data, ./logs")

os.environ["MODEL_SAVE_PATH"] = MODEL_DIR
os.environ["DATA_PATH"] = DATA_DIR
os.environ["LOG_PATH"] = LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def auto_detect_gpu() -> int:
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            logger.info(f"Detected {n} GPU(s): {names}")
        return n
    except ImportError:
        return 0


def auto_n_envs(n_gpus: int, requested: int) -> int:
    if requested > 0:
        return requested
    if n_gpus > 0:
        return 8
    return 2


def main():
    parser = argparse.ArgumentParser(description="Lightning.ai PPO training")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    n_gpus = auto_detect_gpu()
    n_envs = auto_n_envs(n_gpus, args.n_envs)
    logger.info(f"Hardware: {n_gpus} GPU(s) | Using {n_envs} parallel envs")

    if not args.no_download:
        _ensure_data(DATA_DIR, years=2)

    if args.download_only:
        logger.info("--download-only set. Exiting.")
        return

    resume_path = args.resume
    if resume_path is None:
        resume_path = _find_latest_checkpoint(MODEL_DIR)
        if resume_path:
            logger.info(f"Auto-resuming from: {resume_path}")

    logger.info("=" * 60)
    logger.info("  BTC PPO SMC TRAINER — Lightning.ai")
    logger.info("=" * 60)

    from src.main_train import main as train_main

    final_path = train_main(
        model_save_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        total_timesteps=args.timesteps,
        n_envs=n_envs,
        pretrained_path=resume_path,
        learning_rate=args.lr,
    )

    logger.info(f"Training complete! Final model: {final_path}")


def _ensure_data(data_dir: str, years: int):
    from src.utils.data_loader import DataLoader
    loader = DataLoader(data_dir=data_dir, years=years)

    # Check for cached parquet files (both naming conventions)
    cache_5m = Path(data_dir) / "btcusdt_5m.parquet"
    cache_5m_alt = Path(data_dir) / "BTCUSDT_5m.parquet"

    if cache_5m.exists() or cache_5m_alt.exists():
        try:
            loader.load_base_df()
            logger.info(f"Using cached data: {loader.n_candles:,} candles")
            return
        except Exception:
            pass

    logger.info(f"Downloading {years}y of BTCUSDT 5m data...")
    df = loader.download_history(years=years)
    logger.info(f"Downloaded {len(df):,} candles")


def _find_latest_checkpoint(model_dir: str):
    checkpoints = sorted(glob.glob(os.path.join(model_dir, "ppo_btc_*.zip")))
    return checkpoints[-1] if checkpoints else None


if __name__ == "__main__":
    main()
