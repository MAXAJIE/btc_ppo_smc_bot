"""
train_lightning.py
──────────────────
Lightning.ai training entrypoint.

Lightning.ai Studios are cloud-hosted VSCode/JupyterLab environments.
You run this script directly inside the Studio terminal — no wrapper
needed like Modal. The GPU is whatever machine you've selected in the
Studio (e.g. RTX 4090 or A100).

Usage (inside Lightning.ai Studio terminal):
    # Install deps first (run once):
    pip install -r requirements.txt

    # Run offline training:
    python train_lightning.py

    # Resume from checkpoint:
    python train_lightning.py --resume ./models/ppo_btc_200000_steps.zip

    # Quick test run:
    python train_lightning.py --timesteps 50000 --n-envs 1

To programmatically start/stop Lightning.ai Studios from outside
(e.g. in a CI pipeline), see: lightning_sdk_launcher.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# ── Lightning.ai Studio sets up /teamspace/studios/this_studio ───────────────
# We auto-detect if we're running inside a Studio and adjust paths.
IN_LIGHTNING_STUDIO = Path("/teamspace").exists()

if IN_LIGHTNING_STUDIO:
    # Use /teamspace/studios/this_studio for persistent storage on Lightning.ai
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

# Override environment vars used by data_loader, logger, etc.
os.environ["MODEL_SAVE_PATH"] = MODEL_DIR
os.environ["DATA_PATH"] = DATA_DIR
os.environ["LOG_PATH"] = LOG_DIR

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def auto_detect_gpu() -> int:
    """Return number of available GPUs (0 = CPU only)."""
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
        # 针对 Lightning AI 的高性能 GPU 调优
        return 8 if n_gpus == 1 else 16
    return 2


def main():
    parser = argparse.ArgumentParser(description="Lightning.ai PPO training")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--n-envs", type=int, default=0, help="Parallel envs (0=auto)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .zip to resume from")
    parser.add_argument("--download-only", action="store_true", help="Only download data, then exit")
    parser.add_argument("--no-download", action="store_true", help="Skip data download (use cache)")
    args = parser.parse_args()

    # ── Detect hardware ───────────────────────────────────────────────
    n_gpus = auto_detect_gpu()
    n_envs = auto_n_envs(n_gpus, args.n_envs)
    logger.info(f"Hardware: {n_gpus} GPU(s) | Using {n_envs} parallel envs")

    # ── Step 1: Data ──────────────────────────────────────────────────
    if not args.no_download:
        _ensure_data(DATA_DIR, years=2)

    if args.download_only:
        logger.info("--download-only set. Exiting.")
        return

    # ── Step 2: Auto-resume logic ─────────────────────────────────────
    resume_path = args.resume
    if resume_path is None:
        resume_path = _find_latest_checkpoint(MODEL_DIR)
        if resume_path:
            logger.info(f"Auto-resuming from: {resume_path}")

    # ── Step 3: Train ─────────────────────────────────────────────────
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

    )

    logger.info(f"Training complete! Final model: {final_path}")
    logger.info(
        "\nTo use the model for live trading:\n"
        f"  python -m src.main_live --model {final_path}.zip"
    )


def _ensure_data(data_dir: str, years: int):
    """Download historical data if not already cached."""
    from src.utils.data_loader import DataLoader
    loader = DataLoader(data_dir=data_dir)

    cache_path = Path(data_dir) / "BTCUSDT_5m.parquet"
    if cache_path.exists():
        loader.load_base_df()
        logger.info(f"Using cached data: {loader.n_candles:,} candles")
        return

    logger.info(f"Downloading {years}y of BTCUSDT 5m data...")
    df = loader.download_history(years=years)
    logger.info(f"Downloaded {len(df):,} candles")


def _find_latest_checkpoint(model_dir: str) -> str | None:
    """Find the most recent .zip checkpoint in model_dir."""
    import glob
    checkpoints = sorted(glob.glob(os.path.join(model_dir, "ppo_btc_*.zip")))
    return checkpoints[-1] if checkpoints else None


if __name__ == "__main__":
    main()
