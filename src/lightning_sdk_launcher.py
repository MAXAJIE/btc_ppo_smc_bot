"""
lightning_sdk_launcher.py
──────────────────────────
Programmatically start a Lightning.ai Studio, upload code, run training,
and download the resulting model checkpoint.

Requires:
    pip install lightning-sdk
    LIGHTNING_USER_ID and LIGHTNING_API_KEY in .environment

Usage:
    # Start training on Lightning.ai from your local machine:
    python lightning_sdk_launcher.py

    # With options:
    python lightning_sdk_launcher.py --timesteps 1000000
    python lightning_sdk_launcher.py --machine gpu-rtx-4090-x-1
    python lightning_sdk_launcher.py --download-model ./models/from_lightning.zip
    python lightning_sdk_launcher.py --stop    # stop a running Studio

Available machine types (as of 2025):
    cpu
    gpu-rtx-4090-x-1
    gpu-rtx-4090-x-2
    gpu-a100-x-1
    gpu-a100-x-2
    gpu-a100-x-4
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def get_studio():
    """Return a Lightning.ai Studio instance (creates if not exists)."""
    try:
        from lightning_sdk import Studio, Machine
    except ImportError:
        logger.error(
            "lightning-sdk not installed.\n"
            "Install with: pip install lightning-sdk"
        )
        sys.exit(1)

    user_id = os.getenv("LIGHTNING_USER_ID", "")
    if not user_id:
        logger.error(
            "LIGHTNING_USER_ID not set in .environment.\n"
            "Find it at: https://lightning.ai/<username>/home?settings=keys"
        )
        sys.exit(1)

    studio_name = os.getenv("LIGHTNING_STUDIO_NAME", "btc-ppo-trainer")
    return Studio(name=studio_name, user=user_id), Machine


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning.ai Studio launcher")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--machine", default="gpu-rtx-4090-x-1",
        help="Lightning.ai machine type (default: gpu-rtx-4090-x-1)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path inside Studio to resume from (e.g. models/ppo_btc_200000_steps.zip)"
    )
    parser.add_argument(
        "--download-model", default=None, metavar="LOCAL_PATH",
        help="After training, download the final model to this local path"
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop a running Studio"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print Studio status and exit"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    studio, Machine = get_studio()

    # ── Status check ──────────────────────────────────────────────────
    if args.status:
        try:
            status = studio.status
            logger.info(f"Studio status: {status}")
        except Exception as e:
            logger.info(f"Studio not running or inaccessible: {e}")
        return

    # ── Stop ──────────────────────────────────────────────────────────
    if args.stop:
        logger.info("Stopping Studio...")
        try:
            studio.stop()
            logger.info("Studio stopped.")
        except Exception as e:
            logger.warning(f"Stop failed: {e}")
        return

    # ── Start Studio ──────────────────────────────────────────────────
    machine_map = {
        "cpu":                Machine.CPU,
        "gpu-rtx-4090-x-1":  Machine.GPU_RTX_4090,
        "gpu-a100-x-1":      Machine.A100,
    }
    machine = machine_map.get(args.machine)
    if machine is None:
        # Try passing the string directly — SDK may accept it
        machine = args.machine

    logger.info(f"Starting Lightning.ai Studio on {args.machine}...")
    try:
        studio.start(machine=machine)
        logger.info("Studio started.")
    except Exception as e:
        logger.warning(f"Studio start: {e} (may already be running)")

    # ── Install dependencies ──────────────────────────────────────────
    logger.info("Installing dependencies in Studio...")
    try:
        studio.run("pip install -r requirements.txt -q")
        logger.info("Dependencies installed.")
    except Exception as e:
        logger.warning(f"Dep install: {e}")

    # ── Build training command ────────────────────────────────────────
    cmd_parts = ["python", "train_lightning.py", "--no-download"]

    if args.timesteps:
        cmd_parts += ["--timesteps", str(args.timesteps)]
    if args.n_envs:
        cmd_parts += ["--n-envs", str(args.n_envs)]
    if args.resume:
        cmd_parts += ["--resume", args.resume]

    remote_cmd = " ".join(cmd_parts)

    # ── Run training ──────────────────────────────────────────────────
    logger.info(f"Launching training: {remote_cmd}")
    logger.info(
        f"Monitor at: https://lightning.ai/"
        f"{os.getenv('LIGHTNING_USER_ID', '<user>')}/studios/"
        f"{os.getenv('LIGHTNING_STUDIO_NAME', 'btc-ppo-trainer')}"
    )

    try:
        # studio.run() blocks until command completes
        studio.run(remote_cmd)
        logger.info("Training command completed.")
    except Exception as e:
        logger.error(f"Training run failed: {e}")
        sys.exit(1)

    # ── Download model ────────────────────────────────────────────────
    if args.download_model:
        _download_model(studio, args.download_model)

    logger.info("Done.")


def _download_model(studio, local_path: str):
    """
    Download the trained model from the Studio to local_path.
    Lightning.ai Studios persist files at /teamspace/studios/this_studio.
    """
    remote_model = "/teamspace/studios/this_studio/models/ppo_btc_final.zip"
    local_path = str(Path(local_path).expanduser().resolve())

    logger.info(f"Downloading model: {remote_model} → {local_path}")

    try:
        # Use studio.download if available in the SDK version
        if hasattr(studio, "download"):
            studio.download(remote_path=remote_model, local_path=local_path)
            logger.info(f"Model downloaded to {local_path}")
        else:
            logger.info(
                "Auto-download not supported in this lightning-sdk version.\n"
                "To download manually:\n"
                "  1. Open your Studio in the browser\n"
                "  2. Navigate to /teamspace/studios/this_studio/models/\n"
                "  3. Right-click ppo_btc_final.zip → Download\n"
                "  OR use the Lightning.ai CLI:\n"
                "     lit download /teamspace/studios/this_studio/models/ppo_btc_final.zip"
            )
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        logger.info(
            "Download the model manually from your Studio:\n"
            "  /teamspace/studios/this_studio/models/ppo_btc_final.zip"
        )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
