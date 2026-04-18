"""
run_training.py
────────────────
Unified training launcher. Reads TRAIN_BACKEND from .environment and
dispatches to the correct backend.

Usage:
    # Set TRAIN_BACKEND in your .environment (or export it), then:
    python run_training.py

    # Or override inline:
    TRAIN_BACKEND=modal    python run_training.py
    TRAIN_BACKEND=lightning python run_training.py
    TRAIN_BACKEND=local    python run_training.py

    # Pass extra args (forwarded to the backend):
    python run_training.py --timesteps 1000000
    python run_training.py --resume ./models/ppo_btc_200000_steps.zip
    python run_training.py --download-only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import TradeLogger


load_dotenv()

VALID_BACKENDS = ("modal", "lightning", "local")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified PPO training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        choices=VALID_BACKENDS,
        default=None,
        help="Training backend (overrides TRAIN_BACKEND environment var)",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint .zip")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve backend
    backend = args.backend or os.getenv("TRAIN_BACKEND", "local").lower()

    if backend not in VALID_BACKENDS:
        print(f"ERROR: Unknown backend '{backend}'. Choose from: {VALID_BACKENDS}")
        sys.exit(1)

    print(f"\n{'═'*55}")
    print(f"  BTC PPO SMC TRAINING LAUNCHER")
    print(f"  Backend: {backend.upper()}")
    print(f"{'═'*55}\n")

    # ── Dispatch ──────────────────────────────────────────────────────

    if backend == "modal":
        _run_modal(args)

    elif backend == "lightning":
        _run_lightning(args)

    elif backend == "local":
        _run_local(args)


# ─────────────────────────────────────────────────────────────────────────────

def _run_modal(args):
    """
    Dispatch to Modal.com via CLI.
    Requires: pip install modal && modal token new
    """
    _check_command("modal", "pip install modal")

    cmd = ["modal", "run", "train_modal.py"]

    if args.timesteps:
        cmd += ["--timesteps", str(args.timesteps)]
    if args.n_envs:
        cmd += ["--n-envs", str(args.n_envs)]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.download_only:
        cmd += ["--download-only", "true"]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _run_lightning(args):
    """
    Run train_lightning.py locally (inside Lightning.ai Studio terminal)
    or via lightning-sdk for programmatic Studio control.
    """
    # Check if we're already inside a Lightning.ai Studio
    if Path("/teamspace").exists():
        print("Detected Lightning.ai Studio environment. Running directly.\n")
        cmd = [sys.executable, "train_lightning.py"]
    else:
        # Try to launch via lightning-sdk if available
        try:
            import lightning_sdk  # noqa: F401
            print("lightning-sdk available. Launching via SDK...\n")
            _launch_via_lightning_sdk(args)
            return
        except ImportError:
            print(
                "NOTE: Not inside a Lightning.ai Studio and lightning-sdk not installed.\n"
                "Falling back to running train_lightning.py locally.\n"
                "For cloud execution, open a Lightning.ai Studio and run:\n"
                "  python train_lightning.py\n"
            )
            cmd = [sys.executable, "train_lightning.py"]

    if args.timesteps:
        cmd += ["--timesteps", str(args.timesteps)]
    if args.n_envs:
        cmd += ["--n-envs", str(args.n_envs)]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.download_only:
        cmd += ["--download-only"]
    if args.no_download:
        cmd += ["--no-download"]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _run_local(args):
    """Run training directly in the current Python process (no cloud)."""
    from src.main_train import main as train_main

    train_main(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs if args.n_envs > 0 else 1,
        pretrained_path=args.resume,
    )


def _launch_via_lightning_sdk(args):
    """
    Programmatically start a Lightning.ai Studio and run training.
    Requires: pip install lightning-sdk
    Requires: LIGHTNING_USER_ID and LIGHTNING_API_KEY in .environment
    """
    from lightning_sdk import Studio, Machine

    user_id = os.getenv("LIGHTNING_USER_ID")
    api_key = os.getenv("LIGHTNING_API_KEY")

    if not user_id or not api_key:
        print(
            "ERROR: LIGHTNING_USER_ID and LIGHTNING_API_KEY must be set in .environment\n"
            "Find them at: https://lightning.ai/<username>/home?settings=keys"
        )
        sys.exit(1)

    studio_name = "btc-ppo-trainer"
    machine = Machine.GPU_RTX_4090   # change to Machine.CPU for debugging

    print(f"Starting Lightning.ai Studio '{studio_name}' on {machine}...")

    studio = Studio(name=studio_name, teamspace="default", user=user_id)
    studio.start(machine=machine)

    # Build the remote command
    remote_cmd_parts = ["python", "train_lightning.py", "--no-download"]
    if args.timesteps:
        remote_cmd_parts += ["--timesteps", str(args.timesteps)]
    if args.n_envs:
        remote_cmd_parts += ["--n-envs", str(args.n_envs)]
    if args.resume:
        remote_cmd_parts += ["--resume", args.resume]

    remote_cmd = " ".join(remote_cmd_parts)
    print(f"Running in Studio: {remote_cmd}")
    studio.run(remote_cmd)

    print(
        "\nTraining launched in Lightning.ai Studio.\n"
        f"Monitor at: https://lightning.ai/{user_id}/studios/{studio_name}"
    )


# ─────────────────────────────────────────────────────────────────────────────

def _check_command(cmd: str, install_hint: str):
    import shutil
    if shutil.which(cmd) is None:
        print(
            f"ERROR: '{cmd}' not found.\n"
            f"Install with: {install_hint}\n"
            f"Then authenticate: {cmd} token new"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
