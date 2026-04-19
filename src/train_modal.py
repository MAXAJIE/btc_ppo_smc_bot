"""
train_modal.py
──────────────
Modal.com training entrypoint.

Usage:
    # First time setup:
    modal secret create binance-secrets \
        BINANCE_TESTNET_API_KEY=<your_key> \
        BINANCE_TESTNET_API_SECRET=<your_secret>

    # Run offline training:
    modal run train_modal.py

    # Run and keep GPU warm (for iterative debugging):
    modal serve train_modal.py

The trained model and data are persisted in a Modal Volume
so they survive container teardowns.
"""

import modal

# ─────────────────────────────────────────────────────────────────────────────
# App & Image definition
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("btc-ppo-trainer")

# Persistent volume: stores historical data + model checkpoints
volume = modal.Volume.from_name("btc-ppo-vol", create_if_missing=True)
VOLUME_MOUNT = "/vol"

# Container image — debian slim + all Python deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium>=1.0.0",
        "stable-baselines3[extra]>=2.3.0",
        "python-binance>=1.0.19",
        "binance-futures-connector>=4.0.0",
        "arch>=7.0.0",
        "smartmoneyconcepts>=0.0.27",
        "ta>=0.10.2",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "ccxt>=4.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "tensorboard>=2.16.0",
        "pyarrow>=15.0.0",      # for parquet
        "fastparquet>=2024.2.0",
    )
    # Copy the entire project into the container
    .add_local_dir(".", remote_path="/root/btc_ppo_smc_bot", ignore=["__pycache__", "*.pyc", ".git", "logs", "data", "*.csv", "*.png"])
)

# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="h100",                          # Good cost/perf for RL (non-LLM)
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("binance-secrets")],
    timeout=60 * 60 * 23,               # 23h — restart before Modal's 24h limit
    memory=16384,                        # 16 GB RAM
    cpu=16.0,
)
def train_offline(
    total_timesteps: int = 3_000_000,
    n_envs: int = 64,
    resume_from: str = None,
):
    """
    Full offline PPO training run inside a Modal A10G container.

    Data and models are saved to /vol (Modal Volume) so they persist
    between runs. On resume, the latest checkpoint is loaded.
    """
    import sys
    import os
    project_root = "/root/btc_ppo_smc_bot"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    # Override environment vars to use volume paths
    os.environ["MODEL_SAVE_PATH"] = f"{VOLUME_MOUNT}/models"
    os.environ["DATA_PATH"] = f"{VOLUME_MOUNT}/data"
    os.environ["LOG_PATH"] = f"{VOLUME_MOUNT}/logs"

    from src.main_train import main as train_main

    # Check if we should resume from a previous checkpoint
    if resume_from is None:
        import glob
        checkpoints = sorted(
            glob.glob(f"{VOLUME_MOUNT}/models/ppo_btc_*.zip")
        )
        if checkpoints:
            resume_from = checkpoints[-1]
            print(f"Auto-resuming from checkpoint: {resume_from}")

    result_path = train_main(
        model_save_dir=f"{VOLUME_MOUNT}/models",
        data_dir=f"{VOLUME_MOUNT}/data",
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        pretrained_path=resume_from,
    )

    # Commit volume so changes are visible outside the function
    volume.commit()
    print(f"Training complete. Model at: {result_path}")
    return result_path


# ─────────────────────────────────────────────────────────────────────────────
# Download-only function (cheap CPU job)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60 * 2,               # 2h for data download
    memory=8172,
)
def download_data(years: int = 2):
    """Download and cache historical data into the volume."""
    import sys
    import os
    sys.path.insert(0, "/app")
    os.chdir("/app")
    project_root = "/root/btc_ppo_smc_bot"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    # 验证是否能看到 src 文件夹
    if not os.path.exists(os.path.join(project_root, "src")):
        print(f"Error: src folder not found in {project_root}")
        print(f"Contents of {project_root}: {os.listdir(project_root)}")

    from src.utils.data_loader import DataLoader
    loader = DataLoader(data_dir=f"{VOLUME_MOUNT}/data")
    df = loader.download_history(years=years, force=True)
    volume.commit()
    print(f"Downloaded {len(df):,} 5m candles spanning {years} years.")
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint — what runs when you call `modal run train_modal.py`
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    timesteps: int = 3_000_000,
    n_envs: int = 64,
    download_only: bool = False,
    resume: str = None,
):
    """
    Entry point. Steps:
    1. Download data (if not cached in volume)
    2. Run offline PPO training

    Examples:
        modal run train_modal.py                          # full 3M step run
        modal run train_modal.py --timesteps 500000       # quick test run
        modal run train_modal.py --download-only true     # only download data
        modal run train_modal.py --resume /vol/models/ppo_btc_200000_steps.zip
    """
    print("=" * 60)
    print("  BTC PPO SMC TRAINER — Modal.com")
    print("=" * 60)

    # Step 1: Ensure data exists in volume
    print("\n[1/2] Checking / downloading data...")
    n_candles = download_data.remote(years=2)
    print(f"  Data ready: {n_candles:,} candles")

    if download_only:
        print("--download-only flag set. Exiting.")
        return

    # Step 2: Train
    print(f"\n[2/2] Starting training: {timesteps:,} timesteps on A10G GPU...")
    result = train_offline.remote(
        total_timesteps=timesteps,
        n_envs=n_envs,
        resume_from=resume,
    )
    print(f"\nTraining complete! Model saved to: {result}")
    print("\nTo download the model locally, run:")
    print("  modal volume get btc-ppo-vol models/ppo_btc_final.zip ./models/")
