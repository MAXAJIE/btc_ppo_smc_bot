"""
scripts/download_data.py
─────────────────────────
Standalone script to download and cache BTCUSDT 5m historical data.
Run this BEFORE training to pre-populate the data cache.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --years 3
    python scripts/download_data.py --years 1 --force   # force re-download
    python scripts/download_data.py --verify            # just verify cache
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download BTCUSDT historical data")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history to download")
    parser.add_argument("--data-dir", default="./data", help="Directory to save data")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--verify", action="store_true", help="Only verify existing cache, no download")
    args = parser.parse_args()

    from src.utils.data_loader import DataLoader

    loader = DataLoader(data_dir=args.data_dir)

    if args.verify:
        _verify(loader)
        return

    cache_path = Path(args.data_dir) / "BTCUSDT_5m.parquet"
    if cache_path.exists() and not args.force:
        logger.info(f"Cache already exists at {cache_path}")
        loader.load_base_df()
        logger.info(f"  {loader.n_candles:,} candles available")
        _print_summary(loader)
        logger.info("Use --force to re-download.")
        return

    logger.info(f"Downloading {args.years:.1f} years of BTCUSDT 5m data...")
    df = loader.download_history(years=args.years, force=args.force)
    logger.info(f"\nDownload complete!")
    _print_summary(loader)


def _verify(loader):
    """Verify data integrity — check for gaps and NaNs."""
    try:
        loader.load_base_df()
    except FileNotFoundError:
        logger.error("No cached data found. Run without --verify first.")
        sys.exit(1)

    df = loader._base_df
    logger.info(f"Loaded {len(df):,} rows")

    # Check for NaN
    nan_count = df.isna().sum().sum()
    logger.info(f"NaN values: {nan_count}")

    # Check for gaps (missing 5m bars)
    expected_interval = 5 * 60 * 1_000_000_000  # 5min in nanoseconds
    diffs = df.index.diff().dropna()
    gaps = diffs[diffs > pd.Timedelta("10min")]

    import pandas as pd
    if len(gaps) > 0:
        logger.warning(f"Found {len(gaps)} gaps > 10min:")
        for ts, gap in zip(gaps.index[:5], gaps[:5]):
            logger.warning(f"  {ts}: gap = {gap}")
        if len(gaps) > 5:
            logger.warning(f"  ... and {len(gaps)-5} more")
    else:
        logger.info("No significant gaps found ✓")

    _print_summary(loader)


def _print_summary(loader):
    import pandas as pd
    df = loader._base_df
    if df is None or len(df) == 0:
        return

    n_episodes = len(loader.get_episode_start_indices(episode_len=4320, warmup=500))
    size_mb = (Path(os.getenv("DATA_PATH", "./data")) / "BTCUSDT_5m.parquet").stat().st_size / 1e6

    logger.info(f"""
  ─────────────────────────────────────────
  Data Summary
  ─────────────────────────────────────────
  Rows         : {len(df):>12,}
  From         : {str(df.index[0])[:19]}
  To           : {str(df.index[-1])[:19]}
  Duration     : {(df.index[-1] - df.index[0]).days} days
  File size    : {size_mb:.1f} MB
  Episodes (15d): {n_episodes:>10,}  (50% overlap)
  ─────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
