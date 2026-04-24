"""
main_real.py
============
LIVE production trading loop — uses REAL money on Binance USDM Futures.

⚠ PREREQUISITES ⚠
------------------
Before running this, ensure:
  1. Your model has passed the 4-check validator in main_live.py:
       Sharpe > 1.0, Calmar > 2.0, OOS equity > 1.0, MC win rate > 80%
  2. You have set these env vars in .env:
       BINANCE_LIVE_API_KEY=...
       BINANCE_LIVE_API_SECRET=...
  3. Start with --dry-run to confirm logic before real orders
  4. Start with a small balance (e.g. $50–100) to verify execution
  5. Monitor the --dashboard live during first run

Usage
-----
    # Dry run first (no real orders, logs what would happen):
    python -m src.main_real --model ./models/validated_best/ppo_validated_ep5.zip --dry-run

    # Real trading:
    python -m src.main_real --model ./models/validated_best/ppo_validated_ep5.zip

    # With proxy (for geo-restricted servers):
    python -m src.main_real --model ./models/... --proxy http://127.0.0.1:7890

Differences from main_live.py (testnet)
----------------------------------------
  - Uses BinanceRealExecutor (live credentials, not testnet)
  - No PPO fine-tuning (inference only — don't update model in production)
  - Stricter risk limits (1% per-trade risk, 5% daily loss halt)
  - Auto SL+TP bracket placed after every open order
  - Immediate kill-switch at 10% account drawdown (tighter than training)
  - Pre-flight balance check before first trade
  - All actions logged with timestamps + order IDs for audit trail
"""

from __future__ import annotations

import os
import sys
import signal
import logging
from datetime import datetime, timedelta, timezone
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
import numpy as np

from src.environment.binance_testnet_env         import BinanceEnv
from src.execution.binance_real_executor import BinanceRealExecutor
from src.utils.data_loader               import DataLoader
from src.utils.logger                    import TradeLogger
from src.utils.model_validator           import ModelValidator
from src.models.ppo_model                import load_ppo

CANDLE_INTERVAL_S  = 300     # 5 minutes
KILL_DRAWDOWN_LIVE = 0.10    # tighter than training (10% not 15%)
MIN_BALANCE_USDT   = 20.0    # refuse to start with less than this

_SHUTDOWN = False


def _handle_signal(sig, frame):
    global _SHUTDOWN
    logger.warning("Signal %s — shutting down after current candle.", sig)
    _SHUTDOWN = True


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------

def main(
    model_path:          str,
    model_save_dir:      str   = "./models",
    max_runtime_hours:   float = 720.0,
    dry_run:             bool  = False,
    proxies:             Optional[dict] = None,
    revalidate_every_h:  float = 24.0,    # re-run 4-check validator every N hours
):
    global _SHUTDOWN

    cfg    = _load_cfg()
    env_cfg = cfg.get("environment", {})
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    start_time      = datetime.now(timezone.utc)
    deadline        = start_time + timedelta(hours=max_runtime_hours)
    next_revalidate = start_time + timedelta(hours=revalidate_every_h)

    # ── 1. Executor (LIVE) ────────────────────────────────────────────────────
    logger.info("Initialising %s executor …", "DRY-RUN" if dry_run else "LIVE")
    executor = BinanceRealExecutor(
        api_key    = os.getenv("BINANCE_LIVE_API_KEY",    ""),
        api_secret = os.getenv("BINANCE_LIVE_API_SECRET", ""),
        proxies    = proxies,
        dry_run    = dry_run,
    )

    # ── 2. Pre-flight checks ──────────────────────────────────────────────────
    balance = executor.get_equity()
    logger.info("Account balance: %.2f USDT", balance)

    if not dry_run and balance < MIN_BALANCE_USDT:
        raise RuntimeError(
            f"Balance {balance:.2f} USDT < minimum {MIN_BALANCE_USDT:.0f} USDT. "
            "Fund your Binance account before starting."
        )

    # ── 3. Data ───────────────────────────────────────────────────────────────
    logger.info("Loading market data …")
    loader = DataLoader(
        data_dir=os.getenv("DATA_PATH", "./data"), years=1
    )
    try:
        loader.load_base_df()
    except FileNotFoundError:
        logger.info("Downloading 6 months of history …")
        loader.download_history(years=0.5)

    tf_data = loader.tf_data
    logger.info("Data ready: %d 5m bars", loader.n_candles)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run main_live.py first and use a validated_best checkpoint."
        )
    model = load_ppo(model_path)
    logger.info("Model loaded: %s  (steps=%d)", model_path, model.num_timesteps)

    # ── 5. Pre-flight model revalidation ─────────────────────────────────────
    logger.info("Running pre-flight model validation …")
    validator = ModelValidator(model=model, tf_data=tf_data, config=cfg)
    report    = validator.full_report(
        train_end_idx = max(0, loader.n_candles - 2880),
        oos_len       = 1440,
        mc_trials     = 30,
    )

    if not report["is_good"] and not dry_run:
        logger.critical(
            "Model failed pre-flight validation. REFUSING to start live trading.\n"
            "Reasons: %s\n"
            "Use a model from ./models/validated_best/ or re-train.",
            report["reasons"],
        )
        raise SystemExit(1)
    elif not report["is_good"]:
        logger.warning(
            "Model failed validation but DRY-RUN mode — continuing anyway.\n"
            "Reasons: %s", report["reasons"]
        )

    # ── 6. Trade logger ───────────────────────────────────────────────────────
    tlog = TradeLogger(
        log_dir=os.getenv("LOG_PATH", "./logs"),
        run_id="live_real",
    )

    # ── 7. Main loop ──────────────────────────────────────────────────────────
    env          = BinanceEnv(tf_data=tf_data, config=cfg)
    obs, _       = env.reset()
    position     = 0
    entry_price  = 0.0
    equity       = balance
    peak_equity  = balance
    step         = 0
    episode      = 0

    logger.info(
        "\n%s\n  LIVE REAL TRADING STARTED\n  Model: %s\n  Balance: %.2f USDT\n%s",
        "═" * 60, model_path, balance, "═" * 60,
    )

    while not _SHUTDOWN:
        now = datetime.now(timezone.utc)
        if now >= deadline:
            logger.info("Max runtime %.1fh reached.", max_runtime_hours)
            break

        # Wait for next 5m candle
        _wait_candle(CANDLE_INTERVAL_S)

        # Get prediction
        action, _  = model.predict(obs, deterministic=True)
        action      = int(action)

        # Sim env step (for observation update only — not for P&L)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1

        # Live price
        close_px   = float(tf_data["5m"]["close"].iloc[-1])

        # Execute on real exchange
        prev_pos   = position
        position, entry_price, realised_pnl = executor.execute(
            action        = action,
            current_price = close_px,
            position      = position,
            entry_price   = entry_price,
        )

        equity = executor.get_equity() or equity

        # Log
        if abs(realised_pnl) > 1e-6:
            tlog.log_trade(
                episode     = episode,
                step        = step,
                side        = "long" if prev_pos == 1 else "short",
                entry_price = entry_price,
                exit_price  = close_px,
                pnl_pct     = realised_pnl,
                hold_steps  = step,
                equity      = equity,
            )
            logger.info(
                "TRADE CLOSED — side=%s pnl=%.4f%%  equity=%.2f USDT",
                "long" if prev_pos == 1 else "short",
                realised_pnl * 100, equity,
            )

        tlog.log_equity(episode=episode, step=step, equity=equity)

        if step % 288 == 0:
            logger.info(
                "[step %d] equity=%.2f USDT  pos=%s  close=%.2f",
                step, equity,
                "L" if position == 1 else "S" if position == -1 else "-",
                close_px,
            )

        # ── Kill switch ───────────────────────────────────────────────────
        if equity > peak_equity:
            peak_equity = equity

        drawdown = (peak_equity - equity) / max(peak_equity, 1e-8)
        if drawdown >= KILL_DRAWDOWN_LIVE:
            logger.critical(
                "KILL-SWITCH: drawdown %.1f%% >= %.0f%%. "
                "Closing all positions and halting.",
                drawdown * 100, KILL_DRAWDOWN_LIVE * 100,
            )
            executor.close_all()
            break

        # ── Episode boundary ─────────────────────────────────────────────
        if terminated or truncated or step % 4320 == 0:
            episode += 1
            obs, _  = env.reset()
            logger.info("Episode %d complete.", episode)

        # ── Periodic re-validation ────────────────────────────────────────
        if now >= next_revalidate:
            logger.info("Periodic re-validation …")
            re_report = validator.full_report(
                train_end_idx = max(0, loader.n_candles - 2880),
                oos_len       = 720,
                mc_trials     = 20,
            )
            if not re_report["is_good"]:
                logger.warning(
                    "Periodic validation FAILED — consider switching to a "
                    "newer validated model.\nReasons: %s",
                    re_report["reasons"],
                )
            next_revalidate = now + timedelta(hours=revalidate_every_h)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down live trading.")
    executor.close_all()

    summary = tlog.summary()
    logger.info(
        "\n%s\n  TRADING SESSION SUMMARY\n"
        "  Total trades : %d\n"
        "  Win rate     : %.1f%%\n"
        "  Total PnL    : %.2f%%\n"
        "  Max drawdown : %.2f%%\n%s",
        "═" * 60,
        summary.get("total_trades", 0),
        summary.get("win_rate",     0) * 100,
        summary.get("total_pnl_pct", 0),
        summary.get("max_dd",       0) * 100,
        "═" * 60,
    )


# ---------------------------------------------------------------------------

def _wait_candle(interval_s: int) -> None:
    import time
    now  = time.time()
    nxt  = (now // interval_s + 1) * interval_s + 2
    time.sleep(max(0, nxt - now))


def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Live REAL trading")
    p.add_argument("--model",            required=True)
    p.add_argument("--model-dir",        default="./models")
    p.add_argument("--runtime-hours",    type=float, default=720.0)
    p.add_argument("--dry-run",          action="store_true",
                   help="Log orders but don't send them. Use this first!")
    p.add_argument("--proxy",            default=None,
                   help="e.g. http://127.0.0.1:7890")
    p.add_argument("--revalidate-every", type=float, default=24.0,
                   help="Hours between periodic model re-validations")
    args = p.parse_args()

    proxies = {"https": args.proxy, "http": args.proxy} if args.proxy else None

    main(
        model_path         = args.model,
        model_save_dir     = args.model_dir,
        max_runtime_hours  = args.runtime_hours,
        dry_run            = args.dry_run,
        proxies            = proxies,
        revalidate_every_h = args.revalidate_every,
    )