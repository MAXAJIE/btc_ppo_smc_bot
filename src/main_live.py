"""
main_live.py  —  Live testnet fine-tuning loop
===============================================

Bugs fixed in this version
---------------------------
1. Position sync: REDUCE_HALF was halving an already-halved position
   repeatedly (steps 49→158→184→186→194→226 in real logs).
   The executor returns the *real* Binance qty after each call; we now
   track actual_qty separately so we know when the position is really flat.

2. MultiTFFeatureBuilder rebuilt every candle.  The `_base_df` setter was
   triggering `MultiTFFeatureBuilder(tf_data)` on every 5m bar, throwing
   away all rolling momentum history and logging "initialised" every step.
   Fixed: only rebuild if the data shape changes.

3. Short-bias diagnosis: we log action distribution every 50 steps so you
   can see if the model is stuck.  If it shorts >80% of the time, you need
   to retrain (not just fine-tune) with the new swing reward function.
"""

import os
import sys
import signal
import logging
import yaml
from collections import defaultdict
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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.binance_testnet_env    import BinanceEnv
from src.execution.binance_executor import BinanceExecutor
from src.utils.data_loader          import DataLoader
from src.utils.logger               import TradeLogger
from src.utils.model_validator      import ModelValidator
from src.models.ppo_model           import load_ppo, save_ppo

EPISODE_STEPS   = 4320
OOS_LEN         = 1440
MC_TRIALS       = 30       # reduced to keep validation fast
CANDLE_INTERVAL = 300

ACTION_NAMES = {0:"HOLD", 1:"L_FULL", 2:"L_HALF", 3:"S_FULL", 4:"S_HALF",
                5:"CLOSE", 6:"REDUCE"}

_SHUTDOWN = False


def _handle_signal(sig, frame):
    global _SHUTDOWN
    logger.warning("Signal %s — shutting down after current episode.", sig)
    _SHUTDOWN = True


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
def main(
    pretrained_model_path: str,
    model_save_dir:        str   = "./models",
    max_runtime_hours:     float = 720.0,
    walk_forward_days:     int   = 7,
    mc_trials:             int   = MC_TRIALS,
    proxies:               Optional[dict] = None,
):
    global _SHUTDOWN

    cfg      = _load_cfg()
    env_cfg  = cfg.get("environment", {})
    risk_cfg = cfg.get("risk", {})

    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_dir = model_save_dir / "validated_best"
    best_dir.mkdir(exist_ok=True)

    start_time        = datetime.now(timezone.utc)
    deadline          = start_time + timedelta(hours=max_runtime_hours)
    next_walk_forward = start_time + timedelta(days=walk_forward_days)

    # ── 1. Executor ──────────────────────────────────────────────────────────
    logger.info("Connecting to Binance Futures Testnet …")
    executor = BinanceExecutor(
        api_key    = os.getenv("BINANCE_TESTNET_API_KEY",    ""),
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", ""),
        testnet    = True,
        proxies    = proxies,
    )
    balance = executor.get_equity()
    logger.info("Testnet balance: %.2f USDT", balance)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    logger.info("Loading historical data …")
    loader = DataLoader(data_dir=os.getenv("DATA_PATH", "./data"), years=1)
    try:
        loader.load_base_df()
        logger.info("Cached data: %d bars", loader.n_candles)
    except FileNotFoundError:
        logger.info("Downloading 6 months of history …")
        loader.download_history(years=0.5)

    tf_data = loader.tf_data

    # ── 3. Environment (historical — for fine-tune rollout only) ─────────────
    def _make_vec():
        return Monitor(BinanceEnv(tf_data=tf_data, config=cfg))

    vec_env = DummyVecEnv([_make_vec])

    # ── 4. Model ─────────────────────────────────────────────────────────────
    if not Path(pretrained_model_path).exists():
        raise FileNotFoundError(f"Model not found: {pretrained_model_path}")

    model = load_ppo(pretrained_model_path, env=vec_env)
    logger.info(
        "Model loaded (num_timesteps=%d) — starting live fine-tuning loop.",
        model.num_timesteps,
    )

    # ── 5. Validator & trade logger ───────────────────────────────────────────
    validator = ModelValidator(model=model, tf_data=tf_data, config=cfg)
    tlog      = TradeLogger(log_dir=os.getenv("LOG_PATH", "./logs"), run_id="live")

    # ── 6. Ensure any leftover Binance position is flat before we start ──────
    logger.info("Checking for open positions before starting …")
    executor.close_all()

    # ── 7. Main episode loop ─────────────────────────────────────────────────
    episode         = 0
    peak_equity     = max(balance, 1.0)
    validated_saves = 0

    while not _SHUTDOWN:
        now = datetime.now(timezone.utc)
        if now >= deadline:
            logger.info("Max runtime reached.")
            break

        episode += 1
        logger.info("\n%s EPISODE %d %s", "═"*20, episode, "═"*20)

        ep_reward, ep_steps, train_end_idx = _run_episode(
            model    = model,
            loader   = loader,
            executor = executor,
            tlog     = tlog,
            episode  = episode,
            cfg      = cfg,
        )
        logger.info("Episode %d done — reward=%.3f  steps=%d",
                    episode, ep_reward, ep_steps)

        # ── Fine-tune ────────────────────────────────────────────────────────
        # Must train for at least n_steps (4096) or PPO buffer never flushes.
        min_learn = max(ep_steps, model.n_steps)
        logger.info("Fine-tuning for %d steps …", min_learn)
        model.set_env(vec_env)
        model.learn(total_timesteps=min_learn, reset_num_timesteps=False,
                    progress_bar=False)
        save_ppo(model, str(model_save_dir / f"ppo_live_ep{episode}"))

        # ── Validation ───────────────────────────────────────────────────────
        report = validator.full_report(
            train_end_idx = max(0, train_end_idx - OOS_LEN * 2),
            oos_len       = OOS_LEN,
            mc_trials     = mc_trials,
        )
        if report["is_good"]:
            validated_saves += 1
            save_ppo(model, str(best_dir / f"ppo_validated_ep{episode}"))
            logger.info("✓ Validation passed — saved as best. (total: %d)",
                        validated_saves)
        else:
            logger.warning("✗ Validation failed: %s", report["reasons"])

        # ── Kill switch ──────────────────────────────────────────────────────
        current_equity = executor.get_equity() or peak_equity
        peak_equity    = max(peak_equity, current_equity)
        drawdown       = (peak_equity - current_equity) / max(peak_equity, 1e-8)
        kill_dd        = risk_cfg.get("max_drawdown_kill",
                         env_cfg.get("kill_switch_drawdown", 0.15))
        if drawdown >= kill_dd:
            logger.critical("KILL-SWITCH: drawdown %.1f%%. Closing all.",
                            drawdown * 100)
            executor.close_all()
            break

        # ── Walk-forward ─────────────────────────────────────────────────────
        if now >= next_walk_forward:
            _scheduled_walk_forward(validator, train_end_idx, OOS_LEN)
            next_walk_forward = now + timedelta(days=walk_forward_days)

        if _SHUTDOWN:
            break

    executor.close_all()
    save_ppo(model, str(model_save_dir / "ppo_live_final"))
    logger.info("Live loop ended. Validated saves: %d", validated_saves)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    model,
    loader:   DataLoader,
    executor: BinanceExecutor,
    tlog:     TradeLogger,
    episode:  int,
    cfg:      dict,
) -> tuple:

    total_reward  = 0.0
    steps         = 0

    # ── Position tracking (synced from Binance on every trade) ───────────────
    # direction: -1 short / 0 flat / 1 long
    # qty_btc: actual BTC quantity on Binance (needed for REDUCE_HALF tracking)
    position    = 0
    entry_price = 0.0
    qty_btc     = 0.0      # FIX: track actual qty so REDUCE_HALF sync works

    # ── Action distribution counter (for short-bias diagnosis) ───────────────
    action_counts: dict = defaultdict(int)

    # ── Build live env once; update data each candle without full rebuild ─────
    client  = getattr(executor, "client", None)
    live_tf = loader.update_live_data(client=client, lookback=500)
    n_live  = len(live_tf["5m"])

    env = BinanceEnv(tf_data=live_tf, config=cfg,
                     episode_steps=max(1, n_live - 2))
    env.reset()

    prev_n = n_live   # track shape to know when rebuild is needed

    while steps < EPISODE_STEPS and not _SHUTDOWN:

        # ── 1. Wait for next 5m candle ────────────────────────────────────
        _wait_candle(CANDLE_INTERVAL)

        # ── 2. Fetch latest data ──────────────────────────────────────────
        live_tf = loader.update_live_data(client=client, lookback=500)
        n_live  = len(live_tf["5m"])

        # ── 3. Update env data WITHOUT rebuilding MultiTFFeatureBuilder ───
        #       Only rebuild if the TF structure changed (shouldn't happen).
        env.tf     = live_tf
        env._base  = live_tf["5m"]    # bypass setter to skip redundant rebuild
        env._n     = n_live
        if n_live != prev_n:
            # Shape changed — rebuild momentum columns (rare)
            from src.features.multi_tf_features import MultiTFFeatureBuilder
            env._builder = MultiTFFeatureBuilder(live_tf)
            prev_n = n_live

        # Pin cursor to last bar
        env._cur_idx = n_live - 1

        # ── 4. Get observation + predict ──────────────────────────────────
        obs       = env._get_obs()
        action, _ = model.predict(obs, deterministic=False)
        action    = int(action)
        action_counts[action] += 1

        # ── 5. Env step for reward signal ─────────────────────────────────
        _, reward, _, _, info = env.step(action)
        total_reward += float(reward)
        steps        += 1

        # ── 6. Live execution ─────────────────────────────────────────────
        close_px = float(live_tf["5m"]["close"].iloc[-1])
        prev_pos = position
        prev_qty = qty_btc

        new_position, entry_price, realised_pnl = executor.execute(
            action        = action,
            current_price = close_px,
            position      = position,
            entry_price   = entry_price,
        )

        # Sync position direction
        position = new_position

        # FIX: sync qty_btc from Binance after every action
        # This prevents REDUCE_HALF from desynchronising the tracker.
        if position != 0:
            qty_btc = executor._get_open_qty()
        else:
            qty_btc = 0.0

        # Detect ghost position: we think we're in a trade but Binance says 0
        if position != 0 and qty_btc < executor._get_min_qty(close_px):
            logger.warning(
                "Ghost position detected (tracker=%d, binance_qty=%.6f). "
                "Resetting position tracker to flat.",
                position, qty_btc,
            )
            position    = 0
            entry_price = 0.0
            qty_btc     = 0.0

        # ── 7. Log ────────────────────────────────────────────────────────
        if abs(realised_pnl) > 1e-6:
            tlog.log_trade(
                episode     = episode,
                step        = steps,
                side        = "long" if prev_pos == 1 else "short",
                entry_price = entry_price,
                exit_price  = close_px,
                pnl_pct     = realised_pnl,
                hold_steps  = steps,
                equity      = executor.get_equity(),
            )

        tlog.log_equity(episode=episode, step=steps,
                        equity=info.get("equity", 1.0), reward=reward)

        pos_str = "L" if position == 1 else "S" if position == -1 else "-"
        logger.info(
            "  [ep%d step%d] %s | action=%s | pos=%s qty=%.4f | pnl=%.4f%%",
            episode, steps, f"{close_px:.2f}",
            ACTION_NAMES.get(action, str(action)),
            pos_str, qty_btc, realised_pnl * 100,
        )

        # ── 8. Action distribution every 50 steps (short-bias diagnosis) ──
        if steps % 50 == 0:
            total_acts = sum(action_counts.values())
            dist = {ACTION_NAMES[k]: f"{v/total_acts:.0%}"
                    for k, v in sorted(action_counts.items())}
            logger.info("  Action distribution (last %d steps): %s", steps, dist)

            short_actions = action_counts.get(3, 0) + action_counts.get(4, 0)
            short_pct = short_actions / max(total_acts, 1)
            if short_pct > 0.70:
                logger.warning(
                    "  ⚠ SHORT bias detected (%.0f%% of actions). "
                    "This model needs RETRAINING with the new swing reward, "
                    "not just fine-tuning.", short_pct * 100
                )

    final_idx = getattr(env, "_cur_idx", steps)
    return total_reward, steps, final_idx


def _scheduled_walk_forward(validator, train_end_idx, oos_len):
    logger.info("Scheduled walk-forward validation …")
    try:
        report = validator.full_report(
            train_end_idx=train_end_idx, oos_len=oos_len, mc_trials=20
        )
        logger.info(
            "WF — is_good=%s  sharpe=%.2f  calmar=%.2f",
            report["is_good"],
            report["risk"]["sharpe"],
            report["risk"]["calmar"],
        )
    except Exception as e:
        logger.warning("Walk-forward failed: %s", e)


def _wait_candle(interval_s: int) -> None:
    import time
    now = time.time()
    nxt = (now // interval_s + 1) * interval_s + 1.5
    time.sleep(max(0, nxt - now))


def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",         required=True)
    p.add_argument("--model-dir",     default="./models")
    p.add_argument("--runtime-hours", type=float, default=720.0)
    p.add_argument("--wf-days",       type=int,   default=7)
    p.add_argument("--mc-trials",     type=int,   default=MC_TRIALS)
    p.add_argument("--proxy",         default=None)
    args = p.parse_args()

    main(
        pretrained_model_path = args.model,
        model_save_dir        = args.model_dir,
        max_runtime_hours     = args.runtime_hours,
        walk_forward_days     = args.wf_days,
        mc_trials             = args.mc_trials,
        proxies = {"https": args.proxy, "http": args.proxy} if args.proxy else None,
    )