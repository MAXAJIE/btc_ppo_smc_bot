"""
model_validator.py
==================
Four professional methods to validate whether a fine-tuned PPO model is
genuinely better — not just over-fitted to the training window.

Methods
-------
1. Out-of-sample walk-forward test
   Train on [start, split], validate on [split, end].
   Returns reward, equity curve, and drawdown on unseen data.

2. Sharpe & Calmar ratio
   Risk-adjusted return metrics.  Sharpe > 1.0 and Calmar > 2.0 are
   thresholds for a "good" model version.

3. Action stability analysis
   Detects "nervous" models that chop LONG/SHORT on noise.
   Flags excessive direction flips within short windows.

4. Monte Carlo simulation
   Adds micro-noise to prices and reruns the model N times.
   A robust model should be profitable in ≥ 80% of trials.

Usage
-----
    from src.utils.model_validator import ModelValidator

    validator = ModelValidator(model, tf_data, config)

    # After each fine-tune episode:
    report = validator.full_report(train_end_idx=4320, oos_len=1440)

    if report["is_good"]:
        save_ppo(model, "best_model")
    else:
        logger.warning("Fine-tuned model failed validation: %s", report["reasons"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — adjust to be more/less strict
# ---------------------------------------------------------------------------
SHARPE_THRESHOLD        = 0.8    # annualised Sharpe (252 trading days)
CALMAR_THRESHOLD        = 1.0    # annual_return / max_drawdown
FLIP_RATE_THRESHOLD     = 0.30   # max fraction of steps that flip direction
MC_WIN_RATE_THRESHOLD   = 0.70   # >= 70% of MC trials must be profitable
MC_TRIALS               = 100
MC_NOISE_PCT            = 0.001  # ±0.1% random price noise per bar


class ModelValidator:
    """
    Parameters
    ----------
    model   : stable_baselines3.PPO
    tf_data : dict[str, pd.DataFrame]   (output of DataLoader.load())
    config  : dict                       (full config.yaml)
    """

    def __init__(self, model, tf_data: Dict[str, pd.DataFrame], config: dict):
        self.model   = model
        self.tf_data = tf_data
        self.config  = config

    # -----------------------------------------------------------------------
    # Master report
    # -----------------------------------------------------------------------

    def full_report(
        self,
        train_end_idx: int,
        oos_len:       int = 1440,   # ~5 days of 5m bars
        mc_trials:     int = MC_TRIALS,
    ) -> Dict[str, Any]:
        """
        Run all four validation checks and return a unified report.

        Returns
        -------
        dict with keys:
            is_good         : bool   — True if ALL checks pass
            reasons         : list   — human-readable reasons for failures
            oos             : dict   — out-of-sample metrics
            risk            : dict   — Sharpe / Calmar
            action_stability: dict   — flip rate analysis
            monte_carlo     : dict   — MC win rate and equity percentiles
        """
        reasons: List[str] = []

        # ── 1. Out-of-sample ───────────────────────────────────────────────
        oos = self.walk_forward(train_end_idx=train_end_idx, oos_len=oos_len)

        # ── 2. Risk ratios ────────────────────────────────────────────────
        risk = self.risk_ratios(equity_curve=oos["equity_curve"])

        # ── 3. Action stability ───────────────────────────────────────────
        stability = self.action_stability(
            start_idx=train_end_idx, length=min(oos_len, 500)
        )

        # ── 4. Monte Carlo ────────────────────────────────────────────────
        mc = self.monte_carlo(
            start_idx=train_end_idx, length=oos_len, n_trials=mc_trials
        )

        # ── Verdict ───────────────────────────────────────────────────────
        if oos["final_equity"] < 1.0:
            reasons.append(
                f"OOS equity < 1.0 (={oos['final_equity']:.4f}) — model lost money on unseen data"
            )

        if risk["sharpe"] < SHARPE_THRESHOLD:
            reasons.append(
                f"Sharpe {risk['sharpe']:.2f} < threshold {SHARPE_THRESHOLD} — unstable returns"
            )

        if risk["calmar"] < CALMAR_THRESHOLD:
            reasons.append(
                f"Calmar {risk['calmar']:.2f} < threshold {CALMAR_THRESHOLD} — too much drawdown vs return"
            )

        if stability["flip_rate"] > FLIP_RATE_THRESHOLD:
            reasons.append(
                f"Action flip rate {stability['flip_rate']:.2%} > {FLIP_RATE_THRESHOLD:.0%} "
                "— model is chopping / over-reacting to noise"
            )

        if mc["win_rate"] < MC_WIN_RATE_THRESHOLD:
            reasons.append(
                f"Monte Carlo win rate {mc['win_rate']:.1%} < {MC_WIN_RATE_THRESHOLD:.0%} "
                "— model is not robust to minor price variations"
            )

        is_good = len(reasons) == 0

        report = {
            "is_good":          is_good,
            "reasons":          reasons,
            "oos":              oos,
            "risk":             risk,
            "action_stability": stability,
            "monte_carlo":      mc,
        }

        self._log_report(report)
        return report

    # -----------------------------------------------------------------------
    # Method 1 — Out-of-sample walk-forward
    # -----------------------------------------------------------------------

    def walk_forward(
        self, train_end_idx: int, oos_len: int = 1440
    ) -> Dict[str, Any]:
        """
        Evaluate the model on [train_end_idx, train_end_idx + oos_len].
        This slice was never seen during training.

        Returns
        -------
        dict:
            final_equity   float   — equity at end of OOS window
            max_drawdown   float   — worst drawdown fraction in OOS window
            total_reward   float   — cumulative reward
            n_trades       int
            win_rate       float
            equity_curve   list[float]
        """
        env = self._make_env(start_idx=train_end_idx, length=oos_len)
        obs, _ = env.reset()
        done   = False
        equity_curve  = [1.0]
        total_reward  = 0.0
        trades        = []
        prev_pos      = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done          = terminated or truncated
            total_reward += float(reward)
            equity        = float(info.get("equity", equity_curve[-1]))
            equity_curve.append(equity)

            pos = int(info.get("position", 0))
            if prev_pos != 0 and pos == 0:
                trades.append(equity - equity_curve[-2])   # last pnl
            prev_pos = pos

        arr   = np.array(equity_curve)
        peak  = np.maximum.accumulate(arr)
        dd    = (peak - arr) / (peak + 1e-8)
        max_dd = float(dd.max())

        win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0

        return {
            "final_equity": float(arr[-1]),
            "max_drawdown": max_dd,
            "total_reward": total_reward,
            "n_trades":     len(trades),
            "win_rate":     win_rate,
            "equity_curve": equity_curve,
        }

    # -----------------------------------------------------------------------
    # Method 2 — Sharpe & Calmar ratio
    # -----------------------------------------------------------------------

    def risk_ratios(
        self, equity_curve: Optional[List[float]] = None,
        start_idx: int = 0, length: int = 1440,
    ) -> Dict[str, float]:
        """
        Compute annualised Sharpe and Calmar from an equity curve.

        If equity_curve is not provided, runs a fresh episode to generate one.

        Returns
        -------
        dict:
            sharpe   float   — annualised Sharpe ratio
            calmar   float   — annual_return / max_drawdown
            annual_return float
            max_drawdown  float
            sortino  float   — downside-only Sharpe
        """
        if equity_curve is None:
            result = self.walk_forward(train_end_idx=start_idx, oos_len=length)
            equity_curve = result["equity_curve"]

        arr  = np.array(equity_curve, dtype=float)
        rets = np.diff(arr) / (arr[:-1] + 1e-8)   # step returns

        if len(rets) < 2:
            return {"sharpe": 0.0, "calmar": 0.0,
                    "annual_return": 0.0, "max_drawdown": 0.0, "sortino": 0.0}

        # 5m bars → 288 bars/day → 288 * 252 bars/year
        bars_per_year = 288 * 252
        ann_factor    = np.sqrt(bars_per_year)

        mean_ret = float(np.mean(rets))
        std_ret  = float(np.std(rets)) + 1e-8
        sharpe   = float(mean_ret / std_ret * ann_factor)

        # Downside deviation (Sortino)
        down_rets = rets[rets < 0]
        sortino   = float(
            mean_ret / (np.std(down_rets) + 1e-8) * ann_factor
        ) if len(down_rets) > 0 else sharpe

        # Max drawdown
        peak   = np.maximum.accumulate(arr)
        dd     = (peak - arr) / (peak + 1e-8)
        max_dd = float(dd.max())

        # Annual return (extrapolated from episode length)
        total_return  = float(arr[-1] / (arr[0] + 1e-8) - 1.0)
        n_steps       = len(arr)
        annual_return = float(
            (1 + total_return) ** (bars_per_year / max(n_steps, 1)) - 1
        )

        calmar = float(annual_return / (max_dd + 1e-8))

        return {
            "sharpe":        sharpe,
            "calmar":        calmar,
            "annual_return": annual_return,
            "max_drawdown":  max_dd,
            "sortino":       sortino,
        }

    # -----------------------------------------------------------------------
    # Method 3 — Action stability analysis
    # -----------------------------------------------------------------------

    def action_stability(
        self, start_idx: int = 0, length: int = 500
    ) -> Dict[str, Any]:
        """
        Detect whether the model nervously flips LONG/SHORT on noise.

        Metrics
        -------
        flip_rate       : fraction of steps where direction changed
        avg_hold_steps  : average bars held before direction change
        action_counts   : dict of {action_id: count}
        hold_histogram  : list of hold durations in bars
        verdict         : "stable" | "choppy" | "passive"
        """
        env    = self._make_env(start_idx=start_idx, length=length)
        obs, _ = env.reset()
        done   = False

        actions       = []
        positions     = []
        hold_durations = []
        current_hold  = 0
        prev_direction = 0   # -1 short / 0 flat / 1 long

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action     = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done       = terminated or truncated

            actions.append(action)
            pos = int(info.get("position", 0))
            positions.append(pos)

            direction = 1 if pos == 1 else (-1 if pos == -1 else 0)
            if direction != 0 and direction == prev_direction:
                current_hold += 1
            else:
                if current_hold > 0:
                    hold_durations.append(current_hold)
                current_hold   = 0
                prev_direction = direction

        if current_hold > 0:
            hold_durations.append(current_hold)

        n         = len(positions)
        flips     = sum(
            1 for i in range(1, n)
            if positions[i] != 0
            and positions[i-1] != 0
            and positions[i] != positions[i-1]
        )
        flip_rate = flips / max(n, 1)

        action_counts = {i: actions.count(i) for i in range(7)}
        avg_hold = float(np.mean(hold_durations)) if hold_durations else 0.0

        hold_only = action_counts.get(0, 0) / max(n, 1)
        if flip_rate > FLIP_RATE_THRESHOLD:
            verdict = "choppy"
        elif hold_only > 0.95:
            verdict = "passive"   # never trades — also bad
        else:
            verdict = "stable"

        return {
            "flip_rate":        flip_rate,
            "avg_hold_steps":   avg_hold,
            "action_counts":    action_counts,
            "hold_histogram":   hold_durations,
            "hold_pct":         hold_only,
            "verdict":          verdict,
            "n_steps":          n,
        }

    # -----------------------------------------------------------------------
    # Method 4 — Monte Carlo simulation
    # -----------------------------------------------------------------------

    def monte_carlo(
        self,
        start_idx: int = 0,
        length:    int = 1440,
        n_trials:  int = MC_TRIALS,
        noise_pct: float = MC_NOISE_PCT,
    ) -> Dict[str, Any]:
        """
        Inject ±noise_pct random price noise into the data and rerun
        the model n_trials times.  A robust model should be profitable
        in ≥ MC_WIN_RATE_THRESHOLD of trials.

        Returns
        -------
        dict:
            win_rate        float   — fraction of trials ending equity > 1.0
            mean_equity     float
            median_equity   float
            p10_equity      float   — 10th percentile (worst 10%)
            p90_equity      float   — 90th percentile (best 10%)
            std_equity      float
            all_final_equities list[float]
        """
        final_equities = []

        for trial in range(n_trials):
            noisy_tf = _inject_noise(self.tf_data, noise_pct, seed=trial)
            env      = self._make_env(
                start_idx=start_idx, length=length, tf_data_override=noisy_tf
            )
            obs, _   = env.reset()
            done     = False
            equity   = 1.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done   = terminated or truncated
                equity = float(info.get("equity", equity))

            final_equities.append(equity)

        arr      = np.array(final_equities)
        win_rate = float(np.mean(arr > 1.0))

        return {
            "win_rate":            win_rate,
            "mean_equity":         float(arr.mean()),
            "median_equity":       float(np.median(arr)),
            "p10_equity":          float(np.percentile(arr, 10)),
            "p90_equity":          float(np.percentile(arr, 90)),
            "std_equity":          float(arr.std()),
            "all_final_equities":  final_equities,
            "n_trials":            n_trials,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_env(
        self,
        start_idx:        int,
        length:           int,
        tf_data_override: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        from src.environment.binance_testnet_env import BinanceEnv

        tf  = tf_data_override or self.tf_data
        cfg = self.config

        n_bars = len(tf["5m"])
        end    = min(start_idx + length, n_bars - 1)
        actual_len = max(1, end - start_idx)

        # Slice each TF to the window we care about
        ts_start = tf["5m"].index[start_idx]
        ts_end   = tf["5m"].index[end - 1]

        sliced = {}
        for tfk, df in tf.items():
            mask = (df.index >= ts_start) & (df.index <= ts_end)
            sliced[tfk] = df.loc[mask].copy() if mask.any() else df.copy()

        env = BinanceEnv(
            tf_data       = sliced,
            config        = cfg,
            episode_steps = actual_len,
        )
        return env

    def _log_report(self, report: Dict[str, Any]) -> None:
        status = "✓ PASS" if report["is_good"] else "✗ FAIL"
        logger.info("=" * 60)
        logger.info("Model Validation Report — %s", status)
        logger.info("=" * 60)

        oos = report["oos"]
        logger.info(
            "OOS:     equity=%.4f  max_dd=%.2f%%  trades=%d  win_rate=%.1f%%",
            oos["final_equity"], oos["max_drawdown"] * 100,
            oos["n_trades"], oos["win_rate"] * 100,
        )

        risk = report["risk"]
        logger.info(
            "Risk:    sharpe=%.2f  calmar=%.2f  annual_ret=%.1f%%  sortino=%.2f",
            risk["sharpe"], risk["calmar"],
            risk["annual_return"] * 100, risk["sortino"],
        )

        stab = report["action_stability"]
        logger.info(
            "Actions: flip_rate=%.1f%%  avg_hold=%.1f bars  verdict=%s",
            stab["flip_rate"] * 100, stab["avg_hold_steps"], stab["verdict"],
        )
        logger.info("         counts=%s", stab["action_counts"])

        mc = report["monte_carlo"]
        logger.info(
            "MC(%d):  win_rate=%.1f%%  median=%.4f  p10=%.4f  p90=%.4f",
            mc["n_trials"], mc["win_rate"] * 100,
            mc["median_equity"], mc["p10_equity"], mc["p90_equity"],
        )

        if not report["is_good"]:
            for r in report["reasons"]:
                logger.warning("  ✗ %s", r)

        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Noise injection for Monte Carlo
# ---------------------------------------------------------------------------

def _inject_noise(
    tf_data:   Dict[str, pd.DataFrame],
    noise_pct: float,
    seed:      int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Add multiplicative Gaussian noise to all OHLCV price columns.
    Volume is left unchanged (noise in volume would be unrealistic).
    """
    rng    = np.random.default_rng(seed)
    result = {}

    for tf, df in tf_data.items():
        noisy = df.copy()
        price_cols = [c for c in ["open", "high", "low", "close"] if c in noisy.columns]
        noise  = rng.normal(1.0, noise_pct, size=(len(noisy), len(price_cols)))
        noisy[price_cols] = noisy[price_cols].values * noise

        # Enforce OHLC consistency: high >= max(open, close), low <= min(open, close)
        if all(c in noisy.columns for c in ["open", "high", "low", "close"]):
            noisy["high"] = noisy[["high", "open", "close"]].max(axis=1)
            noisy["low"]  = noisy[["low",  "open", "close"]].min(axis=1)

        result[tf] = noisy

    return result