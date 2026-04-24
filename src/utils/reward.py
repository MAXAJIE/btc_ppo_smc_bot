"""
reward.py  —  Swing-trading reward function
============================================

Three core changes from the previous scalping-biased version
-------------------------------------------------------------

1. Entry Tax  (战略降频)
   Every new position costs a flat -0.15 penalty at the moment of entry.
   This forces the model to expect a large enough move to justify the cost.
   A trade must clear ~3% just to break even on its cumulative reward —
   so the model learns to wait for high-conviction setups.

2. Profit Ladder  (目标导向奖励)
   Realised PnL reward is now non-linear and threshold-gated:
     pnl ≥ 3%  →  exponential reward  (pnl×100)^1.5  — big win bonus
     0 < pnl < 3%  →  tiny reward  pnl×10  — "technically profitable but meh"
     pnl < 0%  →  sharp penalty  |pnl|×150  — losses still hurt

3. Survival Bonus + Minimal Intra-Reward  (鼓励长线持有)
   While in a position: +0.001 per step ("stay alive" signal).
   Floating unrealised reward is cut to near-zero so the model is NOT
   tempted to close early just because it's up 1%.

   Combined with gamma=0.999, the model's value function now looks 1000
   steps into the future — essential for multi-day swing trades.

Other structural improvements
------------------------------
•  Entry quality bonus raised to 0.2 (only enter near OB/S&R).
•  Drawdown penalty kept linear + capped (no gradient spikes).
•  Hard reward clip tightened to [-3, 3] to stabilise value loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants — all overridable via config, but these are the swing defaults
# ---------------------------------------------------------------------------

ENTRY_TAX            = 0.15    # flat cost per new trade — forces selectivity
SURVIVAL_BONUS       = 0.001   # per-step bonus for holding a position
INTRA_WIN_SCALE      = 0.002   # near-zero: don't reward floating profit
INTRA_LOSS_SCALE     = 0.005   # near-zero: don't punish floating loss either

# Profit ladder thresholds
PROFIT_TARGET_PCT    = 0.03    # 3% is the "real win" threshold
WIN_SCALE_ABOVE      = 1.5     # exponent for (pnl×100)^WIN_SCALE_ABOVE
WIN_SCALE_BELOW      = 10.0    # linear scale for sub-3% wins
LOSS_SCALE           = 150.0   # harsh but bounded loss punishment

HOLD_PENALTY         = 0.00001 # tiny flat-position cost (not 0 — avoid infinite HOLD)

ENTRY_QUALITY_BONUS  = 0.20    # bonus for entering near OB / S&R level
ENTRY_QUALITY_THRESH = 0.5     # ATR-normalised distance threshold

DRAWDOWN_THRESHOLD   = 0.05
DRAWDOWN_LINEAR_K    = 0.20
DRAWDOWN_CAP         = 0.30

COMMISSION_RATE      = 0.0004
SLIPPAGE_RATE        = 0.0002
FUNDING_RATE         = 0.0001
FUNDING_STEPS        = 480      # 8h in 5m bars

REWARD_CLIP          = 3.0      # hard clip per step


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class RewardState:
    peak_equity:          float = 1.0
    prev_unrealised_pct:  float = 0.0
    step:                 int   = 0
    total_trades:         int   = 0
    trade_hold_steps:     int   = 0   # bars current trade has been open

    def update_peak(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset(self) -> None:
        self.peak_equity         = 1.0
        self.prev_unrealised_pct = 0.0
        self.step                = 0
        self.total_trades        = 0
        self.trade_hold_steps    = 0


# ---------------------------------------------------------------------------
# Primary reward function
# ---------------------------------------------------------------------------

def compute_reward(
    *,
    action:            int,
    prev_position:     int,        # -1 / 0 / 1
    new_position:      int,
    realised_pnl_pct:  float,      # fraction; non-zero only on close
    unrealised_pct:    float,
    equity:            float,
    state:             RewardState,
    # SMC / SNR entry-quality context (ATR-normalised, from observation)
    bull_ob_dist:  float = 0.0,
    bear_ob_dist:  float = 0.0,
    snr_support_1: float = 0.0,
    snr_resist_1:  float = 0.0,
) -> float:

    state.step += 1
    state.update_peak(equity)
    reward = 0.0

    entering = prev_position == 0 and new_position != 0
    exiting  = prev_position != 0 and new_position == 0
    holding  = new_position != 0 and not entering

    # Track how long we've held this trade
    if entering:
        state.trade_hold_steps = 0
    elif holding or exiting:
        state.trade_hold_steps += 1

    # ── 1. Transaction costs (every open step) ──────────────────────────────
    if new_position != 0:
        reward -= COMMISSION_RATE + SLIPPAGE_RATE
    if state.step % FUNDING_STEPS == 0 and new_position != 0:
        reward -= FUNDING_RATE

    # ── 2. Entry Tax + Entry Quality Bonus  (战略降频) ──────────────────────
    if entering:
        state.total_trades += 1

        # ENTRY TAX: flat cost that forces selectivity
        reward -= ENTRY_TAX

        # ENTRY QUALITY: partial refund if we enter near a valid level
        near_bull_ob  = 0.0 < bull_ob_dist  < ENTRY_QUALITY_THRESH
        near_support  = 0.0 < snr_support_1 < ENTRY_QUALITY_THRESH
        near_bear_ob  = 0.0 < bear_ob_dist  < ENTRY_QUALITY_THRESH
        near_resist   = 0.0 < snr_resist_1  < ENTRY_QUALITY_THRESH

        if new_position == 1 and (near_bull_ob or near_support):
            reward += ENTRY_QUALITY_BONUS
        elif new_position == -1 and (near_bear_ob or near_resist):
            reward += ENTRY_QUALITY_BONUS
        # If neither condition met, the entry tax stands unreduced.

    # ── 3. Realised PnL — Profit Ladder  (利润阶梯奖励) ────────────────────
    if abs(realised_pnl_pct) > 1e-6:

        if realised_pnl_pct >= PROFIT_TARGET_PCT:
            # 达到 3% 目标 — 指数级大奖
            reward += (realised_pnl_pct * 100) ** WIN_SCALE_ABOVE

        elif realised_pnl_pct > 0:
            # 盈利但不足 3% — 极小奖励（"做得一般"）
            reward += realised_pnl_pct * WIN_SCALE_BELOW

        else:
            # 亏损 — 严厉惩罚
            reward -= abs(realised_pnl_pct) * LOSS_SCALE

        # Reset unrealised tracker on close
        state.prev_unrealised_pct = 0.0

    # ── 4. Survival Bonus + minimal intra-reward  (鼓励持仓) ────────────────
    elif holding:
        # Survival bonus: small reward for staying in the trade
        reward += SURVIVAL_BONUS

        # Near-zero unrealised shaping — just enough to not completely ignore
        # the direction of price movement, but not enough to trigger early close
        delta = unrealised_pct - state.prev_unrealised_pct
        if delta > 0:
            reward += INTRA_WIN_SCALE  * math.log(1.0 + delta * 100.0)
        elif delta < 0:
            reward -= INTRA_LOSS_SCALE * math.log(1.0 + abs(delta) * 100.0)
        state.prev_unrealised_pct = unrealised_pct

    # ── 5. Flat / no-position penalty  (轻微推动模型去交易) ─────────────────
    else:
        reward -= HOLD_PENALTY
        state.prev_unrealised_pct = 0.0

    # ── 6. Drawdown penalty  (linear + capped) ──────────────────────────────
    dd = max(0.0, (state.peak_equity - equity) / (state.peak_equity + 1e-8))
    if dd > DRAWDOWN_THRESHOLD:
        reward -= min(DRAWDOWN_LINEAR_K * (dd - DRAWDOWN_THRESHOLD), DRAWDOWN_CAP)

    return float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

def compute_step_reward(
    action:           int,
    prev_position:    int,
    new_position:     int,
    realised_pnl_pct: float,
    unrealised_pct:   float,
    equity:           float,
    state:            RewardState,
    **kwargs,
) -> float:
    return compute_reward(
        action=action,
        prev_position=prev_position,
        new_position=new_position,
        realised_pnl_pct=realised_pnl_pct,
        unrealised_pct=unrealised_pct,
        equity=equity,
        state=state,
        **kwargs,
    )


def trade_reward(realised_pnl_pct: float) -> float:
    """Standalone closed-trade reward for testing."""
    if abs(realised_pnl_pct) < 1e-6:
        return 0.0
    if realised_pnl_pct >= PROFIT_TARGET_PCT:
        return (realised_pnl_pct * 100) ** WIN_SCALE_ABOVE
    elif realised_pnl_pct > 0:
        return realised_pnl_pct * WIN_SCALE_BELOW
    else:
        return -abs(realised_pnl_pct) * LOSS_SCALE


def cost_penalty(position: int = 1, step: int = 0) -> float:
    cost = 0.0
    if position != 0:
        cost += COMMISSION_RATE + SLIPPAGE_RATE
    if step % FUNDING_STEPS == 0 and position != 0:
        cost += FUNDING_RATE
    return -cost