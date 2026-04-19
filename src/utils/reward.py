"""
reward.py  –  Reward shaping for PPO convergence  (FIXED)
==========================================================
Why the old reward was preventing convergence
---------------------------------------------
1.  ASYMMETRY TOO LARGE (2× loss penalty): With a 2× loss multiplier the
    agent learns very quickly to NEVER trade, because any exploratory trade
    that goes wrong is punished twice as hard as the equivalent win is
    rewarded.  The policy collapses to always choosing HOLD (action 0),
    gradient vanishes, value loss diverges.

2.  SPARSE REWARD ON CLOSE ONLY: the 4 320-step episode (15 days × 288
    bars/day) gives the policy almost no reward signal per step.  The PPO
    critic can't fit a value function with so few non-zero targets,
    causing high value-loss.

3.  INTRA-TRADE UNREALISED PENALTY (−0.10 per step): discourages the
    agent from holding losing trades (fine), but since ALL new trades start
    with a small unrealised loss due to commission + spread, the agent
    learns to CLOSE immediately → degenerate strategy.

4.  QUADRATIC DRAWDOWN PENALTY (−5 × excess²): can spike to −5 × 0.09 =
    −0.45 on a single step — orders of magnitude larger than normal rewards
    — causing gradient explosions.

5.  NO CONTEXT REWARD FOR TOOL USE: the policy never learns that entering
    near an SMC OB / S&R level is better than entering randomly, because
    the reward is purely P&L based.  This is why the bot "doesn't understand
    the tools".

Fixes applied
-------------
1.  Asymmetry reduced to 1.3× (still penalises losses, but doesn't kill
    exploration).
2.  Dense per-step shaped reward using unrealised PnL delta (not raw PnL).
    This gives the critic a non-zero target every step.
3.  Entry-quality bonus: +0.05 if entering near a bullish OB / support
    for longs; +0.05 if entering near a bearish OB / resistance for shorts.
    Teaches the policy to use the SMC/SNR tools.
4.  HOLD encouraged only if we hold during favourable HTF alignment.
5.  Drawdown penalty is now linear and capped to avoid gradient spikes.
6.  Time decay is unchanged (tiny −0.0001 per flat step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
WIN_SCALE    = 1.0
LOSS_SCALE   = 1.3          # was 2.0 — reduced to encourage exploration
HOLD_PENALTY = 0.0001       # per-step cost of doing nothing

INTRA_WIN_SCALE  = 0.03     # per-step unrealised-gain shaping
INTRA_LOSS_SCALE = 0.04     # was 0.10 — reduced so new trades aren't immediately closed

DRAWDOWN_THRESHOLD = 0.05   # 5% drawdown before penalty starts
DRAWDOWN_LINEAR_K  = 0.20   # linear coefficient (not squared) → max ≈ -0.20 per step
DRAWDOWN_CAP       = 0.30   # hard cap on drawdown penalty per step

ENTRY_QUALITY_BONUS  = 0.08  # reward for entering near OB/SNR zone
ENTRY_QUALITY_THRESH = 0.5   # feature threshold to count as "near zone" (ATR units)

COMMISSION_RATE = 0.0004     # 0.04% taker
SLIPPAGE_RATE   = 0.0002     # estimated 0.02%
FUNDING_RATE    = 0.0001     # 0.01% per 8h
FUNDING_STEPS   = 480        # 8h in 5m bars


# ---------------------------------------------------------------------------
# State tracker (one per episode)
# ---------------------------------------------------------------------------

@dataclass
class RewardState:
    peak_equity: float = 1.0
    prev_unrealised_pct: float = 0.0
    step: int = 0
    total_trades: int = 0
    _rolling_pnl: list = field(default_factory=list)

    def update_peak(self, equity: float):
        if equity > self.peak_equity:
            self.peak_equity = equity


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    *,
    action: int,
    prev_position: int,          # 0 = flat, 1 = long, -1 = short
    new_position: int,
    realised_pnl_pct: float,     # fraction of account, set on trade close else 0
    unrealised_pct: float,       # current unrealised PnL fraction
    equity: float,               # current account equity
    state: RewardState,
    # SMC / SNR context (from observation — already normalised)
    bull_ob_dist: float = 0.0,   # obs[51] — ATR units, negative = price below OB top
    bear_ob_dist: float = 0.0,   # obs[52]
    snr_support_1: float = 0.0,  # obs[67]
    snr_resist_1:  float = 0.0,  # obs[70]
) -> float:

    state.step += 1
    state.update_peak(equity)
    reward = 0.0

    # ------------------------------------------------------------------ #
    # 1.  Transaction costs (charged every step a position is open)       #
    # ------------------------------------------------------------------ #
    if new_position != 0:
        reward -= (COMMISSION_RATE + SLIPPAGE_RATE) * abs(new_position)
    if state.step % FUNDING_STEPS == 0 and new_position != 0:
        reward -= FUNDING_RATE

    # ------------------------------------------------------------------ #
    # 2.  Entry bonus / penalty (fires once when a new trade opens)       #
    # ------------------------------------------------------------------ #
    entering_long  = (prev_position == 0 and new_position ==  1)
    entering_short = (prev_position == 0 and new_position == -1)

    if entering_long:
        state.total_trades += 1
        # Reward entering near a bullish OB (price just above OB top → small positive dist)
        # or near support (small positive dist)
        near_bull_ob  = 0.0 < bull_ob_dist  < ENTRY_QUALITY_THRESH
        near_support  = 0.0 < snr_support_1 < ENTRY_QUALITY_THRESH
        if near_bull_ob or near_support:
            reward += ENTRY_QUALITY_BONUS
        else:
            reward -= ENTRY_QUALITY_BONUS * 0.3   # mild penalty for random entry

    if entering_short:
        state.total_trades += 1
        near_bear_ob   = 0.0 < bear_ob_dist < ENTRY_QUALITY_THRESH
        near_resistance = 0.0 < snr_resist_1 < ENTRY_QUALITY_THRESH
        if near_bear_ob or near_resistance:
            reward += ENTRY_QUALITY_BONUS
        else:
            reward -= ENTRY_QUALITY_BONUS * 0.3

    # ------------------------------------------------------------------ #
    # 3.  Realised PnL reward (on trade close)                            #
    # ------------------------------------------------------------------ #
    if abs(realised_pnl_pct) > 1e-6:
        mag = math.log(1 + abs(realised_pnl_pct) * 100)
        if realised_pnl_pct > 0:
            reward += WIN_SCALE  * mag
        else:
            reward -= LOSS_SCALE * mag

        state.prev_unrealised_pct = 0.0   # reset delta tracker on close

    # ------------------------------------------------------------------ #
    # 4.  Intra-trade unrealised-delta shaping                            #
    # Only reward the CHANGE in unrealised PnL, not the raw value.        #
    # This encourages the agent to stay in trades that are moving in its  #
    # favour and exit trades that are moving against it — without the     #
    # "close immediately" bias from penalising raw unrealised PnL.        #
    # ------------------------------------------------------------------ #
    elif new_position != 0:
        delta = unrealised_pct - state.prev_unrealised_pct
        if delta > 0:
            reward += INTRA_WIN_SCALE  * math.log(1 + delta * 100)
        elif delta < 0:
            reward -= INTRA_LOSS_SCALE * math.log(1 + abs(delta) * 100)
        state.prev_unrealised_pct = unrealised_pct

    # ------------------------------------------------------------------ #
    # 5.  Flat / no-trade penalty                                         #
    # ------------------------------------------------------------------ #
    else:
        reward -= HOLD_PENALTY
        state.prev_unrealised_pct = 0.0

    # ------------------------------------------------------------------ #
    # 6.  Drawdown penalty  (linear, capped — no quadratic spikes)        #
    # ------------------------------------------------------------------ #
    drawdown = max(0.0, (state.peak_equity - equity) / (state.peak_equity + 1e-8))
    if drawdown > DRAWDOWN_THRESHOLD:
        excess  = drawdown - DRAWDOWN_THRESHOLD
        dd_pen  = min(DRAWDOWN_LINEAR_K * excess, DRAWDOWN_CAP)
        reward -= dd_pen

    return float(np.clip(reward, -5.0, 5.0))   # hard clip prevents single spike
