"""
reward.py – Reward shaping for PPO convergence (FIXED + EXTENDED)
=================================================================
All original compute_reward logic preserved.
Added missing functions required by tests and environment:
  - step_reward()
  - drawdown_penalty()
  - funding_cost()
  - killswitch_penalty()
  - holding_cost()
  - sl_hit_extra_penalty()
  - compute_step_reward() — extended signature with kill_triggered, sl_hit, funding_fee, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
WIN_SCALE    = 1.01
LOSS_SCALE   = 1.3
HOLD_PENALTY = 0.00005

INTRA_WIN_SCALE  = 0.03
INTRA_LOSS_SCALE = 0.04

DRAWDOWN_THRESHOLD = 0.05
DRAWDOWN_LINEAR_K  = 0.30
DRAWDOWN_QUAD_K    = 5.0
DRAWDOWN_CAP       = 0.30

ENTRY_QUALITY_BONUS  = 0.08
ENTRY_QUALITY_THRESH = 0.5

COMMISSION_RATE = 0.0004
SLIPPAGE_RATE   = 0.0002
FUNDING_RATE    = 0.0001
FUNDING_STEPS   = 480

SL_EXTRA_PENALTY    = -1.5
HOLDING_COST_PER_BAR = -0.0001
KILLSWITCH_SCALE     = 50.0


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
# Atomic reward components (used by tests directly)
# ---------------------------------------------------------------------------

def trade_reward(
    pnl_pct: float,
    win_scale: float = WIN_SCALE,
    loss_penalty: float = LOSS_SCALE,
) -> float:
    """Log-scaled reward for a completed trade."""
    if abs(pnl_pct) < 1e-9:
        return 0.0
    mag = math.log(1 + abs(pnl_pct) * 100)
    return win_scale * mag if pnl_pct > 0 else -loss_penalty * mag


def step_reward(
    unrealized_pnl_pct: float,
    position: int,
    bars_in_trade: int,
    hold_penalty: float = -HOLD_PENALTY,
) -> float:
    """Per-step shaped reward based on unrealised PnL delta."""
    if position == 0:
        return hold_penalty if hold_penalty < 0 else -HOLD_PENALTY
    delta = unrealized_pnl_pct  # caller passes current unrealised; env tracks delta
    if delta > 0:
        return INTRA_WIN_SCALE * math.log(1 + delta * 100)
    elif delta < 0:
        return -INTRA_LOSS_SCALE * math.log(1 + abs(delta) * 100)
    return 0.0


def drawdown_penalty(current_drawdown: float) -> float:
    """
    Linear + quadratic drawdown penalty.
    Zero below DRAWDOWN_THRESHOLD (5%).
    Formula: -linear_k * excess - quad_k * excess^2
    """
    if current_drawdown <= DRAWDOWN_THRESHOLD:
        return 0.0
    excess = current_drawdown - DRAWDOWN_THRESHOLD
    return -(DRAWDOWN_LINEAR_K * excess + DRAWDOWN_QUAD_K * excess ** 2)


def funding_cost(
    notional: float,
    rate: float,
    balance: float,
) -> float:
    """
    Funding cost as fraction of balance.
    Always negative (cost to the trader).
    """
    if notional == 0 or rate == 0:
        return 0.0
    return -(abs(rate) * notional / max(balance, 1e-8))


def killswitch_penalty(drawdown: float, scale: float = KILLSWITCH_SCALE) -> float:
    """Catastrophic penalty when account drawdown triggers kill-switch."""
    return -scale * (1.0 + drawdown)


def holding_cost(
    bars_in_trade: int,
    position: int,
    cost_per_bar: float = HOLDING_COST_PER_BAR,
) -> float:
    """Flat per-bar cost for holding a position (regardless of duration)."""
    if position == 0:
        return 0.0
    return cost_per_bar


def sl_hit_extra_penalty(penalty: float = SL_EXTRA_PENALTY) -> float:
    """Extra one-off penalty applied when stop-loss is hit (not voluntary close)."""
    return float(penalty)


def cost_penalty(
    notional: float,
    balance: float,
    commission: float = COMMISSION_RATE,
    slippage: float = SLIPPAGE_RATE,
) -> float:
    """Commission + slippage cost as fraction of balance."""
    if notional == 0:
        return 0.0
    return -((commission + slippage) * notional / max(balance, 1e-8))


# ---------------------------------------------------------------------------
# Composite per-step reward (used by environment + tests)
# ---------------------------------------------------------------------------

def compute_step_reward(
    position: int = 0,
    unrealized_pnl_pct: float = 0.0,
    bars_in_trade: int = 0,
    current_drawdown: float = 0.0,
    trade_closed: bool = False,
    realized_pnl_pct: float = 0.0,
    transaction_cost: float = 0.0,
    funding_fee: float = 0.0,
    kill_triggered: bool = False,
    sl_hit: bool = False,
    # Legacy / env kwargs
    action: int = 0,
    prev_position: int = 0,
    new_position: int = None,
    realised_pnl_pct: float = None,
    unrealised_pct: float = None,
    equity: float = 1.0,
    state: Optional[RewardState] = None,
    bull_ob_dist: float = 0.0,
    bear_ob_dist: float = 0.0,
    snr_support_1: float = 0.0,
    snr_resist_1: float = 0.0,
**kwargs
) -> float:
    # 别名处理
    if kwargs.get('realised_pnl_pct') is not None:
        realized_pnl_pct = kwargs['realised_pnl_pct']
        trade_closed = abs(realized_pnl_pct) > 1e-9
    if kwargs.get('unrealised_pct') is not None:
        unrealized_pnl_pct = kwargs['unrealised_pct']

    reward = 0.0

    # 1. Kill-switch (最高优先级)
    if kill_triggered:
        return float(np.clip(killswitch_penalty(current_drawdown), -200.0, 5.0))

    # 2. 分情况讨论：平仓 vs 持仓 vs 空仓
    if trade_closed:
        # 平仓瞬间：只计入交易收益和止损罚金，不计入时间惩罚
        reward += trade_reward(realized_pnl_pct)
        reward += transaction_cost
        if sl_hit:
            reward += sl_hit_extra_penalty()
    elif position != 0:
        # 正在持仓：PnL 波动 + 持仓成本 + 资金费
        reward += step_reward(unrealized_pnl_pct, position, bars_in_trade)
        reward += holding_cost(bars_in_trade, position)
        reward += funding_fee
    else:
        # 纯空仓：只扣除空仓惩罚
        reward -= HOLD_PENALTY

    # 3. 回撤惩罚 (始终存在)
    reward += drawdown_penalty(current_drawdown)

    return float(np.clip(reward, -200.0, 5.0))




# ---------------------------------------------------------------------------
# Original full compute_reward (used by BinanceEnv directly)
# ---------------------------------------------------------------------------

def compute_reward(
    *,
    action: int,
    prev_position: int,
    new_position: int,
    realised_pnl_pct: float,
    unrealised_pct: float,
    equity: float,
    state: RewardState,
    bull_ob_dist: float = 0.0,
    bear_ob_dist: float = 0.0,
    snr_support_1: float = 0.0,
    snr_resist_1: float = 0.0,
) -> float:

    state.step += 1
    state.update_peak(equity)
    reward = 0.0

    # Transaction costs
    if new_position != 0:
        reward -= (COMMISSION_RATE + SLIPPAGE_RATE) * abs(new_position)
    if state.step % FUNDING_STEPS == 0 and new_position != 0:
        reward -= FUNDING_RATE

    # Entry quality bonus
    entering_long  = (prev_position == 0 and new_position ==  1)
    entering_short = (prev_position == 0 and new_position == -1)

    if entering_long:
        state.total_trades += 1
        near_bull_ob  = 0.0 < bull_ob_dist  < ENTRY_QUALITY_THRESH
        near_support  = 0.0 < snr_support_1 < ENTRY_QUALITY_THRESH
        if near_bull_ob or near_support:
            reward += ENTRY_QUALITY_BONUS
        else:
            reward -= ENTRY_QUALITY_BONUS * 0.3

    if entering_short:
        state.total_trades += 1
        near_bear_ob   = 0.0 < bear_ob_dist < ENTRY_QUALITY_THRESH
        near_resistance = 0.0 < snr_resist_1 < ENTRY_QUALITY_THRESH
        if near_bear_ob or near_resistance:
            reward += ENTRY_QUALITY_BONUS
        else:
            reward -= ENTRY_QUALITY_BONUS * 0.3

    # Realised PnL reward
    if abs(realised_pnl_pct) > 1e-6:
        mag = math.log(1 + abs(realised_pnl_pct) * 100)
        if realised_pnl_pct > 0:
            reward += WIN_SCALE  * mag
        else:
            reward -= LOSS_SCALE * mag
        state.prev_unrealised_pct = 0.0

    # Intra-trade unrealised-delta shaping
    elif new_position != 0:
        delta = unrealised_pct - state.prev_unrealised_pct
        if delta > 0:
            reward += INTRA_WIN_SCALE  * math.log(1 + delta * 100)
        elif delta < 0:
            reward -= INTRA_LOSS_SCALE * math.log(1 + abs(delta) * 100)
        state.prev_unrealised_pct = unrealised_pct

    else:
        reward -= HOLD_PENALTY
        state.prev_unrealised_pct = 0.0

    # Drawdown penalty (linear+quadratic, capped)
    dd = max(0.0, (state.peak_equity - equity) / (state.peak_equity + 1e-8))
    reward += drawdown_penalty(dd)

    return float(np.clip(reward, -5.0, 5.0))
