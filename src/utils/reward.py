"""
reward.py  —  Swing-trading reward function (Full Integrated Version)
====================================================================
保留了原始代码中所有的 Feature、注释和辅助函数。
新增：Trailing Stop Penalty (保护浮盈) 与 Smart Exit Bonus (结构位平仓奖励)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Constants — 原有参数全部保留，仅微调 REWARD_CLIP 以适配指数奖励
# ---------------------------------------------------------------------------

ENTRY_TAX            = 0.15    # 战略降频：强制模型筛选高价值入场
SURVIVAL_BONUS       = 0.001   # 鼓励持仓信号
INTRA_WIN_SCALE      = 0.002   # 极小的浮盈奖励
INTRA_LOSS_SCALE     = 0.005   # 极小的浮亏惩罚

# Profit ladder thresholds (利润阶梯)
PROFIT_TARGET_PCT    = 0.03    # 3% 盈利目标
WIN_SCALE_ABOVE      = 1.5     # 超过目标后的指数奖励系数
WIN_SCALE_BELOW      = 10.0    # 低于目标的线性奖励
LOSS_SCALE           = 150.0   # 亏损惩罚

HOLD_PENALTY         = 0.00001 # 空仓惩罚

ENTRY_QUALITY_BONUS  = 0.20    # 结构位入场奖励
ENTRY_QUALITY_THRESH = 0.5     # ATR 归一化距离阈值

DRAWDOWN_THRESHOLD   = 0.05
DRAWDOWN_LINEAR_K    = 0.20
DRAWDOWN_CAP         = 0.30

COMMISSION_RATE      = 0.0004
SLIPPAGE_RATE        = 0.0002
FUNDING_RATE         = 0.0001
FUNDING_STEPS        = 480      # 8小时资金费率周期 (5m K线)

REWARD_CLIP          = 15.0    # 放大裁剪范围，确保 PPO 能够识别 3:1 的数学优势

# --- 新增动态平仓控制参数 ---
TRAILING_STOP_START  = 0.02    # 浮盈达到 2% 时开启利润保护
MAX_RETRACE_ALLOWED  = 0.012   # 允许从最高点回撤 1.2%
SMART_EXIT_BONUS     = 0.25    # 在阻力/支撑位主动平仓奖励


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class RewardState:
    peak_equity:          float = 1.0
    prev_unrealised_pct:  float = 0.0
    peak_unrealised_pct:  float = 0.0  # 追踪单笔交易的最大浮盈点
    step:                 int   = 0
    total_trades:         int   = 0
    trade_hold_steps:     int   = 0

    def reset_trade(self):
        self.prev_unrealised_pct = 0.0
        self.peak_unrealised_pct = 0.0
        self.trade_hold_steps    = 0

    def update_peak(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity


# ---------------------------------------------------------------------------
# Primary reward function (核心重写部分)
# ---------------------------------------------------------------------------

def compute_reward(
    *,
    action:            int,
    prev_position:     int,        # -1 / 0 / 1
    new_position:      int,
    realised_pnl_pct:  float,      # 非零仅在平仓时
    unrealised_pct:    float,
    equity:            float,
    state:             RewardState,
    # SMC / SNR entry-quality context
    bull_ob_dist:  float = 0.0,
    bear_ob_dist:  float = 0.0,
    snr_support_1: float = 0.0,
    snr_resist_1:  float = 0.0,
) -> float:
    state.step += 1
    state.update_peak(equity)
    reward = 0.0

    entering = prev_position == 0 and new_position != 0
    exiting = prev_position != 0 and new_position == 0
    holding = new_position != 0 and not entering

    # ── 1. Transaction costs (交易成本 - 原有逻辑) ──────────────────────────
    if new_position != 0:
        reward -= COMMISSION_RATE + SLIPPAGE_RATE
    if state.step % FUNDING_STEPS == 0 and new_position != 0:
        reward -= FUNDING_RATE

    # ── 2. Entry Tax + Entry Quality Bonus (入场控制 - 原有逻辑) ─────────────
    if entering:
        state.reset_trade()
        state.total_trades += 1
        reward -= ENTRY_TAX

        # ENTRY QUALITY: 如果在结构位进场，返还部分进场税
        near_bull_ob  = 0.0 < bull_ob_dist  < ENTRY_QUALITY_THRESH
        near_support  = 0.0 < snr_support_1 < ENTRY_QUALITY_THRESH
        near_bear_ob  = 0.0 < bear_ob_dist  < ENTRY_QUALITY_THRESH
        near_resist   = 0.0 < snr_resist_1  < ENTRY_QUALITY_THRESH

        if new_position == 1 and (near_bull_ob or near_support):
            reward += ENTRY_QUALITY_BONUS
        elif new_position == -1 and (near_bear_ob or near_resist):
            reward += ENTRY_QUALITY_BONUS

    # ── 3. Realised PnL (利润阶梯奖励 + 新增 Smart Exit) ───────────────────
    if exiting:
        # A. 原有的 Profit Ladder 逻辑
        if realised_pnl_pct >= PROFIT_TARGET_PCT:
            reward += (realised_pnl_pct * 100) ** WIN_SCALE_ABOVE
        elif realised_pnl_pct > 0:
            reward += realised_pnl_pct * WIN_SCALE_BELOW
        else:
            reward -= abs(realised_pnl_pct) * LOSS_SCALE

        # B. 新增：Smart Exit Bonus (在反向结构位主动止盈奖励)
        at_resist  = (0.0 < bear_ob_dist < ENTRY_QUALITY_THRESH) or (0.0 < snr_resist_1 < ENTRY_QUALITY_THRESH)
        at_support = (0.0 < bull_ob_dist < ENTRY_QUALITY_THRESH) or (0.0 < snr_support_1 < ENTRY_QUALITY_THRESH)

        if (prev_position == 1 and at_resist) or (prev_position == -1 and at_support):
            if realised_pnl_pct > 0:
                reward += SMART_EXIT_BONUS

        state.reset_trade()

    # ── 4. Survival Bonus + Dynamic Trailing Penalty (持仓引导) ─────────────
    elif holding:
        state.trade_hold_steps += 1
        reward += SURVIVAL_BONUS

        # 更新最高浮盈
        if unrealised_pct > state.peak_unrealised_pct:
            state.peak_unrealised_pct = unrealised_pct

        # 新增：动态回撤惩罚 (保护利润，控制风险)
        if state.peak_unrealised_pct > TRAILING_STOP_START:
            retrace = state.peak_unrealised_pct - unrealised_pct
            if retrace > MAX_RETRACE_ALLOWED:
                reward -= 0.02 # 持续惩罚直到模型主动平仓

        # 原有的微量浮盈诱导逻辑
        delta = unrealised_pct - state.prev_unrealised_pct
        if delta > 0:
            reward += INTRA_WIN_SCALE  * math.log(1.0 + delta * 100.0)
        elif delta < 0:
            reward -= INTRA_LOSS_SCALE * math.log(1.0 + abs(delta) * 100.0)

        state.prev_unrealised_pct = unrealised_pct

    # ── 5. Flat / no-position penalty (原有逻辑) ──────────────────────────
    else:
        reward -= HOLD_PENALTY
        state.prev_unrealised_pct = 0.0

    # ── 6. Drawdown penalty (原有逻辑) ───────────────────────────────────
    dd = max(0.0, (state.peak_equity - equity) / (state.peak_equity + 1e-8))
    if dd > DRAWDOWN_THRESHOLD:
        reward -= min(DRAWDOWN_LINEAR_K * (dd - DRAWDOWN_THRESHOLD), DRAWDOWN_CAP)

    return float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))


# ---------------------------------------------------------------------------
# Backward-compatibility aliases — 还原被删除的辅助函数
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
    """原有别名函数，确保环境调用不报错。"""
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
    """原有独立测试函数：计算平仓时的静态奖励。"""
    if abs(realised_pnl_pct) < 1e-6:
        return 0.0
    if realised_pnl_pct >= PROFIT_TARGET_PCT:
        return (realised_pnl_pct * 100) ** WIN_SCALE_ABOVE
    elif realised_pnl_pct > 0:
        return realised_pnl_pct * WIN_SCALE_BELOW
    else:
        return -abs(realised_pnl_pct) * LOSS_SCALE


def cost_penalty(position: int = 1, step: int = 0) -> float:
    """原有独立测试函数：计算交易成本。"""
    cost = 0.0
    if position != 0:
        cost += COMMISSION_RATE + SLIPPAGE_RATE
    if step % FUNDING_STEPS == 0 and position != 0:
        cost += FUNDING_RATE
    return -cost