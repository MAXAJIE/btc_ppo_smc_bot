"""
reward.py
─────────
Pure reward calculations — no env state, fully unit-testable.

Reward philosophy
─────────────────
• Wins    : R =  scale     × log(1 + |pnl_pct| × 100)
• Losses  : R = -penalty   × log(1 + |pnl_pct| × 100)   ← 2× amplification
• Holding : tiny time-decay (flat) + per-bar holding cost (in-trade)  [B]
• Kill-sw : R = -kill_penalty_scale × (1 + drawdown)                  [A]
• Costs   : commission + estimated slippage deducted at open/close

Changes vs v1
─────────────
A. killswitch_penalty  — massive proportional punishment on ruin event
B. holding_cost_per_bar — per-bar cost while in position (deters dead-holds)
"""

import numpy as np
import yaml
import os


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_CFG = None


def _r():
    global _CFG
    if _CFG is None:
        _CFG = _load_cfg()["reward"]
    return _CFG


# ─────────────────────────────────────────────────────────────────────────────
# A. Kill-switch punitive penalty
# ─────────────────────────────────────────────────────────────────────────────

def killswitch_penalty(current_drawdown: float, scale: float = None) -> float:
    """
    Fired exactly once when the account kill-switch triggers.

    Formula:  R = -scale × (1 + drawdown)

    At scale=50 and drawdown=0.30:  R = -50 × 1.30 = -65
    At scale=50 and drawdown=0.15:  R = -50 × 1.15 = -57.5

    This is orders of magnitude larger than a normal per-step reward
    (~±0.7 for a trade).  PPO will learn that ruin is an absolute
    catastrophe, not just a large loss — forcing the agent to treat
    drawdown avoidance as a hard constraint rather than a soft preference.

    Parameters
    ----------
    current_drawdown : float
        Account drawdown from peak at the moment of firing [0, 1].
    scale : float | None
        Override config value.
    """
    cfg = _r()
    s = scale if scale is not None else cfg.get("kill_penalty_scale", 50.0)
    penalty = -s * (1.0 + float(current_drawdown))
    return float(penalty)


# ─────────────────────────────────────────────────────────────────────────────
# B. Per-bar holding cost
# ─────────────────────────────────────────────────────────────────────────────

def holding_cost(
    bars_in_trade: int,
    position: int,
    cost_per_bar: float = None,
) -> float:
    """
    Deducted every 5m bar while a position is OPEN.

    Rationale
    ---------
    Without this, the agent can learn to open a trade and then
    ignore SMC exit signals — essentially "dead-holding" through
    adverse moves, hoping for mean-reversion before the kill-switch.

    The cost is flat (not cumulative) so it doesn't compound into
    a dominant signal that overrides directional edge.  At default
    -0.0005/bar, a 2-day hold (576 bars) costs only -0.288 in total
    reward — meaningful but not trade-crushing.

    Parameters
    ----------
    bars_in_trade : int
        How many 5m bars the current position has been open.
    position : int
        -1, 0, or +1.  Cost only applied when position != 0.
    cost_per_bar : float | None
        Override config value.  Should be negative (e.g. -0.0005).
    """
    if position == 0:
        return 0.0

    cfg = _r()
    cpb = cost_per_bar if cost_per_bar is not None else cfg.get("holding_cost_per_bar", -0.0005)
    return float(cpb)  # flat per-bar cost, same regardless of bars_in_trade


# ─────────────────────────────────────────────────────────────────────────────
# Existing reward components (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def trade_reward(pnl_pct: float, win_scale: float = None, loss_penalty: float = None) -> float:
    """
    Reward on trade close.

    pnl_pct : float  — realised PnL as fraction of entry notional (after fees).
    """
    cfg = _r()
    ws = win_scale if win_scale is not None else cfg["win_scale"]
    lp = loss_penalty if loss_penalty is not None else cfg["loss_penalty"]

    x = abs(pnl_pct) * 100.0
    log_val = np.log(1.0 + x)

    if pnl_pct >= 0:
        return float(ws * log_val)
    else:
        return float(-lp * log_val)


def step_reward(
    unrealized_pnl_pct: float,
    position: int,
    bars_in_trade: int,
    hold_penalty: float = None,
) -> float:
    """
    Per-step shaping signal based on unrealised P&L direction.

    Note: this fires for ALL steps regardless of position.
    The separate holding_cost() only fires while in-trade.
    """
    cfg = _r()
    hp = hold_penalty if hold_penalty is not None else cfg["step_hold_penalty"]

    if position == 0:
        return hp  # flat: small negative to discourage never trading

    x = abs(unrealized_pnl_pct) * 100.0
    log_val = np.log(1.0 + x)

    if unrealized_pnl_pct >= 0:
        step_r = 0.05 * log_val
    else:
        step_r = -0.10 * log_val

    return float(step_r + hp)


def cost_penalty(notional: float, balance: float) -> float:
    """One-way transaction cost as a fraction of balance."""
    cfg = _r()
    cost_rate = cfg["commission_rate"] + cfg["slippage_rate"]
    return -float(cost_rate * notional / balance)


def drawdown_penalty(current_drawdown: float) -> float:
    """
    Quadratic ramp penalty as drawdown grows toward the kill threshold.
    Returns 0 below 5%, ramps steeply above.
    """
    if current_drawdown < 0.05:
        return 0.0
    excess = current_drawdown - 0.05
    return -float(5.0 * (excess ** 2))


def funding_cost(position_size_usdt: float, funding_rate: float, balance: float) -> float:
    """Funding fee — deducted every 8h (480 × 5m steps)."""
    cost = position_size_usdt * abs(funding_rate)
    return -float(cost / balance)


# ─────────────────────────────────────────────────────────────────────────────
# Master composite reward — called by the environment every step
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    *,
    position: int,
    unrealized_pnl_pct: float,
    bars_in_trade: int,
    current_drawdown: float,
    trade_closed: bool,
    realized_pnl_pct: float,
    transaction_cost: float,
    funding_fee: float,
    # ── NEW parameters ──────────────────────────────────────────────
    kill_triggered: bool = False,   # [A] True on the step kill-switch fires
) -> float:
    """
    Master reward function called once per environment step.

    Component order and magnitude guide
    ────────────────────────────────────
    kill_triggered   → -50 to -65         (overwhelms everything — once only)
    trade_closed win → +0.3 to +3.5       (log-scaled pnl)
    trade_closed loss→ -0.6 to -7.0       (2× amplified)
    holding_cost     → -0.0005 / bar      [B] new
    step_reward      → ±0 to ±0.3         (unrealised shaping)
    drawdown_penalty → 0 to -0.5          (quadratic ramp)
    transaction_cost → -0.001 to -0.006   (per open/close)
    funding_fee      → -0.001             (every 8h)
    """
    r = 0.0

    # ── [A] Kill-switch punitive reward — fires first, dominates all ──
    if kill_triggered:
        r += killswitch_penalty(current_drawdown)
        # Still add the trade close reward if a position was just closed
        # by the kill-switch — but it will be dwarfed by the penalty above
        if trade_closed:
            r += trade_reward(realized_pnl_pct)
        return float(r)   # short-circuit: nothing else matters this step

    # ── 1. Trade close reward (dominant signal on normal steps) ──────
    if trade_closed:
        r += trade_reward(realized_pnl_pct)

    # ── 2. Per-step unrealised P&L shaping ───────────────────────────
    r += step_reward(unrealized_pnl_pct, position, bars_in_trade)

    # ── 3. [B] Holding cost — per-bar cost while in position ─────────
    r += holding_cost(bars_in_trade, position)

    # ── 4. Transaction costs (already negative, 0 if no trade) ───────
    r += transaction_cost

    # ── 5. Funding fee (already negative, 0 most steps) ──────────────
    r += funding_fee

    # ── 6. Drawdown quadratic ramp ───────────────────────────────────
    r += drawdown_penalty(current_drawdown)

    return float(r)
