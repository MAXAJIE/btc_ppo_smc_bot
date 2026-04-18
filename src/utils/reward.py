"""
reward.py
─────────
Pure reward calculations — no env state, fully unit-testable.

Reward philosophy
─────────────────
• Wins    : R =  scale     × log(1 + |pnl_pct| × 100)
• Losses  : R = -penalty   × log(1 + |pnl_pct| × 100)   ← 2× amplification
• Holding : tiny time-decay to discourage endless open positions
• Costs   : commission + estimated slippage deducted at open/close

The log transformation keeps large wins from dominating the gradient
while the 2× multiplier on losses trains the agent to cut losers fast.
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
# Core reward functions
# ─────────────────────────────────────────────────────────────────────────────

def trade_reward(pnl_pct: float, win_scale: float = None, loss_penalty: float = None) -> float:
    """
    Reward on trade close.

    Parameters
    ----------
    pnl_pct : float
        Realised PnL as a fraction of entry notional (after fees).
        e.g. 0.02 = +2%, -0.015 = -1.5%

    Returns
    -------
    float : reward scalar
    """
    cfg = _r()
    ws = win_scale if win_scale is not None else cfg["win_scale"]
    lp = loss_penalty if loss_penalty is not None else cfg["loss_penalty"]

    x = abs(pnl_pct) * 100.0      # scale: 1% pnl → x=1 → log(2)≈0.69
    log_val = np.log(1.0 + x)

    if pnl_pct >= 0:
        return float(ws * log_val)
    else:
        return float(-lp * log_val)


def step_reward(
    unrealized_pnl_pct: float,
    position: int,          # -1 short, 0 flat, 1 long
    bars_in_trade: int,
    hold_penalty: float = None,
) -> float:
    """
    Small per-step reward to shape intra-episode behaviour.

    • Flat position         → tiny negative (encourage acting)
    • Holding winning trade → tiny positive (log-scaled)
    • Holding losing trade  → tiny negative (log-scaled, amplified)

    Intentionally small so it doesn't dominate trade_reward.
    """
    cfg = _r()
    hp = hold_penalty if hold_penalty is not None else cfg["step_hold_penalty"]

    if position == 0:
        return hp  # flat: small negative to discourage never trading

    x = abs(unrealized_pnl_pct) * 100.0
    log_val = np.log(1.0 + x)

    if unrealized_pnl_pct >= 0:
        step_r = 0.05 * log_val      # very small positive
    else:
        step_r = -0.10 * log_val     # small negative, double weight for losses

    return float(step_r + hp)        # always include base time decay


def cost_penalty(notional: float, balance: float) -> float:
    """
    One-way transaction cost as a fraction of balance.
    Called once on open and once on close.

    cost = (commission + slippage) × notional
    """
    cfg = _r()
    cost_rate = cfg["commission_rate"] + cfg["slippage_rate"]
    return -float(cost_rate * notional / balance)


def drawdown_penalty(current_drawdown: float, kill_threshold: float = 0.15) -> float:
    """
    Additional penalty as drawdown approaches the kill-switch threshold.
    Quadratic ramp so the agent learns to fear large drawdowns.

    Returns 0 until drawdown > 5%, then ramps up steeply.
    """
    if current_drawdown < 0.05:
        return 0.0
    excess = current_drawdown - 0.05
    return -float(5.0 * (excess ** 2))


def funding_cost(position_size_usdt: float, funding_rate: float, balance: float) -> float:
    """
    Funding fee deducted every 8h (480 5m steps).

    funding_rate: typical BTCUSDT rate ≈ 0.0001 (0.01%) per 8h.
    Positive rate → longs pay shorts. Negative → shorts pay longs.
    Agent pays if aligned with the dominant side.
    """
    cost = position_size_usdt * abs(funding_rate)
    return -float(cost / balance)


# ─────────────────────────────────────────────────────────────────────────────
# Composite: called by the environment
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
) -> float:
    """
    Master reward function called once per environment step.

    Parameters
    ----------
    position : int
        Current position direction after action: -1, 0, +1
    unrealized_pnl_pct : float
        Unrealized PnL / entry notional (may be 0 if flat)
    bars_in_trade : int
        How many 5m bars the current trade has been open
    current_drawdown : float
        Account drawdown from peak (0.0 → 1.0)
    trade_closed : bool
        Whether a trade was just closed this step
    realized_pnl_pct : float
        If trade_closed, the realised PnL fraction (after fees)
    transaction_cost : float
        Pre-computed cost penalty (already negative)
    funding_fee : float
        Pre-computed funding fee (already negative, 0 if not funding step)
    """
    r = 0.0

    # 1. Trade close reward (dominant signal)
    if trade_closed:
        r += trade_reward(realized_pnl_pct)

    # 2. Per-step holding signal
    r += step_reward(unrealized_pnl_pct, position, bars_in_trade)

    # 3. Transaction costs (already negative)
    r += transaction_cost

    # 4. Funding (already negative)
    r += funding_fee

    # 5. Drawdown penalty
    r += drawdown_penalty(current_drawdown)

    return float(r)
