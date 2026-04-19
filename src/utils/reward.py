"""
reward.py
─────────
Pure reward calculations — no env state, fully unit-testable.

Reward philosophy
─────────────────
• Wins    :  R =  scale     × log(1 + |pnl_pct| × 100)
• Losses  :  R = -penalty   × log(1 + |pnl_pct| × 100)   ← 2× amplification
• SL hit  :  R += sl_hit_extra_penalty                    [1] new — extra sting
• DD/step :  R += -dd_scale × max(0, dd - threshold)      [1] new — linear ramp
• Holding :  flat per-bar cost while in-trade              [B]
• Kill-sw :  R  = -kill_scale × (1 + drawdown)            [A] overwhelms all

Changes vs v2
─────────────
1. drawdown_penalty() — now two-component:
     linear proportional above 5% (fires every step, tiny)
   + quadratic acceleration near kill threshold
2. sl_hit_extra_penalty() — additional fixed penalty on SL trigger
   (forces PPO: "entering a bad trade and getting SL'd is worse than
    voluntarily exiting at the same loss")
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
# [A] Kill-switch punitive penalty  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def killswitch_penalty(current_drawdown: float, scale: float = None) -> float:
    """
    Fired exactly once when the account kill-switch triggers.
    R = -scale × (1 + drawdown)

    At scale=50 and drawdown=0.30 → R = -65.
    Overwhelms all other reward components — ruin is never worth it.
    """
    cfg = _r()
    s = scale if scale is not None else cfg.get("kill_penalty_scale", 50.0)
    return float(-s * (1.0 + float(current_drawdown)))


# ─────────────────────────────────────────────────────────────────────────────
# [B] Per-bar holding cost  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def holding_cost(bars_in_trade: int, position: int, cost_per_bar: float = None) -> float:
    """Flat negative reward per bar while a position is open."""
    if position == 0:
        return 0.0
    cfg = _r()
    cpb = cost_per_bar if cost_per_bar is not None else cfg.get("holding_cost_per_bar", -0.0005)
    return float(cpb)


# ─────────────────────────────────────────────────────────────────────────────
# [1] Per-step drawdown penalty — new in v3
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_penalty(current_drawdown: float) -> float:
    """
    Two-component penalty applied every step.

    Component 1 — Linear proportional  (new)
    ─────────────────────────────────────────
    Fires every step once drawdown > threshold (default 5%).
    R₁ = -dd_per_step_scale × (dd - threshold)

    Magnitude guide (default scale=0.30):
      dd =  6% → -0.30 × 0.01 = -0.003   (nearly silent)
      dd = 10% → -0.30 × 0.05 = -0.015   (gentle nudge)
      dd = 14% → -0.30 × 0.09 = -0.027   (increasingly loud)

    This is intentionally tiny — it's a continuous pressure, not a shock.
    The agent will feel it accumulate over hundreds of steps of bad behaviour.

    Component 2 — Quadratic acceleration  (unchanged from v2)
    ────────────────────────────────────────────────────────
    R₂ = -5 × (excess²)     where excess = max(0, dd - 0.05)

    Near the kill threshold, R₂ dominates, amplifying the urgency.
    At dd=14%: R₂ = -5 × 0.09² = -0.0405
    """
    cfg = _r()
    threshold = float(cfg.get("drawdown_linear_threshold", 0.05))
    lin_scale  = float(cfg.get("drawdown_per_step_scale",  0.30))

    excess = max(0.0, current_drawdown - threshold)

    r_linear    = -lin_scale * excess                 # component 1
    r_quadratic = -5.0 * (excess ** 2)                # component 2

    return float(r_linear + r_quadratic)


# ─────────────────────────────────────────────────────────────────────────────
# [1] SL-hit extra penalty — new in v3
# ─────────────────────────────────────────────────────────────────────────────

def sl_hit_extra_penalty(penalty: float = None) -> float:
    """
    Additional fixed penalty applied when a stop-loss order is triggered.

    This fires ON TOP OF the normal trade_reward(loss), making SL hits
    meaningfully worse than a voluntary exit at the same PnL.

    Why this matters
    ────────────────
    Without this, hitting a 3% SL and voluntarily closing at -3% have
    identical reward.  The agent has no incentive to exit before SL fires.
    With this penalty, the agent learns: "if my SMC setup is invalidated,
    close before the hard SL — because letting it get hit costs extra."

    Magnitude calibration (default -1.5):
      Voluntary -3% exit:   trade_reward(-0.03) ≈ -2.77
      SL at -3%:            trade_reward(-0.03) + sl_extra ≈ -2.77 + -1.50 = -4.27
      Ratio:                1.55× worse than voluntary exit

    Do NOT set so large that the agent never opens trades to avoid SL risk.
    -1.5 is calibrated so SL pain is real but doesn't dominate win signals.
    """
    cfg = _r()
    p = penalty if penalty is not None else cfg.get("sl_hit_extra_penalty", -1.5)
    return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Core trade & step components  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def trade_reward(pnl_pct: float, win_scale: float = None, loss_penalty: float = None) -> float:
    """Log-scaled reward on trade close. Losses penalised 2×."""
    cfg = _r()
    ws = win_scale    if win_scale    is not None else cfg["win_scale"]
    lp = loss_penalty if loss_penalty is not None else cfg["loss_penalty"]
    x = abs(pnl_pct) * 100.0
    log_val = np.log(1.0 + x)
    return float(ws * log_val) if pnl_pct >= 0 else float(-lp * log_val)


def step_reward(unrealized_pnl_pct: float, position: int,
                bars_in_trade: int, hold_penalty: float = None) -> float:
    """Tiny per-step shaping from unrealised P&L direction."""
    cfg = _r()
    hp = hold_penalty if hold_penalty is not None else cfg["step_hold_penalty"]
    if position == 0:
        return hp
    x = abs(unrealized_pnl_pct) * 100.0
    log_val = np.log(1.0 + x)
    step_r = 0.05 * log_val if unrealized_pnl_pct >= 0 else -0.10 * log_val
    return float(step_r + hp)


def cost_penalty(notional: float, balance: float) -> float:
    """One-way transaction cost (commission + slippage) as fraction of balance."""
    cfg = _r()
    rate = cfg["commission_rate"] + cfg["slippage_rate"]
    return -float(rate * notional / balance)


def funding_cost(position_size_usdt: float, funding_rate: float, balance: float) -> float:
    """Funding fee — deducted every 8h (480 × 5m steps)."""
    return -float(position_size_usdt * abs(funding_rate) / balance)


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
    kill_triggered: bool = False,    # [A]
    sl_hit: bool = False,            # [1] new
) -> float:
    """
    Master reward function called once per environment step.

    Priority / magnitude guide (v3)
    ────────────────────────────────────────────────────────────────
    kill_triggered   → -57 to -65         fires once, returns immediately
    ────────────────────────────────────────────────────────────────
    trade_close win  → +0.3  to +3.5
    trade_close loss → -0.6  to -7.0
    sl_hit_extra     → -1.5  (added on SL-triggered closes only)     [1]
    ────────────────────────────────────────────────────────────────
    holding_cost     → -0.0005 / bar      while in trade             [B]
    step_reward      → ±0    to ±0.3
    drawdown_linear  → -0    to -0.027    proportional above 5%      [1]
    drawdown_quad    → -0    to -0.04     quadratic acceleration
    tx cost          → -0.001 to -0.006
    funding fee      → -0.001 (every 8h)
    ────────────────────────────────────────────────────────────────
    """
    r = 0.0

    # ── [A] Kill-switch — fires first, short-circuits everything ──────
    if kill_triggered:
        r += killswitch_penalty(current_drawdown)
        if trade_closed:
            r += trade_reward(realized_pnl_pct)
        return float(r)

    # ── 1. Trade close reward ─────────────────────────────────────────
    if trade_closed:
        r += trade_reward(realized_pnl_pct)
        if sl_hit:
            r += sl_hit_extra_penalty()          # [1] extra sting for SL

    # ── 2. Per-step unrealised P&L shaping ───────────────────────────
    r += step_reward(unrealized_pnl_pct, position, bars_in_trade)

    # ── 3. [B] Holding cost ───────────────────────────────────────────
    r += holding_cost(bars_in_trade, position)

    # ── 4. Transaction costs ──────────────────────────────────────────
    r += transaction_cost

    # ── 5. Funding fee ────────────────────────────────────────────────
    r += funding_fee

    # ── 6. [1] Drawdown penalty (linear + quadratic) ─────────────────
    r += drawdown_penalty(current_drawdown)

    return float(r)
