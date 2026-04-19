"""
tests/test_reward.py
─────────────────────
Unit tests for src/utils/reward.py.
No network, no Binance, no GPU required.

FIXES:
  - All imports corrected to match functions now present in reward.py:
      trade_reward, step_reward, cost_penalty, drawdown_penalty,
      funding_cost, compute_step_reward, killswitch_penalty,
      holding_cost, sl_hit_extra_penalty

Run:
    pytest tests/test_reward.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np

from src.utils.reward import (
    trade_reward,
    step_reward,
    cost_penalty,
    drawdown_penalty,
    funding_cost,
    compute_step_reward,
    killswitch_penalty,
    holding_cost,
    sl_hit_extra_penalty,
)


# ─────────────────────────────────────────────────────────────────────────────
# trade_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeReward:

    def test_win_is_positive(self):
        r = trade_reward(pnl_pct=0.02)
        assert r > 0, f"Expected positive reward for win, got {r}"

    def test_loss_is_negative(self):
        r = trade_reward(pnl_pct=-0.02)
        assert r < 0, f"Expected negative reward for loss, got {r}"

    def test_loss_magnitude_greater_than_win(self):
        win = trade_reward(pnl_pct=0.02)
        loss = trade_reward(pnl_pct=-0.02)
        assert abs(loss) > abs(win)

    def test_loss_penalty_exactly_scale(self):
        win  = trade_reward(pnl_pct=0.05, win_scale=1.0, loss_penalty=2.0)
        loss = trade_reward(pnl_pct=-0.05, win_scale=1.0, loss_penalty=2.0)
        ratio = abs(loss) / abs(win)
        assert abs(ratio - 2.0) < 1e-9, f"Expected ratio=2.0, got {ratio:.6f}"

    def test_zero_pnl(self):
        r = trade_reward(pnl_pct=0.0)
        assert r == 0.0

    def test_large_win_finite(self):
        r = trade_reward(pnl_pct=0.50)
        assert np.isfinite(r) and r > 0

    def test_large_loss_finite(self):
        r = trade_reward(pnl_pct=-0.50)
        assert np.isfinite(r) and r < 0

    def test_log_scaling(self):
        r1 = trade_reward(pnl_pct=0.01)
        r2 = trade_reward(pnl_pct=0.02)
        assert r2 < 2 * r1, "Reward should be sublinear in PnL (log scale)"

    def test_custom_scales(self):
        r = trade_reward(pnl_pct=0.01, win_scale=3.0)
        r_default = trade_reward(pnl_pct=0.01, win_scale=1.0)
        assert r > r_default


# ─────────────────────────────────────────────────────────────────────────────
# step_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestStepReward:

    def test_flat_position_negative(self):
        r = step_reward(unrealized_pnl_pct=0.0, position=0, bars_in_trade=0)
        assert r < 0

    def test_winning_long_positive_component(self):
        r = step_reward(unrealized_pnl_pct=0.02, position=1, bars_in_trade=5)
        r_flat = step_reward(0.0, 0, 0)
        assert r > r_flat

    def test_losing_trade_more_negative_than_flat(self):
        r = step_reward(unrealized_pnl_pct=-0.02, position=-1, bars_in_trade=5)
        r_flat = step_reward(0.0, 0, 0)
        assert r < r_flat

    def test_hold_penalty_small_negative(self):
        r = step_reward(0.0, 0, 0)
        assert -0.01 < r < 0


# ─────────────────────────────────────────────────────────────────────────────
# cost_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestCostPenalty:

    def test_always_negative(self):
        r = cost_penalty(notional=1000.0, balance=10000.0)
        assert r < 0

    def test_proportional_to_notional(self):
        r1 = cost_penalty(1000.0, 10000.0)
        r2 = cost_penalty(2000.0, 10000.0)
        assert abs(r2) == pytest.approx(abs(r1) * 2, rel=1e-5)

    def test_zero_notional(self):
        r = cost_penalty(0.0, 10000.0)
        assert r == 0.0

    def test_rate_is_commission_plus_slippage(self):
        r = cost_penalty(1000.0, 10000.0)
        expected = -(0.0004 + 0.0002) * 1000.0 / 10000.0
        assert r == pytest.approx(expected, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# drawdown_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownPenalty:

    def test_zero_for_small_drawdown(self):
        assert drawdown_penalty(0.0) == 0.0
        assert drawdown_penalty(0.04) == 0.0
        assert drawdown_penalty(0.049) == 0.0

    def test_negative_above_threshold(self):
        r = drawdown_penalty(0.10)
        assert r < 0

    def test_increases_with_drawdown(self):
        r1 = drawdown_penalty(0.08)
        r2 = drawdown_penalty(0.12)
        assert r2 < r1

    def test_linear_component_exact(self):
        """At dd=0.08: excess=0.03, linear=-0.30*0.03, quad=-5*(0.03)^2."""
        r = drawdown_penalty(0.08)
        expected = -0.30 * 0.03 + -5.0 * (0.03 ** 2)
        assert r == pytest.approx(expected, rel=1e-6)

    def test_strictly_increasing_with_drawdown(self):
        dds = [0.05, 0.07, 0.09, 0.11, 0.13, 0.14]
        penalties = [drawdown_penalty(d) for d in dds]
        for i in range(1, len(penalties)):
            assert penalties[i] < penalties[i-1]


# ─────────────────────────────────────────────────────────────────────────────
# funding_cost
# ─────────────────────────────────────────────────────────────────────────────

class TestFundingCost:

    def test_always_negative(self):
        r = funding_cost(10000.0, 0.0001, 10000.0)
        assert r < 0

    def test_zero_position(self):
        r = funding_cost(0.0, 0.0001, 10000.0)
        assert r == 0.0

    def test_zero_rate(self):
        r = funding_cost(10000.0, 0.0, 10000.0)
        assert r == 0.0

    def test_negative_rate_same_cost(self):
        r_pos = funding_cost(10000.0, 0.0001, 10000.0)
        r_neg = funding_cost(10000.0, -0.0001, 10000.0)
        assert r_pos == pytest.approx(r_neg, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# killswitch_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestKillswitchPenalty:

    def test_always_negative(self):
        for dd in [0.10, 0.15, 0.20, 0.30, 0.50]:
            r = killswitch_penalty(dd)
            assert r < 0

    def test_proportional_to_drawdown(self):
        r_low  = killswitch_penalty(0.15)
        r_high = killswitch_penalty(0.30)
        assert r_high < r_low

    def test_formula_exact(self):
        scale = 50.0
        dd    = 0.30
        expected = -scale * (1.0 + dd)
        result   = killswitch_penalty(dd, scale=scale)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_magnitude_dominates_trade_reward(self):
        best_trade = trade_reward(pnl_pct=0.10)
        penalty    = killswitch_penalty(0.15)
        assert abs(penalty) > abs(best_trade) * 5

    def test_custom_scale(self):
        r1 = killswitch_penalty(0.20, scale=50.0)
        r2 = killswitch_penalty(0.20, scale=100.0)
        assert abs(r2) == pytest.approx(abs(r1) * 2, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# holding_cost
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldingCost:

    def test_zero_when_flat(self):
        assert holding_cost(bars_in_trade=0, position=0) == 0.0
        assert holding_cost(bars_in_trade=50, position=0) == 0.0

    def test_negative_when_in_trade(self):
        r = holding_cost(bars_in_trade=1, position=1)
        assert r < 0
        r = holding_cost(bars_in_trade=1, position=-1)
        assert r < 0

    def test_flat_per_bar(self):
        r1 = holding_cost(1,   position=1)
        r2 = holding_cost(100, position=1)
        r3 = holding_cost(500, position=1)
        assert r1 == pytest.approx(r2, rel=1e-6)
        assert r1 == pytest.approx(r3, rel=1e-6)

    def test_custom_cost(self):
        r = holding_cost(1, 1, cost_per_bar=-0.001)
        assert r == pytest.approx(-0.001, rel=1e-6)

    def test_small_enough_not_to_dominate(self):
        total_holding_cost = sum(holding_cost(i, 1) for i in range(576))
        typical_trade = trade_reward(0.02)
        assert abs(total_holding_cost) < abs(typical_trade) * 5


# ─────────────────────────────────────────────────────────────────────────────
# sl_hit_extra_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestSLHitPenalty:

    def test_returns_negative_float(self):
        r = sl_hit_extra_penalty()
        assert isinstance(r, float) and r < 0.0

    def test_default_value(self):
        r = sl_hit_extra_penalty()
        assert r == pytest.approx(-1.5, rel=1e-6)

    def test_custom_value(self):
        assert sl_hit_extra_penalty(-2.0) == pytest.approx(-2.0)
        assert sl_hit_extra_penalty(-0.5) == pytest.approx(-0.5)

    def test_sl_exit_worse_than_voluntary(self):
        voluntary = trade_reward(-0.03)
        sl_exit   = trade_reward(-0.03) + sl_hit_extra_penalty()
        assert sl_exit < voluntary
        gap = abs(sl_exit) - abs(voluntary)
        assert gap == pytest.approx(1.5, rel=1e-4)

    def test_sl_not_so_large_it_deters_all_trades(self):
        sl_loss = trade_reward(-0.03) + sl_hit_extra_penalty()
        win     = trade_reward(0.02)
        assert win * 3 > abs(sl_loss)


# ─────────────────────────────────────────────────────────────────────────────
# compute_step_reward (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeStepReward:

    def _base(self, **kw):
        d = dict(
            position=0, unrealized_pnl_pct=0.0, bars_in_trade=0,
            current_drawdown=0.0, trade_closed=False, realized_pnl_pct=0.0,
            transaction_cost=0.0, funding_fee=0.0,
        )
        d.update(kw)
        return d

    def test_flat_hold_negative(self):
        r = compute_step_reward(**self._base())
        assert r < 0

    def test_win_trade_close_positive(self):
        r = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=0.03, transaction_cost=-0.001,
        ))
        assert r > 0

    def test_loss_trade_close_negative(self):
        r = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03, transaction_cost=-0.001,
        ))
        assert r < 0

    def test_high_drawdown_amplifies_loss(self):
        r_low_dd = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.02, current_drawdown=0.03,
        ))
        r_high_dd = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.02, current_drawdown=0.12,
        ))
        assert r_high_dd < r_low_dd

    def test_output_is_scalar_float(self):
        r = compute_step_reward(**self._base())
        assert isinstance(r, float) and np.isfinite(r)

    def test_funding_fee_reduces_reward(self):
        r_no  = compute_step_reward(**self._base(position=1, unrealized_pnl_pct=0.01))
        r_fee = compute_step_reward(**self._base(position=1, unrealized_pnl_pct=0.01, funding_fee=-0.01))
        assert r_fee < r_no

    def test_kill_triggered_dominates(self):
        r_kill = compute_step_reward(**self._base(kill_triggered=True, current_drawdown=0.30))
        r_norm = compute_step_reward(**self._base(trade_closed=True, realized_pnl_pct=-0.10))
        assert r_kill < r_norm

    def test_sl_hit_worsens_reward(self):
        r_vol = compute_step_reward(**self._base(trade_closed=True, realized_pnl_pct=-0.03))
        r_sl  = compute_step_reward(**self._base(trade_closed=True, realized_pnl_pct=-0.03, sl_hit=True))
        assert r_sl < r_vol
        assert r_vol - r_sl == pytest.approx(1.5, rel=1e-4)

    def test_sl_hit_only_on_trade_close(self):
        r_sl_no_close    = compute_step_reward(**self._base(sl_hit=True))
        r_no_sl_no_close = compute_step_reward(**self._base())
        assert r_sl_no_close == pytest.approx(r_no_sl_no_close, rel=1e-8)

    def test_dd_above_5pct_penalises(self):
        r_above = compute_step_reward(**self._base(current_drawdown=0.08))
        r_below = compute_step_reward(**self._base(current_drawdown=0.04))
        assert r_above < r_below

    def test_holding_cost_deducted_when_in_trade(self):
        r_in  = compute_step_reward(**self._base(position=1,  unrealized_pnl_pct=0.0))
        r_out = compute_step_reward(**self._base(position=0,  unrealized_pnl_pct=0.0))
        assert r_in < r_out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
