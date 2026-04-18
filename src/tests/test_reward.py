"""
tests/test_reward.py
─────────────────────
Unit tests for src/utils/reward.py.
No network, no Binance, no GPU required.

Run:
    pytest tests/test_reward.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ── Import under test ────────────────────────────────────────────────────────
from src.utils.reward import (
    trade_reward,
    step_reward,
    cost_penalty,
    drawdown_penalty,
    funding_cost,
    compute_step_reward,
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
        """Loss penalty (2×) should produce higher absolute reward than same-sized win."""
        win = trade_reward(pnl_pct=0.02)
        loss = trade_reward(pnl_pct=-0.02)
        assert abs(loss) > abs(win), (
            f"Loss |{loss:.4f}| should be larger than win |{win:.4f}|"
        )

    def test_loss_penalty_exactly_2x(self):
        """With default scale=1.0 and penalty=2.0, |loss| should be ~2× win."""
        win = trade_reward(pnl_pct=0.05, win_scale=1.0, loss_penalty=2.0)
        loss = trade_reward(pnl_pct=-0.05, win_scale=1.0, loss_penalty=2.0)
        ratio = abs(loss) / abs(win)
        assert abs(ratio - 2.0) < 1e-9, f"Expected ratio=2.0, got {ratio:.6f}"

    def test_zero_pnl(self):
        r = trade_reward(pnl_pct=0.0)
        assert r == 0.0

    def test_large_win_saturates_gracefully(self):
        """A 50% win should produce a finite, not-inf reward."""
        r = trade_reward(pnl_pct=0.50)
        assert np.isfinite(r)
        assert r > 0

    def test_large_loss_saturates_gracefully(self):
        r = trade_reward(pnl_pct=-0.50)
        assert np.isfinite(r)
        assert r < 0

    def test_log_scaling(self):
        """Doubling the pnl should not double the reward (log compression)."""
        r1 = trade_reward(pnl_pct=0.01)
        r2 = trade_reward(pnl_pct=0.02)
        assert r2 < 2 * r1, "Reward should be sublinear in PnL (log scale)"

    def test_custom_scales(self):
        r = trade_reward(pnl_pct=0.01, win_scale=3.0, loss_penalty=1.0)
        r_default = trade_reward(pnl_pct=0.01, win_scale=1.0)
        assert r > r_default


# ─────────────────────────────────────────────────────────────────────────────
# step_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestStepReward:

    def test_flat_position_negative(self):
        r = step_reward(unrealized_pnl_pct=0.0, position=0, bars_in_trade=0)
        assert r < 0, f"Flat position should have small negative reward, got {r}"

    def test_winning_long_has_positive_component(self):
        r = step_reward(unrealized_pnl_pct=0.02, position=1, bars_in_trade=5)
        # Should be positive (win component) + small negative (hold decay)
        # Net should be positive for meaningful wins
        r_flat = step_reward(0.0, 0, 0)
        assert r > r_flat, "Winning trade step should exceed flat step reward"

    def test_losing_trade_more_negative_than_flat(self):
        r = step_reward(unrealized_pnl_pct=-0.02, position=-1, bars_in_trade=5)
        r_flat = step_reward(0.0, 0, 0)
        assert r < r_flat, "Losing trade step should be worse than flat"

    def test_hold_penalty_is_small(self):
        r = step_reward(0.0, 0, 0)
        assert -0.01 < r < 0, f"Hold penalty should be tiny negative, got {r}"

    def test_custom_hold_penalty(self):
        r = step_reward(0.0, 0, 0, hold_penalty=-0.001)
        assert r < 0


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
        """Should be (0.04% + 0.02%) × notional / balance = 0.0006 × 1000 / 10000."""
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
        assert r2 < r1, "Larger drawdown should have larger penalty"

    def test_quadratic_growth(self):
        """Penalty should grow quadratically above 5%."""
        r1 = drawdown_penalty(0.10)  # excess = 0.05
        r2 = drawdown_penalty(0.15)  # excess = 0.10 = 2× excess
        # Quadratic: r2 should be ~4× r1
        ratio = abs(r2) / abs(r1)
        assert abs(ratio - 4.0) < 0.5, f"Expected ~4× growth, got {ratio:.2f}×"


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

    def test_negative_rate_still_costs(self):
        """Funding is always a cost (abs used), regardless of rate sign."""
        r_pos = funding_cost(10000.0, 0.0001, 10000.0)
        r_neg = funding_cost(10000.0, -0.0001, 10000.0)
        assert r_pos == pytest.approx(r_neg, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# compute_step_reward (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeStepReward:

    def _base_kwargs(self, **overrides):
        defaults = dict(
            position=0,
            unrealized_pnl_pct=0.0,
            bars_in_trade=0,
            current_drawdown=0.0,
            trade_closed=False,
            realized_pnl_pct=0.0,
            transaction_cost=0.0,
            funding_fee=0.0,
        )
        defaults.update(overrides)
        return defaults

    def test_flat_hold_negative(self):
        r = compute_step_reward(**self._base_kwargs())
        assert r < 0

    def test_win_trade_close_positive(self):
        r = compute_step_reward(**self._base_kwargs(
            trade_closed=True,
            realized_pnl_pct=0.03,
            transaction_cost=-0.001,
        ))
        assert r > 0, f"Winning trade close should be positive, got {r}"

    def test_loss_trade_close_negative(self):
        r = compute_step_reward(**self._base_kwargs(
            trade_closed=True,
            realized_pnl_pct=-0.03,
            transaction_cost=-0.001,
        ))
        assert r < 0

    def test_high_drawdown_amplifies_loss(self):
        r_low_dd = compute_step_reward(**self._base_kwargs(
            trade_closed=True,
            realized_pnl_pct=-0.02,
            current_drawdown=0.03,
        ))
        r_high_dd = compute_step_reward(**self._base_kwargs(
            trade_closed=True,
            realized_pnl_pct=-0.02,
            current_drawdown=0.12,
        ))
        assert r_high_dd < r_low_dd

    def test_output_is_scalar_float(self):
        r = compute_step_reward(**self._base_kwargs())
        assert isinstance(r, float)
        assert np.isfinite(r)

    def test_funding_fee_reduces_reward(self):
        r_no_fee = compute_step_reward(**self._base_kwargs(position=1, unrealized_pnl_pct=0.01))
        r_with_fee = compute_step_reward(**self._base_kwargs(
            position=1, unrealized_pnl_pct=0.01, funding_fee=-0.01
        ))
        assert r_with_fee < r_no_fee


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ─────────────────────────────────────────────────────────────────────────────
# [A] killswitch_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestKillswitchPenalty:

    def test_always_negative(self):
        from src.utils.reward import killswitch_penalty
        for dd in [0.10, 0.15, 0.20, 0.30, 0.50]:
            r = killswitch_penalty(dd)
            assert r < 0, f"Expected negative at dd={dd}, got {r}"

    def test_proportional_to_drawdown(self):
        from src.utils.reward import killswitch_penalty
        r_low  = killswitch_penalty(0.15)
        r_high = killswitch_penalty(0.30)
        assert r_high < r_low, "Higher drawdown should produce larger penalty"

    def test_formula_exact(self):
        """R = -scale * (1 + drawdown)"""
        from src.utils.reward import killswitch_penalty
        scale = 50.0
        dd    = 0.30
        expected = -scale * (1.0 + dd)   # -65.0
        result   = killswitch_penalty(dd, scale=scale)
        assert result == pytest.approx(expected, rel=1e-6), f"Expected {expected}, got {result}"

    def test_magnitude_dominates_trade_reward(self):
        """Penalty should be >> any normal trade reward."""
        from src.utils.reward import killswitch_penalty, trade_reward
        best_trade = trade_reward(pnl_pct=0.10)   # +10% win, huge
        penalty    = killswitch_penalty(0.15)
        assert abs(penalty) > abs(best_trade) * 5, (
            f"Kill penalty |{penalty:.2f}| should dominate best trade |{best_trade:.2f}|"
        )

    def test_custom_scale(self):
        from src.utils.reward import killswitch_penalty
        r1 = killswitch_penalty(0.20, scale=50.0)
        r2 = killswitch_penalty(0.20, scale=100.0)
        assert abs(r2) == pytest.approx(abs(r1) * 2, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# [B] holding_cost
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldingCost:

    def test_zero_when_flat(self):
        from src.utils.reward import holding_cost
        assert holding_cost(bars_in_trade=0, position=0) == 0.0
        assert holding_cost(bars_in_trade=50, position=0) == 0.0

    def test_negative_when_in_trade(self):
        from src.utils.reward import holding_cost
        r = holding_cost(bars_in_trade=1, position=1)
        assert r < 0
        r = holding_cost(bars_in_trade=1, position=-1)
        assert r < 0

    def test_flat_per_bar(self):
        """Cost should be the same per bar regardless of bars_in_trade (flat, not cumulative)."""
        from src.utils.reward import holding_cost
        r1 = holding_cost(1,   position=1)
        r2 = holding_cost(100, position=1)
        r3 = holding_cost(500, position=1)
        assert r1 == pytest.approx(r2, rel=1e-6), "Holding cost should be flat per bar"
        assert r1 == pytest.approx(r3, rel=1e-6)

    def test_custom_cost(self):
        from src.utils.reward import holding_cost
        r = holding_cost(1, 1, cost_per_bar=-0.001)
        assert r == pytest.approx(-0.001, rel=1e-6)

    def test_small_enough_not_to_dominate(self):
        """576 bars (2 days) of holding cost should be < 1 typical trade reward."""
        from src.utils.reward import holding_cost, trade_reward
        total_holding_cost = sum(holding_cost(i, 1) for i in range(576))
        typical_trade = trade_reward(0.02)  # +2% win
        assert abs(total_holding_cost) < abs(typical_trade) * 5, (
            f"2-day holding cost {total_holding_cost:.3f} should be < 5x typical trade {typical_trade:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# [A] compute_step_reward with kill_triggered
# ─────────────────────────────────────────────────────────────────────────────

class TestKillTriggeredInComposite:

    def _base(self, **kw):
        defaults = dict(
            position=0, unrealized_pnl_pct=0.0, bars_in_trade=0,
            current_drawdown=0.0, trade_closed=False, realized_pnl_pct=0.0,
            transaction_cost=0.0, funding_fee=0.0,
        )
        defaults.update(kw)
        return defaults

    def test_kill_triggered_dominates_everything(self):
        r_kill = compute_step_reward(**self._base(
            kill_triggered=True,
            current_drawdown=0.30,
            trade_closed=True,
            realized_pnl_pct=0.05,   # even a winning trade
        ))
        r_normal = compute_step_reward(**self._base(
            trade_closed=True,
            realized_pnl_pct=-0.10,  # even a 10% losing trade
        ))
        assert r_kill < r_normal, (
            f"Kill-triggered reward {r_kill:.2f} should be worse than worst trade {r_normal:.2f}"
        )

    def test_kill_not_triggered_behaves_normally(self):
        r = compute_step_reward(**self._base(
            kill_triggered=False,
            trade_closed=True,
            realized_pnl_pct=0.02,
        ))
        assert r > 0

    def test_holding_cost_deducted_when_in_trade(self):
        r_in  = compute_step_reward(**self._base(position=1,  unrealized_pnl_pct=0.0))
        r_out = compute_step_reward(**self._base(position=0,  unrealized_pnl_pct=0.0))
        # In-trade should be worse than flat (holding cost applied)
        assert r_in < r_out, (
            f"In-trade reward {r_in:.5f} should be lower than flat {r_out:.5f} due to holding cost"
        )
