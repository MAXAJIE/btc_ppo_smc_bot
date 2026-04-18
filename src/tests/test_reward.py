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
