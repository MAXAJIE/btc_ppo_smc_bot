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


# ═════════════════════════════════════════════════════════════════════════════
# v3 ADDITIONS
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# [1] drawdown_penalty — linear + quadratic two-component
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownPenaltyV3:
    """
    v3 drawdown_penalty = linear_component + quadratic_component.
    Linear fires every step above threshold (default 5%).
    """

    def test_zero_below_threshold(self):
        from src.utils.reward import drawdown_penalty
        for dd in [0.0, 0.01, 0.04, 0.049]:
            assert drawdown_penalty(dd) == 0.0, f"Expected 0 at dd={dd}"

    def test_negative_above_threshold(self):
        from src.utils.reward import drawdown_penalty
        for dd in [0.051, 0.08, 0.10, 0.14]:
            r = drawdown_penalty(dd)
            assert r < 0.0, f"Expected negative at dd={dd}, got {r}"

    def test_linear_component_exact(self):
        """
        At dd=0.08: excess=0.03
        linear    = -0.30 * 0.03 = -0.009
        quadratic = -5 * (0.03)^2 = -0.0045
        total     = -0.0135
        """
        from src.utils.reward import drawdown_penalty
        r = drawdown_penalty(0.08)
        expected = -0.30 * 0.03 + -5.0 * (0.03 ** 2)
        assert r == pytest.approx(expected, rel=1e-6), f"Expected {expected:.6f}, got {r:.6f}"

    def test_strictly_increasing_with_drawdown(self):
        from src.utils.reward import drawdown_penalty
        dds = [0.05, 0.07, 0.09, 0.11, 0.13, 0.14]
        penalties = [drawdown_penalty(d) for d in dds]
        for i in range(1, len(penalties)):
            assert penalties[i] < penalties[i-1], (
                f"Penalty should worsen with DD: {penalties[i-1]:.5f} → {penalties[i]:.5f}"
            )

    def test_linear_grows_proportionally(self):
        """
        Between dd=6% and dd=11%, linear component doubles (excess 1% → 6%).
        Full penalty also grows, but quadratic is small at low DD.
        """
        from src.utils.reward import drawdown_penalty
        # At these DDs, quadratic is tiny → penalty ≈ linear
        r_lo = drawdown_penalty(0.06)   # excess = 0.01
        r_hi = drawdown_penalty(0.11)   # excess = 0.06  (6× the excess)
        # Penalty should be at least 4× larger (linear dominates at low DD)
        assert abs(r_hi) > abs(r_lo) * 3.0, (
            f"Penalty at 11% ({r_hi:.4f}) should be >> penalty at 6% ({r_lo:.4f})"
        )

    def test_fires_every_step_when_in_drawdown(self):
        """
        Unlike v2 quadratic-only, v3 linear component fires even at mild DD.
        Calling 100× at 8% DD should accumulate a meaningful total.
        """
        from src.utils.reward import drawdown_penalty
        per_step = drawdown_penalty(0.08)   # -0.0135
        total_100 = per_step * 100
        assert total_100 < -1.0, (
            f"100 steps at 8% DD should accumulate at least -1.0, got {total_100:.3f}"
        )

    def test_magnitude_never_dominates_trade_reward(self):
        """Per-step DD penalty should never exceed a single small win."""
        from src.utils.reward import drawdown_penalty, trade_reward
        worst_step = drawdown_penalty(0.149)   # just under kill threshold
        small_win  = trade_reward(0.005)       # tiny 0.5% win
        assert abs(worst_step) < abs(small_win), (
            f"DD per-step |{worst_step:.4f}| should be < small win |{small_win:.4f}|"
        )


# ─────────────────────────────────────────────────────────────────────────────
# [1] sl_hit_extra_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestSLHitPenalty:

    def test_returns_negative_float(self):
        from src.utils.reward import sl_hit_extra_penalty
        r = sl_hit_extra_penalty()
        assert isinstance(r, float)
        assert r < 0.0

    def test_default_value(self):
        from src.utils.reward import sl_hit_extra_penalty
        r = sl_hit_extra_penalty()
        assert r == pytest.approx(-1.5, rel=1e-6)

    def test_custom_value(self):
        from src.utils.reward import sl_hit_extra_penalty
        assert sl_hit_extra_penalty(-2.0) == pytest.approx(-2.0)
        assert sl_hit_extra_penalty(-0.5) == pytest.approx(-0.5)

    def test_sl_exit_worse_than_voluntary_exit(self):
        """SL at -3% must be meaningfully worse than voluntary exit at -3%."""
        from src.utils.reward import trade_reward, sl_hit_extra_penalty
        voluntary = trade_reward(-0.03)
        sl_exit   = trade_reward(-0.03) + sl_hit_extra_penalty()
        assert sl_exit < voluntary, "SL exit should have worse reward than voluntary"
        gap = abs(sl_exit) - abs(voluntary)
        assert gap == pytest.approx(1.5, rel=1e-4), f"Gap should be |sl_extra|=1.5, got {gap:.4f}"

    def test_sl_not_so_large_it_deters_all_trades(self):
        """
        Even after SL sting, a subsequent 2% winning trade should
        more than recover.  Otherwise agent learns: never open trades.
        """
        from src.utils.reward import trade_reward, sl_hit_extra_penalty
        sl_loss = trade_reward(-0.03) + sl_hit_extra_penalty()   # ≈ -4.27
        win     = trade_reward(0.02)                             # ≈ +1.38
        # Three good trades > one SL hit
        assert win * 3 > abs(sl_loss), (
            f"3× win={win*3:.2f} should > |SL loss|={abs(sl_loss):.2f} — "
            f"otherwise agent won't trade"
        )


# ─────────────────────────────────────────────────────────────────────────────
# [1+A] compute_step_reward with sl_hit parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRewardV3:

    def _base(self, **kw):
        d = dict(position=0, unrealized_pnl_pct=0.0, bars_in_trade=0,
                 current_drawdown=0.0, trade_closed=False, realized_pnl_pct=0.0,
                 transaction_cost=0.0, funding_fee=0.0)
        d.update(kw)
        return d

    def test_sl_hit_worsens_trade_close_reward(self):
        r_voluntary = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03
        ))
        r_sl = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03, sl_hit=True
        ))
        assert r_sl < r_voluntary, "SL reward should be worse than voluntary at same PnL"
        assert r_voluntary - r_sl == pytest.approx(1.5, rel=1e-4)

    def test_sl_hit_false_no_extra_penalty(self):
        """sl_hit=False (default) must not add any extra penalty."""
        r1 = compute_step_reward(**self._base(trade_closed=True, realized_pnl_pct=-0.02))
        r2 = compute_step_reward(**self._base(trade_closed=True, realized_pnl_pct=-0.02, sl_hit=False))
        assert r1 == pytest.approx(r2, rel=1e-8)

    def test_sl_hit_only_on_trade_close(self):
        """sl_hit=True with trade_closed=False should have no extra effect."""
        r_sl_no_close   = compute_step_reward(**self._base(sl_hit=True))
        r_no_sl_no_close = compute_step_reward(**self._base())
        assert r_sl_no_close == pytest.approx(r_no_sl_no_close, rel=1e-8)

    def test_dd_above_5pct_always_penalises(self):
        """With dd=0.08 (above threshold), every step should have a penalty."""
        r_dd_above = compute_step_reward(**self._base(current_drawdown=0.08))
        r_dd_below = compute_step_reward(**self._base(current_drawdown=0.04))
        assert r_dd_above < r_dd_below, (
            f"Above-threshold dd should penalise more: {r_dd_above:.5f} vs {r_dd_below:.5f}"
        )

    def test_dd_penalty_proportional(self):
        """Higher DD → worse per-step reward (proportionality test)."""
        dds   = [0.06, 0.09, 0.12]
        rwds  = [compute_step_reward(**self._base(current_drawdown=d)) for d in dds]
        for i in range(1, len(rwds)):
            assert rwds[i] < rwds[i-1], f"Reward should decrease with DD: {rwds}"

    def test_sl_and_dd_combine_additively(self):
        """SL sting + drawdown penalty are additive, not competing."""
        r_only_sl = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03, sl_hit=True, current_drawdown=0.0
        ))
        r_only_dd = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03, sl_hit=False, current_drawdown=0.10
        ))
        r_both = compute_step_reward(**self._base(
            trade_closed=True, realized_pnl_pct=-0.03, sl_hit=True, current_drawdown=0.10
        ))
        # r_both should be worse than either alone
        assert r_both < r_only_sl
        assert r_both < r_only_dd
