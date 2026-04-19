# BTC PPO SMC Bot — Fix Summary

## Root Causes → Fixes

---

### Bug 1 — MTF Data Fetcher Only Returned 5m Data

**Root cause** (`src/utils/data_loader.py`):  
The downloader fetched only `5m` candles and the MTF builder was trying to
resample them on-the-fly.  Resampling from 5m to 1d with `pandas.resample`
produces `NaN` for any bar that doesn't perfectly land on a calendar boundary
(timezone issues, gaps, etc.).  The NaN propagated into the entire observation
vector, making the 90-dim obs a mix of real values and zeros/NaN.

**Fix**:
- Each timeframe (`5m`, `15m`, `1h`, `4h`, `1d`) is downloaded **independently**
  via separate `ccxt.fetch_ohlcv()` calls and cached to separate `.parquet` files.
- `build_aligned_dataset()` uses `pd.merge_asof` (as-of join) to align all TFs
  to the 5m index with no look-ahead.
- `MultiTFFeatureBuilder` accepts the full `dict[str, pd.DataFrame]` and does a
  safe indexed lookup for each TF at every step.

---

### Bug 2 — PPO Reward Preventing Convergence

**Root cause** (`src/utils/reward.py`):

| Issue | Effect |
|-------|--------|
| 2× asymmetric loss penalty | Policy collapses to HOLD after ~100k steps |
| Reward only on trade close | Critic has almost no signal per step → high value-loss |
| Intra-trade −0.10× unrealised penalty | Agent learned to close every trade immediately (round-trip churn) |
| Quadratic drawdown (−5 × excess²) | Single spike of −0.45 caused gradient explosion |
| No entropy bonus (ent_coef=0) | Policy became deterministic far too early |

**Fixes**:
1. **Loss scale reduced to 1.3×** (from 2.0). Still penalises losses more than wins,
   but not so harshly that the agent stops trading.
2. **Dense per-step reward via unrealised PnL delta** — only the *change* in
   unrealised PnL is shaped each step (not the raw value), so new trades aren't
   immediately penalised.
3. **Entry-quality bonus ±0.08** — agent gets +0.08 when it opens a trade near
   a valid SMC Order Block or S&R level, and −0.024 for random entry.  This is
   the mechanism that teaches the bot to understand its tools.
4. **Drawdown penalty is now linear and capped** at 0.30 per step (no quadratic
   spikes).
5. **`ent_coef = 0.01`** in `config.yaml` — the most important single-line fix.

---

### Bug 3 — SMC/SNR/AMT Features Were Giving Zero/Noise Signal

**Root cause** (`smc_features.py`, `snr_features.py`, `amt_features.py`):

- **SMC**: OB detection compared every candle to every other — nearly every bar
  qualified as an OB → feature was always 1.0 → zero information.
- **SNR**: returned raw BTC prices (~60 000) in the observation vector alongside
  normalised values near 0 → policy network couldn't make sense of the scale.
- **AMT**: POC/VAH/VAL as raw prices — same problem.
- **GARCH**: returned raw GARCH parameters (omega ~1e-6) — effectively zero.

**Fixes**:
- **SMC**: proper swing-high / swing-low detection with a `SWING_LOOKBACK`
  window.  Returns ATR-normalised *distance* to nearest OB (float, not bool).
- **SNR**: returns ATR-normalised distance from close to each S&R level.
- **AMT**: rolling 100-bar volume profile (not whole-history), returns
  ATR-normalised POC/VAH/VAL distances plus Value Area width and volume rank.
- **GARCH**: returns daily volatility %, vol regime ratio, Kelly fraction,
  and fit confidence — all in [0, 1] range.

---

### Bug 4 — PPO Hyperparameters Fighting Convergence

**Root cause** (`config/config.yaml`):

| Param | Old | New | Reason |
|-------|-----|-----|--------|
| `learning_rate` | 3e-4 | 1e-4 | Reduces oscillation with 90-dim mixed-scale obs |
| `n_steps` | 2048 | 4096 | Episode is 4320 steps; 2048 gave <1 full episode per update |
| `batch_size` | 64 | 256 | Larger batches reduce value-loss variance |
| `n_epochs` | 10 | 5 | Fewer epochs prevents overfitting the small rollout buffer |
| `clip_range` | 0.2 | 0.15 | Tighter clip for conservative early updates |
| `ent_coef` | 0.0 | **0.01** | **Critical**: was causing HOLD-collapse |
| `vf_coef` | 0.5 | 0.7 | More gradient to the critic to fit value function |
| `gamma` | 0.99 | 0.995 | Longer credit assignment for 15-day episodes |
| `gae_lambda` | 0.95 | 0.97 | More Monte-Carlo-like returns while critic is under-fitted |

---

## Expected Outcome After Fixes

| Metric | Before | After (expected) |
|--------|--------|-----------------|
| Value loss | Diverging / high | Steadily decreasing |
| Episode reward | Flat / slightly negative | Gradually increasing after ~500k steps |
| Action distribution | HOLD ~95% of time | Balanced across actions |
| SMC tool usage | Ignored | Entry quality correlated with OB/SNR proximity |
| MTF features | Zeros/NaN in obs[15:51] | Populated with real HTF context |

## Files Changed

```
src/utils/data_loader.py          ← fetch all 5 TFs independently
src/features/multi_tf_features.py ← as-of lookup, NaN-safe, proper normalisation
src/features/smc_features.py      ← swing-based OB/FVG/BOS, ATR-normalised distances
src/features/snr_features.py      ← pivot-based S&R, ATR-normalised distances
src/features/amt_features.py      ← rolling vol profile, ATR-normalised distances
src/features/garch_kelly.py       ← normalised output [0,1] range
src/utils/reward.py               ← fixed asymmetry, dense shaping, entry bonus
src/env/binance_testnet_env.py    ← wires all modules, passes SMC ctx to reward
config/config.yaml                ← convergence-tuned hyperparameters
```
