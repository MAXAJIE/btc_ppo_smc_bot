# ₿ BTC PPO SMC Trading Bot

> BTCUSDT USDT-M Futures | Isolated Margin | Max 3× Leverage  
> PPO agent with Smart Money Concepts · GARCH + Kelly · Multi-timeframe

---

## Architecture

```
Observation (90-dim)                    Action Space (Discrete 7)
──────────────────────────────────────  ─────────────────────────
[00:15] OHLCV momentum  — 5m  (15)      0 = HOLD
[15:25] OHLCV momentum  — 15m (10)      1 = LONG  full Kelly
[25:35] OHLCV momentum  — 1h  (10)      2 = LONG  half Kelly
[35:43] OHLCV momentum  — 4h  (8)       3 = SHORT full Kelly
[43:51] OHLCV momentum  — 1D  (8)       4 = SHORT half Kelly
[51:59] SMC features    — 5m  (8)       5 = CLOSE position
[59:67] SMC features    — 1h  (8)       6 = REDUCE 50%
[67:73] S&R levels      — 1h  (6)
[73:79] AMT/Vol Profile — 4h  (6)      Reward
[79:83] GARCH + Kelly       (4)       ──────────────────────────
[83:90] Position state      (7)        Win:  +scale × log(1 + |pnl|×100)
                                       Loss: −penalty × log(1 + |pnl|×100)
Episode = 4,320 steps = 15 days 5m              (penalty = 2× scale)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd btc_ppo_smc_bot
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env:
#   BINANCE_TESTNET_API_KEY=...
#   BINANCE_TESTNET_API_SECRET=...
#   TRAIN_BACKEND=local   # or modal or lightning
```

Get testnet keys at: https://testnet.binancefuture.com  
(**Futures testnet** — not the spot testnet, they are different.)

### 3. Choose training backend

| Backend | When to use | How to run |
|---|---|---|
| `local` | Development / quick tests | `python run_training.py` |
| `modal` | Cloud GPU (serverless) | `modal run train_modal.py` |
| `lightning` | Lightning.ai Studio GPU | Open Studio → `python train_lightning.py` |

---

## Training Workflows

### Offline warm-up (required first step)

Downloads 2 years of BTCUSDT 5m data and trains PPO for ~3M steps.

```bash
# Local
TRAIN_BACKEND=local python run_training.py

# Modal (A10G GPU, ~4-6h for 3M steps)
TRAIN_BACKEND=modal python run_training.py

# Lightning.ai (inside Studio terminal)
TRAIN_BACKEND=lightning python run_training.py
```

Resume from checkpoint:
```bash
python run_training.py --resume ./models/ppo_btc_200000_steps.zip
```

### Live testnet fine-tuning

After offline training produces `./models/ppo_btc_final.zip`:

```bash
python -m src.main_live --model ./models/ppo_btc_final.zip
```

Runs continuously on real testnet 5m candle closes.  
PPO updates after each 15-day episode.  
Walk-forward validation fires every 7 days.

---

## Modal Setup

```bash
pip install modal
modal token new                     # authenticate

# Create the Binance secrets
modal secret create binance-secrets \
    BINANCE_TESTNET_API_KEY=your_key \
    BINANCE_TESTNET_API_SECRET=your_secret

# Run
modal run train_modal.py

# After training: download the model
modal volume get btc-ppo-vol models/ppo_btc_final.zip ./models/
```

The Modal Volume `btc-ppo-vol` persists all data and checkpoints.  
If the 23h timeout fires, just re-run — it auto-resumes from the latest checkpoint.

---

## Lightning.ai Setup

1. Create an account at https://lightning.ai
2. Create a new Studio with a GPU machine (RTX 4090 recommended)
3. Clone your repo into the Studio
4. In the Studio terminal:

```bash
pip install -r requirements.txt
python train_lightning.py
```

To control Studios programmatically from outside:
```bash
pip install lightning-sdk
# Add LIGHTNING_USER_ID and LIGHTNING_API_KEY to .env
TRAIN_BACKEND=lightning python run_training.py
```

---

## Live Dashboard

```bash
streamlit run src/utils/dashboard.py -- --log-dir ./logs
```

Shows: equity curve · drawdown · PnL distribution · last 50 trades

---

## Project Structure

```
btc_ppo_smc_bot/
├── .env.example
├── requirements.txt
├── run_training.py          ← unified launcher (LOCAL / MODAL / LIGHTNING)
├── train_modal.py           ← Modal.com entrypoint
├── train_lightning.py       ← Lightning.ai entrypoint
├── config/
│   └── config.yaml          ← all hyperparameters
├── src/
│   ├── main_train.py        ← offline warm-up training
│   ├── main_live.py         ← live testnet fine-tuning
│   ├── env/
│   │   └── binance_testnet_env.py   ← Gymnasium environment
│   ├── features/
│   │   ├── smc_features.py          ← SMC (OB, FVG, BOS/CHOCH)
│   │   ├── amt_features.py          ← Volume Profile (POC, VAH, VAL)
│   │   ├── snr_features.py          ← S&R pivot levels
│   │   ├── garch_kelly.py           ← GARCH + fractional Kelly
│   │   └── multi_tf_features.py     ← 90-dim observation builder
│   ├── execution/
│   │   └── binance_executor.py      ← Testnet order execution
│   ├── models/
│   │   └── ppo_model.py             ← PPO build / load / eval
│   └── utils/
│       ├── reward.py                ← Log reward + loss punishment
│       ├── data_loader.py           ← Historical OHLCV download
│       ├── logger.py                ← Trade journal + equity tracker
│       └── dashboard.py             ← Streamlit monitor
└── data/                            ← Auto-created, .parquet cache
└── logs/                            ← Auto-created, CSV trade logs
└── models/                          ← Auto-created, .zip checkpoints
```

---

## Reward Function

```
Step reward = f(trade outcome, position, costs, drawdown)

On trade close:
  win:   R =  1.0 × log(1 + |pnl_pct| × 100)
  loss:  R = −2.0 × log(1 + |pnl_pct| × 100)   ← 2× amplification

Per step (intra-trade):
  open win trade:  tiny +   (0.05 × log(1 + unrealized))
  open loss trade: tiny −   (0.10 × log(1 + unrealized))
  flat:            −0.0001  (time decay — discourages never trading)

Costs deducted every step:
  commission: 0.04% taker
  slippage:   0.02% estimated
  funding:    every 8h (480 × 5m steps)

Drawdown penalty:
  0 until drawdown > 5%, then −5 × (excess²)
```

---

## Safety

- Hard stop-loss per trade: **−3%**
- Hard take-profit per trade: **+6%**
- Force-close after **2 days** (576 × 5m bars) in any single trade
- Account kill-switch at **−15% drawdown** (closes all positions, halts bot)
- Quarter-Kelly position sizing (never bets more than 25% of Kelly optimal)
- Maximum **3× leverage** hard cap

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `stable-baselines3` | PPO implementation |
| `gymnasium` | RL environment interface |
| `smartmoneyconcepts` | SMC: order blocks, FVG, BOS/CHOCH |
| `arch` | GARCH(1,1) volatility forecasting |
| `python-binance` | Binance REST + WebSocket |
| `ccxt` | Historical data download |
| `streamlit` + `plotly` | Live monitoring dashboard |
| `modal` *(optional)* | Cloud GPU training |
| `lightning-sdk` *(optional)* | Lightning.ai Studio control |
