# ⬡ ForexChautari

**Forex Market Prediction and Analysis ML Model**

A full-stack real-world forex trading platform with ML signals, multi-pair support,
user accounts, subscription plans, and direct Oanda broker integration.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env
# Edit .env with your Oanda, Telegram, and Alpha Vantage keys

# 3. Fetch data and train models for all 4 pairs
python train_all.py --fetch

# 4. Start the API backend (Terminal 1)
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

# 5. Start the dashboard (Terminal 2)
streamlit run app/main.py

# 6. Open browser → http://localhost:8501
#    Default admin login: admin / admin123
```

---

## Architecture

```
forex-ml-platform/
├── app/
│   ├── main.py              Entry point — routes to admin or user dashboard
│   ├── auth.py              Login, register, session management
│   ├── admin_panel.py       Admin dashboard (6 tabs)
│   ├── user_dashboard.py    User dashboard (6 tabs, plan-gated)
│   └── api.py               FastAPI REST backend (10 endpoints)
├── src/
│   ├── database.py          SQLite — users, trades, signals, audit log
│   ├── oanda_client.py      Oanda v20 REST client
│   ├── trading_engine.py    Order placement, SL/TP, position sizing
│   ├── paper_trader.py      Automated signal → trade pipeline
│   ├── multi_pair_manager.py  Multi-pair fetch, train, signal
│   ├── regime_detector.py   ADX trend/range regime filter
│   ├── alerter.py           Telegram notifications
│   ├── features.py          Technical indicators (23 features)
│   ├── model.py             Random Forest with regularisation
│   └── ...
├── config/
│   └── settings.py          All config and plan limits
├── tests/
│   └── test_all.py          81 unit tests
├── train_all.py             Train models for all 4 pairs
├── run_scheduler.py         Automated daily signal + trade scheduler
└── .env.example             Environment variable template
```

---

## Subscription Plans

| Feature                  | Free | Basic | Pro | Enterprise |
|--------------------------|------|-------|-----|------------|
| Pairs                    | 1    | 2     | 4   | All        |
| Signal cards             | ✅   | ✅    | ✅  | ✅         |
| Price charts + MACD/RSI  | ✅   | ✅    | ✅  | ✅         |
| Walk-forward results     | ✗    | ✅    | ✅  | ✅         |
| Connect Oanda account    | ✗    | ✗     | ✅  | ✅         |
| Place / close trades     | ✗    | ✗     | ✅  | ✅         |
| Auto-trading             | ✗    | ✗     | ✅  | ✅         |
| Telegram alerts          | ✗    | ✗     | ✅  | ✅         |
| Trade history + P&L      | ✗    | ✗     | ✅  | ✅         |

---

## Pairs Supported

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD

---

## API Endpoints

| Method | Path                  | Description                  |
|--------|-----------------------|------------------------------|
| GET    | /health               | System liveness check        |
| GET    | /predict/latest       | Signal for one pair          |
| GET    | /predict/all          | Signals for all pairs        |
| GET    | /portfolio/signals    | Ranked portfolio signals     |
| GET    | /portfolio/health     | Model health all pairs       |
| GET    | /model-info           | Model metadata               |
| POST   | /retrain              | Retrain one or all models    |
| POST   | /fetch-data           | Fetch candles from Oanda     |
| GET    | /history              | OHLC history for a pair      |
| GET    | /walk-forward         | Walk-forward results         |

Interactive docs: http://127.0.0.1:8000/docs

---

## Automated Scheduler

```bash
# Run in background — signals + trades every day automatically
python run_scheduler.py
```

Schedule (UTC):
- **08:05** — Fetch latest candles for all pairs
- **08:10** — Run ML models → regime filter → risk check → place trades
- **08:30** — Send daily account summary to Telegram
- **Monday 00:01** — Retrain all models on fresh data

---

## Running Tests

```bash
python tests/test_all.py
# Ran 81 tests — OK
```

---

## Live Trading — v1 Restriction

Live Oanda account connections are **disabled by default** for all users.

| Account type | Practice | Live |
|---|---|---|
| Regular users (any plan) | ✅ Allowed | 🔒 Always blocked |
| Admin | ✅ Allowed | ⚠️ Requires `LIVE_TRADING_ENABLED=true` in `.env` |

**Why?** The v1 Random Forest model achieves ~51–53% walk-forward accuracy,
which is statistically close to a coin flip. Connecting real money to an
automated system with this edge profile carries significant financial risk.

To enable live trading (admins only, at your own risk):

```env
LIVE_TRADING_ENABLED=true
```

This flag is enforced at the database layer — it cannot be bypassed via the
UI, the REST API, or by calling `add_trading_account()` from a script.

## Default Credentials & First-Run Setup

| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |

**The platform enforces a password change before it is usable.**

On first startup:
1. All non-health API endpoints return `503 Setup Required`
2. The admin panel shows a setup wizard instead of any other tab
3. A `CRITICAL` warning is written to the application log

The block is lifted automatically once the admin password is changed
via the wizard. There is no way to skip this step.

Minimum password requirements enforced by the wizard:
- 12 characters or more
- Cannot be `admin123`

---

## Tech Stack

- **ML**: scikit-learn Random Forest (regularised, walk-forward validated)
- **Backend**: FastAPI + uvicorn
- **Frontend**: Streamlit + Plotly
- **Database**: SQLite (via stdlib sqlite3)
- **Broker**: Oanda v20 REST API
- **Alerts**: Telegram Bot API
- **Scheduler**: schedule library
