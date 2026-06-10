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

## Deployment Checklist

Complete these steps **in order** before going live. Skipping any step
leaves user credentials or trading accounts exposed.

### Step 1 — Generate required secrets

```bash
# JWT signing key (signs every login token)
python -c "import secrets; print(secrets.token_hex(32))"

# Field encryption key (encrypts Oanda API keys at rest)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add both values to `.env`:

```env
JWT_SECRET=<output of first command>
FIELD_ENCRYPTION_KEY=<output of second command>
```

**Back up `FIELD_ENCRYPTION_KEY` somewhere separate from the database.**
If the key is lost, all stored Oanda API keys become permanently unreadable
and every user must reconnect their trading account.

### Step 2 — Change the admin password

On first startup the admin account uses the default password `admin123`.
The platform blocks all non-health API endpoints and shows a setup wizard
until you change it.

1. Open http://localhost:8501
2. Log in with `admin` / `admin123`
3. Complete the setup wizard (minimum 12 characters, cannot be `admin123`)

### Step 3 — Connect your Oanda account

1. Sign up at https://www.oanda.com/register (free practice account)
2. Go to **My Account → Manage API Access → Generate Token**
3. Add your token and account ID to `.env`:

```env
OANDA_API_KEY=your_token_here
OANDA_ACCOUNT_ID=101-001-XXXXXXX-001
OANDA_ENVIRONMENT=practice
```

### Step 4 — Encrypt stored API keys (REQUIRED for existing deployments)

> **This step applies if you are upgrading an existing deployment that already
> has users with trading accounts in the database.**
> Fresh deployments where no trading accounts have been added yet can skip this
> step — `add_trading_account()` already encrypts new keys automatically.

Every time the server starts, `init_db()` scans `trading_accounts` for rows
whose `api_key_enc` column is not prefixed with `enc:v1:`. If any plaintext
rows are found:

- A `CRITICAL` log line is written identifying the affected row IDs
- `platform_settings.PLAINTEXT_API_KEYS_FOUND` is set to `"true"`
- The admin panel shows an orange warning banner with the count and fix command

**To fix:**

```bash
# 1. Back up the database FIRST
cp data/forexchautari.db data/forexchautari.db.bak

# 2. Verify FIELD_ENCRYPTION_KEY is set in .env
grep FIELD_ENCRYPTION_KEY .env

# 3. Dry run — shows what would change, writes nothing
python scripts/migrate_encrypt_keys.py --dry-run

# 4. Run the actual migration
python scripts/migrate_encrypt_keys.py
```

Expected output:
```
ForexChautari — API key encryption migration
==================================================
Total accounts :  3
Already encrypted: 0
Need migration :  3

  [id=1] user_id=1 account=My Practice Account (101-001-...) key_starts_with=abc123...
  [id=2] user_id=2 account=User Account (101-001-...) key_starts_with=def456...

Encrypting 3 rows...
  ✓ Encrypted account id=1
  ✓ Encrypted account id=2

Migration complete: 3 encrypted, 0 errors
All API keys are now encrypted.
```

After a successful migration, restart the server. The `CRITICAL` log line and
admin banner will not appear on subsequent startups.

### Step 5 — Verify the setup

```bash
# Run the test suite (81 tests, no Oanda account required)
python tests/test_all.py

# Check the API health endpoint
curl http://127.0.0.1:8000/health
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
│   │                          init_db() runs 3 security audits on every start:
│   │                            _audit_admin_default_password()
│   │                            _audit_plaintext_api_keys()        ← NEW
│   │                            (backfills user_trading_settings)
│   ├── encryption.py        Fernet field encryption for API keys
│   ├── oanda_client.py      Oanda v20 REST client
│   ├── trading_engine.py    Order placement, SL/TP, position sizing
│   ├── paper_trader.py      Automated signal → trade pipeline
│   ├── multi_pair_manager.py  Multi-pair fetch, train, signal
│   ├── regime_detector.py   ADX trend/range regime filter
│   ├── alerter.py           Telegram notifications
│   ├── features.py          Technical indicators (23 features)
│   ├── model.py             Random Forest with regularisation
│   └── ...
├── scripts/
│   ├── migrate_encrypt_keys.py   Encrypt plaintext API keys (one-time, REQUIRED)
│   └── rotate_encryption_key.py  Re-encrypt under a new key (key rotation)
├── config/
│   └── settings.py          All config and plan limits
├── tests/
│   └── test_all.py          81 unit tests
├── train_all.py             Train models for all 4 pairs
├── run_scheduler.py         Automated daily signal + trade scheduler
└── .env.example             Environment variable template
```

---

## Security Model

### API key encryption at rest

Oanda API keys give full programmatic access to a trading account, including
the ability to place and close positions. Storing them in plaintext means that
anyone who can read the SQLite file (backups, misconfigured file permissions,
a compromised server) can immediately start trading on behalf of every user.

All new keys written by `add_trading_account()` are encrypted with
[Fernet](https://cryptography.io/en/latest/fernet/) (AES-128-CBC + HMAC-SHA256)
before the INSERT. The ciphertext is prefixed with `enc:v1:` so the migration
script and `is_encrypted()` helper can distinguish encrypted rows from legacy
plaintext rows without attempting decryption.

The encryption key (`FIELD_ENCRYPTION_KEY`) must be:
- Stored in `.env` — never in the database or version control
- Backed up separately from the database file
- Rotated if compromised using `scripts/rotate_encryption_key.py`

### Startup security audits

`init_db()` runs two read-only security checks on every server start and
writes the results into `platform_settings` for the admin panel to display:

| Setting key | What it checks | Fix |
|---|---|---|
| `ADMIN_PASSWORD_CHANGED` | Default password still in use | Admin panel setup wizard |
| `PLAINTEXT_API_KEYS_FOUND` | Unencrypted API keys in DB | `migrate_encrypt_keys.py` |

Both checks write a `CRITICAL` log line so they are visible in any log
aggregator (Datadog, CloudWatch, etc.) without needing to open the dashboard.

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

---

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
