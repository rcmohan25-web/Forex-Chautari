import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME    = "Forex Market Prediction and Analysis ML Model"
APP_BRAND   = "ForexChautari"
APP_VERSION = "1.0.0"

# ── Multi-pair config ─────────────────────────────────────────────────────────
PAIRS = {
    "EUR_USD": {"spread": 0.00010, "units": 1000, "pip": 0.0001},
    "GBP_USD": {"spread": 0.00015, "units": 1000, "pip": 0.0001},
    "USD_JPY": {"spread": 0.01500, "units": 1000, "pip": 0.0100},
    "AUD_USD": {"spread": 0.00015, "units": 1000, "pip": 0.0001},
}
ACTIVE_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

def data_path(pair):  return f"data/{pair}.csv"
def model_path(pair): return f"models/{pair}_model.pkl"
def meta_path(pair):  return f"models/{pair}_metadata.json"
def wf_path(pair):    return f"models/{pair}_wf_results.csv"

# ── Legacy paths (backward compat) ───────────────────────────────────────────
DATA_PATH         = "data/EURUSD.csv"
MODEL_PATH        = "models/model.pkl"
METADATA_PATH     = "models/metadata.json"
WF_RESULTS_PATH   = "models/walk_forward_results.csv"
LOG_PATH          = "logs/app.log"
PAPER_TRADES_PATH = "data/paper_trades.json"
SIGNALS_LOG_PATH  = "data/signals_log.csv"
DB_PATH           = os.getenv("DB_PATH", "data/forexchautari.db")
DATABASE_URL      = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ── External APIs ─────────────────────────────────────────────────────────────
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
OANDA_API_KEY        = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT    = os.getenv("OANDA_ENVIRONMENT", "practice")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Model defaults ────────────────────────────────────────────────────────────
DEFAULT_SIGNAL_THRESHOLD = 0.55
DEFAULT_SPREAD_COST      = 0.0001
DEFAULT_WF_TRAIN_SIZE    = 300
DEFAULT_WF_TEST_SIZE     = 100
DEFAULT_WF_STEP_SIZE     = 100

# ── Risk management (user-configurable, subject to hard ceilings below) ───────
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_MAX_DAILY_LOSS = 0.02
DEFAULT_MAX_POSITIONS  = 3
DEFAULT_UNITS          = 1000

# ── Hard risk ceilings (CANNOT be overridden by user settings) ────────────────
# These are enforced at the platform layer before every order, regardless of
# what any user or admin has configured. They exist to prevent a misconfigured
# or compromised account from causing outsized losses.
#
# HARD_MAX_POSITIONS     : absolute cap on open positions per account
# HARD_MAX_RISK_PCT      : max fraction of balance risked on a single trade
#                          (units × sl_pips × pip_value / balance)
# HARD_MAX_DAILY_LOSS_PCT: unrealised drawdown threshold that triggers the
#                          daily kill switch — auto-trading is halted for the
#                          rest of the UTC day and a Telegram alert is sent
HARD_MAX_POSITIONS      = 5     # never more than 5 open positions at once
HARD_MAX_RISK_PCT       = 0.02  # never risk more than 2% of balance per trade
HARD_MAX_DAILY_LOSS_PCT = 0.05  # kill switch fires at 5% unrealised drawdown

# ── Subscription plan limits ──────────────────────────────────────────────────
PLAN_LIMITS = {
    "free":       {"pairs": 1, "auto_trade": False, "price": "$0/mo"},
    "basic":      {"pairs": 2, "auto_trade": False, "price": "$9/mo"},
    "pro":        {"pairs": 4, "auto_trade": True,  "price": "$29/mo"},
    "enterprise": {"pairs": 99,"auto_trade": True,  "price": "Custom"},
}

# ── Live trading gate ─────────────────────────────────────────────────────────
# False by default. Set LIVE_TRADING_ENABLED=true in .env only after
# thorough review. v1 models have ~51-53% walk-forward accuracy.
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
