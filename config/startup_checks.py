"""
ForexChautari — config/startup_checks.py

Fail-fast startup validation. Called before any server or dashboard
starts accepting traffic. Prints a clear list of every missing or
placeholder value rather than crashing mid-request on the first one
that gets used.

Usage (add to the top of app/api.py, app/main.py, run_scheduler.py):
    from config.startup_checks import validate_env
    validate_env()
"""

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("startup_checks")

# ── Sentinel values that ship in .env.example ────────────────────────────────
# These are the placeholder strings a developer copies verbatim without filling
# in. Treat them the same as an unset variable.
_PLACEHOLDERS = {
    "JWT_SECRET":            {"GENERATE_AND_PASTE_JWT_SECRET_HERE"},
    "FIELD_ENCRYPTION_KEY":  {"GENERATE_AND_PASTE_KEY_HERE"},
    "OANDA_API_KEY":         {"YOUR_OANDA_TOKEN_HERE"},
    "OANDA_ACCOUNT_ID":      {"101-001-XXXXXXX-001"},
    "TELEGRAM_BOT_TOKEN":    {"YOUR_BOT_TOKEN_HERE"},
    "TELEGRAM_CHAT_ID":      {"YOUR_CHAT_ID_HERE"},
    "ALPHAVANTAGE_API_KEY":  {"YOUR_AV_KEY_HERE"},
}

# ── Variable definitions ──────────────────────────────────────────────────────
# Each entry is:
#   key          : env var name
#   required     : True  → SystemExit if missing
#                  False → warning only
#   min_length   : optional minimum character length (catches truncated secrets)
#   generate_cmd : shown in the error message so the developer knows what to run

_ENV_SPEC = [
    {
        "key":          "JWT_SECRET",
        "required":     True,
        "min_length":   32,
        "description":  "Signs and verifies all JWT access/refresh tokens.",
        "generate_cmd": 'python -c "import secrets; print(secrets.token_hex(32))"',
    },
    {
        "key":          "FIELD_ENCRYPTION_KEY",
        "required":     True,
        "min_length":   44,   # Fernet keys are always 44 base64 chars
        "description":  "Encrypts Oanda API keys stored in the database.",
        "generate_cmd": (
            'python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        ),
    },
    {
        "key":          "OANDA_API_KEY",
        "required":     True,
        "min_length":   10,
        "description":  "Oanda v20 REST API bearer token.",
        "generate_cmd": "Log in to oanda.com → My Account → Manage API Access",
    },
    {
        "key":          "OANDA_ACCOUNT_ID",
        "required":     True,
        "min_length":   5,
        "description":  "Oanda account ID (e.g. 101-001-XXXXXXX-001).",
        "generate_cmd": "Find it on your Oanda dashboard under Account Summary",
    },
    # ── Optional but warn if completely absent ─────────────────────────────
    {
        "key":          "TELEGRAM_BOT_TOKEN",
        "required":     False,
        "description":  "Telegram bot token — alerts will be silently disabled.",
        "generate_cmd": "Open Telegram → @BotFather → /newbot",
    },
    {
        "key":          "TELEGRAM_CHAT_ID",
        "required":     False,
        "description":  "Telegram chat ID — alerts will be silently disabled.",
        "generate_cmd": (
            "Send a message to your bot, then visit "
            "https://api.telegram.org/bot<TOKEN>/getUpdates"
        ),
    },
    {
        "key":          "ALPHAVANTAGE_API_KEY",
        "required":     False,
        "description":  "Alpha Vantage key — only needed if using AV as data source.",
        "generate_cmd": "https://www.alphavantage.co/support/#api-key (free)",
    },
    {
        "key":          "DATABASE_URL",
        "required":     False,
        "description":  "Database connection string (Postgres recommended for production).",
        "generate_cmd": "postgresql://user:pass@host:5432/forexchautari",
    },
]


# ── Public API ────────────────────────────────────────────────────────────────

def validate_env(*, die_on_warnings: bool = False) -> None:
    """
    Validate all required and optional environment variables.

    Required variables that are missing or still set to their placeholder
    value cause an immediate SystemExit with a full diagnostic message.

    Optional variables that are missing emit a WARNING log line so
    operators know which features will be disabled.

    Parameters
    ----------
    die_on_warnings : bool
        If True, treat missing optional variables as fatal too.
        Useful in CI/CD pipelines that want a fully-configured environment.
    """
    hard_failures: list[str] = []   # required + missing/placeholder
    soft_warnings: list[str] = []   # optional + missing/placeholder
    length_failures: list[str] = [] # present but too short (truncated secret)

    for spec in _ENV_SPEC:
        key         = spec["key"]
        required    = spec["required"]
        min_length  = spec.get("min_length", 0)
        description = spec["description"]
        gen_cmd     = spec.get("generate_cmd", "")

        raw = os.getenv(key, "").strip()

        # ── Check 1: missing or placeholder ───────────────────────────────────
        placeholders = _PLACEHOLDERS.get(key, set())
        is_empty_or_placeholder = not raw or raw in placeholders

        if is_empty_or_placeholder:
            msg = (
                f"  • {key}\n"
                f"    Purpose : {description}\n"
                f"    Generate: {gen_cmd}"
            )
            if required or die_on_warnings:
                hard_failures.append(msg)
            else:
                soft_warnings.append(f"  • {key} — {description}")
            continue   # no point checking length if value is empty

        # ── Check 2: present but suspiciously short ────────────────────────────
        if min_length and len(raw) < min_length:
            length_failures.append(
                f"  • {key} is only {len(raw)} chars — "
                f"expected at least {min_length}. "
                f"Was it truncated in .env?\n"
                f"    Generate: {gen_cmd}"
            )

    # ── Emit soft warnings ─────────────────────────────────────────────────────
    for w in soft_warnings:
        logger.warning("Optional env var not set — %s", w.strip())

    # ── Collect all fatal problems ─────────────────────────────────────────────
    all_fatal = hard_failures + length_failures

    if all_fatal:
        separator = "\n" + "─" * 68 + "\n"
        message = separator.join([
            (
                "\n"
                "╔══════════════════════════════════════════════════════════════════╗\n"
                "║           FOREXCHAUTARI — STARTUP VALIDATION FAILED             ║\n"
                "╚══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                "The following required environment variables are missing, set to\n"
                "their placeholder value, or too short to be valid.\n"
                "\n"
                "Add them to your .env file and restart.\n"
            ),
            "\n".join(all_fatal),
            (
                "Copy .env.example to .env if you have not done so:\n"
                "    cp .env.example .env\n"
                "\n"
                "Then fill in each value above and restart the server.\n"
            ),
        ])
        # Print directly to stderr so it is visible even if logging is not
        # yet configured (e.g. very early in the startup sequence).
        print(message, file=sys.stderr)
        raise SystemExit(1)

    # ── Optional: if DATABASE_URL points to Postgres, verify the shared
    # SQLAlchemy ENGINE can actually connect. This fails fast at startup
    # rather than when the first query is attempted at runtime.
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url and not database_url.startswith("sqlite"):
        try:
            from sqlalchemy import text
            from src.db_engine import ENGINE

            with ENGINE.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database engine connectivity check passed: ENGINE.connect() ✓")
        except Exception as exc:  # pragma: no cover - startup failure path
            message = (
                "\n"
                "╔══════════════════════════════════════════════════════════════════╗\n"
                "║       FOREXCHAUTARI — DATABASE ENGINE CONNECTIVITY FAILED        ║\n"
                "╚══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                f"Could not connect using the application ENGINE to:\n"
                f"  {database_url}\n\n"
                f"Error:\n  {type(exc).__name__}: {exc}\n\n"
                "Troubleshooting:\n"
                "  1. Verify DATABASE_URL in .env is correct.\n"
                "  2. Ensure the Postgres server is running and reachable.\n"
                "  3. Confirm the database and user exist and credentials are valid.\n"
                "\n"
                "To use SQLite instead, unset DATABASE_URL in .env and restart.\n"
            )
            print(message, file=sys.stderr)
            raise SystemExit(1)


def warn_if_debug_settings_in_production() -> None:
    """
    Separately check for settings that are safe in development but
    dangerous in production. Call this after validate_env() if you
    want belt-and-suspenders production hardening.
    """
    warnings_found = []

    # Live trading enabled — warn loudly
    if os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true":
        warnings_found.append(
            "  • LIVE_TRADING_ENABLED=true — real money is at risk. "
            "The v1 model has ~51-53% walk-forward accuracy."
        )

    # DATABASE_URL in a temp location (SQLite only)
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url and database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite:///", "")
        if db_path.startswith("/tmp"):
            warnings_found.append(
                f"  • DATABASE_URL={database_url} — database is in /tmp and will be lost on reboot."
            )

    # Short JWT secret
    jwt_secret = os.getenv("JWT_SECRET", "")
    if jwt_secret and len(jwt_secret) < 32:
        warnings_found.append(
            "  • JWT_SECRET is shorter than 32 characters — "
            "use `python -c \"import secrets; print(secrets.token_hex(32))\"`"
        )

    for w in warnings_found:
        logger.warning("Production warning: %s", w.strip())


def verify_database_connectivity() -> None:
    """
    If DATABASE_URL is set to a PostgreSQL connection, verify it's reachable.
    This is a fail-fast check — if the database is down, the app should not start.

    Call this during startup (after validate_env()) so bad connections are caught
    before the app starts accepting traffic or processing background jobs.

    Raises SystemExit if the connection cannot be established.
    """
    database_url = os.getenv("DATABASE_URL", "").strip()

    # Skip check for SQLite or unset DATABASE_URL
    if not database_url or database_url.startswith("sqlite"):
        return

    # Try to establish a test connection
    try:
        from sqlalchemy import create_engine, text
        test_engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={},
            future=True,
            echo=False,
        )
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(
            f"Database connectivity check passed: {database_url.split('@')[0]}@... ✓"
        )
    except Exception as exc:
        message = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║       FOREXCHAUTARI — DATABASE CONNECTIVITY CHECK FAILED         ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Could not connect to database at:\n"
            f"  {database_url}\n"
            f"\n"
            f"Error:\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"\n"
            f"Troubleshooting:\n"
            f"  1. Verify the DATABASE_URL is correct in .env\n"
            f"  2. Check that the PostgreSQL server is running\n"
            f"  3. Verify network connectivity and firewall rules\n"
            f"  4. Ensure the database and user exist (run migrations if needed)\n"
            f"\n"
            f"To use SQLite instead, unset DATABASE_URL in .env and restart.\n"
        )
        print(message, file=sys.stderr)
        raise SystemExit(1)
