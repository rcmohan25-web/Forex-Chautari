"""
Database layer for ForexChautari.

Dialect-agnostic: works with SQLite (default) or PostgreSQL via DATABASE_URL.
All queries use named params (:name) for Postgres/SQLite compatibility.

Concurrency & Locking:
  - SQLite: uses WAL mode at the engine level (db_engine.py), enabling
    concurrent reads and single-writer transactions.
  - PostgreSQL: MVCC (Multi-Version Concurrency Control) + row-level locking
    are handled natively; no pragmas needed. Connections are pooled to prevent
    exhausting max_connections limits.

Imports from src.db_engine:
  - get_db: context manager for transactional connections
  - execute, fetchone, fetchall: dict-based query helpers
  - pk_column, IS_POSTGRES, IS_SQLITE: dialect detection
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from src.logger import get_logger
from src.encryption import encrypt, decrypt, is_encrypted
from src.db_engine import get_db, execute, fetchone, fetchall, pk_column, IS_POSTGRES, IS_SQLITE

logger = get_logger("database")


# ── Default password sentinel ─────────────────────────────────────────────────
_DEFAULT_ADMIN_PASSWORD = "admin123"


def _audit_admin_default_password(conn) -> None:
    """
    Called once per process startup inside init_db().
    Checks whether the admin account still uses the seeded password.
    Writes the result into platform_settings so the API middleware and
    admin panel can both read it without hitting the users table.

    Two possible states written to platform_settings:
      ADMIN_PASSWORD_CHANGED = "true"   — password has been changed, all clear
      ADMIN_PASSWORD_CHANGED = "false"  — still on default, block non-health endpoints
    """
    now = datetime.utcnow().isoformat()
    admin_row = fetchone(conn,
        "SELECT password_hash, salt FROM users WHERE username='admin' AND is_active=1")

    if not admin_row:
        return

    still_default = verify_password(
        _DEFAULT_ADMIN_PASSWORD,
        admin_row["salt"],
        admin_row["password_hash"],
    )

    flag_value = "false" if still_default else "true"

    execute(conn, """
        INSERT INTO platform_settings (key, value, updated_at)
        VALUES (:k, :v, :now)
        ON CONFLICT(key) DO UPDATE
            SET value = excluded.value,
                updated_at = excluded.updated_at
    """, {"k": "ADMIN_PASSWORD_CHANGED", "v": flag_value, "now": now})

    if still_default:
        logger.critical(
            "\n".join([
                "=" * 70,
                "CRITICAL SECURITY WARNING",
                "The admin account is still using the default password 'admin123'.",
                "All non-health API endpoints are BLOCKED until you change it.",
                "Log in to the admin panel to complete first-run setup.",
                "=" * 70,
            ])
        )
    else:
        logger.info("Admin password check passed — default password has been changed.")


def _audit_plaintext_api_keys(conn) -> None:
    """
    Called once per process startup inside init_db().

    Scans every active row in trading_accounts and counts how many have
    api_key_enc values that are NOT prefixed with "enc:v1:" — i.e. they
    are still stored as plaintext, meaning the encrypt-at-rest migration
    has not been run yet.

    Actions taken:
      • Writes PLAINTEXT_API_KEYS_FOUND = "true" / "false" into
        platform_settings so the admin panel can surface a banner.
      • Logs a CRITICAL message with the exact count and the command
        needed to fix it, so operators catch it in log aggregators.
      • Does NOT auto-encrypt here: the migration script is a deliberate,
        operator-confirmed action (it needs FIELD_ENCRYPTION_KEY to be
        set and the DB to be backed up first).

    This function is intentionally read-only with respect to the
    trading_accounts table so it can never corrupt stored credentials.
    """
    now = datetime.utcnow().isoformat()

    rows = fetchall(conn,
        "SELECT id, user_id, api_key_enc FROM trading_accounts WHERE is_active = 1")

    plaintext_ids: list[int] = [
        row["id"] for row in rows
        if row["api_key_enc"] and not is_encrypted(row["api_key_enc"])
    ]

    found = len(plaintext_ids) > 0
    flag  = "true" if found else "false"

    execute(conn, """
        INSERT INTO platform_settings (key, value, updated_at)
        VALUES (:k, :v, :now)
        ON CONFLICT(key) DO UPDATE
            SET value      = excluded.value,
                updated_at = excluded.updated_at
    """, {"k": "PLAINTEXT_API_KEYS_FOUND", "v": flag, "now": now})

    if found:
        # Also store the count so the admin panel can show a specific number.
        execute(conn, """
            INSERT INTO platform_settings (key, value, updated_at)
            VALUES (:k, :v, :now)
            ON CONFLICT(key) DO UPDATE
                SET value      = excluded.value,
                    updated_at = excluded.updated_at
        """, {"k": "PLAINTEXT_API_KEYS_COUNT", "v": str(len(plaintext_ids)), "now": now})

        logger.critical(
            "\n".join([
                "=" * 70,
                "CRITICAL SECURITY WARNING — PLAINTEXT API KEYS IN DATABASE",
                f"{len(plaintext_ids)} trading account(s) have their Oanda API key",
                "stored as PLAINTEXT. Anyone who reads the database file can steal",
                "live Oanda credentials and place or close trades on behalf of users.",
                "",
                "To encrypt all plaintext keys, run:",
                "    python scripts/migrate_encrypt_keys.py",
                "",
                "Before running the migration:",
                "  1. Ensure FIELD_ENCRYPTION_KEY is set in .env",
                "  2. Back up the database:",
                "       cp data/forexchautari.db data/forexchautari.db.bak",
                "  3. Run the migration (dry-run first is safe):",
                "       python scripts/migrate_encrypt_keys.py --dry-run",
                "       python scripts/migrate_encrypt_keys.py",
                "",
                f"Affected trading_account row IDs: {plaintext_ids}",
                "=" * 70,
            ])
        )
    else:
        logger.info(
            "API key encryption check passed — "
            f"all {len(rows)} active trading account(s) use encrypted keys."
        )


def init_db():
    pk = pk_column()
    with get_db() as conn:
        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS users (
            id            {pk},
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            is_active     INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT NOT NULL,
            last_login    TEXT,
            full_name     TEXT,
            phone         TEXT
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id          {pk},
            user_id     INTEGER NOT NULL REFERENCES users(id),
            plan        TEXT NOT NULL DEFAULT 'free',
            status      TEXT NOT NULL DEFAULT 'active',
            started_at  TEXT NOT NULL,
            expires_at  TEXT,
            auto_trade  INTEGER NOT NULL DEFAULT 0,
            max_pairs   INTEGER NOT NULL DEFAULT 1,
            UNIQUE(user_id)
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS trading_accounts (
            id              {pk},
            user_id         INTEGER NOT NULL REFERENCES users(id),
            account_name    TEXT NOT NULL,
            broker          TEXT NOT NULL DEFAULT 'oanda',
            api_key_enc     TEXT NOT NULL,
            account_id      TEXT NOT NULL,
            environment     TEXT NOT NULL DEFAULT 'practice',
            is_active       INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT NOT NULL,
            verified_at     TEXT
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS trades (
            id              {pk},
            user_id         INTEGER REFERENCES users(id),
            pair            TEXT NOT NULL,
            signal          TEXT NOT NULL,
            entry_price     REAL,
            exit_price      REAL,
            units           INTEGER,
            pnl             REAL,
            status          TEXT NOT NULL DEFAULT 'open',
            trade_type      TEXT NOT NULL DEFAULT 'auto',
            broker_trade_id TEXT,
            opened_at       TEXT NOT NULL,
            closed_at       TEXT
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS signals_log (
            id          {pk},
            pair        TEXT NOT NULL,
            signal      TEXT NOT NULL,
            prob_up     REAL,
            confidence  TEXT,
            regime      TEXT,
            tradeable   INTEGER,
            price       REAL,
            created_at  TEXT NOT NULL
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          {pk},
            user_id     INTEGER REFERENCES users(id),
            event       TEXT NOT NULL,
            detail      TEXT,
            ip_address  TEXT,
            created_at  TEXT NOT NULL
        )""")

        execute(conn, f"""
        CREATE TABLE IF NOT EXISTS notifications (
            id          {pk},
            user_id     INTEGER REFERENCES users(id),
            title       TEXT NOT NULL,
            message     TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'info',
            is_read     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        )""")

        execute(conn, """
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            role        TEXT NOT NULL,
            expires     TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )""")

        # ── Task 3.3 migration: add outcome tracking to signals_log ──────────
        # Must be done in separate transactions because PostgreSQL aborts the
        # transaction on any error, even if caught. Use separate get_db() contexts
        # for the ALTER TABLE statements so failures don't poison the main init.

    # End main transaction

    # Attempt to add outcome and exit_price columns if they don't exist
    # (e.g., during upgrades). Each one gets its own transaction.
    try:
        with get_db() as conn:
            execute(conn, "ALTER TABLE signals_log ADD COLUMN outcome INTEGER")
    except Exception:
        pass  # column already exists

    try:
        with get_db() as conn:
            execute(conn, "ALTER TABLE signals_log ADD COLUMN exit_price REAL")
    except Exception:
        pass  # column already exists

    # Resume main transaction for remaining initialization tasks
    with get_db() as conn:
        execute(conn, """
        CREATE TABLE IF NOT EXISTS user_trading_settings (
            user_id              INTEGER PRIMARY KEY REFERENCES users(id),
            mode                 TEXT NOT NULL DEFAULT 'signals_only',
            auto_trade_enabled   INTEGER NOT NULL DEFAULT 0,
            trading_account_id   INTEGER REFERENCES trading_accounts(id),
            threshold            REAL NOT NULL DEFAULT 0.55,
            risk_pct             REAL NOT NULL DEFAULT 0.01,
            sl_pips              REAL NOT NULL DEFAULT 20,
            tp_pips              REAL NOT NULL DEFAULT 40,
            units                INTEGER NOT NULL DEFAULT 1000,
            max_positions        INTEGER NOT NULL DEFAULT 3,
            use_regime_filter    INTEGER NOT NULL DEFAULT 1,
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL
        )""")

        execute(conn, """
        CREATE TABLE IF NOT EXISTS platform_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )""")

        admin = fetchone(conn, "SELECT id FROM users WHERE username=:u", {"u": "admin"})
        if not admin:
            salt    = secrets.token_hex(32)
            pw_hash = hash_password("admin123", salt)
            now     = datetime.utcnow().isoformat()
            execute(conn, """
                INSERT INTO users (username,email,password_hash,salt,role,created_at,full_name)
                VALUES (:u,:e,:h,:s,'admin',:now,'System Administrator')
            """, {"u": "admin", "e": "admin@forexchautari.com", "h": pw_hash, "s": salt, "now": now})

            uid = fetchone(conn, "SELECT id FROM users WHERE username=:u", {"u": "admin"})["id"]

            execute(conn, """
                INSERT INTO subscriptions (user_id,plan,status,started_at,auto_trade,max_pairs)
                VALUES (:uid,'enterprise','active',:now,1,99)
            """, {"uid": uid, "now": now})

            execute(conn, """
                INSERT INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (:uid,'signals_only',0,0.55,0.01,20,40,1000,3,1,:now,:now)
                ON CONFLICT (user_id) DO NOTHING
            """, {"uid": uid, "now": now})

            logger.critical(
                "SECURITY: Default admin account created with password 'admin123'. "
                "You MUST change this before using the system."
            )

        # ── Security audits (run on every startup) ─────────────────────────
        _audit_admin_default_password(conn)
        _audit_plaintext_api_keys(conn)

        # Backfill settings for existing users after upgrades.
        now = datetime.utcnow().isoformat()
        rows = fetchall(conn, """
            SELECT u.id FROM users u
            LEFT JOIN user_trading_settings s ON s.user_id=u.id
            WHERE s.user_id IS NULL
        """)
        for row in rows:
            execute(conn, """
                INSERT INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (:uid,'signals_only',0,0.55,0.01,20,40,1000,3,1,:now,:now)
                ON CONFLICT (user_id) DO NOTHING
            """, {"uid": row["id"], "now": now})

        default_settings = {
            "auto_fetch_enabled": "1",
            "auto_train_enabled": "1",
            "auto_trade_enabled": "1",
            "fetch_count": "100",
            "fetch_time_utc": "08:05",
            "signal_check_time_utc": "08:10",
            "daily_summary_time_utc": "08:30",
            "train_time_utc": "00:01",
            "train_weekday_utc": "0",
            "minimum_wf_accuracy": "0.53",
            "minimum_profit_factor": "1.2",
        }
        for key, value in default_settings.items():
            execute(conn, """
                INSERT INTO platform_settings (key,value,updated_at)
                VALUES (:k,:v,:now)
                ON CONFLICT (key) DO NOTHING
            """, {"k": key, "v": value, "now": now})

    logger.info(f"Database ready ({'PostgreSQL' if IS_POSTGRES else 'SQLite'})")


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return dk.hex()

_hash_password = hash_password

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return secrets.compare_digest(hash_password(password, salt), stored_hash)


# ── Sessions ──────────────────────────────────────────────────────────────────

SESSIONS: dict = {}

def create_session(user_id: int, role: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=12)).isoformat()
    SESSIONS[token] = {
        "user_id": user_id,
        "role":    role,
        "expires": expires,
    }
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, """
            INSERT INTO sessions (token,user_id,role,expires,created_at)
            VALUES (:token,:uid,:role,:exp,:now)
            ON CONFLICT(token) DO UPDATE SET user_id=excluded.user_id, role=excluded.role, expires=excluded.expires
        """, {"token": token, "uid": user_id, "role": role, "exp": expires, "now": now})
    return token

def get_session(token: str) -> dict | None:
    s = SESSIONS.get(token)
    if not s:
        with get_db() as conn:
            row = fetchone(conn,
                "SELECT token,user_id,role,expires FROM sessions WHERE token=:t",
                {"t": token})
            if not row:
                return None
            s = row
            SESSIONS[token] = s
    if datetime.utcnow().isoformat() > s["expires"]:
        SESSIONS.pop(token, None)
        with get_db() as conn:
            execute(conn, "DELETE FROM sessions WHERE token=:t", {"t": token})
        return None
    return s

def destroy_session(token: str):
    SESSIONS.pop(token, None)
    with get_db() as conn:
        execute(conn, "DELETE FROM sessions WHERE token=:t", {"t": token})


# ── Users ─────────────────────────────────────────────────────────────────────

PLAN_LIMITS = {
    "free":       {"auto_trade": 0, "max_pairs": 1},
    "basic":      {"auto_trade": 0, "max_pairs": 2},
    "pro":        {"auto_trade": 1, "max_pairs": 4},
    "enterprise": {"auto_trade": 1, "max_pairs": 99},
}

def create_user(username: str, email: str, password: str,
                full_name: str = "", phone: str = "",
                role: str = "user", plan: str = "free") -> dict:
    salt    = secrets.token_hex(32)
    pw_hash = hash_password(password, salt)
    now     = datetime.utcnow().isoformat()
    limits  = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    with get_db() as conn:
        execute(conn, """
            INSERT INTO users (username,email,password_hash,salt,role,created_at,full_name,phone)
            VALUES (:u,:e,:h,:s,:role,:now,:full_name,:phone)
        """, {"u": username, "e": email, "h": pw_hash, "s": salt, "role": role, "now": now, "full_name": full_name, "phone": phone})
        
        uid = fetchone(conn, "SELECT id FROM users WHERE username=:u", {"u": username})["id"]
        
        execute(conn, """
            INSERT INTO subscriptions (user_id,plan,status,started_at,auto_trade,max_pairs)
            VALUES (:uid,:plan,'active',:now,:at,:mp)
        """, {"uid": uid, "plan": plan, "now": now, "at": limits["auto_trade"], "mp": limits["max_pairs"]})
        
        execute(conn, """
            INSERT INTO user_trading_settings
            (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
             max_positions,use_regime_filter,created_at,updated_at)
            VALUES (:uid,'signals_only',0,0.55,0.01,20,40,1000,3,1,:now,:now)
        """, {"uid": uid, "now": now})
        
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,created_at)
            VALUES (:uid,:event,:detail,:now)
        """, {"uid": uid, "event": "register", "detail": f"New {role} account: {email}", "now": now})
        
        execute(conn, """
            INSERT INTO notifications (user_id,title,message,type,created_at)
            VALUES (:uid,:title,:msg,:type,:now)
        """, {"uid": uid, "title": "Welcome to ForexChautari!", "msg": f"Your {plan.title()} account is ready. Explore the signals tab to get started.", "type": "success", "now": now})
    
    return {"id": uid, "username": username, "role": role, "plan": plan}


def authenticate_user(username: str, password: str, ip: str = "") -> dict | None:
    with get_db() as conn:
        row = fetchone(conn,
            "SELECT * FROM users WHERE (username=:u OR email=:u) AND is_active=1",
            {"u": username})
        if not row:
            return None
        if not verify_password(password, row["salt"], row["password_hash"]):
            execute(conn, """
                INSERT INTO audit_log (user_id,event,detail,ip_address,created_at)
                VALUES (:uid,:event,:detail,:ip,:now)
            """, {"uid": row["id"], "event": "login_fail", "detail": "Bad password", "ip": ip, "now": datetime.utcnow().isoformat()})
            return None
        
        now = datetime.utcnow().isoformat()
        execute(conn, "UPDATE users SET last_login=:now WHERE id=:id", {"now": now, "id": row["id"]})
        
        sub = fetchone(conn, "SELECT * FROM subscriptions WHERE user_id=:id", {"id": row["id"]})
        
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,ip_address,created_at)
            VALUES (:uid,:event,:detail,:ip,:now)
        """, {"uid": row["id"], "event": "login_ok", "detail": "Successful login", "ip": ip, "now": now})
        
        return {
            "id":         row["id"],
            "username":   row["username"],
            "email":      row["email"],
            "full_name":  row["full_name"] or "",
            "phone":      row["phone"] or "",
            "role":       row["role"],
            "plan":       sub["plan"] if sub else "free",
            "auto_trade": bool(sub["auto_trade"]) if sub else False,
            "max_pairs":  sub["max_pairs"] if sub else 1,
        }


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        row = fetchone(conn, """
            SELECT u.*, s.plan, s.status, s.auto_trade, s.max_pairs
            FROM users u LEFT JOIN subscriptions s ON s.user_id=u.id
            WHERE u.id=:id
        """, {"id": user_id})
        return row


def get_all_users() -> list:
    with get_db() as conn:
        rows = fetchall(conn, """
            SELECT u.id, u.username, u.email, u.full_name, u.phone, u.role,
                   u.is_active, u.last_login, u.created_at,
                   s.plan, s.status, s.auto_trade, s.max_pairs
            FROM users u LEFT JOIN subscriptions s ON s.user_id=u.id
            ORDER BY u.created_at DESC
        """)
        return rows


def update_user_plan(user_id: int, plan: str, admin_id: int):
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    now    = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, """
            UPDATE subscriptions SET plan=:plan, auto_trade=:at, max_pairs=:mp WHERE user_id=:uid
        """, {"plan": plan, "at": limits["auto_trade"], "mp": limits["max_pairs"], "uid": user_id})
        
        if not limits["auto_trade"]:
            execute(conn, """
                UPDATE user_trading_settings
                SET mode='signals_only', auto_trade_enabled=0, updated_at=:now
                WHERE user_id=:uid
            """, {"now": now, "uid": user_id})
        
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (:admin,:event,:detail,:now)
        """, {"admin": admin_id, "event": "plan_change", "detail": f"Changed user {user_id} to {plan}", "now": now})
        
        execute(conn, """
            INSERT INTO notifications (user_id,title,message,type,created_at)
            VALUES (:uid,:title,:msg,:type,:now)
        """, {"uid": user_id, "title": "Plan Updated", "msg": f"Your plan has been updated to {plan.title()}. Refresh to see new features.", "type": "info", "now": now})


def update_user_profile(user_id: int, full_name: str = None,
                        phone: str = None, email: str = None):
    fields, vals = [], {}
    if full_name is not None: fields.append("full_name=:fn"); vals["fn"] = full_name
    if phone is not None:     fields.append("phone=:ph");     vals["ph"] = phone
    if email is not None:     fields.append("email=:em");     vals["em"] = email
    if not fields:
        return
    vals["uid"] = user_id
    with get_db() as conn:
        execute(conn, f"UPDATE users SET {','.join(fields)} WHERE id=:uid", vals)


def update_user_password(user_id: int, new_password: str):
    salt    = secrets.token_hex(32)
    pw_hash = hash_password(new_password, salt)
    now     = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, "UPDATE users SET password_hash=:h, salt=:s WHERE id=:id",
                     {"h": pw_hash, "s": salt, "id": user_id})
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (:uid,:event,:detail,:now)
        """, {"uid": user_id, "event": "password_change", "detail": "Password changed by user", "now": now})


def deactivate_user(user_id: int, admin_id: int):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, "UPDATE users SET is_active=0 WHERE id=:id", {"id": user_id})
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (:admin,:event,:detail,:now)
        """, {"admin": admin_id, "event": "deactivate", "detail": f"Deactivated user {user_id}", "now": now})


def reactivate_user(user_id: int, admin_id: int):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, "UPDATE users SET is_active=1 WHERE id=:id", {"id": user_id})
        execute(conn, """
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (:admin,:event,:detail,:now)
        """, {"admin": admin_id, "event": "reactivate", "detail": f"Reactivated user {user_id}", "now": now})


# ── Trading accounts ──────────────────────────────────────────────────────────

def add_trading_account(
    user_id:      int,
    account_name: str,
    api_key:      str,
    account_id:   str,
    environment:  str = "practice",
    is_admin:     bool = False,
) -> int:
    """
    Store a new Oanda trading account for a user.

    Live environment policy (v1):
      - Live trading is disabled for all non-admin users unconditionally.
      - Even for admins, live trading requires LIVE_TRADING_ENABLED=true in .env.
      - This is enforced here at the DB layer, not just in the UI, so it cannot
        be bypassed by crafting a direct API call or calling this function
        from a script.

    The api_key is encrypted with Fernet (AES-128-CBC + HMAC-SHA256)
    before it is written to the database.
    """
    if environment == "live":
        live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
        if not live_enabled:
            raise ValueError(
                "Live trading is disabled in this deployment. "
                "Set LIVE_TRADING_ENABLED=true in .env to enable it. "
                "WARNING: The v1 model has ~51-53% walk-forward accuracy — "
                "live trading with real money is not recommended."
            )
        if not is_admin:
            raise ValueError(
                "Live trading accounts can only be connected by administrators. "
                "Please use practice mode, or contact your admin."
            )
        logger.warning(
            f"LIVE trading account being added: user_id={user_id} "
            f"account_id={account_id} — admin override active"
        )

    now = datetime.utcnow().isoformat()

    # Encrypt before the value ever touches the database.
    encrypted_key = encrypt(api_key)

    with get_db() as conn:
        execute(conn, """
            INSERT INTO trading_accounts
                (user_id, account_name, broker, api_key_enc,
                 account_id, environment, created_at, verified_at)
            VALUES (:uid, :name, 'oanda', :key, :acc_id, :env, :now, :now)
        """, {"uid": user_id, "name": account_name, "key": encrypted_key, "acc_id": account_id, "env": environment, "now": now})

        row = fetchone(conn,
            "SELECT id FROM trading_accounts WHERE user_id=:uid ORDER BY id DESC LIMIT 1",
            {"uid": user_id})

    new_id = row["id"]
    logger.info(
        f"Trading account added: user_id={user_id} "
        f"account_id={account_id} env={environment} db_id={new_id}"
    )
    return new_id


def get_trading_accounts(user_id: int) -> list:
    """
    Return all active trading accounts for a user with decrypted API keys.

    decrypt() handles legacy plaintext rows transparently (returns as-is),
    so the app stays fully functional during the gap between deploying this
    code and running the migration script.

    If decryption fails for a row (wrong key / corrupted data), that account
    is excluded from the result with an error log — the rest still work.
    """
    with get_db() as conn:
        rows = fetchall(conn,
            "SELECT * FROM trading_accounts WHERE user_id=:uid AND is_active=1",
            {"uid": user_id})

    result = []
    for row in rows:
        try:
            row["api_key_enc"] = decrypt(row["api_key_enc"])
        except RuntimeError as exc:
            logger.error(
                f"Failed to decrypt API key for trading_account "
                f"id={row.get('id')} user_id={user_id}: {exc}"
            )
            continue
        result.append(row)

    return result


def remove_trading_account(account_id: int, user_id: int):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn,
            "UPDATE trading_accounts SET is_active=0 WHERE id=:id AND user_id=:uid",
            {"id": account_id, "uid": user_id})
        execute(conn, """
            UPDATE user_trading_settings
            SET mode='signals_only', auto_trade_enabled=0, trading_account_id=NULL, updated_at=:now
            WHERE user_id=:uid AND trading_account_id=:acc_id
        """, {"now": now, "uid": user_id, "acc_id": account_id})


# ── Trading settings ──────────────────────────────────────────────────────────

TRADING_SETTING_FIELDS = {
    "mode",
    "auto_trade_enabled",
    "trading_account_id",
    "threshold",
    "risk_pct",
    "sl_pips",
    "tp_pips",
    "units",
    "max_positions",
    "use_regime_filter",
}


def ensure_trading_settings(user_id: int) -> dict:
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        row = fetchone(conn,
            "SELECT * FROM user_trading_settings WHERE user_id=:uid", {"uid": user_id})
        if not row:
            execute(conn, """
                INSERT INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (:uid,'signals_only',0,0.55,0.01,20,40,1000,3,1,:now,:now)
            """, {"uid": user_id, "now": now})
            row = fetchone(conn,
                "SELECT * FROM user_trading_settings WHERE user_id=:uid", {"uid": user_id})
        return row


def get_user_trading_settings(user_id: int) -> dict:
    row = ensure_trading_settings(user_id)
    row["auto_trade_enabled"] = bool(row.get("auto_trade_enabled"))
    row["use_regime_filter"] = bool(row.get("use_regime_filter"))
    return row


def update_user_trading_settings(user_id: int, **settings):
    allowed = {k: v for k, v in settings.items() if k in TRADING_SETTING_FIELDS}
    if not allowed:
        return

    if allowed.get("mode") not in (None, "signals_only", "manual", "auto"):
        raise ValueError("Invalid trading mode")

    if "auto_trade_enabled" in allowed:
        allowed["auto_trade_enabled"] = int(bool(allowed["auto_trade_enabled"]))
    if "use_regime_filter" in allowed:
        allowed["use_regime_filter"] = int(bool(allowed["use_regime_filter"]))

    allowed["updated_at"] = datetime.utcnow().isoformat()
    fields = ",".join(f"{k}=:{k}" for k in allowed.keys())
    ensure_trading_settings(user_id)
    with get_db() as conn:
        execute(conn, f"UPDATE user_trading_settings SET {fields} WHERE user_id=:uid", {**allowed, "uid": user_id})


def get_auto_trade_users() -> list:
    """Return active users eligible for scheduled auto-trading."""
    with get_db() as conn:
        rows = fetchall(conn, """
            SELECT u.id, u.username, u.email, u.role,
                   s.plan, s.auto_trade AS plan_auto_trade,
                   ts.mode, ts.auto_trade_enabled, ts.trading_account_id,
                   ts.threshold, ts.risk_pct, ts.sl_pips, ts.tp_pips,
                   ts.units, ts.max_positions, ts.use_regime_filter
            FROM users u
            JOIN subscriptions s ON s.user_id=u.id
            JOIN user_trading_settings ts ON ts.user_id=u.id
            JOIN trading_accounts ta ON ta.user_id=u.id
                AND ta.is_active=1
                AND (ts.trading_account_id IS NULL OR ts.trading_account_id=ta.id)
            WHERE u.is_active=1
              AND s.status='active'
              AND s.auto_trade=1
              AND ts.auto_trade_enabled=1
              AND ts.mode='auto'
            GROUP BY u.id, u.username, u.email, u.role,
                   s.plan, s.auto_trade,
                   ts.mode, ts.auto_trade_enabled, ts.trading_account_id,
                   ts.threshold, ts.risk_pct, ts.sl_pips, ts.tp_pips,
                   ts.units, ts.max_positions, ts.use_regime_filter
            ORDER BY u.id
        """)
        result = []
        for row in rows:
            row["auto_trade_enabled"] = bool(row["auto_trade_enabled"])
            row["use_regime_filter"] = bool(row["use_regime_filter"])
            result.append(row)
        return result


# ── Platform settings ─────────────────────────────────────────────────────────

def get_platform_settings() -> dict:
    with get_db() as conn:
        rows = fetchall(conn, "SELECT key,value FROM platform_settings")
        return {r["key"]: r["value"] for r in rows}


def is_admin_password_default() -> bool:
    """
    Returns True if the admin account still has the seeded 'admin123' password.
    Reads from platform_settings (written by _audit_admin_default_password at
    startup) so callers never need to touch the users table directly.
    """
    settings = get_platform_settings()
    return settings.get("ADMIN_PASSWORD_CHANGED", "false").strip().lower() != "true"


def has_plaintext_api_keys() -> bool:
    """
    Returns True if any active trading account still has an unencrypted API key.
    Reads from platform_settings (written by _audit_plaintext_api_keys at startup).
    Fast: single indexed key lookup.
    """
    settings = get_platform_settings()
    return settings.get("PLAINTEXT_API_KEYS_FOUND", "false").strip().lower() == "true"


def plaintext_api_keys_count() -> int:
    """Return the number of active trading accounts with plaintext API keys."""
    settings = get_platform_settings()
    try:
        return int(settings.get("PLAINTEXT_API_KEYS_COUNT", "0"))
    except (ValueError, TypeError):
        return 0


def update_platform_settings(settings: dict):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        for key, value in settings.items():
            execute(conn, """
                INSERT INTO platform_settings (key,value,updated_at)
                VALUES (:k,:v,:now)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, {"k": str(key), "v": str(value), "now": now})


def setting_bool(settings: dict, key: str, default: bool = False) -> bool:
    val = str(settings.get(key, "1" if default else "0")).strip().lower()
    return val in ("1", "true", "yes", "on")


# ── Trades ────────────────────────────────────────────────────────────────────

def log_trade(user_id: int, pair: str, signal: str, entry_price: float,
              units: int, trade_type: str = "auto",
              broker_trade_id: str = "") -> int:
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, """
            INSERT INTO trades
            (user_id,pair,signal,entry_price,units,trade_type,broker_trade_id,opened_at,status)
            VALUES (:uid,:pair,:signal,:price,:units,:type,:bid,:now,'open')
        """, {"uid": user_id, "pair": pair, "signal": signal, "price": entry_price, "units": units, "type": trade_type, "bid": broker_trade_id, "now": now})
        
        row = fetchone(conn,
            "SELECT id FROM trades WHERE user_id=:uid ORDER BY id DESC LIMIT 1", {"uid": user_id})
        return row["id"]


def close_trade(trade_id: int, exit_price: float, pnl: float):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        execute(conn, """
            UPDATE trades SET exit_price=:ep,pnl=:pnl,status='closed',closed_at=:now WHERE id=:id
        """, {"ep": exit_price, "pnl": pnl, "now": now, "id": trade_id})


def get_user_trades(user_id: int | None = None, limit: int = 50) -> list:
    """Get trades. user_id=None returns all trades (admin use)."""
    with get_db() as conn:
        if user_id is None:
            rows = fetchall(conn, """
                SELECT t.*, u.username FROM trades t
                LEFT JOIN users u ON u.id=t.user_id
                ORDER BY t.opened_at DESC LIMIT :limit
            """, {"limit": limit})
        else:
            rows = fetchall(conn, """
                SELECT * FROM trades WHERE user_id=:uid
                ORDER BY opened_at DESC LIMIT :limit
            """, {"uid": user_id, "limit": limit})
        return rows


def get_trade_stats(user_id: int | None = None) -> dict:
    """Return aggregated trade statistics."""
    with get_db() as conn:
        if user_id is not None:
            total = execute(conn, "SELECT COUNT(*) FROM trades WHERE user_id=:uid", {"uid": user_id}).scalar()
            closed = fetchone(conn,
                "SELECT COUNT(*) as count,SUM(pnl) as total_pnl,AVG(pnl) as avg_pnl FROM trades WHERE user_id=:uid AND status='closed'",
                {"uid": user_id})
            wins = execute(conn, "SELECT COUNT(*) FROM trades WHERE user_id=:uid AND pnl>0", {"uid": user_id}).scalar()
        else:
            total = execute(conn, "SELECT COUNT(*) FROM trades").scalar()
            closed = fetchone(conn, "SELECT COUNT(*) as count,SUM(pnl) as total_pnl,AVG(pnl) as avg_pnl FROM trades WHERE status='closed'")
            wins = execute(conn, "SELECT COUNT(*) FROM trades WHERE pnl>0").scalar()

        closed_count = closed["count"] if closed else 0
        return {
            "total_trades":  total,
            "closed_trades": closed_count,
            "total_pnl":     round(float(closed["total_pnl"] or 0), 2) if closed else 0.0,
            "avg_pnl":       round(float(closed["avg_pnl"] or 0), 2) if closed else 0.0,
            "wins":          wins,
            "win_rate":      round(wins / closed_count * 100, 1) if closed_count else 0,
        }


# ── Signals ───────────────────────────────────────────────────────────────────

def log_signal(pair: str, signal: str, prob_up: float, confidence: str,
               regime: str, tradeable: bool, price: float):
    with get_db() as conn:
        recent = fetchone(conn, """
            SELECT id FROM signals_log
            WHERE pair=:pair AND signal=:sig AND ABS(prob_up - :prob) < 0.00001
              AND confidence=:conf AND regime=:reg AND tradeable=:tradeable
              AND created_at >= :time
            ORDER BY created_at DESC LIMIT 1
        """, {
            "pair": pair, "sig": signal, "prob": prob_up, "conf": confidence, "reg": regime, "tradeable": int(tradeable),
            "time": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
        })
        if recent:
            return
        execute(conn, """
            INSERT INTO signals_log
            (pair,signal,prob_up,confidence,regime,tradeable,price,created_at)
            VALUES (:pair,:sig,:prob,:conf,:reg,:tradeable,:price,:now)
        """, {"pair": pair, "sig": signal, "prob": prob_up, "conf": confidence, "reg": regime,
              "tradeable": int(tradeable), "price": price, "now": datetime.utcnow().isoformat()})


def get_signals_log(limit: int = 100, pair: str = None) -> list:
    with get_db() as conn:
        if pair:
            rows = fetchall(conn,
                "SELECT * FROM signals_log WHERE pair=:pair ORDER BY created_at DESC LIMIT :limit",
                {"pair": pair, "limit": limit})
        else:
            rows = fetchall(conn,
                "SELECT * FROM signals_log ORDER BY created_at DESC LIMIT :limit", {"limit": limit})
        return rows


# ── Audit & notifications ─────────────────────────────────────────────────────

def get_audit_log(limit: int = 100) -> list:
    with get_db() as conn:
        rows = fetchall(conn, """
            SELECT a.*, u.username FROM audit_log a
            LEFT JOIN users u ON u.id=a.user_id
            ORDER BY a.created_at DESC LIMIT :limit
        """, {"limit": limit})
        return rows


def get_notifications(user_id: int, unread_only: bool = False) -> list:
    with get_db() as conn:
        if unread_only:
            rows = fetchall(conn,
                "SELECT * FROM notifications WHERE user_id=:uid AND is_read=0 ORDER BY created_at DESC LIMIT 20",
                {"uid": user_id})
        else:
            rows = fetchall(conn,
                "SELECT * FROM notifications WHERE user_id=:uid ORDER BY created_at DESC LIMIT 20",
                {"uid": user_id})
        return rows


def mark_notifications_read(user_id: int):
    with get_db() as conn:
        execute(conn, "UPDATE notifications SET is_read=1 WHERE user_id=:uid", {"uid": user_id})


# ── Platform stats ────────────────────────────────────────────────────────────

def get_platform_stats() -> dict:
    with get_db() as conn:
        total_users   = execute(conn, "SELECT COUNT(*) FROM users WHERE is_active=1").scalar()
        total_trades  = execute(conn, "SELECT COUNT(*) FROM trades").scalar()
        total_signals = execute(conn, "SELECT COUNT(*) FROM signals_log").scalar()
        total_pnl     = execute(conn, "SELECT SUM(pnl) FROM trades WHERE status='closed'").scalar()
        plan_counts   = fetchall(conn,
            "SELECT plan,COUNT(*) as cnt FROM subscriptions GROUP BY plan")
        today_start   = datetime.utcnow().strftime("%Y-%m-%d")
        new_today     = fetchone(conn,
            "SELECT COUNT(*) AS c FROM users WHERE created_at >= :today",
            {"today": today_start})["c"]
        return {
            "total_users":   total_users,
            "total_trades":  total_trades,
            "total_signals": total_signals,
            "total_pnl":     round(float(total_pnl or 0), 2),
            "new_today":     new_today,
            "plans":         {r["plan"]: r["cnt"] for r in plan_counts},
        }


def get_latest_unresolved_signal(pair: str) -> dict | None:
    """
    Return the most recent tradeable signal for `pair` whose outcome has not
    yet been resolved (outcome IS NULL).

    Only signals with a recorded price > 0 are returned — we need the entry
    price to compute whether the prediction was correct.

    Called by PaperTrader._resolve_previous_signal_outcome() at the start of
    every signal check cycle.
    """
    with get_db() as conn:
        row = fetchone(conn, """
            SELECT *
            FROM   signals_log
            WHERE  pair      = :pair
              AND  tradeable = 1
              AND  outcome   IS NULL
              AND  price     IS NOT NULL
              AND  price     > 0
            ORDER  BY created_at DESC
            LIMIT  1
        """, {"pair": pair})
    return row


def resolve_signal_outcome(signal_id: int, outcome: int, exit_price: float) -> None:
    """
    Record the outcome of a previously unresolved paper signal.

    outcome   — 1 = win (price moved in predicted direction), 0 = loss
    exit_price — live mid price at time of resolution (next signal check)

    Called by PaperTrader._resolve_previous_signal_outcome().
    """
    with get_db() as conn:
        execute(conn, """
            UPDATE signals_log
            SET    outcome    = :outcome,
                   exit_price = :exit_price
            WHERE  id = :id
              AND  outcome IS NULL   -- idempotency guard
        """, {"outcome": outcome, "exit_price": exit_price, "id": signal_id})


def get_paper_signal_stats(pair: str, since_date: str = "2000-01-01") -> dict:
    """
    Return aggregated paper signal statistics for `pair` since `since_date`.

    Only tradeable=1 signals with a resolved outcome are counted; regime-blocked
    and below-threshold signals are excluded from the win-rate calculation so
    the number reflects what the auto-trader would actually have done.

    Returns:
        resolved   — total resolved tradeable signals
        wins       — number of winning signals
        losses     — number of losing signals
        win_rate   — float 0–1 (0.0 if no resolved signals)

    Called by paper_validator.check_and_promote_model().
    """
    with get_db() as conn:
        row = fetchone(conn, """
            SELECT
                COUNT(*)                                            AS resolved_count,
                SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END)       AS wins
            FROM   signals_log
            WHERE  pair      = :pair
              AND  tradeable = 1
              AND  outcome   IS NOT NULL
              AND  created_at >= :since
        """, {"pair": pair, "since": since_date})

    resolved = int(row["resolved_count"] or 0) if row else 0
    wins     = int(row["wins"]          or 0) if row else 0
    win_rate = wins / resolved if resolved > 0 else 0.0

    return {
        "resolved": resolved,
        "wins":     wins,
        "losses":   resolved - wins,
        "win_rate": round(win_rate, 4),
    }
