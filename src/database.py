"""
SQLite database layer for ForexChautari.
All fixes applied:
  - get_user_trades accepts optional user_id (None = all users for admin)
  - _hash_password exported as public verify/hash functions
  - notifications table added
  - reactivate_user added
  - get_user_by_id added
  - update_user_profile added
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from contextlib import contextmanager
from src.logger import get_logger

logger = get_logger("database")

DB_PATH = os.getenv("DB_PATH", "data/forexchautari.db")


@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
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
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            plan        TEXT NOT NULL DEFAULT 'free',
            status      TEXT NOT NULL DEFAULT 'active',
            started_at  TEXT NOT NULL,
            expires_at  TEXT,
            auto_trade  INTEGER NOT NULL DEFAULT 0,
            max_pairs   INTEGER NOT NULL DEFAULT 1,
            UNIQUE(user_id)
        );

        CREATE TABLE IF NOT EXISTS trading_accounts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL REFERENCES users(id),
            account_name    TEXT NOT NULL,
            broker          TEXT NOT NULL DEFAULT 'oanda',
            api_key_enc     TEXT NOT NULL,
            account_id      TEXT NOT NULL,
            environment     TEXT NOT NULL DEFAULT 'practice',
            is_active       INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT NOT NULL,
            verified_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
        );

        CREATE TABLE IF NOT EXISTS signals_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pair        TEXT NOT NULL,
            signal      TEXT NOT NULL,
            prob_up     REAL,
            confidence  TEXT,
            regime      TEXT,
            tradeable   INTEGER,
            price       REAL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            event       TEXT NOT NULL,
            detail      TEXT,
            ip_address  TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            title       TEXT NOT NULL,
            message     TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'info',
            is_read     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            role        TEXT NOT NULL,
            expires     TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

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
        );

        CREATE TABLE IF NOT EXISTS platform_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        """)

        admin = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
        if not admin:
            salt    = secrets.token_hex(32)
            pw_hash = hash_password("admin123", salt)
            now     = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO users (username,email,password_hash,salt,role,created_at,full_name)
                VALUES (?,?,?,?,'admin',?,'System Administrator')
            """, ("admin","admin@forexchautari.com", pw_hash, salt, now))
            uid = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()["id"]
            conn.execute("""
                INSERT INTO subscriptions (user_id,plan,status,started_at,auto_trade,max_pairs)
                VALUES (?,'enterprise','active',?,1,99)
            """, (uid, now))
            conn.execute("""
                INSERT OR IGNORE INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (?,'signals_only',0,0.55,0.01,20,40,1000,3,1,?,?)
            """, (uid, now, now))
            logger.info("Default admin created — username: admin  password: admin123")

        # Backfill settings for existing users after upgrades.
        now = datetime.utcnow().isoformat()
        rows = conn.execute("""
            SELECT u.id FROM users u
            LEFT JOIN user_trading_settings s ON s.user_id=u.id
            WHERE s.user_id IS NULL
        """).fetchall()
        for row in rows:
            conn.execute("""
                INSERT INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (?,'signals_only',0,0.55,0.01,20,40,1000,3,1,?,?)
            """, (row["id"], now, now))

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
            "minimum_wf_accuracy": "0.51",
            "minimum_profit_factor": "1.05",
        }
        for key, value in default_settings.items():
            conn.execute("""
                INSERT OR IGNORE INTO platform_settings (key,value,updated_at)
                VALUES (?,?,?)
            """, (key, value, now))

    logger.info(f"Database ready at {DB_PATH}")


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return dk.hex()

# Keep private alias for internal backward compat
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
        conn.execute("""
            INSERT OR REPLACE INTO sessions (token,user_id,role,expires,created_at)
            VALUES (?,?,?,?,?)
        """, (token, user_id, role, expires, now))
    return token

def get_session(token: str) -> dict | None:
    s = SESSIONS.get(token)
    if not s:
        with get_db() as conn:
            row = conn.execute(
                "SELECT token,user_id,role,expires FROM sessions WHERE token=?",
                (token,)
            ).fetchone()
            if not row:
                return None
            s = dict(row)
            SESSIONS[token] = s
    if datetime.utcnow().isoformat() > s["expires"]:
        SESSIONS.pop(token, None)
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        return None
    return s

def destroy_session(token: str):
    SESSIONS.pop(token, None)
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


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
        conn.execute("""
            INSERT INTO users (username,email,password_hash,salt,role,created_at,full_name,phone)
            VALUES (?,?,?,?,?,?,?,?)
        """, (username, email, pw_hash, salt, role, now, full_name, phone))
        uid = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()["id"]
        conn.execute("""
            INSERT INTO subscriptions (user_id,plan,status,started_at,auto_trade,max_pairs)
            VALUES (?,?,'active',?,?,?)
        """, (uid, plan, now, limits["auto_trade"], limits["max_pairs"]))
        conn.execute("""
            INSERT INTO user_trading_settings
            (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
             max_positions,use_regime_filter,created_at,updated_at)
            VALUES (?,'signals_only',0,0.55,0.01,20,40,1000,3,1,?,?)
        """, (uid, now, now))
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,created_at)
            VALUES (?,?,?,?)
        """, (uid, "register", f"New {role} account: {email}", now))
        conn.execute("""
            INSERT INTO notifications (user_id,title,message,type,created_at)
            VALUES (?,?,?,?,?)
        """, (uid, "Welcome to ForexChautari!",
              f"Your {plan.title()} account is ready. Explore the signals tab to get started.",
              "success", now))
    return {"id": uid, "username": username, "role": role, "plan": plan}


def authenticate_user(username: str, password: str, ip: str = "") -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE (username=? OR email=?) AND is_active=1",
            (username, username)
        ).fetchone()
        if not row:
            return None
        if not verify_password(password, row["salt"], row["password_hash"]):
            conn.execute("""
                INSERT INTO audit_log (user_id,event,detail,ip_address,created_at)
                VALUES (?,?,?,?,?)
            """, (row["id"],"login_fail","Bad password", ip, datetime.utcnow().isoformat()))
            return None
        now = datetime.utcnow().isoformat()
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (now, row["id"]))
        sub = conn.execute("SELECT * FROM subscriptions WHERE user_id=?", (row["id"],)).fetchone()
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,ip_address,created_at)
            VALUES (?,?,?,?,?)
        """, (row["id"],"login_ok","Successful login", ip, now))
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
        row = conn.execute("""
            SELECT u.*, s.plan, s.status, s.auto_trade, s.max_pairs
            FROM users u LEFT JOIN subscriptions s ON s.user_id=u.id
            WHERE u.id=?
        """, (user_id,)).fetchone()
        return dict(row) if row else None


def get_all_users() -> list:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username, u.email, u.full_name, u.phone, u.role,
                   u.is_active, u.last_login, u.created_at,
                   s.plan, s.status, s.auto_trade, s.max_pairs
            FROM users u LEFT JOIN subscriptions s ON s.user_id=u.id
            ORDER BY u.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def update_user_plan(user_id: int, plan: str, admin_id: int):
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    now    = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("""
            UPDATE subscriptions SET plan=?, auto_trade=?, max_pairs=? WHERE user_id=?
        """, (plan, limits["auto_trade"], limits["max_pairs"], user_id))
        if not limits["auto_trade"]:
            conn.execute("""
                UPDATE user_trading_settings
                SET mode='signals_only', auto_trade_enabled=0, updated_at=?
                WHERE user_id=?
            """, (now, user_id))
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (?,?,?,?)
        """, (admin_id, "plan_change", f"Changed user {user_id} to {plan}", now))
        conn.execute("""
            INSERT INTO notifications (user_id,title,message,type,created_at)
            VALUES (?,?,?,?,?)
        """, (user_id, "Plan Updated",
              f"Your plan has been updated to {plan.title()}. Refresh to see new features.",
              "info", now))


def update_user_profile(user_id: int, full_name: str = None,
                        phone: str = None, email: str = None):
    fields, vals = [], []
    if full_name is not None: fields.append("full_name=?"); vals.append(full_name)
    if phone is not None:     fields.append("phone=?");     vals.append(phone)
    if email is not None:     fields.append("email=?");     vals.append(email)
    if not fields:
        return
    vals.append(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {','.join(fields)} WHERE id=?", vals)


def update_user_password(user_id: int, new_password: str):
    salt    = secrets.token_hex(32)
    pw_hash = hash_password(new_password, salt)
    now     = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?",
                     (pw_hash, salt, user_id))
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (?,?,?,?)
        """, (user_id, "password_change", "Password changed by user", now))


def deactivate_user(user_id: int, admin_id: int):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (?,?,?,?)
        """, (admin_id, "deactivate", f"Deactivated user {user_id}", now))


def reactivate_user(user_id: int, admin_id: int):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active=1 WHERE id=?", (user_id,))
        conn.execute("""
            INSERT INTO audit_log (user_id,event,detail,created_at) VALUES (?,?,?,?)
        """, (admin_id, "reactivate", f"Reactivated user {user_id}", now))


# ── Trading accounts ──────────────────────────────────────────────────────────

def add_trading_account(user_id: int, account_name: str, api_key: str,
                        account_id: str, environment: str = "practice") -> int:
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO trading_accounts
            (user_id,account_name,broker,api_key_enc,account_id,environment,created_at,verified_at)
            VALUES (?,?,'oanda',?,?,?,?,?)
        """, (user_id, account_name, api_key, account_id, environment, now, now))
        row = conn.execute(
            "SELECT id FROM trading_accounts WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,)
        ).fetchone()
        return row["id"]


def get_trading_accounts(user_id: int) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trading_accounts WHERE user_id=? AND is_active=1", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def remove_trading_account(account_id: int, user_id: int):
    with get_db() as conn:
        conn.execute(
            "UPDATE trading_accounts SET is_active=0 WHERE id=? AND user_id=?",
            (account_id, user_id)
        )
        conn.execute("""
            UPDATE user_trading_settings
            SET mode='signals_only', auto_trade_enabled=0, trading_account_id=NULL, updated_at=?
            WHERE user_id=? AND trading_account_id=?
        """, (datetime.utcnow().isoformat(), user_id, account_id))


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
        row = conn.execute(
            "SELECT * FROM user_trading_settings WHERE user_id=?", (user_id,)
        ).fetchone()
        if not row:
            conn.execute("""
                INSERT INTO user_trading_settings
                (user_id,mode,auto_trade_enabled,threshold,risk_pct,sl_pips,tp_pips,units,
                 max_positions,use_regime_filter,created_at,updated_at)
                VALUES (?,'signals_only',0,0.55,0.01,20,40,1000,3,1,?,?)
            """, (user_id, now, now))
            row = conn.execute(
                "SELECT * FROM user_trading_settings WHERE user_id=?", (user_id,)
            ).fetchone()
        return dict(row)


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
    fields = ",".join(f"{k}=?" for k in allowed.keys())
    vals = list(allowed.values()) + [user_id]
    ensure_trading_settings(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE user_trading_settings SET {fields} WHERE user_id=?", vals)


def get_auto_trade_users() -> list:
    """Return active users eligible for scheduled auto-trading."""
    with get_db() as conn:
        rows = conn.execute("""
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
            GROUP BY u.id
            ORDER BY u.id
        """).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["auto_trade_enabled"] = bool(d["auto_trade_enabled"])
            d["use_regime_filter"] = bool(d["use_regime_filter"])
            result.append(d)
        return result


# ── Platform settings ─────────────────────────────────────────────────────────

def get_platform_settings() -> dict:
    with get_db() as conn:
        rows = conn.execute("SELECT key,value FROM platform_settings").fetchall()
        return {r["key"]: r["value"] for r in rows}


def update_platform_settings(settings: dict):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        for key, value in settings.items():
            conn.execute("""
                INSERT INTO platform_settings (key,value,updated_at)
                VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (str(key), str(value), now))


def setting_bool(settings: dict, key: str, default: bool = False) -> bool:
    val = str(settings.get(key, "1" if default else "0")).strip().lower()
    return val in ("1", "true", "yes", "on")


# ── Trades ────────────────────────────────────────────────────────────────────

def log_trade(user_id: int, pair: str, signal: str, entry_price: float,
              units: int, trade_type: str = "auto",
              broker_trade_id: str = "") -> int:
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO trades
            (user_id,pair,signal,entry_price,units,trade_type,broker_trade_id,opened_at,status)
            VALUES (?,?,?,?,?,?,?,?,'open')
        """, (user_id, pair, signal, entry_price, units, trade_type, broker_trade_id, now))
        row = conn.execute(
            "SELECT id FROM trades WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
        ).fetchone()
        return row["id"]


def close_trade(trade_id: int, exit_price: float, pnl: float):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("""
            UPDATE trades SET exit_price=?,pnl=?,status='closed',closed_at=? WHERE id=?
        """, (exit_price, pnl, now, trade_id))


def get_user_trades(user_id: int | None = None, limit: int = 50) -> list:
    """Get trades. user_id=None returns all trades (admin use)."""
    with get_db() as conn:
        if user_id is None:
            rows = conn.execute("""
                SELECT t.*, u.username FROM trades t
                LEFT JOIN users u ON u.id=t.user_id
                ORDER BY t.opened_at DESC LIMIT ?
            """, (limit,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM trades WHERE user_id=?
                ORDER BY opened_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_trade_stats(user_id: int | None = None) -> dict:
    """Return aggregated trade statistics."""
    with get_db() as conn:
        if user_id is not None:
            total  = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            closed = conn.execute(
                "SELECT COUNT(*),SUM(pnl),AVG(pnl) FROM trades WHERE user_id=? AND status='closed'",
                (user_id,)
            ).fetchone()
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE user_id=? AND pnl>0", (user_id,)
            ).fetchone()[0]
        else:
            total  = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            closed = conn.execute(
                "SELECT COUNT(*),SUM(pnl),AVG(pnl) FROM trades WHERE status='closed'"
            ).fetchone()
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE pnl>0"
            ).fetchone()[0]

        closed_count = closed[0] or 0
        return {
            "total_trades":  total,
            "closed_trades": closed_count,
            "total_pnl":     round(float(closed[1] or 0), 2),
            "avg_pnl":       round(float(closed[2] or 0), 2),
            "wins":          wins,
            "win_rate":      round(wins / closed_count * 100, 1) if closed_count else 0,
        }


# ── Signals ───────────────────────────────────────────────────────────────────

def log_signal(pair: str, signal: str, prob_up: float, confidence: str,
               regime: str, tradeable: bool, price: float):
    with get_db() as conn:
        recent = conn.execute("""
            SELECT id FROM signals_log
            WHERE pair=? AND signal=? AND ABS(prob_up - ?) < 0.00001
              AND confidence=? AND regime=? AND tradeable=?
              AND created_at >= ?
            ORDER BY created_at DESC LIMIT 1
        """, (
            pair, signal, prob_up, confidence, regime, int(tradeable),
            (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
        )).fetchone()
        if recent:
            return
        conn.execute("""
            INSERT INTO signals_log
            (pair,signal,prob_up,confidence,regime,tradeable,price,created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (pair, signal, prob_up, confidence, regime,
              int(tradeable), price, datetime.utcnow().isoformat()))


def get_signals_log(limit: int = 100, pair: str = None) -> list:
    with get_db() as conn:
        if pair:
            rows = conn.execute(
                "SELECT * FROM signals_log WHERE pair=? ORDER BY created_at DESC LIMIT ?",
                (pair, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signals_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ── Audit & notifications ─────────────────────────────────────────────────────

def get_audit_log(limit: int = 100) -> list:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT a.*, u.username FROM audit_log a
            LEFT JOIN users u ON u.id=a.user_id
            ORDER BY a.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_notifications(user_id: int, unread_only: bool = False) -> list:
    with get_db() as conn:
        q = "WHERE user_id=? AND is_read=0" if unread_only else "WHERE user_id=?"
        rows = conn.execute(
            f"SELECT * FROM notifications {q} ORDER BY created_at DESC LIMIT 20",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def mark_notifications_read(user_id: int):
    with get_db() as conn:
        conn.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (user_id,))


# ── Platform stats ────────────────────────────────────────────────────────────

def get_platform_stats() -> dict:
    with get_db() as conn:
        total_users   = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
        total_trades  = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        total_signals = conn.execute("SELECT COUNT(*) FROM signals_log").fetchone()[0]
        total_pnl     = conn.execute("SELECT SUM(pnl) FROM trades WHERE status='closed'").fetchone()[0]
        plan_counts   = conn.execute(
            "SELECT plan,COUNT(*) as cnt FROM subscriptions GROUP BY plan"
        ).fetchall()
        new_today     = conn.execute(
            "SELECT COUNT(*) FROM users WHERE created_at >= date('now')"
        ).fetchone()[0]
        return {
            "total_users":   total_users,
            "total_trades":  total_trades,
            "total_signals": total_signals,
            "total_pnl":     round(float(total_pnl or 0), 2),
            "new_today":     new_today,
            "plans":         {r["plan"]: r["cnt"] for r in plan_counts},
        }
