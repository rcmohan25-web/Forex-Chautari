"""
src/db_engine.py — single SQLAlchemy engine shared by the whole app.

DATABASE_URL drives the dialect:
  - sqlite:///data/forexchautari.db   (default, test fallback)
  - postgresql://user:pass@host/db    (production)

Connection pool sizing:
  - SQLite: check_same_thread=False to allow Streamlit's per-request threads
  - PostgreSQL: pool_size=5, max_overflow=10 to prevent exhausting Postgres's
    max_connections limit. Adjust based on expected concurrent users:
      pool_size    = number of persistent connections (API + scheduler + Streamlit)
      max_overflow = temporary connections beyond pool_size during load spikes

get_db() yields a connection inside a transaction (commits on success,
rolls back on exception) — same contract as the old sqlite3 get_db().

Rows are returned as SQLAlchemy RowMapping objects, which behave like
dicts: row["username"], dict(row), row.get(...).
"""

import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from src.logger import get_logger

logger = get_logger("db_engine")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.getenv('DB_PATH', 'data/forexchautari.db')}")

_connect_args = {}
_engine_kwargs = {
    "pool_pre_ping": True,   # Verify connections are alive before reusing them
    "future": True,          # Use SQLAlchemy 2.0-style SQL constructs
}

if DATABASE_URL.startswith("sqlite"):
    # SQLite: single-file database, allow use across Streamlit's per-request threads
    _connect_args = {"check_same_thread": False}
else:
    # PostgreSQL: use connection pooling to avoid exhausting max_connections
    # Tune pool_size and max_overflow based on your deployment:
    #   - API server: ~10-20 concurrent requests
    #   - Streamlit: ~5-10 concurrent dashboard users
    #   - Scheduler: ~1-2 background jobs
    # Rule of thumb: pool_size = expected steady-state connections,
    #                max_overflow = peak_connections - pool_size
    _engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
    _engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))

ENGINE = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs
)

IS_POSTGRES = ENGINE.dialect.name == "postgresql"
IS_SQLITE   = ENGINE.dialect.name == "sqlite"


@contextmanager
def get_db():
    """
    Yields a SQLAlchemy Connection wrapped in a transaction.
    Use:
        with get_db() as conn:
            row = conn.execute(text("SELECT * FROM users WHERE id=:id"), {"id": 1}).mappings().fetchone()
    """
    conn = ENGINE.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()


def execute(conn, sql: str, params: dict | None = None):
    """Shorthand for conn.execute(text(sql), params or {})."""
    return conn.execute(text(sql), params or {})


def fetchone(conn, sql: str, params: dict | None = None):
    row = execute(conn, sql, params).mappings().fetchone()
    return dict(row) if row else None


def fetchall(conn, sql: str, params: dict | None = None):
    return [dict(r) for r in execute(conn, sql, params).mappings().fetchall()]


def returning_id(conn, table: str, pk_col: str = "id"):
    """
    Get the id of the most recently inserted row in this table.
    Postgres: use `RETURNING id` on the INSERT instead when possible.
    SQLite fallback: SELECT last_insert_rowid().
    """
    if IS_POSTGRES:
        row = execute(conn, f"SELECT lastval()").scalar()
        return row
    return execute(conn, "SELECT last_insert_rowid()").scalar()


def pk_column() -> str:
    """Primary key DDL fragment for the current dialect."""
    return "SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"


def now_func() -> str:
    """Return an SQL expression usable in DDL/queries — we use Python timestamps
    everywhere in this app, so this is rarely needed, but kept for completeness."""
    return "CURRENT_TIMESTAMP"


def reset_engine() -> None:
    """
    Rebuild the ENGINE from the current DATABASE_URL environment variable.
    
    Used by test fixtures to switch between temporary SQLite databases
    or to reconnect to a different database without restarting the process.
    
    Example:
        import os
        os.environ["DATABASE_URL"] = "sqlite:////tmp/test.db"
        reset_engine()  # ENGINE now points to /tmp/test.db
    """
    global ENGINE, IS_POSTGRES, IS_SQLITE
    
    # Re-read environment
    new_url = os.getenv("DATABASE_URL", f"sqlite:///{os.getenv('DB_PATH', 'data/forexchautari.db')}")
    
    # Rebuild connection arguments
    new_connect_args = {}
    new_engine_kwargs = {
        "pool_pre_ping": True,
        "future": True,
    }
    
    if new_url.startswith("sqlite"):
        new_connect_args = {"check_same_thread": False}
    else:
        new_engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
        new_engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # Dispose of old engine (close all connections)
    ENGINE.dispose()
    
    # Create new engine
    ENGINE = create_engine(
        new_url,
        connect_args=new_connect_args,
        **new_engine_kwargs
    )
    
    # Update dialect flags
    IS_POSTGRES = ENGINE.dialect.name == "postgresql"
    IS_SQLITE   = ENGINE.dialect.name == "sqlite"
    
    logger.info(f"Engine reset to: {new_url.split('@')[0]}@... ({'PostgreSQL' if IS_POSTGRES else 'SQLite'})")
