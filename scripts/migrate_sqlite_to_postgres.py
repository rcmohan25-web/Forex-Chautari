"""
One-time data migration: copy all rows from the existing SQLite database
into a fresh PostgreSQL database (schema must already be created by
running `init_db()` against DATABASE_URL pointing at Postgres).

Usage:
    # 1. Create the postgres schema
    DATABASE_URL=postgresql://user:pass@host/db python -c "from src.database import init_db; init_db()"

    # 2. Copy the data
    python scripts/migrate_sqlite_to_postgres.py \
        --sqlite-path data/forexchautari.db \
        --postgres-url postgresql://user:pass@host/db
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, text, inspect

TABLES_IN_ORDER = [
    "users", "subscriptions", "trading_accounts", "trades",
    "signals_log", "audit_log", "notifications", "sessions",
    "user_trading_settings", "platform_settings",
]


def run(sqlite_path: str, postgres_url: str, truncate_first: bool = True):
    src = create_engine(f"sqlite:///{sqlite_path}")
    dst = create_engine(postgres_url)

    with src.connect() as sconn, dst.connect() as dconn:
        trans = dconn.begin()
        try:
            for table in TABLES_IN_ORDER:
                cols = [c["name"] for c in inspect(src).get_columns(table)]
                rows = sconn.execute(text(f"SELECT * FROM {table}")).mappings().all()

                if truncate_first:
                    dconn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

                if not rows:
                    print(f"  {table}: 0 rows")
                    continue

                col_list  = ",".join(cols)
                param_list = ",".join(f":{c}" for c in cols)
                insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({param_list})"

                for row in rows:
                    dconn.execute(text(insert_sql), dict(row))

                # Fix sequence so future SERIAL inserts don't collide
                if "id" in cols:
                    dconn.execute(text(
                        f"SELECT setval(pg_get_serial_sequence('{table}','id'), "
                        f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
                    ))

                print(f"  {table}: {len(rows)} rows")

            trans.commit()
            print("\nMigration complete.")
        except Exception:
            trans.rollback()
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite-path", default="data/forexchautari.db")
    p.add_argument("--postgres-url", required=True)
    args = p.parse_args()
    run(args.sqlite_path, args.postgres_url)
