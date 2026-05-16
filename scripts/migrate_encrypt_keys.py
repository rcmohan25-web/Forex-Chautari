"""
ForexChautari — scripts/migrate_encrypt_keys.py
================================================
One-time migration: encrypt all plaintext API keys already in the database.

Run this ONCE after deploying src/encryption.py and setting
FIELD_ENCRYPTION_KEY in .env. Running it multiple times is safe —
already-encrypted rows are detected by the "enc:v1:" prefix and skipped.

Usage:
    python scripts/migrate_encrypt_keys.py

    # Dry run — shows what would change without writing anything:
    python scripts/migrate_encrypt_keys.py --dry-run

Pre-flight checklist:
  1. FIELD_ENCRYPTION_KEY is set in .env
  2. cryptography is installed:  pip install cryptography
  3. The database file exists at the path in DB_PATH (or default)
  4. BACK UP THE DATABASE FIRST:
       cp data/forexchautari.db data/forexchautari.db.bak
"""

import sys
import os
import argparse

# Allow running from the project root: python scripts/migrate_encrypt_keys.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.encryption import encrypt, is_encrypted, _ENC_PREFIX
from src.database import get_db
from src.logger import get_logger

logger = get_logger("migrate_encrypt_keys")


def run(dry_run: bool = False) -> None:
    print()
    print("ForexChautari — API key encryption migration")
    print("=" * 50)

    if dry_run:
        print("DRY RUN — no changes will be written\n")

    # Verify the encryption key is present before touching the database.
    key = os.getenv("FIELD_ENCRYPTION_KEY", "").strip()
    if not key:
        print("ERROR: FIELD_ENCRYPTION_KEY is not set in .env")
        print("Generate one with:")
        print('  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
        sys.exit(1)

    print(f"Using database: {os.getenv('DB_PATH', 'data/forexchautari.db')}\n")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, user_id, account_name, account_id, api_key_enc FROM trading_accounts"
        ).fetchall()

    total      = len(rows)
    already_ok = 0
    to_migrate = []

    for row in rows:
        if is_encrypted(row["api_key_enc"]):
            already_ok += 1
        else:
            to_migrate.append(row)

    print(f"Total accounts :  {total}")
    print(f"Already encrypted: {already_ok}")
    print(f"Need migration :  {len(to_migrate)}\n")

    if not to_migrate:
        print("Nothing to do — all keys are already encrypted.")
        return

    # Show what will be migrated (never print the actual key).
    for row in to_migrate:
        key_preview = row["api_key_enc"][:6] + "..." if row["api_key_enc"] else "(empty)"
        print(
            f"  [id={row['id']}] user_id={row['user_id']} "
            f"account={row['account_name']} ({row['account_id']}) "
            f"key_starts_with={key_preview}"
        )

    if dry_run:
        print(f"\nDry run complete — {len(to_migrate)} rows would be encrypted.")
        return

    print(f"\nEncrypting {len(to_migrate)} rows...")

    migrated = 0
    errors   = 0

    with get_db() as conn:
        for row in to_migrate:
            try:
                encrypted = encrypt(row["api_key_enc"])
                conn.execute(
                    "UPDATE trading_accounts SET api_key_enc = ? WHERE id = ?",
                    (encrypted, row["id"]),
                )
                migrated += 1
                print(f"  ✓ Encrypted account id={row['id']}")
            except Exception as exc:
                errors += 1
                print(f"  ✗ FAILED for account id={row['id']}: {exc}")
                logger.error(f"Migration failed for account id={row['id']}: {exc}")

    print()
    print(f"Migration complete: {migrated} encrypted, {errors} errors")

    if errors:
        print("\nWARNING: Some rows failed. Check the logs and re-run after fixing the issue.")
        sys.exit(1)
    else:
        print("\nAll API keys are now encrypted.")
        print("The FIELD_ENCRYPTION_KEY in .env is the only way to read them.")
        print("Back it up somewhere safe and separate from the database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt stored Oanda API keys")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing to the database",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
