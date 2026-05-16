"""
ForexChautari — scripts/rotate_encryption_key.py
=================================================
Re-encrypts all stored API keys under a NEW encryption key.

Use this if:
  - The current FIELD_ENCRYPTION_KEY was accidentally exposed
  - You are rotating keys as part of a security policy

Steps:
  1. Generate a new key:
       python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  2. Run this script with both the old and new keys:
       python scripts/rotate_encryption_key.py \
           --old-key <current_key> \
           --new-key <new_key>
  3. Update FIELD_ENCRYPTION_KEY in .env to the new key
  4. Restart all services

The script is transactional — if any row fails, the entire operation is
rolled back and the database is left unchanged.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from cryptography.fernet import Fernet, InvalidToken
from src.encryption import _ENC_PREFIX
from src.database import get_db


def rotate(old_key: str, new_key: str, dry_run: bool = False) -> None:
    print()
    print("ForexChautari — encryption key rotation")
    print("=" * 50)

    if dry_run:
        print("DRY RUN — no changes will be written\n")

    # Validate both keys before touching the database.
    try:
        f_old = Fernet(old_key.encode())
    except Exception as exc:
        print(f"ERROR: --old-key is not a valid Fernet key: {exc}")
        sys.exit(1)

    try:
        f_new = Fernet(new_key.encode())
    except Exception as exc:
        print(f"ERROR: --new-key is not a valid Fernet key: {exc}")
        sys.exit(1)

    if old_key == new_key:
        print("ERROR: --old-key and --new-key are identical. Nothing to do.")
        sys.exit(1)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, api_key_enc FROM trading_accounts"
        ).fetchall()

    encrypted_rows = [r for r in rows if r["api_key_enc"].startswith(_ENC_PREFIX)]
    print(f"Total rows      : {len(rows)}")
    print(f"Encrypted rows  : {len(encrypted_rows)}")
    print(f"Plaintext rows  : {len(rows) - len(encrypted_rows)} (will be skipped)\n")

    if not encrypted_rows:
        print("No encrypted rows found. Run migrate_encrypt_keys.py first.")
        return

    if dry_run:
        print(f"Would re-encrypt {len(encrypted_rows)} rows.")
        return

    # Re-encrypt all rows in a single transaction. If any fails, nothing is saved.
    print(f"Re-encrypting {len(encrypted_rows)} rows...")

    updates = []
    for row in encrypted_rows:
        token = row["api_key_enc"][len(_ENC_PREFIX):]
        try:
            plaintext = f_old.decrypt(token.encode()).decode()
        except InvalidToken:
            print(f"\nERROR: Could not decrypt row id={row['id']} with --old-key.")
            print("Aborting — no changes written.")
            sys.exit(1)

        new_token     = f_new.encrypt(plaintext.encode()).decode()
        new_ciphertext = f"{_ENC_PREFIX}{new_token}"
        updates.append((new_ciphertext, row["id"]))

    with get_db() as conn:
        conn.executemany(
            "UPDATE trading_accounts SET api_key_enc = ? WHERE id = ?",
            updates,
        )

    print(f"\n✓ Rotated {len(updates)} rows successfully.")
    print("\nNext steps:")
    print("  1. Update FIELD_ENCRYPTION_KEY in .env to the new key")
    print("  2. Delete the old key from wherever it was stored")
    print("  3. Restart all services (uvicorn, scheduler, streamlit)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate the field encryption key")
    parser.add_argument("--old-key", required=True, help="Current Fernet key")
    parser.add_argument("--new-key", required=True, help="New Fernet key to rotate to")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    rotate(args.old_key, args.new_key, args.dry_run)
