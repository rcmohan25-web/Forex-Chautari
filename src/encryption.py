"""
ForexChautari — Field-level encryption for sensitive stored values.

Used to encrypt Oanda API keys before they are written to the database,
so that a stolen SQLite file does not expose live trading credentials.

Setup (one-time):
  1. Generate a key:
       python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  2. Add it to .env:
       FIELD_ENCRYPTION_KEY=your_generated_key_here
  3. Run the migration script once to encrypt existing plaintext rows:
       python scripts/migrate_encrypt_keys.py

The key must be backed up securely and separately from the database.
If the key is lost, all stored API keys become unreadable and every user
must reconnect their trading account.
"""

import os
import base64
from cryptography.fernet import Fernet, InvalidToken
from src.logger import get_logger

logger = get_logger("encryption")

# Prefix written into every encrypted value so we can tell at a glance
# (and in the migration script) whether a value is already encrypted.
_ENC_PREFIX = "enc:v1:"


def _get_fernet() -> Fernet:
    """
    Build a Fernet instance from the environment key.
    Raises RuntimeError clearly so misconfiguration is obvious at startup,
    not buried in a database write error.
    """
    raw_key = os.getenv("FIELD_ENCRYPTION_KEY", "").strip()
    if not raw_key:
        raise RuntimeError(
            "FIELD_ENCRYPTION_KEY is not set.\n"
            "Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"\n"
            "Then add it to your .env file."
        )
    try:
        # Fernet expects a URL-safe base64-encoded 32-byte key.
        # Validate it eagerly so we fail at startup, not at first write.
        return Fernet(raw_key.encode())
    except Exception as exc:
        raise RuntimeError(
            f"FIELD_ENCRYPTION_KEY is set but is not a valid Fernet key: {exc}\n"
            "Re-generate it with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        ) from exc


def encrypt(plaintext: str) -> str:
    """
    Encrypt a plaintext string and return a prefixed ciphertext string.

    The prefix "enc:v1:" allows the migration script and decrypt() to
    detect whether a value has already been encrypted, preventing
    double-encryption if a row is accidentally processed twice.

    Returns:
        "enc:v1:<fernet_token>"
    """
    if not plaintext:
        return plaintext

    # Safety guard: if somehow already encrypted, return as-is.
    if plaintext.startswith(_ENC_PREFIX):
        logger.warning("encrypt() called on a value that is already encrypted — skipping")
        return plaintext

    token = _get_fernet().encrypt(plaintext.encode()).decode()
    return f"{_ENC_PREFIX}{token}"


def decrypt(ciphertext: str) -> str:
    """
    Decrypt a value produced by encrypt().

    Handles three cases gracefully:
      1. Normal encrypted value  → decrypts and returns plaintext.
      2. Legacy plaintext value  → returns as-is (migration not yet run).
      3. Empty / None            → returns as-is.

    The legacy fallback means the app stays functional during the window
    between deploying this code and running the migration script.

    Raises:
        RuntimeError  if the value looks encrypted but decryption fails
                      (wrong key, corrupted data).
    """
    if not ciphertext:
        return ciphertext

    if not ciphertext.startswith(_ENC_PREFIX):
        # Value has not been encrypted yet — legacy row before migration.
        logger.debug("decrypt() called on a plaintext value (pre-migration row) — returning as-is")
        return ciphertext

    token = ciphertext[len(_ENC_PREFIX):]
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError(
            "Failed to decrypt a stored API key. "
            "This usually means the FIELD_ENCRYPTION_KEY in .env does not match "
            "the key that was used to encrypt the data. "
            "Check that the key has not been rotated without a re-encryption migration."
        ) from exc


def is_encrypted(value: str) -> bool:
    """Return True if the value was produced by encrypt()."""
    return isinstance(value, str) and value.startswith(_ENC_PREFIX)


def rotate_key(old_key: str, new_key: str, ciphertext: str) -> str:
    """
    Re-encrypt a single value under a new key.

    Used by the key-rotation migration script when you need to change
    the encryption key (e.g. after a suspected key compromise).

    Args:
        old_key:    The current FIELD_ENCRYPTION_KEY value (base64 string).
        new_key:    The new key to encrypt under.
        ciphertext: A value previously produced by encrypt().

    Returns:
        A new encrypted value under new_key.
    """
    if not ciphertext.startswith(_ENC_PREFIX):
        raise ValueError("rotate_key() requires an already-encrypted value")

    token = ciphertext[len(_ENC_PREFIX):]
    plaintext = Fernet(old_key.encode()).decrypt(token.encode()).decode()
    new_token = Fernet(new_key.encode()).encrypt(plaintext.encode()).decode()
    return f"{_ENC_PREFIX}{new_token}"
