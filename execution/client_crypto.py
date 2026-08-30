"""
Encryption for client Alpaca credentials, and password hashing for the
client portal login — kept in one small module since both exist for the
same reason (this file is now one of the highest-stakes ones in the repo,
alongside broker_alpaca.py and circuit_breakers.py: a bug here either
leaks every client's brokerage credentials or lets someone log into
another client's portal).

Credentials: Fernet (symmetric, authenticated) encryption keyed by
CLIENT_KEY_ENCRYPTION_KEY (config/settings.py) — an env var, never stored
in the database itself, so a DB dump alone doesn't unlock anything.
Encrypt/decrypt both fail loudly if the key is unset rather than falling
back to something predictable.

Passwords: PBKDF2-HMAC-SHA256 with a random per-password salt (stdlib
`hashlib`, no new dependency) — client portal passwords are operator-set
(see the "you set/reset manually" decision), not the encryption keys
protecting brokerage access, so a slower, more exotic KDF wasn't judged
worth a new dependency here. 600,000 iterations matches OWASP's current
PBKDF2-SHA256 recommendation.
"""
from __future__ import annotations

import hashlib
import hmac
import os

from cryptography.fernet import Fernet, InvalidToken

from config.settings import settings

_PBKDF2_ITERATIONS = 600_000
_SALT_BYTES = 16


def _fernet() -> Fernet:
    key = settings.client_key_encryption_key
    if not key:
        raise RuntimeError(
            "CLIENT_KEY_ENCRYPTION_KEY is not set — refusing to encrypt or decrypt client "
            "brokerage credentials with no key configured. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and set it as an env var."
        )
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as e:
        raise RuntimeError("CLIENT_KEY_ENCRYPTION_KEY is set but is not a valid Fernet key.") from e


def encrypt_credential(plaintext: str) -> bytes:
    """Encrypts a single Alpaca API key or secret for storage in `clients`."""
    return _fernet().encrypt(plaintext.encode("utf-8"))


def decrypt_credential(ciphertext: bytes) -> str:
    """
    Decrypts a value written by encrypt_credential. Raises RuntimeError
    (never returns garbage) if CLIENT_KEY_ENCRYPTION_KEY was rotated or the
    stored value is corrupt — trading with a wrongly-decrypted credential is
    worse than failing loudly.
    """
    try:
        return _fernet().decrypt(bytes(ciphertext)).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError(
            "Could not decrypt a client credential — CLIENT_KEY_ENCRYPTION_KEY may have "
            "changed since it was stored, or the stored value is corrupt."
        ) from e


def hash_password(password: str) -> str:
    """
    Returns a self-contained string ("pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>")
    -- the iteration count travels with the hash so it can be raised later
    without invalidating passwords hashed under the old count.
    """
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison against a hash produced by hash_password."""
    try:
        algo, iterations_s, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)
