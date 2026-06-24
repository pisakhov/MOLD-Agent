import base64
import hashlib
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet


def secret_key():
    value = os.getenv("SECRET_KEY")
    if value:
        return value
    path = Path(os.getenv("SECRET_KEY_FILE", ".data/secret_key"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_urlsafe(48))
        path.chmod(0o600)
    return path.read_text().strip()


def fernet():
    key = base64.urlsafe_b64encode(hashlib.sha256(secret_key().encode()).digest())
    return Fernet(key)


def encrypt(value: str) -> str:
    return fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    return fernet().decrypt(value.encode()).decode()
