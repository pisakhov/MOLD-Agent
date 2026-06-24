import hmac
import os

from fastapi import Request

from app.crypto import secret_key

COOKIE_NAME = "mold_session"


def password():
    return os.getenv("MOLD_PASSWORD", "")


def enabled():
    return bool(password())


def token():
    return hmac.new(secret_key().encode(), password().encode(), "sha256").hexdigest()


def authed(request: Request):
    return not enabled() or hmac.compare_digest(request.cookies.get(COOKIE_NAME, ""), token())
