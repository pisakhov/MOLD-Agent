import base64
import hashlib
import json
import os
import time
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import store

router = APIRouter()

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"

_pkce_store: dict[str, str] = {}


class ExchangeBody(BaseModel):
    callback_url: str


def _generate_pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


@router.get("/auth-url")
def auth_url():
    verifier, challenge = _generate_pkce()
    state = os.urandom(16).hex()
    _pkce_store[state] = verifier
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex-auth",
    }
    return {"auth_url": f"{AUTHORIZE_URL}?{urlencode(params)}", "state": state}


@router.post("/exchange")
async def exchange(body: ExchangeBody):
    params = parse_qs(urlparse(body.callback_url).query)
    code = next(iter(params.get("code", [])), "")
    state = next(iter(params.get("state", [])), "")
    verifier = _pkce_store.pop(state, None)
    if not verifier or not code:
        raise HTTPException(status_code=400, detail="Invalid callback URL or expired state. Start Connect Codex again.")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(TOKEN_URL, data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            })
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail=f"OpenAI token exchange failed: {exc}") from exc

    data = response.json()
    tokens = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires": int(time.time() + data["expires_in"]),
    }
    if data.get("id_token"):
        tokens["id_token"] = data["id_token"]
    encoded = json.dumps(tokens)

    existing = next((p for p in store.list_providers() if p["provider_type"] == "codex"), None)
    if existing:
        return store.update_provider(existing["id"], api_key=encoded)
    return store.create_provider("Codex Subscription", "codex", None, encoded)
