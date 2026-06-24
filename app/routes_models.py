import asyncio
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import store
from app.llm import build_llm

router = APIRouter()


class ProviderCreate(BaseModel):
    name: str
    provider_type: str
    base_url: Optional[str] = None
    api_key: str


class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    provider_type: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None


class ModelCreate(BaseModel):
    provider_id: str
    display_name: str
    model_id: str
    enabled: bool = True
    is_default: bool = False


class ModelUpdate(BaseModel):
    display_name: Optional[str] = None
    model_id: Optional[str] = None
    enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class ChainEntry(BaseModel):
    model_id: str
    timeout: int = 120
    retries: int = 1


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


@router.get("/providers")
def providers():
    return {"providers": store.list_providers()}


@router.post("/providers")
def create_provider(body: ProviderCreate):
    return store.create_provider(body.name, body.provider_type, body.base_url, body.api_key)


@router.patch("/providers/{provider_id}")
def update_provider(provider_id: str, body: ProviderUpdate):
    data = body.model_dump(exclude_none=True)
    result = store.update_provider(provider_id, **data)
    if not result:
        raise HTTPException(status_code=404, detail="Provider not found")
    return result


@router.delete("/providers/{provider_id}")
def delete_provider(provider_id: str):
    if not store.delete_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"status": "ok"}


@router.get("/providers/{provider_id}/detect")
async def detect_models(provider_id: str):
    provider = store.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    from app.crypto import decrypt
    api_key = decrypt(provider["api_key_encrypted"])
    try:
        if provider["provider_type"] == "anthropic":
            models = await asyncio.to_thread(fetch_anthropic_models, api_key, provider.get("base_url"))
        elif provider["provider_type"] == "google":
            models = await asyncio.to_thread(fetch_google_models, api_key)
        else:
            models = await asyncio.to_thread(fetch_openai_models, provider.get("base_url") or "https://api.openai.com/v1", api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"models": models}


def fetch_openai_models(base_url: str, api_key: str):
    res = requests.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    res.raise_for_status()
    return [{"id": m["id"], "display_name": m.get("id")} for m in sorted(res.json().get("data", []), key=lambda x: x["id"])]


def fetch_anthropic_models(api_key: str, base_url: str | None):
    res = requests.get(
        f"{(base_url or 'https://api.anthropic.com').rstrip('/')}/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=10,
    )
    if not res.ok:
        return [
            {"id": "claude-opus-4-5", "display_name": "Claude Opus 4.5"},
            {"id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"},
            {"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5"},
        ]
    return [{"id": m["id"], "display_name": m.get("display_name") or m["id"]} for m in res.json().get("data", [])]


def fetch_google_models(api_key: str):
    res = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": api_key}, timeout=10)
    res.raise_for_status()
    return [
        {"id": m["name"], "display_name": m.get("displayName") or m["name"]}
        for m in res.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


@router.get("/")
def models(provider_id: Optional[str] = None):
    return {"models": store.list_models(provider_id)}


@router.post("/")
def create_model(body: ModelCreate):
    return store.create_model(body.provider_id, body.display_name, body.model_id, body.enabled, body.is_default)


@router.patch("/{model_id}")
def update_model(model_id: str, body: ModelUpdate):
    result = store.update_model(model_id, **body.model_dump(exclude_none=True))
    if not result:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.delete("/{model_id}")
def delete_model(model_id: str):
    if not store.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "ok"}


@router.post("/{model_id}/test")
async def test_model(model_id: str):
    info = store.get_model_with_key(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found or disabled")
    llm = build_llm(info["provider_type"], info["model_id"], info["api_key"], info.get("base_url"), max_tokens=16)
    try:
        reply = await llm.ainvoke("Reply with the single word: pong")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "reply": extract_text(reply.content)}


@router.get("/chain")
def get_chain():
    return {"chain": store.get_chain()}


@router.put("/chain")
def set_chain(body: list[ChainEntry]):
    store.set_chain([x.model_dump() for x in body])
    return {"status": "ok"}
