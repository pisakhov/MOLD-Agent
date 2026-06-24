import asyncio
import json
import random
import time
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field

from app import store

DEFAULT_TEMPERATURE = 0.6
QUIET = {"callbacks": []}


def retry_delay():
    return random.uniform(2, 8)


class FallbackChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    members: list = Field(default_factory=list)
    retries: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "mold_fallback_chain"

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "FallbackChatModel":
        return self.model_copy(update={"members": [m.bind_tools(tools, **kwargs) for m in self.members]})

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        last_error = None
        for member, retries in zip(self.members, self.retries):
            for attempt in range(retries + 1):
                try:
                    return ChatResult(generations=[ChatGeneration(message=member.invoke(messages, config=QUIET))])
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        time.sleep(retry_delay())
        raise last_error

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        last_error = None
        for member, retries in zip(self.members, self.retries):
            for attempt in range(retries + 1):
                try:
                    return ChatResult(generations=[ChatGeneration(message=await member.ainvoke(messages, config=QUIET))])
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        await asyncio.sleep(retry_delay())
        raise last_error

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for chunk in self.members[0].stream(messages, config=QUIET):
            yield ChatGenerationChunk(message=chunk)

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self.members[0].astream(messages, config=QUIET):
            yield ChatGenerationChunk(message=chunk)


def build_llm(provider_type: str, model_id: str, api_key: str, base_url: str | None = None, max_tokens: int | None = None, timeout: int | None = None, max_retries: int = 0, provider_id: str | None = None):
    if provider_type == "codex":
        from app.codex import CodexChatModel, get_account_id, refresh_tokens
        tokens, changed = refresh_tokens(json.loads(api_key))
        if changed and provider_id:
            store.update_provider(provider_id, api_key=json.dumps(tokens))
        return CodexChatModel(model=model_id, access_token=tokens["access_token"], account_id=get_account_id(tokens["access_token"]), max_output_tokens=max_tokens)
    if provider_type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(model=model_id, api_key=api_key, base_url=base_url, temperature=DEFAULT_TEMPERATURE, timeout=timeout or 60, max_retries=max_retries)
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)
    if provider_type == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(model=model_id, api_key=api_key, timeout=timeout or 60, max_retries=max_retries)
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)
    from langchain_openai import ChatOpenAI
    kwargs = dict(model=model_id, openai_api_key=api_key, base_url=base_url or "https://api.openai.com/v1", temperature=DEFAULT_TEMPERATURE, timeout=timeout or 60, max_retries=max_retries)
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def build_chain_llm(max_tokens: int | None = None):
    members, retries = [], []
    for entry in store.get_chain():
        info = store.get_model_with_key(entry["model_id"])
        if info:
            members.append(build_llm(info["provider_type"], info["model_id"], info["api_key"], info.get("base_url"), max_tokens=max_tokens, timeout=entry.get("timeout", 120), provider_id=info.get("provider_id")))
            retries.append(entry.get("retries", 1))
    if not members:
        defaults = [m for m in store.list_models() if m["enabled"] and m["is_default"]]
        for model in defaults[:1]:
            info = store.get_model_with_key(model["id"])
            if info:
                members.append(build_llm(info["provider_type"], info["model_id"], info["api_key"], info.get("base_url"), max_tokens=max_tokens, provider_id=info.get("provider_id")))
                retries.append(1)
    return FallbackChatModel(members=members, retries=retries) if members else None
