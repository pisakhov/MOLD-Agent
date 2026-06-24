import base64
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Sequence

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH = Path.home() / ".codex" / "auth.json"
CODEX_MODELS = [
    "gpt-5.1", "gpt-5.1-codex-max", "gpt-5.1-codex-mini",
    "gpt-5.2", "gpt-5.2-codex",
    "gpt-5.3-codex", "gpt-5.3-codex-spark",
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.5",
]


def get_account_id(access_token: str) -> str:
    payload = access_token.split(".")[1]
    padded = payload + "=" * (4 - len(payload) % 4)
    data = json.loads(base64.urlsafe_b64decode(padded))
    return data["https://api.openai.com/auth"]["chatgpt_account_id"]


def token_expiry(access_token: str) -> int:
    payload = access_token.split(".")[1]
    padded = payload + "=" * (4 - len(payload) % 4)
    return int(json.loads(base64.urlsafe_b64decode(padded)).get("exp", time.time() + 3600))


def normalize_tokens(value: str | dict) -> dict:
    data = json.loads(value) if isinstance(value, str) else value
    tokens = data.get("tokens") if isinstance(data, dict) and "tokens" in data else data
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise ValueError("Codex auth must include access_token and refresh_token")
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires": int(tokens.get("expires") or token_expiry(tokens["access_token"])),
    }


def load_auth_tokens() -> dict | None:
    raw = os.getenv("CODEX_AUTH_JSON")
    if raw:
        return normalize_tokens(raw)
    raw_b64 = os.getenv("CODEX_AUTH_JSON_B64")
    if raw_b64:
        return normalize_tokens(base64.b64decode(raw_b64).decode())
    if CODEX_AUTH.exists():
        return normalize_tokens(json.loads(CODEX_AUTH.read_text()))
    return None


def refresh_tokens(tokens: dict) -> tuple[dict, bool]:
    if tokens["expires"] > time.time() + 60:
        return tokens, False
    res = httpx.post(TOKEN_URL, data={
        "grant_type": "refresh_token",
        "refresh_token": tokens["refresh_token"],
        "client_id": CLIENT_ID,
    })
    res.raise_for_status()
    data = res.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires": int(time.time() + data["expires_in"]),
    }, True


def to_responses_tool(tool: Any) -> dict:
    from langchain_core.utils.function_calling import convert_to_openai_function
    fn = convert_to_openai_function(tool)
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }


class CodexChatModel(BaseChatModel):
    model: str
    access_token: str
    account_id: str
    tools: list = Field(default_factory=list)
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: bool = True
    verbosity: str = "low"

    @property
    def _llm_type(self) -> str:
        return "codex"

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "CodexChatModel":
        return self.model_copy(update={"tools": [to_responses_tool(t) for t in tools]})

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "chatgpt-account-id": self.account_id,
            "originator": "codex-auth",
            "OpenAI-Beta": "responses=experimental",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

    def _to_input(self, messages: List[BaseMessage]) -> tuple[list, str]:
        system = "You are a helpful assistant."
        parts = []
        seen_call_ids: set[str] = set()
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else "".join(c.get("text", "") for c in msg.content if isinstance(c, dict))
            if isinstance(msg, SystemMessage):
                system = content
            elif isinstance(msg, HumanMessage):
                parts.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    if content:
                        parts.append({"role": "assistant", "content": content})
                    for tc in msg.tool_calls:
                        seen_call_ids.add(tc["id"])
                        parts.append({"type": "function_call", "call_id": tc["id"], "name": tc["name"], "arguments": json.dumps(tc["args"])})
                else:
                    parts.append({"role": "assistant", "content": content})
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in seen_call_ids:
                parts.append({"type": "function_call_output", "call_id": msg.tool_call_id, "output": content})
        return parts, system

    def _body(self, input_msgs: list, system: str) -> dict:
        body = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": system,
            "input": input_msgs,
            "text": {"verbosity": self.verbosity},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.max_output_tokens:
            body["max_output_tokens"] = self.max_output_tokens
        if self.tools:
            body["tools"] = self.tools
        return body

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        input_msgs, system = self._to_input(messages)
        text, tool_calls, pending = "", [], {}
        with httpx.Client(timeout=600) as client:
            with client.stream("POST", CODEX_URL, headers=self._headers(), json=self._body(input_msgs, system)) as res:
                res.raise_for_status()
                for line in res.iter_lines():
                    text, tool_calls, pending = self._handle_line(line, text, tool_calls, pending)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, tool_calls=tool_calls))])

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        input_msgs, system = self._to_input(messages)
        indexes, args_seen, next_index = {}, {}, 0
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", CODEX_URL, headers=self._headers(), json=self._body(input_msgs, system)) as res:
                res.raise_for_status()
                async for line in res.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        return
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "error":
                        raise RuntimeError(f"Codex API error: {event.get('message', event)}")
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", "")
                            indexes[call_id] = next_index
                            args_seen[call_id] = False
                            next_index += 1
                            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{"name": item.get("name", ""), "args": "", "id": call_id, "index": indexes[call_id]}]))
                    elif event_type == "response.function_call_arguments.delta":
                        call_id, delta = event.get("call_id", ""), event.get("delta", "")
                        if call_id in indexes and delta:
                            args_seen[call_id] = True
                            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": delta, "id": None, "index": indexes[call_id]}]))
                    elif event_type in ("response.function_call_arguments.done", "response.output_item.done"):
                        item = event.get("item", {})
                        call_id = event.get("call_id") or item.get("call_id", "")
                        arguments = event.get("arguments") or item.get("arguments", "")
                        if call_id in indexes and arguments and not args_seen.get(call_id):
                            args_seen[call_id] = True
                            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": arguments, "id": None, "index": indexes[call_id]}]))

    def _handle_line(self, line: str, text: str, tool_calls: list, pending: dict) -> tuple[str, list, dict]:
        if not line.startswith("data: "):
            return text, tool_calls, pending
        raw = line[6:]
        if raw == "[DONE]":
            return text, tool_calls, pending
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return text, tool_calls, pending
        event_type = event.get("type", "")
        if event_type == "error":
            raise RuntimeError(f"Codex API error: {event.get('message', event)}")
        if event_type == "response.output_text.delta":
            text += event.get("delta", "")
        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                pending[item.get("call_id", "")] = {"name": item.get("name", ""), "args": ""}
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id", "")
            if call_id in pending:
                pending[call_id]["args"] += event.get("delta", "")
        elif event_type in ("response.function_call_arguments.done", "response.output_item.done"):
            item = event.get("item", {})
            call_id = event.get("call_id") or item.get("call_id", "")
            if call_id in pending:
                call = pending.pop(call_id)
                args = call["args"] or event.get("arguments") or item.get("arguments") or "{}"
                tool_calls.append({"id": call_id, "name": call["name"], "args": json.loads(args), "type": "tool_call"})
        return text, tool_calls, pending
