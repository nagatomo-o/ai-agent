from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import ollama


class OllamaConnectionError(Exception):
    """Ollama サーバーへの接続または応答取得に失敗した場合のエラー。"""


@dataclass
class ChatChunk:
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def parse_chunk(raw_chunk: Any) -> ChatChunk:
    msg = getattr(raw_chunk, "message", None)
    if msg is None and isinstance(raw_chunk, dict):
        msg = raw_chunk.get("message", {})
    if msg is None:
        return ChatChunk()

    if isinstance(msg, dict):
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        raw_calls = msg.get("tool_calls") or []
    else:
        content = getattr(msg, "content", None) or ""
        thinking = getattr(msg, "thinking", None) or ""
        raw_calls = getattr(msg, "tool_calls", None) or []

    parsed_calls: list[dict[str, Any]] = []
    for call in raw_calls:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", {})
        parsed_calls.append({"name": str(name), "arguments": dict(args or {})})

    return ChatChunk(content=content, thinking=thinking, tool_calls=parsed_calls)


class OllamaClientWrapper:
    def __init__(self, host: str, sdk_client: Any | None = None) -> None:
        self.host = host
        self._client = sdk_client or ollama.AsyncClient(host=host)
        # モデル名 -> capabilities。None は SDK 応答に capabilities 情報がないこと
        self._capabilities_cache: dict[str, list[str] | None] = {}

    async def _fetch_capabilities(self, model: str) -> list[str] | None:
        if model in self._capabilities_cache:
            return self._capabilities_cache[model]
        try:
            response = await self._client.show(model)
        except ollama.ResponseError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise OllamaConnectionError(
                    f"モデル '{model}' が見つかりません。`ollama run {model}` で取得するか、別モデルを選択してください。"
                ) from exc
            raise OllamaConnectionError(f"Ollama がエラーを返しました: {exc}") from exc
        except (httpx.HTTPError, OSError, ollama.RequestError) as exc:
            raise OllamaConnectionError(
                "Ollama サーバーに接続できません。`ollama serve` または Ollama アプリが起動しているか確認してください。"
            ) from exc

        if isinstance(response, dict):
            raw = response.get("capabilities")
        else:
            raw = getattr(response, "capabilities", None)
        capabilities = None if raw is None else [str(item) for item in raw]
        self._capabilities_cache[model] = capabilities
        return capabilities

    async def model_capabilities(self, model: str) -> list[str]:
        capabilities = await self._fetch_capabilities(model)
        return list(capabilities) if capabilities is not None else []

    async def supports_tools(self, model: str) -> bool:
        """tools を含む、または capabilities 情報がない（不明な）場合は True を返す。"""
        capabilities = await self._fetch_capabilities(model)
        return capabilities is None or "tools" in capabilities

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.list()
        except (httpx.HTTPError, OSError, ollama.RequestError, ollama.ResponseError) as exc:
            raise OllamaConnectionError(
                "Ollama サーバーに接続できません。`ollama serve` または Ollama アプリが起動しているか確認してください。"
            ) from exc

        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        names: list[str] = []
        for item in models or []:
            name = (
                item.get("model") or item.get("name")
                if isinstance(item, dict)
                else getattr(item, "model", None) or getattr(item, "name", None)
            )
            if name:
                names.append(str(name))
        return names

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ChatChunk]:
        try:
            stream = await self._client.chat(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                stream=True,
            )
            async for raw in stream:
                yield parse_chunk(raw)
        except ollama.ResponseError as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                raise OllamaConnectionError(
                    f"モデル '{model}' が見つかりません。`ollama run {model}` で取得するか、別モデルを選択してください。"
                ) from exc
            raise OllamaConnectionError(f"Ollama がエラーを返しました: {exc}") from exc
        except (httpx.HTTPError, OSError, ollama.RequestError) as exc:
            raise OllamaConnectionError(
                "Ollama サーバーとの通信に失敗しました。サーバーの起動状態を確認してください。"
            ) from exc
