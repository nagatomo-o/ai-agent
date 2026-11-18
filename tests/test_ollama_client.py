from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.ollama_client import OllamaClientWrapper, OllamaConnectionError, parse_chunk


def test_parse_chunk_extracts_content_thinking_and_tool_calls() -> None:
    raw = SimpleNamespace(
        message=SimpleNamespace(
            content="こんにちは",
            thinking="考えています",
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="list_files",
                        arguments={"path": "."},
                    )
                )
            ],
        )
    )
    parsed = parse_chunk(raw)
    assert parsed.content == "こんにちは"
    assert parsed.thinking == "考えています"
    assert parsed.tool_calls == [{"name": "list_files", "arguments": {"path": "."}}]


@pytest.mark.asyncio
async def test_list_models_raises_connection_error_on_connect_failure() -> None:
    class BrokenSdkClient:
        async def list(self):
            raise httpx.ConnectError("connection refused")

    wrapper = OllamaClientWrapper(host="http://127.0.0.1:11434", sdk_client=BrokenSdkClient())
    with pytest.raises(OllamaConnectionError):
        await wrapper.list_models()


class ShowSdkClient:
    def __init__(self, responses=None, error=None) -> None:
        self.responses = responses or {}
        self.error = error
        self.calls: list[str] = []

    async def show(self, model: str):
        self.calls.append(model)
        if self.error is not None:
            raise self.error
        return self.responses[model]


def _wrapper(sdk) -> OllamaClientWrapper:
    return OllamaClientWrapper(host="http://127.0.0.1:11434", sdk_client=sdk)


@pytest.mark.asyncio
async def test_supports_tools_true_when_capabilities_include_tools() -> None:
    sdk = ShowSdkClient({"qwen3:4b": SimpleNamespace(capabilities=["completion", "tools"])})
    assert await _wrapper(sdk).supports_tools("qwen3:4b") is True


@pytest.mark.asyncio
async def test_supports_tools_false_without_tools_capability() -> None:
    sdk = ShowSdkClient({"gemma3:4b": SimpleNamespace(capabilities=["completion", "vision"])})
    wrapper = _wrapper(sdk)
    assert await wrapper.model_capabilities("gemma3:4b") == ["completion", "vision"]
    assert await wrapper.supports_tools("gemma3:4b") is False


@pytest.mark.asyncio
async def test_supports_tools_true_when_capabilities_missing() -> None:
    sdk = ShowSdkClient(
        {
            "a": SimpleNamespace(),
            "b": SimpleNamespace(capabilities=None),
            "c": {"details": {}},
        }
    )
    wrapper = _wrapper(sdk)
    assert await wrapper.model_capabilities("a") == []
    assert await wrapper.supports_tools("a") is True
    assert await wrapper.supports_tools("b") is True
    assert await wrapper.supports_tools("c") is True


@pytest.mark.asyncio
async def test_supports_tools_reads_dict_response() -> None:
    sdk = ShowSdkClient({"g": {"capabilities": ["completion"]}})
    assert await _wrapper(sdk).supports_tools("g") is False


@pytest.mark.asyncio
async def test_model_capabilities_cached_per_model() -> None:
    sdk = ShowSdkClient({"m": SimpleNamespace(capabilities=["tools"])})
    wrapper = _wrapper(sdk)
    await wrapper.supports_tools("m")
    await wrapper.supports_tools("m")
    await wrapper.model_capabilities("m")
    assert sdk.calls == ["m"]


@pytest.mark.asyncio
async def test_model_capabilities_404_raises_not_found() -> None:
    import ollama

    sdk = ShowSdkClient(error=ollama.ResponseError("not found", 404))
    with pytest.raises(OllamaConnectionError, match="見つかりません"):
        await _wrapper(sdk).model_capabilities("nope")


@pytest.mark.asyncio
async def test_model_capabilities_connect_failure_raises_connection_error() -> None:
    sdk = ShowSdkClient(error=httpx.ConnectError("refused"))
    with pytest.raises(OllamaConnectionError):
        await _wrapper(sdk).supports_tools("m")
