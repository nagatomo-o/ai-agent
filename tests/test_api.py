from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentConflictError, AgentService
from app.config import Settings
from app.main import create_app
from app.ollama_client import ChatChunk
from app.tools.registry import create_default_registry


class FakeOllamaClient:
    def __init__(self, turns: list[list[ChatChunk]]) -> None:
        self._turns = list(turns)

    async def list_models(self) -> list[str]:
        return ["qwen3:4b", "qwen3:8b", "gemma3:4b"]

    async def supports_tools(self, model: str) -> bool:
        return model != "gemma3:4b"

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncIterator[ChatChunk]:
        for chunk in self._turns.pop(0):
            yield chunk


def _make_parts(
    tmp_path: Path,
    turns: list[list[ChatChunk]] | None = None,
) -> tuple[TestClient, AgentService]:
    settings = Settings(
        ollama_host="http://127.0.0.1:11434",
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
    )
    fake = FakeOllamaClient(turns or [[ChatChunk(content="ok")]])
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model=settings.default_model,
        default_workspace=settings.default_workspace,
        max_tool_steps=settings.max_tool_steps,
    )
    app = create_app(agent_service=service, ollama_client=fake, settings=settings)
    return TestClient(app), service


def _make_client(tmp_path: Path, turns: list[list[ChatChunk]] | None = None) -> TestClient:
    return _make_parts(tmp_path, turns)[0]


_WRITE_TURNS = [
    [
        ChatChunk(
            tool_calls=[
                {
                    "name": "write_file",
                    "arguments": {"path": "memo.txt", "content": "hi"},
                }
            ]
        )
    ],
    [ChatChunk(content="完了しました")],
]


def test_health_and_session_creation_and_workspace_update(tmp_path: Path) -> None:
    sub = tmp_path / "project_a"
    sub.mkdir()
    client = _make_client(tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ollama_connected"] is True
    assert "qwen3:4b" in body["models"]

    created = client.post("/api/sessions", json={})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    bad_ws = client.patch(
        f"/api/sessions/{sid}/workspace",
        json={"workspace_root": str(tmp_path / "missing_dir")},
    )
    assert bad_ws.status_code == 400

    good_ws = client.patch(
        f"/api/sessions/{sid}/workspace",
        json={"workspace_root": str(sub)},
    )
    assert good_ws.status_code == 200
    assert Path(good_ws.json()["workspace_root"]) == sub.resolve()


def test_chat_and_approval_sse_endpoints(tmp_path: Path) -> None:
    client = _make_client(tmp_path, turns=_WRITE_TURNS)
    sid = client.post("/api/sessions", json={}).json()["session_id"]

    resp = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "memo.txt を作って"})
    assert resp.status_code == 200
    assert "event: approval_required" in resp.text

    conflict = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "続き"})
    assert conflict.status_code == 409

    detail = client.get(f"/api/sessions/{sid}").json()
    approval_id = detail["pending_approval"]["approval_id"]

    approve = client.post(
        f"/api/sessions/{sid}/approvals/{approval_id}",
        json={"decision": "approve"},
    )
    assert approve.status_code == 200
    assert "event: done" in approve.text
    assert (tmp_path / "memo.txt").read_text(encoding="utf-8") == "hi"


def test_unknown_session_returns_404(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/chat", json={"prompt": "a"}).status_code == 404
    assert (
        client.post("/api/sessions/nope/approvals/x", json={"decision": "approve"}).status_code
        == 404
    )


def test_empty_prompt_returns_400(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    resp = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "   "})
    assert resp.status_code == 400


def test_chat_returns_409_when_session_busy(tmp_path: Path) -> None:
    client, service = _make_parts(tmp_path)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    service.get_session(sid).busy = True

    resp = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "こんにちは"})
    assert resp.status_code == 409


def test_approval_returns_409_when_busy_or_unmatched(tmp_path: Path) -> None:
    client, service = _make_parts(tmp_path, turns=_WRITE_TURNS)
    sid = client.post("/api/sessions", json={}).json()["session_id"]

    # 承認待ちが無い状態
    none_pending = client.post(
        f"/api/sessions/{sid}/approvals/unknown", json={"decision": "approve"}
    )
    assert none_pending.status_code == 409

    client.post(f"/api/sessions/{sid}/chat", json={"prompt": "memo.txt を作って"})
    session = service.get_session(sid)
    assert session.pending_approval is not None
    approval_id = session.pending_approval.approval_id

    wrong_id = client.post(
        f"/api/sessions/{sid}/approvals/other", json={"decision": "approve"}
    )
    assert wrong_id.status_code == 409

    session.busy = True
    busy = client.post(
        f"/api/sessions/{sid}/approvals/{approval_id}", json={"decision": "approve"}
    )
    assert busy.status_code == 409
    assert not (tmp_path / "memo.txt").exists()

    session.busy = False
    ok = client.post(
        f"/api/sessions/{sid}/approvals/{approval_id}", json={"decision": "reject"}
    )
    assert ok.status_code == 200
    assert "event: done" in ok.text


def test_approval_invalid_decision_is_rejected_before_streaming(tmp_path: Path) -> None:
    client, service = _make_parts(tmp_path, turns=_WRITE_TURNS)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    client.post(f"/api/sessions/{sid}/chat", json={"prompt": "memo.txt を作って"})
    approval_id = service.get_session(sid).pending_approval.approval_id

    resp = client.post(
        f"/api/sessions/{sid}/approvals/{approval_id}", json={"decision": "maybe"}
    )
    assert resp.status_code in (400, 422)
    assert service.get_session(sid).pending_approval is not None


def test_workspace_update_returns_409_when_busy_or_pending(tmp_path: Path) -> None:
    sub = tmp_path / "project_b"
    sub.mkdir()
    client, service = _make_parts(tmp_path, turns=_WRITE_TURNS)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    session = service.get_session(sid)

    session.busy = True
    busy = client.patch(f"/api/sessions/{sid}/workspace", json={"workspace_root": str(sub)})
    assert busy.status_code == 409
    session.busy = False

    client.post(f"/api/sessions/{sid}/chat", json={"prompt": "memo.txt を作って"})
    pending = client.patch(f"/api/sessions/{sid}/workspace", json={"workspace_root": str(sub)})
    assert pending.status_code == 409


def test_conflict_error_from_service_before_stream_becomes_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, service = _make_parts(tmp_path)
    sid = client.post("/api/sessions", json={}).json()["session_id"]

    async def raising_run_chat(session_id: str, prompt: str):
        raise AgentConflictError("race")
        yield  # pragma: no cover

    monkeypatch.setattr(service, "run_chat", raising_run_chat)
    resp = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "こんにちは"})
    assert resp.status_code == 409


def test_client_disconnect_resets_busy(tmp_path: Path) -> None:
    class SlowOllama(FakeOllamaClient):
        async def chat_stream(self, model, messages, tools):
            yield ChatChunk(content="a")
            await asyncio.sleep(30)
            yield ChatChunk(content="b")

    settings = Settings(
        ollama_host="http://127.0.0.1:11434",
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    slow = SlowOllama([])
    service = AgentService(
        ollama_client=slow,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    client = TestClient(create_app(agent_service=service, ollama_client=slow, settings=settings))
    sid = client.post("/api/sessions", json={}).json()["session_id"]

    with client.stream("POST", f"/api/sessions/{sid}/chat", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event: token"):
                break
        assert service.get_session(sid).busy is True
    # ストリームを閉じた後は busy が解除されている
    assert service.get_session(sid).busy is False


def test_unexpected_error_after_stream_start_emits_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.agent import AgentEvent

    client, service = _make_parts(tmp_path)
    sid = client.post("/api/sessions", json={}).json()["session_id"]

    async def failing_run_chat(session_id: str, prompt: str):
        yield AgentEvent("status", {"message": "開始"})
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "run_chat", failing_run_chat)
    resp = client.post(f"/api/sessions/{sid}/chat", json={"prompt": "こんにちは"})
    assert resp.status_code == 200
    assert "event: status" in resp.text
    assert "event: error" in resp.text
    assert "処理中に想定外のエラーが発生しました。" in resp.text
    assert "boom" in resp.text


def test_index_returns_404_when_index_html_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.main as main_module

    static_dir = tmp_path / "static_empty"
    static_dir.mkdir()
    monkeypatch.setattr(main_module, "STATIC_DIR", static_dir)
    client = _make_client(tmp_path)
    assert client.get("/").status_code == 404


def test_module_level_app_exists() -> None:
    import app.main as main_module

    assert main_module.app is not None


def test_static_pages_are_served(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "ローカルLLM エージェント" in res_index.text

    res_css = client.get("/static/frame.css")
    assert res_css.status_code == 200
    assert "--accent" in res_css.text

def test_health_includes_model_capabilities(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    body = client.get("/api/health").json()

    assert "gemma3:4b" in body["models"]
    assert body["model_capabilities"] == {
        "qwen3:4b": {"tools": True},
        "qwen3:8b": {"tools": True},
        "gemma3:4b": {"tools": False},
    }


def test_health_capability_failure_defaults_to_tools_true(tmp_path: Path) -> None:
    from app.ollama_client import OllamaConnectionError

    class FlakyOllama(FakeOllamaClient):
        async def supports_tools(self, model: str) -> bool:
            if model == "qwen3:8b":
                raise OllamaConnectionError("boom")
            return await super().supports_tools(model)

    settings = Settings(
        ollama_host="http://127.0.0.1:11434",
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    flaky = FlakyOllama([])
    service = AgentService(
        ollama_client=flaky,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    client = TestClient(create_app(agent_service=service, ollama_client=flaky, settings=settings))
    body = client.get("/api/health").json()

    assert body["ollama_connected"] is True
    assert body["model_capabilities"]["qwen3:8b"] == {"tools": True}
    assert body["model_capabilities"]["gemma3:4b"] == {"tools": False}


def test_session_includes_tools_enabled_flag(tmp_path: Path) -> None:
    settings = Settings(
        ollama_host="http://127.0.0.1:11434",
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    fake = FakeOllamaClient([])
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        model_supports_tools=fake.supports_tools,
    )
    client = TestClient(create_app(agent_service=service, ollama_client=fake, settings=settings))

    default_session = client.post("/api/sessions", json={}).json()
    assert default_session["tools_enabled"] is True

    chat_only = client.post("/api/sessions", json={"model": "gemma3:4b"}).json()
    assert chat_only["tools_enabled"] is False
    sid = chat_only["session_id"]
    assert client.get(f"/api/sessions/{sid}").json()["tools_enabled"] is False

    switched = client.patch(f"/api/sessions/{sid}/model", json={"model": "qwen3:8b"}).json()
    assert switched["tools_enabled"] is True
    back = client.patch(f"/api/sessions/{sid}/model", json={"model": "gemma3:4b"}).json()
    assert back["tools_enabled"] is False

class LineupOllama(FakeOllamaClient):
    async def list_models(self) -> list[str]:
        return ["gemma3:4b", "qwen3:8b"]


def _make_lineup_client(
    tmp_path: Path,
    fake: FakeOllamaClient,
    default_model: str = "qwen3:4b",
) -> TestClient:
    settings = Settings(
        ollama_host="http://127.0.0.1:11434",
        default_model=default_model,
        default_workspace=tmp_path,
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model=default_model,
        default_workspace=tmp_path,
    )
    return TestClient(create_app(agent_service=service, ollama_client=fake, settings=settings))


def test_health_model_options_include_lineup_and_extra_installed(tmp_path: Path) -> None:
    from app.config import MODEL_LINEUP

    body = _make_lineup_client(tmp_path, LineupOllama([])).get("/api/health").json()
    options = body["model_options"]
    by_name = {o["name"]: o for o in options}

    for name in MODEL_LINEUP:
        assert name in by_name
    assert [o["name"] for o in options] == [*MODEL_LINEUP, "qwen3:8b"]
    assert by_name["gemma3:4b"] == {"name": "gemma3:4b", "installed": True, "tools": False}
    assert by_name["gemma4:e4b"] == {"name": "gemma4:e4b", "installed": False, "tools": None}
    assert by_name["qwen3:4b"] == {"name": "qwen3:4b", "installed": False, "tools": None}
    assert by_name["qwen3:8b"] == {"name": "qwen3:8b", "installed": True, "tools": True}
    assert body["models"] == ["gemma3:4b", "qwen3:8b"]
    assert body["default_model"] == "qwen3:4b"


def test_health_default_model_resolved_falls_back_to_installed(tmp_path: Path) -> None:
    body = _make_lineup_client(tmp_path, LineupOllama([])).get("/api/health").json()
    installed = {o["name"] for o in body["model_options"] if o["installed"]}
    assert body["default_model_resolved"] in installed
    assert body["default_model_resolved"] == "gemma3:4b"


def test_health_default_model_resolved_prefers_installed_default(tmp_path: Path) -> None:
    client = _make_lineup_client(tmp_path, LineupOllama([]), default_model="qwen3:8b")
    assert client.get("/api/health").json()["default_model_resolved"] == "qwen3:8b"


def test_health_default_model_resolved_uses_first_installed_when_no_lineup_match(
    tmp_path: Path,
) -> None:
    class OnlyExtra(FakeOllamaClient):
        async def list_models(self) -> list[str]:
            return ["zeta:1b", "alpha:1b"]

    body = _make_lineup_client(tmp_path, OnlyExtra([])).get("/api/health").json()
    assert body["default_model_resolved"] == "zeta:1b"


def test_health_default_model_resolved_when_nothing_installed(tmp_path: Path) -> None:
    class Empty(FakeOllamaClient):
        async def list_models(self) -> list[str]:
            return []

    body = _make_lineup_client(tmp_path, Empty([])).get("/api/health").json()
    assert body["default_model_resolved"] == "qwen3:4b"
    assert [o["installed"] for o in body["model_options"]] == [False, False, False]


def test_health_model_options_survive_capability_failure(tmp_path: Path) -> None:
    from app.ollama_client import OllamaConnectionError

    class Flaky(LineupOllama):
        async def supports_tools(self, model: str) -> bool:
            if model == "qwen3:8b":
                raise OllamaConnectionError("boom")
            return await super().supports_tools(model)

    body = _make_lineup_client(tmp_path, Flaky([])).get("/api/health").json()
    by_name = {o["name"]: o for o in body["model_options"]}
    assert by_name["qwen3:8b"]["tools"] is True
    assert by_name["gemma3:4b"]["tools"] is False


def test_settings_default_model_uses_lineup_head(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import MODEL_LINEUP, get_settings

    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    assert get_settings().default_model == MODEL_LINEUP[0] == "gemma4:e4b"
    monkeypatch.setenv("LOCAL_LLM_MODEL", "custom:1b")
    assert get_settings().default_model == "custom:1b"
