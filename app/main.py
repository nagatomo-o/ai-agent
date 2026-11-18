from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

from app.agent import (
    AgentConflictError,
    AgentEvent,
    AgentService,
    AgentSession,
    SessionNotFoundError,
    WorkspacePathError,
)
from app.config import MODEL_LINEUP, Settings, get_settings
from app.ollama_client import OllamaClientWrapper, OllamaConnectionError
from app.tools.registry import ToolRegistry, create_default_registry

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

BUSY_MESSAGE = "このセッションは別の処理を実行中です。"
PENDING_MESSAGE = "承認待ちの操作を先に承認または拒否してください。"
NO_APPROVAL_MESSAGE = "対象の承認待ち操作が見つかりません。"


class CreateSessionRequest(BaseModel):
    model: str | None = None
    workspace_root: str | None = None


class UpdateWorkspaceRequest(BaseModel):
    workspace_root: str


class UpdateModelRequest(BaseModel):
    model: str


class ChatRequest(BaseModel):
    prompt: str


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]


def _format_sse(event: AgentEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


def _serialize_session(session: AgentSession, tools_enabled: bool = True) -> dict[str, Any]:
    pending = None
    if session.pending_approval is not None:
        pending = {
            "approval_id": session.pending_approval.approval_id,
            "name": session.pending_approval.tool_name,
            "arguments": session.pending_approval.arguments,
            **session.pending_approval.preview,
        }
    visible_messages = [
        msg for msg in session.messages if msg.get("role") in ("user", "assistant")
    ]
    return {
        "session_id": session.session_id,
        "model": session.model,
        "workspace_root": str(session.workspace_root),
        "messages": visible_messages,
        "pending_approval": pending,
        "tools_enabled": tools_enabled,
    }


class _AgentStreamingResponse(StreamingResponse):
    """クライアント切断などで中断されたときも、エージェントのジェネレーターを確実に閉じる。"""

    def __init__(
        self,
        source: AsyncGenerator[AgentEvent, None],
        first_event: AgentEvent | None,
    ) -> None:
        self._source = source
        self._stream = self._event_stream(source, first_event)
        super().__init__(self._stream, media_type="text/event-stream")

    @staticmethod
    async def _event_stream(
        source: AsyncGenerator[AgentEvent, None],
        first_event: AgentEvent | None,
    ) -> AsyncIterator[str]:
        async with aclosing(source) as events:
            try:
                if first_event is not None:
                    yield _format_sse(first_event)
                async for event in events:
                    yield _format_sse(event)
            except Exception as exc:  # noqa: BLE001 CancelledError と GeneratorExit は捕捉しない
                error_event = AgentEvent(
                    "error",
                    {
                        "message": "処理中に想定外のエラーが発生しました。",
                        "detail": str(exc),
                    },
                )
                yield _format_sse(error_event)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # キャンセル中でも後始末を完了させ、セッションの busy を確実に解除する
            with anyio.CancelScope(shield=True):
                await self._stream.aclose()
                await self._source.aclose()


async def _open_agent_stream(source: AsyncGenerator[AgentEvent, None]) -> StreamingResponse:
    """最初のイベントを取り出してからレスポンスを返し、開始前の競合や入力不正を HTTP エラーにする。"""
    try:
        first_event: AgentEvent | None = await anext(source)
    except StopAsyncIteration:
        first_event = None
    except AgentConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BaseException:
        await source.aclose()
        raise
    return _AgentStreamingResponse(source, first_event)


def create_app(
    agent_service: AgentService | None = None,
    ollama_client: OllamaClientWrapper | None = None,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    reg = registry or create_default_registry()
    client = ollama_client or OllamaClientWrapper(host=cfg.ollama_host)
    service = agent_service or AgentService(
        ollama_client=client,
        registry=reg,
        default_model=cfg.default_model,
        default_workspace=cfg.default_workspace,
        max_tool_steps=cfg.max_tool_steps,
        model_supports_tools=client.supports_tools,
    )

    api = FastAPI(title="Local LLM Agent")

    async def _tools_enabled(model: str) -> bool:
        # 判定に失敗したモデルは従来どおりツールを使えるものとして扱う
        try:
            return await service.supports_tools(model)
        except OllamaConnectionError:
            return True

    async def _session_payload(session: AgentSession) -> dict[str, Any]:
        return _serialize_session(session, await _tools_enabled(session.model))

    @api.get("/api/health")
    async def health() -> dict[str, Any]:
        try:
            models = await client.list_models()
            connected = True
            error = None
        except OllamaConnectionError as exc:
            models = []
            connected = False
            error = str(exc)
        capabilities: dict[str, dict[str, bool]] = {}
        for name in models:
            try:
                capabilities[name] = {"tools": await client.supports_tools(name)}
            except Exception:  # noqa: BLE001 1モデルの失敗で health 全体を壊さない
                capabilities[name] = {"tools": True}
        installed = set(models)
        option_names = [*MODEL_LINEUP, *(n for n in models if n not in MODEL_LINEUP)]
        model_options = [
            {
                "name": name,
                "installed": name in installed,
                "tools": capabilities[name]["tools"] if name in installed else None,
            }
            for name in option_names
        ]
        if cfg.default_model in installed:
            resolved = cfg.default_model
        else:
            resolved = next(
                (n for n in MODEL_LINEUP if n in installed),
                models[0] if models else cfg.default_model,
            )
        return {
            "ollama_connected": connected,
            "ollama_host": cfg.ollama_host,
            "error": error,
            "models": models,
            "model_capabilities": capabilities,
            "model_options": model_options,
            "default_model": cfg.default_model,
            "default_model_resolved": resolved,
            "default_workspace": str(cfg.default_workspace),
            "max_tool_steps": cfg.max_tool_steps,
            "tools": reg.describe_tools(),
        }

    @api.post("/api/sessions")
    async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        try:
            ws = Path(req.workspace_root) if req.workspace_root else None
            session = service.create_session(model=req.model, workspace_root=ws)
        except WorkspacePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _session_payload(session)

    @api.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return await _session_payload(session)

    @api.patch("/api/sessions/{session_id}/workspace")
    async def update_workspace(session_id: str, req: UpdateWorkspaceRequest) -> dict[str, Any]:
        try:
            current = service.get_session(session_id)
            if current.busy:
                raise AgentConflictError(BUSY_MESSAGE)
            session = service.update_workspace(session_id, req.workspace_root)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AgentConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkspacePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _session_payload(session)

    @api.patch("/api/sessions/{session_id}/model")
    async def update_model(session_id: str, req: UpdateModelRequest) -> dict[str, Any]:
        try:
            session = service.update_model(session_id, req.model)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _session_payload(session)

    @api.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, req: ChatRequest) -> StreamingResponse:
        try:
            session = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if session.busy:
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        if session.pending_approval is not None:
            raise HTTPException(status_code=409, detail=PENDING_MESSAGE)
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="メッセージが空です。")

        return await _open_agent_stream(service.run_chat(session_id, req.prompt))

    @api.post("/api/sessions/{session_id}/approvals/{approval_id}")
    async def approve(session_id: str, approval_id: str, req: ApprovalRequest) -> StreamingResponse:
        try:
            session = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if session.busy:
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        pending = session.pending_approval
        if pending is None or pending.approval_id != approval_id:
            raise HTTPException(status_code=409, detail=NO_APPROVAL_MESSAGE)

        return await _open_agent_stream(
            service.resolve_approval(session_id, approval_id, req.decision)
        )

    if STATIC_DIR.exists():

        @api.get("/")
        async def index() -> FileResponse:
            index_file = STATIC_DIR / "index.html"
            if not index_file.exists():
                raise HTTPException(status_code=404)
            return FileResponse(index_file)

        api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return api


app = create_app()
