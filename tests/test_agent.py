from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.agent import AgentConflictError, AgentService
from app.ollama_client import ChatChunk, OllamaConnectionError
from app.tools.registry import create_default_registry


class FakeOllamaClient:
    def __init__(self, turns: list[list[ChatChunk]]) -> None:
        self._turns = list(turns)
        self.recorded_messages: list[list[dict]] = []
        self.recorded_tools: list[list[dict] | None] = []

    async def list_models(self) -> list[str]:
        return ["qwen3:4b"]

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncIterator[ChatChunk]:
        self.recorded_messages.append([dict(m) for m in messages])
        self.recorded_tools.append(tools)
        chunks = self._turns.pop(0)
        for chunk in chunks:
            yield chunk


@pytest.mark.asyncio
async def test_agent_executes_read_tool_automatically(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("secret line\n", encoding="utf-8")
    fake = FakeOllamaClient(
        turns=[
            [
                ChatChunk(
                    thinking="ファイルを確認します",
                    tool_calls=[{"name": "read_file", "arguments": {"path": "hello.txt"}}],
                )
            ],
            [ChatChunk(content="中身は secret line です。")],
        ]
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "hello.txt を読んで")]
    names = [ev.event for ev in events]

    assert "tool_call" in names
    assert "tool_result" in names
    assert "approval_required" not in names
    assert events[-1].event == "done"
    assert any(
        ev.event == "token" and "secret line" in ev.data["delta"]
        for ev in events
    )


@pytest.mark.asyncio
async def test_agent_pauses_for_write_approval_and_resumes_on_approve(tmp_path: Path) -> None:
    fake = FakeOllamaClient(
        turns=[
            [
                ChatChunk(
                    tool_calls=[
                        {
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "ok\n"},
                        }
                    ]
                )
            ],
            [ChatChunk(content="out.txt を作成しました。")],
        ]
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "out.txt を作って")]
    approval_events = [ev for ev in events if ev.event == "approval_required"]
    assert len(approval_events) == 1
    approval_id = approval_events[0].data["approval_id"]
    assert not (tmp_path / "out.txt").exists()

    with pytest.raises(AgentConflictError):
        async for _ in service.run_chat(session.session_id, "別の質問"):
            pass

    resume_events = [
        ev
        async for ev in service.resolve_approval(
            session.session_id,
            approval_id,
            "approve",
        )
    ]
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok\n"
    assert resume_events[-1].event == "done"


@pytest.mark.asyncio
async def test_agent_handles_rejected_write_without_modifying_file(tmp_path: Path) -> None:
    fake = FakeOllamaClient(
        turns=[
            [
                ChatChunk(
                    tool_calls=[
                        {
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "bad\n"},
                        }
                    ]
                )
            ],
            [ChatChunk(content="作成を中止しました。")],
        ]
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "out.txt を作って")]
    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]

    resume_events = [
        ev
        async for ev in service.resolve_approval(
            session.session_id,
            approval_id,
            "reject",
        )
    ]
    assert not (tmp_path / "out.txt").exists()
    assert resume_events[-1].event == "done"


@pytest.mark.asyncio
async def test_agent_stops_at_max_tool_steps(tmp_path: Path) -> None:
    infinite_turns = [
        [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])]
        for _ in range(3)
    ] + [[ChatChunk(content="上限で終了しました。")]]
    fake = FakeOllamaClient(turns=infinite_turns)
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=2,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "繰り返し調べて")]
    tool_calls = [ev for ev in events if ev.event == "tool_call"]
    assert len(tool_calls) == 2
    assert events[-1].event == "done"


SKIPPED_FOR_APPROVAL = (
    "先行する承認待ちの操作のため、このツールは未実行です。必要であれば承認後に改めて依頼してください。"
)
SKIPPED_FOR_BUDGET = "ツール実行上限に達したため、このツールは未実行です。"
INTERRUPTED = "中断されたため未実行です。"
BUDGET_HINT = "ツール実行上限に達しました。これまでに得た情報だけで回答してください。"


def _service(tmp_path: Path, turns: list[list[ChatChunk]], max_tool_steps: int = 8, registry=None):
    fake = FakeOllamaClient(turns=turns)
    service = AgentService(
        ollama_client=fake,
        registry=registry or create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=max_tool_steps,
    )
    return fake, service, service.create_session()


def _write_and_read_turn() -> list[ChatChunk]:
    return [
        ChatChunk(
            tool_calls=[
                {"name": "write_file", "arguments": {"path": "out.txt", "content": "ok\n"}},
                {"name": "read_file", "arguments": {"path": "hello.txt"}},
            ]
        )
    ]


@pytest.mark.asyncio
async def test_agent_answers_remaining_calls_when_paused_for_approval(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    fake = FakeOllamaClient(
        turns=[
            [
                ChatChunk(
                    tool_calls=[
                        {"name": "write_file", "arguments": {"path": "out.txt", "content": "ok\n"}},
                        {"name": "read_file", "arguments": {"path": "hello.txt"}},
                    ]
                )
            ],
            [ChatChunk(content="完了しました。")],
        ]
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "作って読んで")]
    names = [ev.event for ev in events]

    # tool_call は approval_required の直前に1回だけ出る。スキップされた呼び出しのイベントは出ない
    tool_call_events = [ev for ev in events if ev.event == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].data["name"] == "write_file"
    assert tool_call_events[0].data["requires_approval"] is True
    assert names.index("tool_call") + 1 == names.index("approval_required")
    assert "tool_result" not in names

    # 承認待ちの間は、先頭の呼び出しより前にスキップ応答を積まない
    assert [m for m in session.messages if m["role"] == "tool"] == []
    assert session.pending_approval is not None
    assert [c["name"] for c in session.pending_approval.deferred_calls] == ["read_file"]

    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]
    resume_events = [
        ev async for ev in service.resolve_approval(session.session_id, approval_id, "approve")
    ]
    assert resume_events[-1].event == "done"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok\n"

    # 再開後は assistant の tool_calls と同じ順序で tool メッセージが並ぶ
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_messages] == ["write_file", "read_file"]
    assert tool_messages[0]["content"] != SKIPPED_FOR_APPROVAL
    assert tool_messages[1]["content"] == SKIPPED_FOR_APPROVAL
    # 後続の read_file は実行されない
    assert not [ev for ev in resume_events if ev.event == "tool_call"]
    assert [ev.data["name"] for ev in resume_events if ev.event == "tool_result"] == ["write_file"]


@pytest.mark.asyncio
async def test_agent_answers_remaining_calls_when_step_budget_exhausted(tmp_path: Path) -> None:
    fake = FakeOllamaClient(
        turns=[
            [
                ChatChunk(
                    tool_calls=[
                        {"name": "list_files", "arguments": {"path": "."}},
                        {"name": "list_files", "arguments": {"path": "."}},
                    ]
                )
            ],
            [ChatChunk(content="上限に達しました。")],
        ]
    )
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=1,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "2回調べて")]

    assert len([ev for ev in events if ev.event == "tool_call"]) == 1
    assert len([ev for ev in events if ev.event == "tool_result"]) == 1
    assert events[-1].event == "done"

    assistant_calls = [m for m in session.messages if m.get("tool_calls")]
    assert len(assistant_calls) == 1
    assert len(assistant_calls[0]["tool_calls"]) == 2
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["content"] != SKIPPED_FOR_BUDGET
    assert tool_messages[1] == {
        "role": "tool",
        "tool_name": "list_files",
        "content": SKIPPED_FOR_BUDGET,
    }
    # 最後のモデル呼び出しには、全ツール呼び出しへの応答が揃った履歴が渡される
    last_history = fake.recorded_messages[-1]
    assert [m["role"] for m in last_history[-4:]] == ["assistant", "tool", "tool", "system"]
    assert last_history[-1]["content"] == BUDGET_HINT
    # ヒントは最終呼び出しにだけ渡し、履歴には残さない
    assert all(m.get("content") != BUDGET_HINT for m in session.messages)


@pytest.mark.asyncio
async def test_rejected_approval_keeps_tool_message_order(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    fake, service, session = _service(
        tmp_path, [_write_and_read_turn(), [ChatChunk(content="中止しました。")]]
    )
    events = [ev async for ev in service.run_chat(session.session_id, "作って読んで")]
    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]
    resume = [
        ev async for ev in service.resolve_approval(session.session_id, approval_id, "reject")
    ]

    assert resume[-1].event == "done"
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_messages] == ["write_file", "read_file"]
    assert "拒否" in tool_messages[0]["content"]
    assert tool_messages[1]["content"] == SKIPPED_FOR_APPROVAL
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_tool_calls_are_dropped_when_tools_are_disabled(tmp_path: Path) -> None:
    fake, service, session = _service(
        tmp_path,
        [
            [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])],
            [
                ChatChunk(
                    content="まだ調べます。",
                    tool_calls=[{"name": "list_files", "arguments": {"path": "."}}],
                )
            ],
        ],
        max_tool_steps=1,
    )
    events = [ev async for ev in service.run_chat(session.session_id, "調べて")]

    assert events[-1].event == "done"
    assert len([ev for ev in events if ev.event == "tool_call"]) == 1
    assert len([ev for ev in events if ev.event == "tool_result"]) == 1
    assert [m["role"] for m in session.messages[-3:]] == ["assistant", "tool", "assistant"]
    final_assistant = session.messages[-1]
    assert final_assistant["content"] == "まだ調べます。"
    assert "tool_calls" not in final_assistant
    # 応答のない tool_calls が履歴に残らない
    for index, message in enumerate(session.messages):
        if message.get("tool_calls"):
            following = session.messages[index + 1 : index + 1 + len(message["tool_calls"])]
            assert [m["role"] for m in following] == ["tool"] * len(message["tool_calls"])


@pytest.mark.asyncio
async def test_tool_message_is_appended_before_tool_result_event(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    fake, service, session = _service(
        tmp_path,
        [
            [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])],
            [ChatChunk(content="完了")],
        ],
    )
    seen = 0
    async for ev in service.run_chat(session.session_id, "調べて"):
        if ev.event == "tool_result":
            seen += 1
            assert session.messages[-1]["role"] == "tool"
            assert session.messages[-1]["content"] == ev.data["output"]
    assert seen == 1


@pytest.mark.asyncio
async def test_resolve_approval_appends_messages_before_tool_result_event(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    fake, service, session = _service(
        tmp_path, [_write_and_read_turn(), [ChatChunk(content="完了")]]
    )
    events = [ev async for ev in service.run_chat(session.session_id, "作って読んで")]
    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]
    seen = 0
    async for ev in service.resolve_approval(session.session_id, approval_id, "approve"):
        if ev.event == "tool_result":
            seen += 1
            assert [m["tool_name"] for m in session.messages if m["role"] == "tool"] == [
                "write_file",
                "read_file",
            ]
    assert seen == 1


@pytest.mark.asyncio
async def test_closing_run_chat_answers_unanswered_calls(tmp_path: Path) -> None:
    fake, service, session = _service(
        tmp_path,
        [
            [
                ChatChunk(
                    tool_calls=[
                        {"name": "list_files", "arguments": {"path": "."}},
                        {"name": "list_files", "arguments": {"path": "."}},
                    ]
                )
            ]
        ],
    )
    gen = service.run_chat(session.session_id, "調べて")
    async for ev in gen:
        if ev.event == "tool_call":
            break
    await gen.aclose()

    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_name": "list_files", "content": INTERRUPTED},
        {"role": "tool", "tool_name": "list_files", "content": INTERRUPTED},
    ]
    assert session.busy is False


@pytest.mark.asyncio
async def test_closing_resolve_approval_answers_unanswered_calls(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    fake, service, session = _service(tmp_path, [_write_and_read_turn()])
    events = [ev async for ev in service.run_chat(session.session_id, "作って読んで")]
    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]

    gen = service.resolve_approval(session.session_id, approval_id, "approve")
    async for ev in gen:
        if ev.event == "status":
            break
    await gen.aclose()

    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_name": "write_file", "content": INTERRUPTED},
        {"role": "tool", "tool_name": "read_file", "content": INTERRUPTED},
    ]
    assert not (tmp_path / "out.txt").exists()
    assert session.pending_approval is None
    assert session.busy is False


@pytest.mark.asyncio
async def test_unexpected_tool_error_is_returned_to_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = create_default_registry()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(registry, "execute", boom)
    fake, service, session = _service(
        tmp_path,
        [
            [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])],
            [ChatChunk(content="失敗しました。")],
        ],
        registry=registry,
    )
    events = [ev async for ev in service.run_chat(session.session_id, "調べて")]

    expected = "エラー: 予期しないエラーが発生しました: boom"
    result = [ev for ev in events if ev.event == "tool_result"][0]
    assert result.data["ok"] is False
    assert result.data["output"] == expected
    assert [m for m in session.messages if m["role"] == "tool"][0]["content"] == expected
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_budget_hint_is_added_after_approved_last_step(tmp_path: Path) -> None:
    fake, service, session = _service(
        tmp_path,
        [
            [
                ChatChunk(
                    tool_calls=[
                        {"name": "write_file", "arguments": {"path": "out.txt", "content": "ok\n"}}
                    ]
                )
            ],
            [ChatChunk(content="作成しました。")],
        ],
        max_tool_steps=1,
    )
    events = [ev async for ev in service.run_chat(session.session_id, "作って")]
    approval_id = [ev for ev in events if ev.event == "approval_required"][0].data["approval_id"]
    [ev async for ev in service.resolve_approval(session.session_id, approval_id, "approve")]

    assert fake.recorded_messages[-1][-1] == {"role": "system", "content": BUDGET_HINT}
    assert fake.recorded_messages[-1][-2]["role"] == "tool"


@pytest.mark.asyncio
async def test_no_budget_hint_while_tools_are_available(tmp_path: Path) -> None:
    fake, service, session = _service(
        tmp_path,
        [
            [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])],
            [ChatChunk(content="完了")],
        ],
        max_tool_steps=8,
    )
    [ev async for ev in service.run_chat(session.session_id, "調べて")]
    assert all(m.get("content") != BUDGET_HINT for m in fake.recorded_messages[-1])


@pytest.mark.asyncio
async def test_busy_session_refuses_concurrent_calls_and_resets(tmp_path: Path) -> None:
    fake, service, session = _service(tmp_path, [[ChatChunk(content="応答")]])
    first = service.run_chat(session.session_id, "一つ目")
    await first.__anext__()
    assert session.busy is True

    with pytest.raises(AgentConflictError):
        async for _ in service.run_chat(session.session_id, "二つ目"):
            pass
    with pytest.raises(AgentConflictError):
        async for _ in service.resolve_approval(session.session_id, "x", "approve"):
            pass

    await first.aclose()
    assert session.busy is False


@pytest.mark.asyncio
async def test_busy_is_reset_after_completion_and_errors(tmp_path: Path) -> None:
    fake, service, session = _service(tmp_path, [[ChatChunk(content="応答")]])
    [ev async for ev in service.run_chat(session.session_id, "質問")]
    assert session.busy is False

    class DownClient:
        async def list_models(self) -> list[str]:
            return []

        async def chat_stream(self, model, messages, tools):
            raise OllamaConnectionError("down")
            yield  # pragma: no cover

    down_service = AgentService(
        ollama_client=DownClient(),
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    down_session = down_service.create_session()
    events = [ev async for ev in down_service.run_chat(down_session.session_id, "質問")]
    assert events[-1].event == "error"
    assert down_session.busy is False

    class BrokenClient(DownClient):
        async def chat_stream(self, model, messages, tools):
            raise RuntimeError("bug")
            yield  # pragma: no cover

    broken_service = AgentService(
        ollama_client=BrokenClient(),
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    broken_session = broken_service.create_session()
    with pytest.raises(RuntimeError):
        async for _ in broken_service.run_chat(broken_session.session_id, "質問"):
            pass
    assert broken_session.busy is False


@pytest.mark.asyncio
async def test_budget_hint_does_not_persist_into_later_questions(tmp_path: Path) -> None:
    fake, service, session = _service(
        tmp_path,
        [
            [ChatChunk(tool_calls=[{"name": "list_files", "arguments": {"path": "."}}])],
            [ChatChunk(content="一つ目の回答です。")],
            [ChatChunk(content="二つ目の回答です。")],
        ],
        max_tool_steps=1,
    )
    [ev async for ev in service.run_chat(session.session_id, "調べて")]
    assert fake.recorded_messages[-1][-1] == {"role": "system", "content": BUDGET_HINT}
    assert all(m.get("content") != BUDGET_HINT for m in session.messages)

    [ev async for ev in service.run_chat(session.session_id, "別の質問")]
    second_call_messages = fake.recorded_messages[-1]
    assert second_call_messages[-1] == {"role": "user", "content": "別の質問"}
    assert all(m.get("content") != BUDGET_HINT for m in second_call_messages)


@pytest.mark.asyncio
async def test_unexpected_preview_error_is_returned_to_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = create_default_registry()

    def boom(*args, **kwargs):
        raise RuntimeError("preview bug")

    monkeypatch.setattr(registry, "build_preview", boom)
    fake, service, session = _service(
        tmp_path,
        [
            [
                ChatChunk(
                    tool_calls=[
                        {"name": "write_file", "arguments": {"path": "out.txt", "content": "ok\n"}}
                    ]
                )
            ],
            [ChatChunk(content="失敗しました。")],
        ],
        registry=registry,
    )
    events = [ev async for ev in service.run_chat(session.session_id, "作って")]

    expected = "エラー: 予期しないエラーが発生しました: preview bug"
    assert "approval_required" not in [ev.event for ev in events]
    result = [ev for ev in events if ev.event == "tool_result"][0]
    assert result.data["ok"] is False
    assert result.data["output"] == expected
    assert [m for m in session.messages if m["role"] == "tool"][0]["content"] == expected
    assert session.pending_approval is None
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_model_without_tool_support_is_not_offered_tools(tmp_path: Path) -> None:
    fake = FakeOllamaClient(turns=[[ChatChunk(content="こんにちは")]])
    checked: list[str] = []

    async def supports_tools(model: str) -> bool:
        checked.append(model)
        return False

    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="gemma3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
        model_supports_tools=supports_tools,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "こんにちは")]

    assert checked == ["gemma3:4b"]
    assert not fake.recorded_tools[0]
    assert events[-1].event == "done"
    assert any(ev.event == "token" and ev.data["delta"] == "こんにちは" for ev in events)


@pytest.mark.asyncio
async def test_model_with_tool_support_is_offered_tools(tmp_path: Path) -> None:
    fake = FakeOllamaClient(turns=[[ChatChunk(content="はい")]])

    async def supports_tools(model: str) -> bool:
        return True

    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
        max_tool_steps=8,
        model_supports_tools=supports_tools,
    )
    session = service.create_session()
    events = [ev async for ev in service.run_chat(session.session_id, "こんにちは")]

    assert fake.recorded_tools[0]
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_tools_are_offered_by_default_without_support_check(tmp_path: Path) -> None:
    fake = FakeOllamaClient(turns=[[ChatChunk(content="はい")]])
    service = AgentService(
        ollama_client=fake,
        registry=create_default_registry(),
        default_model="qwen3:4b",
        default_workspace=tmp_path,
    )
    session = service.create_session()
    [ev async for ev in service.run_chat(session.session_id, "こんにちは")]

    assert fake.recorded_tools[0]
    assert await service.supports_tools("qwen3:4b") is True
