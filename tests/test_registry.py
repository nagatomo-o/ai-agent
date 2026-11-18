from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.filesystem import ToolExecutionError
from app.tools.registry import ToolDefinition, ToolRegistry, create_default_registry


def test_default_registry_contains_five_tools_with_expected_approval_flags() -> None:
    registry = create_default_registry()
    info = {item["name"]: item for item in registry.describe_tools()}

    assert set(info.keys()) == {
        "list_files",
        "search_files",
        "read_file",
        "write_file",
        "edit_file",
    }
    assert info["list_files"]["requires_approval"] is False
    assert info["search_files"]["requires_approval"] is False
    assert info["read_file"]["requires_approval"] is False
    assert info["write_file"]["requires_approval"] is True
    assert info["edit_file"]["requires_approval"] is True


def test_registry_ollama_tools_format() -> None:
    registry = create_default_registry()
    tools = registry.ollama_tools()
    assert len(tools) == 5
    first = tools[0]
    assert first["type"] == "function"
    assert "name" in first["function"]
    assert "parameters" in first["function"]


def test_registry_preview_and_execute(tmp_path: Path) -> None:
    registry = create_default_registry()
    preview = registry.build_preview(
        "write_file",
        tmp_path,
        {"path": "hello.txt", "content": "world"},
    )
    assert preview["action_type"] == "create"
    assert not (tmp_path / "hello.txt").exists()

    result = registry.execute(
        "write_file",
        tmp_path,
        {"path": "hello.txt", "content": "world"},
    )
    assert "hello.txt" in result
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "world"


def test_registry_rejects_unknown_tool(tmp_path: Path) -> None:
    registry = create_default_registry()
    with pytest.raises(ToolExecutionError):
        registry.execute("delete_all", tmp_path, {})


def test_registry_edit_file_preview_and_execute(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    registry = create_default_registry()
    preview = registry.build_preview(
        "edit_file",
        tmp_path,
        {"path": "sample.txt", "old_text": "beta", "new_text": "gamma"},
    )
    assert preview["action_type"] == "edit"
    assert "-beta" in preview["diff"]
    assert "+gamma" in preview["diff"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"

    result = registry.execute(
        "edit_file",
        tmp_path,
        {"path": "sample.txt", "old_text": "beta", "new_text": "gamma"},
    )
    assert "sample.txt" in result
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_registry_preview_without_preview_handler_returns_execute_fallback(tmp_path: Path) -> None:
    registry = create_default_registry()
    preview = registry.build_preview("list_files", tmp_path, {})
    assert preview == {"action_type": "execute", "path": "", "diff": "", "content_preview": ""}


def test_registry_rejects_invalid_arguments_and_unknown_get(tmp_path: Path) -> None:
    registry = create_default_registry()
    with pytest.raises(ToolExecutionError):
        registry.get("missing_tool")
    with pytest.raises(ToolExecutionError):
        registry.execute("read_file", tmp_path, {"unexpected_arg": 123})
    with pytest.raises(ToolExecutionError):
        registry.build_preview("write_file", tmp_path, {"unexpected_arg": 123})


def test_registry_custom_registration(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="echo tool",
            requires_approval=False,
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            handler=lambda _root, msg="hi": msg,
        )
    )
    assert registry.get("echo").name == "echo"
    assert registry.execute("echo", tmp_path, {"msg": "hello"}) == "hello"
