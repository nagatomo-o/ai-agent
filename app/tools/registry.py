from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.tools.filesystem import (
    ToolExecutionError,
    edit_file,
    list_files,
    preview_edit_file,
    preview_write_file,
    read_file,
    search_files,
    write_file,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    requires_approval: bool
    parameters: dict[str, Any]
    handler: Callable[..., str]
    preview_handler: Callable[..., dict[str, Any]] | None = None

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolExecutionError(f"未登録のツールです: {name}")
        return self._tools[name]

    def ollama_tools(self) -> list[dict[str, Any]]:
        return [tool.to_ollama_tool() for tool in self._tools.values()]

    def describe_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_approval": tool.requires_approval,
            }
            for tool in self._tools.values()
        ]

    def build_preview(
        self,
        name: str,
        workspace_root: Path,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        tool = self.get(name)
        if tool.preview_handler is None:
            return {"action_type": "execute", "path": "", "diff": "", "content_preview": ""}
        try:
            return tool.preview_handler(workspace_root, **arguments)
        except TypeError as exc:
            raise ToolExecutionError(f"ツール引数が不正です ({name}): {exc}") from exc

    def execute(
        self,
        name: str,
        workspace_root: Path,
        arguments: dict[str, Any],
    ) -> str:
        tool = self.get(name)
        try:
            return tool.handler(workspace_root, **arguments)
        except TypeError as exc:
            raise ToolExecutionError(f"ツール引数が不正です ({name}): {exc}") from exc


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="list_files",
            description="作業フォルダ内のファイルとフォルダ一覧を取得する。",
            requires_approval=False,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "一覧を取得する相対パス（既定は '.'）。",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "探索する最大階層数（既定は 2）。",
                    },
                },
            },
            handler=list_files,
        )
    )
    registry.register(
        ToolDefinition(
            name="search_files",
            description="作業フォルダ内でファイル名とテキスト内容を部分一致検索する。",
            requires_approval=False,
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索する文字列。",
                    },
                    "path": {
                        "type": "string",
                        "description": "検索開始ディレクトリ（既定は '.'）。",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "対象ファイル名パターン（例: '*.py'、既定は '*'）。",
                    },
                },
            },
            handler=search_files,
        )
    )
    registry.register(
        ToolDefinition(
            name="read_file",
            description="作業フォルダ内のテキストファイルを行番号付きで読み込む。",
            requires_approval=False,
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "読み込むファイルの相対パス。",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "読み込み開始行番号（1 始まり）。",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "読み込む最大行数（既定は 200）。",
                    },
                },
            },
            handler=read_file,
        )
    )
    registry.register(
        ToolDefinition(
            name="write_file",
            description="作業フォルダ内にファイルを新規作成または全体上書きする（実行前に承認が必要）。",
            requires_approval=True,
            parameters={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "作成・上書きするファイルの相対パス。",
                    },
                    "content": {
                        "type": "string",
                        "description": "書き込むファイル全文。",
                    },
                },
            },
            handler=write_file,
            preview_handler=preview_write_file,
        )
    )
    registry.register(
        ToolDefinition(
            name="edit_file",
            description="既存ファイル内の一意に一致する文字列を置換する（実行前に承認が必要）。",
            requires_approval=True,
            parameters={
                "type": "object",
                "required": ["path", "old_text", "new_text"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "編集するファイルの相対パス。",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "置換前の文字列（ファイル内で1箇所のみ一致する必要がある）。",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "置換後の文字列。",
                    },
                },
            },
            handler=edit_file,
            preview_handler=preview_edit_file,
        )
    )
    return registry
