from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ドロップダウンに常に表示するモデル。未取得のモデルは選択不可として表示する。
MODEL_LINEUP: tuple[str, ...] = ("gemma4:e4b", "qwen3:4b", "gemma3:4b")


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    default_model: str
    default_workspace: Path
    max_tool_steps: int = 8


def get_settings() -> Settings:
    workspace_env = os.environ.get("LOCAL_LLM_WORKSPACE")
    default_workspace = (
        Path(workspace_env).resolve()
        if workspace_env
        else Path.cwd().resolve()
    )
    return Settings(
        ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        default_model=os.environ.get("LOCAL_LLM_MODEL", MODEL_LINEUP[0]),
        default_workspace=default_workspace,
        max_tool_steps=int(os.environ.get("LOCAL_LLM_MAX_STEPS", "8")),
    )
