from __future__ import annotations

import difflib
import fnmatch
import os
from pathlib import Path

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_LIST_ENTRIES = 200
MAX_SEARCH_HITS = 50
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_READ_FILE_BYTES = 2_000_000
MAX_READ_LINES = 400


class WorkspaceSecurityError(Exception):
    """作業フォルダ外へのアクセスが要求された場合のエラー。"""


class ToolExecutionError(Exception):
    """ツール実行時の入力不備や対象未存在などのエラー。"""


def resolve_safe_path(workspace_root: Path, target_path: str) -> Path:
    if isinstance(target_path, str) and "\x00" in target_path:
        raise WorkspaceSecurityError("パスに不正なNULL文字が含まれています。")
    try:
        root = workspace_root.resolve()
        cleaned = (target_path or ".").strip() or "."
        if "\x00" in cleaned:
            raise WorkspaceSecurityError("パスに不正なNULL文字が含まれています。")
        candidate = Path(cleaned)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    except WorkspaceSecurityError:
        raise
    except (ValueError, OSError) as exc:
        raise WorkspaceSecurityError(
            f"パスを安全に解決できませんでした: {target_path!r}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise WorkspaceSecurityError(
            f"作業フォルダ外のパスにはアクセスできません: {target_path}"
        )
    return resolved


def _rel_posix(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "." if str(rel) == "." else rel.as_posix()


def list_files(workspace_root: Path, path: str = ".", max_depth: int = 2) -> str:
    try:
        depth_val = int(max_depth)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            f"max_depth は整数で指定してください: {max_depth!r}"
        ) from exc

    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if not target.exists():
        raise ToolExecutionError(f"パスが存在しません: {path}")
    if not target.is_dir():
        raise ToolExecutionError(f"ディレクトリではありません: {path}")

    depth_limit = max(1, min(depth_val, 6))
    entries: list[str] = []
    truncated = False

    def walk(current: Path, depth: int) -> None:
        nonlocal truncated
        if depth > depth_limit or truncated:
            return
        try:
            children = sorted(
                current.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError as exc:
            raise ToolExecutionError(f"ディレクトリを読み取れません: {exc}") from exc

        for child in children:
            if child.name in IGNORED_DIRS:
                continue
            resolved_child = child.resolve()
            if not resolved_child.is_relative_to(root):
                continue
            rel = _rel_posix(resolved_child, root)
            suffix = "/" if resolved_child.is_dir() else ""
            entries.append(f"{rel}{suffix}")
            if len(entries) >= MAX_LIST_ENTRIES:
                truncated = True
                return
            if resolved_child.is_dir() and depth < depth_limit:
                walk(resolved_child, depth + 1)

    walk(target, 1)
    if not entries:
        return "（該当するファイル・フォルダはありません）"
    header = f"一覧 ({_rel_posix(target, root)}, depth<={depth_limit}):"
    footer = "\n...（件数上限に達したため省略しました）" if truncated else ""
    return header + "\n" + "\n".join(entries) + footer


def _is_probably_binary(file_path: Path) -> bool:
    try:
        chunk = file_path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in chunk


def search_files(
    workspace_root: Path,
    query: str,
    path: str = ".",
    file_pattern: str = "*",
) -> str:
    q = (query or "").strip()
    if not q:
        raise ToolExecutionError("検索キーワード query を指定してください。")

    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if not target.exists():
        raise ToolExecutionError(f"検索対象パスが存在しません: {path}")

    q_lower = q.lower()
    hits: list[str] = []

    def check_file(file_path: Path) -> None:
        if len(hits) >= MAX_SEARCH_HITS:
            return
        resolved = file_path.resolve()
        if not resolved.is_relative_to(root):
            return
        if not fnmatch.fnmatch(resolved.name, file_pattern or "*"):
            return
        rel = _rel_posix(resolved, root)
        if q_lower in rel.lower():
            hits.append(f"[ファイル名一致] {rel}")
            if len(hits) >= MAX_SEARCH_HITS:
                return
        try:
            size = resolved.stat().st_size
        except OSError:
            return
        if size > MAX_SEARCH_FILE_BYTES or _is_probably_binary(resolved):
            return
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = resolved.read_text(encoding="cp932")
            except (UnicodeDecodeError, OSError):
                return
        except OSError:
            return

        for idx, line in enumerate(text.splitlines(), start=1):
            if q_lower in line.lower():
                snippet = line.strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                hits.append(f"{rel}:{idx}: {snippet}")
                if len(hits) >= MAX_SEARCH_HITS:
                    return

    if target.is_file():
        check_file(target)
    else:
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for fname in sorted(filenames):
                if len(hits) >= MAX_SEARCH_HITS:
                    break
                check_file(Path(dirpath) / fname)
            if len(hits) >= MAX_SEARCH_HITS:
                break

    if not hits:
        return f"検索結果なし (query={q!r})"
    return "\n".join(hits)


def read_file(
    workspace_root: Path,
    path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> str:
    try:
        start = max(1, int(start_line))
        limit = max(1, min(int(max_lines), MAX_READ_LINES))
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            f"start_line と max_lines は整数で指定してください: start_line={start_line!r}, max_lines={max_lines!r}"
        ) from exc

    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if not target.exists():
        raise ToolExecutionError(f"ファイルが存在しません: {path}")
    if not target.is_file():
        raise ToolExecutionError(f"ファイルではありません: {path}")
    try:
        file_size = target.stat().st_size
    except OSError as exc:
        raise ToolExecutionError(f"ファイル情報を取得できません: {exc}") from exc
    if file_size > MAX_READ_FILE_BYTES:
        raise ToolExecutionError(
            f"ファイルサイズが上限 ({MAX_READ_FILE_BYTES} バイト) を超えています: {path} ({file_size} バイト)"
        )
    if _is_probably_binary(target):
        raise ToolExecutionError(f"バイナリファイルのため読み込めません: {path}")

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = target.read_text(encoding="cp932")
        except (UnicodeDecodeError, OSError) as exc:
            raise ToolExecutionError(f"テキストとして読み込めません: {exc}") from exc
    except OSError as exc:
        raise ToolExecutionError(f"ファイルを読み込めません: {exc}") from exc

    lines = text.splitlines()
    total = len(lines)
    end = min(total, start + limit - 1)

    if total == 0:
        return f"{_rel_posix(target, root)} (空ファイル)"
    if start > total:
        raise ToolExecutionError(
            f"開始行 {start} が総行数 {total} を超えています: {path}"
        )

    numbered = [
        f"{idx:>4} | {lines[idx - 1]}"
        for idx in range(start, end + 1)
    ]
    header = f"ファイル: {_rel_posix(target, root)} (表示: {start}-{end} / 全 {total} 行)"
    return header + "\n" + "\n".join(numbered)


def _read_text_preserving_newlines(target: Path) -> tuple[str, str]:
    try:
        file_size = target.stat().st_size
    except OSError as exc:
        raise ToolExecutionError(f"ファイル情報を取得できません: {exc}") from exc
    if file_size > MAX_READ_FILE_BYTES:
        raise ToolExecutionError(
            f"ファイルサイズが上限 ({MAX_READ_FILE_BYTES} バイト) を超えています: {target.name} ({file_size} バイト)"
        )
    try:
        with target.open("r", encoding="utf-8", newline="") as fh:
            return fh.read(), "utf-8"
    except UnicodeDecodeError:
        try:
            with target.open("r", encoding="cp932", newline="") as fh:
                return fh.read(), "cp932"
        except (UnicodeDecodeError, OSError) as exc:
            raise ToolExecutionError(f"テキストとして読み込めません: {exc}") from exc
    except OSError as exc:
        raise ToolExecutionError(f"ファイルを読み込めません: {exc}") from exc


def _build_unified_diff(rel_path: str, before: str, after: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )
    return "\n".join(diff_lines) if diff_lines else "（差分なし）"


def preview_write_file(workspace_root: Path, path: str, content: str) -> dict[str, object]:
    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if target.exists() and target.is_dir():
        raise ToolExecutionError(f"ディレクトリには書き込めません: {path}")
    rel = _rel_posix(target, root)
    existed = target.exists()
    if existed:
        if _is_probably_binary(target):
            raise ToolExecutionError(f"バイナリファイルは上書き対象にできません: {path}")
        before, _ = _read_text_preserving_newlines(target)
        diff = _build_unified_diff(rel, before, content)
        return {
            "action_type": "overwrite",
            "path": rel,
            "diff": diff,
            "content_preview": content[:2000],
            "line_count": len(content.splitlines()),
        }
    return {
        "action_type": "create",
        "path": rel,
        "diff": "",
        "content_preview": content[:2000],
        "line_count": len(content.splitlines()),
    }


def write_file(workspace_root: Path, path: str, content: str) -> str:
    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if target.exists() and target.is_dir():
        raise ToolExecutionError(f"ディレクトリには書き込めません: {path}")
    existed = target.exists()
    if existed and _is_probably_binary(target):
        raise ToolExecutionError(f"バイナリファイルは上書き対象にできません: {path}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
    except OSError as exc:
        raise ToolExecutionError(f"ファイルを書き込めません: {exc}") from exc
    rel = _rel_posix(target, root)
    action = "上書き" if existed else "作成"
    return f"ファイルを{action}しました: {rel} ({len(content.splitlines())} 行)"


def _prepare_edit(
    workspace_root: Path,
    path: str,
    old_text: str,
    new_text: str,
) -> tuple[Path, str, str, str, str]:
    if not old_text:
        raise ToolExecutionError("置換対象 old_text は空にできません。")
    root = workspace_root.resolve()
    target = resolve_safe_path(root, path)
    if not target.exists():
        raise ToolExecutionError(f"ファイルが存在しません: {path}")
    if not target.is_file():
        raise ToolExecutionError(f"ファイルではありません: {path}")
    if _is_probably_binary(target):
        raise ToolExecutionError(f"バイナリファイルは編集できません: {path}")

    before, encoding = _read_text_preserving_newlines(target)
    count = before.count(old_text)
    if count == 0:
        raise ToolExecutionError(f"old_text がファイル内に見つかりません: {path}")
    if count > 1:
        raise ToolExecutionError(
            f"old_text が {count} 箇所見つかりました。一意に特定できるよう前後の行を含めて指定してください。"
        )
    after = before.replace(old_text, new_text, 1)
    rel = _rel_posix(target, root)
    return target, rel, before, after, encoding


def preview_edit_file(
    workspace_root: Path,
    path: str,
    old_text: str,
    new_text: str,
) -> dict[str, object]:
    _, rel, before, after, _ = _prepare_edit(workspace_root, path, old_text, new_text)
    return {
        "action_type": "edit",
        "path": rel,
        "diff": _build_unified_diff(rel, before, after),
        "content_preview": "",
        "line_count": len(after.splitlines()),
    }


def edit_file(
    workspace_root: Path,
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    target, rel, _, after, encoding = _prepare_edit(workspace_root, path, old_text, new_text)
    try:
        with target.open("w", encoding=encoding, newline="") as fh:
            fh.write(after)
    except OSError as exc:
        raise ToolExecutionError(f"ファイルを書き込めません: {exc}") from exc
    return f"ファイルを部分更新しました: {rel}"
