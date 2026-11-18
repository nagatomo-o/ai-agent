from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.filesystem import (
    MAX_READ_FILE_BYTES,
    ToolExecutionError,
    WorkspaceSecurityError,
    edit_file,
    list_files,
    preview_edit_file,
    preview_write_file,
    read_file,
    resolve_safe_path,
    search_files,
    write_file,
)


def test_resolve_safe_path_allows_inside_workspace(tmp_path: Path) -> None:
    sub = tmp_path / "src"
    sub.mkdir()
    target = resolve_safe_path(tmp_path, "src/main.py")
    assert target == (tmp_path / "src" / "main.py").resolve()


def test_resolve_safe_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(tmp_path, "../outside.txt")


def test_resolve_safe_path_rejects_outside_absolute_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(tmp_path, str(outside))


def test_list_files_excludes_ignored_dirs_and_respects_depth(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "sub" / "deep").mkdir()
    (tmp_path / "sub" / "deep" / "too_deep.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x\n", encoding="utf-8")

    output = list_files(tmp_path, ".", max_depth=2)
    assert "app.py" in output
    assert "sub/" in output
    assert "sub/nested.txt" in output
    assert "too_deep.txt" not in output
    assert ".git" not in output


def test_search_files_matches_content_and_skips_binary(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Line 1\nLaravel Route\nLine 3\n", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"\x00\x01\x02Laravel\xff")

    output = search_files(tmp_path, "laravel")
    assert "notes.md:2: Laravel Route" in output
    assert "image.bin" not in output


def test_search_files_matches_filename(tmp_path: Path) -> None:
    (tmp_path / "user_controller.py").write_text("# empty\n", encoding="utf-8")
    output = search_files(tmp_path, "controller")
    assert "user_controller.py" in output


def test_read_file_returns_numbered_lines(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    output = read_file(tmp_path, "sample.txt", start_line=2, max_lines=2)
    assert "2 | beta" in output
    assert "3 | gamma" in output
    assert "alpha" not in output
    assert "delta" not in output


def test_read_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError):
        read_file(tmp_path, "missing.txt")


def test_resolve_safe_path_rejects_symlink_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = ws / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink 作成権限がないためスキップ: {exc}")

    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(ws, "escape.txt")


def test_resolve_safe_path_rejects_symlink_escape_via_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside_secret.txt"
    orig_resolve = Path.resolve
    ws_resolved = orig_resolve(ws)
    outside_resolved = orig_resolve(outside)

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == ws_resolved / "escape.txt":
            return outside_resolved
        return orig_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(ws, "escape.txt")


def test_resolve_safe_path_rejects_null_byte_and_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(tmp_path, "bad\x00path.txt")

    orig_resolve = Path.resolve
    ws_resolved = orig_resolve(tmp_path)

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == ws_resolved / "broken":
            raise OSError("simulated resolve error")
        return orig_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(WorkspaceSecurityError):
        resolve_safe_path(tmp_path, "broken")


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    huge = tmp_path / "huge.txt"
    huge.write_bytes(b"a" * (MAX_READ_FILE_BYTES + 1))
    with pytest.raises(ToolExecutionError):
        read_file(tmp_path, "huge.txt")


def test_list_files_rejects_invalid_max_depth(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError):
        list_files(tmp_path, ".", max_depth="invalid")  # type: ignore[arg-type]


def test_read_file_rejects_invalid_line_arguments(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    with pytest.raises(ToolExecutionError):
        read_file(tmp_path, "sample.txt", start_line="abc")  # type: ignore[arg-type]
    with pytest.raises(ToolExecutionError):
        read_file(tmp_path, "sample.txt", max_lines=None)  # type: ignore[arg-type]


def test_preview_and_write_file_create_and_overwrite(tmp_path: Path) -> None:
    create_preview = preview_write_file(tmp_path, "docs/note.txt", "hello\nworld\n")
    assert create_preview["action_type"] == "create"
    assert create_preview["path"] == "docs/note.txt"
    assert "hello" in create_preview["content_preview"]

    msg = write_file(tmp_path, "docs/note.txt", "hello\nworld\n")
    assert "作成" in msg
    assert (tmp_path / "docs" / "note.txt").read_text(encoding="utf-8") == "hello\nworld\n"

    overwrite_preview = preview_write_file(tmp_path, "docs/note.txt", "hello\nupdated\n")
    assert overwrite_preview["action_type"] == "overwrite"
    assert "-world" in overwrite_preview["diff"]
    assert "+updated" in overwrite_preview["diff"]

    msg2 = write_file(tmp_path, "docs/note.txt", "hello\nupdated\n")
    assert "上書き" in msg2


def test_preview_and_edit_file_unique_match(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    preview = preview_edit_file(tmp_path, "app.py", "return 'hi'", "return 'hello'")
    assert preview["action_type"] == "edit"
    assert "-    return 'hi'" in preview["diff"]
    assert "+    return 'hello'" in preview["diff"]

    result = edit_file(tmp_path, "app.py", "return 'hi'", "return 'hello'")
    assert "更新" in result
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hello'\n"


def test_edit_file_rejects_missing_or_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "dup.txt"
    target.write_text("foo\nfoo\n", encoding="utf-8")

    with pytest.raises(ToolExecutionError):
        preview_edit_file(tmp_path, "dup.txt", "bar", "baz")

    with pytest.raises(ToolExecutionError):
        preview_edit_file(tmp_path, "dup.txt", "foo", "baz")

    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "dup.txt", "bar", "baz")

    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "dup.txt", "foo", "baz")


def test_write_tools_reject_directory_and_binary_file(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    with pytest.raises(ToolExecutionError):
        preview_write_file(tmp_path, "subdir", "content")
    with pytest.raises(ToolExecutionError):
        write_file(tmp_path, "subdir", "content")

    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b"\x00\x01\x02hello")
    with pytest.raises(ToolExecutionError):
        preview_write_file(tmp_path, "data.bin", "new text")
    with pytest.raises(ToolExecutionError):
        write_file(tmp_path, "data.bin", "new text")
    with pytest.raises(ToolExecutionError):
        preview_edit_file(tmp_path, "data.bin", "hello", "world")
    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "data.bin", "hello", "world")


def test_edit_tools_reject_empty_old_text_and_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    with pytest.raises(ToolExecutionError):
        preview_edit_file(tmp_path, "note.txt", "", "world")
    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "note.txt", "", "world")

    with pytest.raises(ToolExecutionError):
        preview_edit_file(tmp_path, "missing.txt", "hello", "world")
    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "missing.txt", "hello", "world")


def test_write_and_edit_tools_reject_outside_workspace_path(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceSecurityError):
        preview_write_file(tmp_path, "../escape.txt", "x")
    with pytest.raises(WorkspaceSecurityError):
        write_file(tmp_path, "../escape.txt", "x")
    with pytest.raises(WorkspaceSecurityError):
        preview_edit_file(tmp_path, "../escape.txt", "a", "b")
    with pytest.raises(WorkspaceSecurityError):
        edit_file(tmp_path, "../escape.txt", "a", "b")


def test_write_file_and_edit_file_preserve_lf_newlines(tmp_path: Path) -> None:
    write_file(tmp_path, "lf.txt", "line1\nline2\n")
    target = tmp_path / "lf.txt"
    assert target.read_bytes() == b"line1\nline2\n"

    edit_file(tmp_path, "lf.txt", "line2", "line2_edited\nline3")
    assert target.read_bytes() == b"line1\nline2_edited\nline3\n"


def test_edit_file_preserves_cp932_and_crlf_and_wraps_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cp932_file = tmp_path / "sjis.txt"
    cp932_file.write_bytes("こんにちは\r\n世界\r\n".encode("cp932"))

    preview = preview_edit_file(tmp_path, "sjis.txt", "世界", "日本")
    assert preview["action_type"] == "edit"
    edit_file(tmp_path, "sjis.txt", "世界", "日本")
    assert cp932_file.read_bytes() == "こんにちは\r\n日本\r\n".encode("cp932")

    (tmp_path / "blocked").write_text("file", encoding="utf-8")
    with pytest.raises(ToolExecutionError):
        write_file(tmp_path, "blocked/child.txt", "data")

    orig_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if self == cp932_file and "w" in mode:
            raise OSError("disk full")
        return orig_open(self, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(ToolExecutionError):
        edit_file(tmp_path, "sjis.txt", "日本", "東京")
