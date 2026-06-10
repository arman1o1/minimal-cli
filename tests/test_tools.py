from __future__ import annotations

import sys
from pathlib import Path

import pytest

from minimal_cli.tools import (
    fetch_url,
    grep_search,
    list_directory,
    list_files_recursive,
    read_file,
    replace_in_file,
    run_command,
    write_file,
)


def test_read_and_write_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "a.txt"
    msg = write_file(str(file_path), "hello\nworld\n")
    assert "Wrote" in msg
    out = read_file(str(file_path), 2, 2)
    assert "world" in out


def test_read_binary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "b.bin"
    file_path.write_bytes(b"\x00\x01\x02")
    out = read_file(str(file_path))
    assert "binary file" in out


def test_list_directory_respects_gitignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("skip.txt\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("x", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("y", encoding="utf-8")
    out = list_directory(str(tmp_path))
    assert "keep.txt" in out
    assert "skip.txt" not in out


def test_list_directory_hides_default_ignore_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src").mkdir()
    out = list_directory(str(tmp_path))
    assert "src" in out
    assert ".git" not in out
    assert "__pycache__" not in out
    assert "node_modules" not in out


def test_list_files_recursive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    out = list_files_recursive(".", max_depth=2)
    assert "src" in out
    assert "app.py" in out


def test_run_command_confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = run_command(f'"{sys.executable}" -c "print(\'ok\')"', confirm_callback=lambda _c: True)
    assert "ok" in out


def test_run_command_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = run_command("echo hi", confirm_callback=lambda _c: False)
    assert "cancelled" in out.lower()


def test_grep_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "x.py"
    f.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    out = grep_search("alpha", str(tmp_path), include="*.py")
    assert "x.py" in out


def test_grep_search_missing_path_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = grep_search("alpha", str(tmp_path / "missing"))
    assert "Error:" in out


def test_grep_search_rg_error_is_not_reported_as_no_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "x.py"
    f.write_text("alpha\n", encoding="utf-8")

    class DummyCompletedProcess:
        returncode = 2
        stdout = ""
        stderr = "regex parse error"

    monkeypatch.setattr("minimal_cli.tools.subprocess.run", lambda *args, **kwargs: DummyCompletedProcess())
    out = grep_search("alpha", str(tmp_path), include="*.py")
    assert "Error:" in out
    assert "No matches found" not in out


def test_replace_in_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "r.txt"
    f.write_text("hello world", encoding="utf-8")
    msg = replace_in_file(str(f), "world", "there")
    assert "Replaced" in msg
    assert f.read_text(encoding="utf-8") == "hello there"


def test_replace_in_file_multiple_requires_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "r2.txt"
    f.write_text("x x", encoding="utf-8")
    msg = replace_in_file(str(f), "x", "y")
    assert "allow_multiple" in msg


def test_file_tools_reject_paths_outside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.chdir(workspace)

    assert "escapes workspace" in read_file(str(outside))
    assert "escapes workspace" in write_file(str(outside), "changed")
    assert "escapes workspace" in replace_in_file(str(outside), "secret", "changed")
    assert outside.read_text(encoding="utf-8") == "secret"


def test_run_command_rejects_cwd_outside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    monkeypatch.chdir(workspace)

    out = run_command("echo hi", cwd=str(outside), confirm_callback=lambda _c: True)
    assert "escapes workspace" in out


def test_fetch_url_html(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body><h1>Title</h1><script>x</script></body></html>"

        def raise_for_status(self):
            return None

    import socket
    monkeypatch.setattr(
        "minimal_cli.tools.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    )
    monkeypatch.setattr("minimal_cli.tools.httpx.Client.get", lambda *args, **kwargs: DummyResponse())
    out = fetch_url("https://example.com")
    assert "URL:" in out
    assert "Title" in out
    assert "script" not in out.lower()


def test_fetch_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        import httpx
        raise httpx.HTTPError("boom")

    import socket
    monkeypatch.setattr(
        "minimal_cli.tools.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    )
    monkeypatch.setattr("minimal_cli.tools.httpx.Client.get", boom)
    out = fetch_url("https://example.com")
    assert "Error:" in out


def test_fetch_url_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class DummyResponse:
        status_code = 404
        headers = {"content-type": "text/html"}
        text = "<html><body>missing</body></html>"

        def raise_for_status(self):
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(status_code=404, request=request)
            raise httpx.HTTPStatusError("404", request=request, response=response)

    import socket
    monkeypatch.setattr(
        "minimal_cli.tools.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    )
    monkeypatch.setattr("minimal_cli.tools.httpx.Client.get", lambda *args, **kwargs: DummyResponse())
    out = fetch_url("https://example.com")
    assert "Error: HTTP 404" in out


def test_clean_html_with_bs4() -> None:
    from minimal_cli.tools import _clean_html_with_bs4
    html = "<html><head><style>body {color: red;}</style></head><body><noscript>hidden</noscript><main><h1>Title</h1><script>alert(1)</script></main></body></html>"
    cleaned = _clean_html_with_bs4(html)
    assert "Title" in cleaned
    assert "alert" not in cleaned
    assert "noscript" not in cleaned
    assert "style" not in cleaned
    assert "main" in cleaned


def test_should_ignore_with_pathspec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from minimal_cli.tools import _load_gitignore_spec, _should_ignore
    (tmp_path / ".gitignore").write_text("**/ignored_dir/\n*.log\n", encoding="utf-8")
    
    spec = _load_gitignore_spec(tmp_path)
    
    # Test ignored log file
    assert _should_ignore(tmp_path / "app.log", spec) is True
    # Test kept text file
    assert _should_ignore(tmp_path / "app.txt", spec) is False
    
    # Test nested ignored folder
    nested_dir = tmp_path / "src" / "ignored_dir"
    nested_dir.mkdir(parents=True)
    assert _should_ignore(nested_dir, spec) is True


def test_file_tools_directory_checks_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    
    # Test read_file negative bounds
    f = tmp_path / "bounds.txt"
    f.write_text("line1\nline2\n", encoding="utf-8")
    assert "Error:" in read_file(str(f), start_line=0)
    assert "Error:" in read_file(str(f), start_line=-1)
    assert "Error:" in read_file(str(f), end_line=0)
    assert "Error:" in read_file(str(f), end_line=-5)
    assert "no content in requested range" in read_file(str(f), start_line=5, end_line=2)

    # Test directory inputs for write_file and replace_in_file
    dir_path = tmp_path / "mydir"
    dir_path.mkdir()
    
    assert "is a directory" in write_file(str(dir_path), "some content")
    assert "is a directory" in replace_in_file(str(dir_path), "old", "new")


def test_security_protections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    
    # Test fetch_url blocks local / private URLs (SSRF protection)
    assert "blocked" in fetch_url("http://127.0.0.1")
    assert "blocked" in fetch_url("http://localhost")
    assert "blocked" in fetch_url("http://169.254.169.254/metadata")
    assert "blocked" in fetch_url("http://192.168.1.1")

    # Test grep_search handles options gracefully without argument injection
    f = tmp_path / "y.txt"
    f.write_text("--version information\n", encoding="utf-8")
    out = grep_search("--version", str(tmp_path))
    assert "y.txt" in out or "No matches found" in out or "No matches" in out


def test_replace_in_file_unicode_decode_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "binary_error.bin"
    f.write_bytes(b"\xff\xfe\x00\x01\x80\x81")
    
    out = replace_in_file(str(f), "old", "new")
    assert "Error modifying" in out


def test_dns_rebinding_ssrf_resolves_and_verifies_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_cli.tools import _resolve_and_verify_url
    
    def mock_getaddrinfo_private(host, port, *args, **kwargs):
        import socket
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        
    monkeypatch.setattr("minimal_cli.tools.socket.getaddrinfo", mock_getaddrinfo_private)
    
    with pytest.raises(ValueError, match="private/loopback/link-local"):
        _resolve_and_verify_url("http://example.com")
