from __future__ import annotations

import fnmatch
import ipaddress
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, urljoin

import httpcore
import httpx
import pathspec
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown

MAX_READ_LINES = 2000
MAX_GREP_RESULTS = 50
MAX_CMD_OUTPUT = 10000

# Directories always excluded from listing and grep (even without .gitignore)
DEFAULT_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".tox", ".mypy_cache", ".pytest_cache"}

# Patterns that warrant a UI warning before execution.
# NOTE: This is a UX hint only, NOT a security boundary. These patterns are
# trivially bypassable (e.g. base64 payloads, python -c, variable expansion).
# The actual security gate is the user confirmation callback.
_DANGEROUS_PATTERNS = re.compile(
    r"rm\s+-[rf]|rm\s+/|rmdir|del\s+/|format\s|mkfs|dd\s+if=|>\s*/dev/"
    r"|curl\s+.*\|\s*sh|wget\s+.*\|\s*sh|chmod\s+777"
    r"|shutdown|reboot|halt|init\s+0",
    re.IGNORECASE,
)


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _safe_path(path: str, *, must_exist: bool = False) -> Path:
    root = _workspace_root()
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes workspace: {path}") from None
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    return resolved


def _path_error(exc: Exception) -> str:
    return f"Error: {exc}"


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
        return b"\x00" in chunk
    except OSError:
        return False


def _line_numbered(text: str, start_line: int = 1) -> str:
    lines = text.splitlines()
    return "\n".join(f"{start_line + idx:>6} | {line}" for idx, line in enumerate(lines))


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    try:
        file_path = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if not file_path.exists():
        return f"Error: file does not exist: {path}"
    if file_path.is_dir():
        return f"Error: path is a directory: {path}"
    if _is_binary(file_path):
        return f"{path}: binary file (not displayed)"

    if start_line is not None and start_line <= 0:
        return "Error: start_line must be a positive integer"
    if end_line is not None and end_line <= 0:
        return "Error: end_line must be a positive integer"

    import itertools
    start = max(0, (start_line or 1) - 1)
    end = end_line
    
    snippet = []
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            limit = min(end, start + MAX_READ_LINES) if end is not None else start + MAX_READ_LINES
            if limit < start:
                return f"{path}: no content in requested range"
            for line in itertools.islice(f, start, limit):
                snippet.append(line.rstrip('\n'))
    except (OSError, ValueError) as exc:
        return f"Error reading {path}: {exc}"

    if not snippet:
        return f"{path}: no content in requested range"
    return _line_numbered("\n".join(snippet), start + 1)


def write_file(path: str, content: str) -> str:
    try:
        file_path = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if file_path.is_dir():
        return f"Error: path is a directory: {path}"
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"
    except OSError as exc:
        return f"Error writing {path}: {exc}"


def _load_gitignore_spec(base: Path) -> pathspec.PathSpec:
    root = _workspace_root()
    patterns = [".git/"]
    gitignore = root / ".gitignore"
    if gitignore.exists():
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
            patterns.extend(lines)
        except OSError:
            pass
    if base != root:
        sub_gitignore = base / ".gitignore"
        if sub_gitignore.exists():
            try:
                lines = sub_gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
                patterns.extend(lines)
            except OSError:
                pass
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def _should_ignore(path: Path, spec: pathspec.PathSpec) -> bool:
    for part in path.parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True
    root = _workspace_root()
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return False
    rel_str = rel_path.as_posix()
    if not rel_str or rel_str == ".":
        return False
    if path.is_dir() and not rel_str.endswith("/"):
        rel_str += "/"
    return spec.match_file(rel_str)


def list_directory(path: str = ".") -> str:
    try:
        base = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if not base.exists() or not base.is_dir():
        return f"Error: directory does not exist: {path}"

    spec = _load_gitignore_spec(base)
    entries: list[str] = []
    for item in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if _should_ignore(item, spec):
            continue
        icon = "[DIR]" if item.is_dir() else "[FILE]"
        size = "-" if item.is_dir() else f"{item.stat().st_size}B"
        entries.append(f"{icon} {item.name} ({size})")
    return "\n".join(entries) if entries else "(empty directory)"


def list_files_recursive(path: str = ".", max_depth: int = 3) -> str:
    """List files recursively up to a certain depth."""
    try:
        base = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if not base.exists() or not base.is_dir():
        return f"Error: directory does not exist: {path}"

    spec = _load_gitignore_spec(base)
    entries: list[str] = []

    def _walk(current_path: Path, depth: int):
        if depth > max_depth:
            return
        
        try:
            items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for item in items:
            if _should_ignore(item, spec):
                continue
            if item.name.startswith(".") and item.name != ".gitignore" and item.name not in DEFAULT_IGNORE_DIRS:
                continue

            indent = "  " * (depth - 1)
            icon = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{indent}{icon} {item.name}")
            
            if item.is_dir():
                _walk(item, depth + 1)

    _walk(base, 1)
    return "\n".join(entries) if entries else "(empty directory)"


def run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 60,
    confirm_callback: Callable[[str], bool] | None = None,
) -> str:
    # Warn on dangerous commands
    is_dangerous = bool(_DANGEROUS_PATTERNS.search(command))
    try:
        workdir = _safe_path(cwd or ".")
    except ValueError as exc:
        return _path_error(exc)
    if not workdir.exists() or not workdir.is_dir():
        return f"Error: cwd is not a directory: {cwd}"

    def _default_confirm(cmd: str) -> bool:
        if is_dangerous:
            print(f"\033[91m!! DANGEROUS: {cmd}\033[0m")
        from rich.prompt import Confirm
        from minimal_cli.display import console
        return Confirm.ask(f"Run command in {workdir}? [bold]{cmd}[/bold]", console=console, default=False)

    confirmer = confirm_callback or _default_confirm
    if not confirmer(command):
        return "Command cancelled by user."

    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"

    output = (proc.stdout or "") + (proc.stderr or "")
    if len(output) > MAX_CMD_OUTPUT:
        output = output[:MAX_CMD_OUTPUT] + "\n...[truncated]"
    return f"exit_code={proc.returncode}\n{output}".strip()


def grep_search(pattern: str, path: str = ".", include: str | None = None) -> str:
    try:
        base = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if not base.exists():
        return f"Error: path does not exist: {path}"

    # Try ripgrep first (fast) — skip hidden dirs by default for consistency
    cmd = ["rg", "-n", "--no-heading", "--no-hidden"]
    if include:
        cmd.extend(["-g", include])
    cmd.extend(["--", pattern, str(base)])

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
        out = proc.stdout.strip()
        if proc.returncode == 0:
            lines = out.splitlines()[:MAX_GREP_RESULTS]
            return "\n".join(lines)
        if proc.returncode == 1:
            return "No matches found."
        err = proc.stderr.strip()
        message = f"Error: grep search failed (exit_code={proc.returncode})"
        if err:
            message = f"{message}\n{err}"
        return message
    except FileNotFoundError:
        pass  # ripgrep not installed — fall through to Python
    except subprocess.TimeoutExpired:
        return "Error: grep search timed out after 20s"
    except subprocess.SubprocessError as exc:
        return f"Error: grep search failed: {exc}"

    # Python fallback (works everywhere)
    return _python_grep(pattern, str(base), include)


def _python_grep(pattern: str, path: str, include: str | None) -> str:
    """Pure-Python grep fallback when ripgrep is unavailable."""
    try:
        compiled = re.compile(pattern)
    except re.error:
        compiled = re.compile(re.escape(pattern))

    base = Path(path)
    if not base.exists():
        return f"Error: path does not exist: {path}"
    files: list[Path] = []
    errors: list[str] = []

    if base.is_file():
        files = [base]
    else:
        def _on_walk_error(exc: OSError) -> None:
            errors.append(str(exc))

        for root, dirs, filenames in os.walk(base, onerror=_on_walk_error):
            # Skip hidden directories and default ignore dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in DEFAULT_IGNORE_DIRS]
            for fname in filenames:
                if include and not fnmatch.fnmatch(fname, include):
                    continue
                files.append(Path(root) / fname)

    results: list[str] = []
    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"{fpath}: {exc}")
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if compiled.search(line):
                results.append(f"{fpath}:{i}:{line}")
                if len(results) >= MAX_GREP_RESULTS:
                    return "\n".join(results)

    if errors and not results:
        return "Error: grep search failed\n" + "\n".join(errors[:5])
    return "\n".join(results) if results else "No matches found."


def replace_in_file(path: str, old_text: str, new_text: str, allow_multiple: bool = False) -> str:
    try:
        file_path = _safe_path(path)
    except ValueError as exc:
        return _path_error(exc)
    if not file_path.exists():
        return f"Error: file does not exist: {path}"
    if file_path.is_dir():
        return f"Error: path is a directory: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return "Error: old_text not found"
        if count > 1 and not allow_multiple:
            return f"Error: old_text found {count} times; set allow_multiple=True"

        replaced = content.replace(old_text, new_text)
        file_path.write_text(replaced, encoding="utf-8")
        return f"Replaced {count} occurrence(s) in {path}"
    except (OSError, ValueError) as exc:
        return f"Error modifying {path}: {exc}"


def _clean_html_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    main = soup.find("main") or soup.body or soup
    return str(main)


def _resolve_and_verify_url(url: str) -> tuple[str, int, str]:
    """Resolve and verify hostname in URL. Returns (hostname, port, ip).

    Raises ValueError if the URL is invalid, resolves to a private/loopback IP,
    or DNS resolution fails.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Scheme {parsed.scheme} is not supported")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid hostname")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addr_info = socket.getaddrinfo(hostname, port)
    except Exception as exc:
        raise ValueError(f"Failed to resolve DNS for {hostname}: {exc}") from exc

    safe_ip = None
    for family, _, _, _, sockaddr in addr_info:
        ip = sockaddr[0]
        if family == socket.AF_INET:
            ip_obj = ipaddress.IPv4Address(ip)
        elif family == socket.AF_INET6:
            ip_obj = ipaddress.IPv6Address(ip)
        else:
            continue

        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            raise ValueError(f"IP {ip} is private/loopback/link-local")

        if not safe_ip:
            safe_ip = ip

    if not safe_ip:
        raise ValueError(f"No valid IP address resolved for {hostname}")

    return hostname, port, safe_ip


class _PinnedTransport(httpcore.ConnectionPool):
    """HTTP transport that connects to a pre-resolved IP instead of re-resolving DNS.

    This prevents TOCTOU / DNS-rebinding attacks where the hostname resolves
    to a safe IP during validation but a private IP at connection time.
    """

    def __init__(self, pinned_ip: str, **kwargs):
        super().__init__(**kwargs)
        self._pinned_ip = pinned_ip

    def _get_connection(
        self,
        origin: httpcore.Origin,
    ) -> httpcore.ConnectionInterface:
        # Override the origin to use the pinned IP for the actual connection,
        # while keeping the original Host header intact.
        pinned_origin = httpcore.Origin(
            scheme=origin.scheme,
            host=self._pinned_ip.encode("ascii"),
            port=origin.port,
        )
        return super()._get_connection(pinned_origin)


def fetch_url(url: str, max_length: int = 20000) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    current_url = url
    response = None

    for _ in range(6):  # max 5 redirects
        try:
            hostname, port, ip = _resolve_and_verify_url(current_url)
        except ValueError as exc:
            return f"Error: URL resolves to a private or invalid IP address, or is otherwise blocked: {exc}"

        # Create a per-hop client with a pinned transport so the connection
        # goes to the validated IP, not a re-resolved (potentially different) one.
        transport = _PinnedTransport(pinned_ip=ip)
        try:
            with httpx.Client(
                timeout=30,
                transport=transport,
                follow_redirects=False,
            ) as client:
                try:
                    response = client.get(current_url, headers=headers)
                except httpx.TimeoutException:
                    return f"Error: timeout while fetching {current_url}"
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code if exc.response is not None else "unknown"
                    return f"Error: HTTP {status_code} while fetching {current_url}"
                except httpx.HTTPError as exc:
                    return f"Error: request failed for {current_url}: {exc}"
        except Exception as exc:
            return f"Error: connection failed for {current_url}: {exc}"

        is_redirect = response.status_code in (301, 302, 303, 307, 308)
        if is_redirect:
            redirect_url = response.headers.get("location")
            if not redirect_url:
                break
            current_url = urljoin(current_url, redirect_url)
        else:
            break
    else:
        return "Error: too many redirects"

    if response is None:
        return f"Error: failed to fetch {url}"

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return f"Error: HTTP {status_code} while fetching {current_url}"

    content_type = response.headers.get("content-type", "unknown")
    body = response.text

    if "html" in content_type:
        body = to_markdown(_clean_html_with_bs4(body), heading_style="ATX")

    if len(body) > max_length:
        body = body[:max_length] + "\n...[truncated]"

    return (
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Content-Type: {content_type}\n\n"
        f"{body}"
    )


def tool_registry(command_timeout: int = 60) -> dict[str, Callable[..., str]]:
    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
        "list_files_recursive": list_files_recursive,
        "run_command": lambda command, cwd=None: run_command(command, cwd=cwd, timeout=command_timeout),
        "grep_search": grep_search,
        "replace_in_file": replace_in_file,
        "fetch_url": fetch_url,
    }
