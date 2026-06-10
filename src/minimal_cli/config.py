from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".minimal-cli"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass(slots=True)
class AppConfig:
    api_key: str
    model: str = "gemini-3.1-flash-lite"
    verbose: bool = False
    max_tool_calls: int = 25
    command_timeout: int = 60
    system_prompt: str | None = None
    auto_save_sessions: bool = True
    max_sessions: int = 50
    auto_resume: bool = False
    context_max_tokens: int = 100_000
    context_keep_recent: int = 4




def load_config_file(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _secure_path_win32(path: Path) -> None:
    """Restricts file or directory permissions to the current owner on Windows."""
    if sys.platform != "win32":
        return
    import subprocess
    try:
        username = os.environ.get("USERNAME")
        if not username:
            import getpass
            username = getpass.getuser()
        if username:
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:F"],
                capture_output=True,
                check=False,
                creationflags=0x08000000,
            )
    except Exception:
        pass


def save_config_file(config: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
    else:
        _secure_path_win32(path.parent)

    content = json.dumps(config, indent=2) + "\n"
    if sys.platform != "win32":
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            return
        except OSError:
            pass

    path.write_text(content, encoding="utf-8")
    if sys.platform == "win32":
        _secure_path_win32(path)


def resolve_config(
    *,
    cli_model: str | None = None,
    cli_api_key: str | None = None,
    prompt_for_key: bool = True,
    config_path: Path = CONFIG_PATH,
) -> AppConfig:
    """Resolve configuration from CLI flags, env vars, and file.

    Priority:
      1. CLI flag
      2. Environment variable
      3. Config file
    """
    file_config = load_config_file(config_path)

    # Resolve API key
    api_key = cli_api_key or os.environ.get("GEMINI_API_KEY") or file_config.get("api_key", "")
    
    if not api_key and prompt_for_key:
        if not sys.stdin or not sys.stdin.isatty():
            raise ValueError("No GEMINI_API_KEY found.")
        try:
            import getpass
            api_key = getpass.getpass("Enter Gemini API key: ").strip()
        except EOFError:
            pass
        if api_key:
            merged = {**file_config, "api_key": api_key}
            save_config_file(merged, config_path)
            
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found.")

    env_verbose = os.environ.get("GEMINI_VERBOSE") or os.environ.get("MINIMAL_CLI_VERBOSE")
    verbose_val = file_config.get("verbose", False)
    if env_verbose is not None:
        verbose_val = env_verbose.lower() in ("true", "1", "yes", "on")

    return AppConfig(
        api_key=api_key or "",
        model=cli_model or os.environ.get("GEMINI_MODEL") or file_config.get("model") or "gemini-3.1-flash-lite",
        verbose=verbose_val,
        max_tool_calls=file_config.get("max_tool_calls", 25),
        command_timeout=file_config.get("command_timeout", 60),
        system_prompt=file_config.get("system_prompt"),
        auto_save_sessions=file_config.get("auto_save_sessions", True),
        max_sessions=file_config.get("max_sessions", 50),
        auto_resume=file_config.get("auto_resume", False),
        context_max_tokens=file_config.get("context_max_tokens", 100_000),
        context_keep_recent=file_config.get("context_keep_recent", 4),
    )
