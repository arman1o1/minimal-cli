from __future__ import annotations
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def ensure_utf8_stdout() -> None:
    """Force UTF-8 stdout/stderr on Windows to prevent cp1252 encoding errors.

    Must be called once at CLI startup, BEFORE any Rich output.
    Not called at import time to avoid breaking pytest capture.
    """
    import io
    import sys

    if sys.platform != "win32":
        return
    if not hasattr(sys.stdout, "buffer"):
        return
    # Don't re-wrap if already UTF-8
    if getattr(sys.stdout, "encoding", "").lower().replace("-", "") == "utf8":
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def print_markdown(text: str) -> None:
    console.print(Markdown(text))


def print_tool_call(name: str, details: str) -> None:
    console.print(f"[bold]>[/bold] [cyan]{name}[/cyan]: {details}")


def print_error(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def print_token_usage(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None = None,
    thoughts_tokens: int | None = None,
    tool_calls: int = 0,
    elapsed: float | None = None,
    session_tokens: int | None = None,
    session_tools: int | None = None,
) -> None:
    parts = []
    tot = total_tokens or ((input_tokens or 0) + (output_tokens or 0))
    tot_str = f"{tot:,}" if tot else "?"
    in_str = f"{input_tokens:,}" if input_tokens is not None else "?"
    out_str = f"{output_tokens:,}" if output_tokens is not None else "?"
    
    parts.append(f"Tokens: {tot_str} ({in_str} in, {out_str} out)")
    
    if thoughts_tokens:
        parts.append(f"Thoughts: {thoughts_tokens:,}")
        
    parts.append(f"Tools: {tool_calls}")
    
    if session_tokens is not None or session_tools is not None:
        sess_tok_str = f"{session_tokens:,}" if session_tokens is not None else "?"
        sess_tools_str = f"{session_tools}" if session_tools is not None else "0"
        parts.append(f"Session: {sess_tok_str} tokens | {sess_tools_str} tools")

    if elapsed is not None:
        parts.append(f"Elapsed: {elapsed:.2f}s")
        
    console.print(f"[dim]{' | '.join(parts)}[/dim]")


# ---------------------------------------------------------------------------
# Interactive arrow-key selector
# ---------------------------------------------------------------------------
def _read_key_windows():
    """Read a single keypress on Windows. Returns 'up'/'down'/'enter'/'esc' or None."""
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        ch2 = msvcrt.getwch()
        return {"H": "up", "P": "down"}.get(ch2)
    if ch == "\r":
        return "enter"
    if ch in ("\x1b", "\x03"):
        return "esc"
    return None


def _read_key_unix():
    """Read a single keypress on Unix. Returns 'up'/'down'/'enter'/'esc' or None."""
    import sys
    import termios
    import tty
    import select

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Check if there is more input available on stdin (0.05 second timeout)
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                seq = sys.stdin.read(2)
                return {"[A": "up", "[B": "down"}.get(seq)
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "esc"
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive_select(
    options: list[tuple[str, str]],
    current: int = 0,
) -> int | None:
    """Show an arrow-key selector. Returns selected index, or None on Esc/Ctrl+C."""
    import sys

    read_key = _read_key_windows if sys.platform == "win32" else _read_key_unix
    n = len(options)
    selected = current

    def _render(first: bool = False) -> None:
        if not first:
            # Move cursor up to overwrite previous render
            sys.stdout.write(f"\033[{n}A")
        for i, (name, desc) in enumerate(options):
            sys.stdout.write("\033[2K")  # clear line
            if i == selected:
                sys.stdout.write(f"  \033[1;37;44m ▸ {name} \033[0m  \033[2m{desc}\033[0m\n")
            else:
                sys.stdout.write(f"    {name}  \033[2m{desc}\033[0m\n")
        sys.stdout.flush()

    _render(first=True)
    try:
        while True:
            key = read_key()
            if key == "up":
                selected = (selected - 1) % n
                _render()
            elif key == "down":
                selected = (selected + 1) % n
                _render()
            elif key == "enter":
                return selected
            elif key == "esc":
                return None
    except (EOFError, KeyboardInterrupt):
        return None


def print_sessions_table(sessions: list) -> None:
    """Print a Rich table of saved sessions."""
    if not sessions:
        console.print("[dim]No saved sessions.[/dim]")
        return

    table = Table(title="Saved Sessions", show_header=True, padding=(0, 1))
    table.add_column("#", style="bold", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Messages", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Model", style="dim")
    table.add_column("Last Active", style="dim")
    table.add_column("Project", style="dim")

    from datetime import datetime

    for i, s in enumerate(sessions, 1):
        # Parse and format the timestamp
        updated = s.updated_at
        try:
            dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            updated = dt.astimezone().strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        project = Path(s.project_dir).name
        table.add_row(
            str(i),
            s.name,
            str(s.message_count),
            f"{getattr(s, 'total_tokens', 0):,}",
            str(getattr(s, 'total_tool_calls', 0)),
            s.model,
            updated,
            project,
        )

    console.print(table)
