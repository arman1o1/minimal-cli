from __future__ import annotations

from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from rich.table import Table

from google.genai.errors import APIError

from . import __version__
from .agent import MinimalAgent
from .config import resolve_config, CONFIG_DIR
from .display import console, ensure_utf8_stdout, interactive_select, print_error, print_token_usage, print_sessions_table


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


SLASH_COMMANDS: dict[str, str] = {
    "/help":     "Show this help",
    "/model":    "Switch model — /model <name> or /model <number>",
    "/clear":    "Clear conversation history",
    "/config":   "Show current config",
    "/verbose":  "Toggle verbose tool-call logging",
    "/add":      "Inject file contents into context — /add <path>",
    "/save":     "Save conversation to a JSON file — /save <path>",
    "/load":     "Load conversation from a JSON file — /load <path>",
    "/sessions": "List saved sessions",
    "/resume":   "Resume a previous session — /resume [id|number]",
    "/rename":   "Name the current session — /rename <name>",
}


class SlashCommandCompleter(Completer):
    def __init__(self, commands: dict[str, str]) -> None:
        self.commands = commands
        self.path_completer = PathCompleter()

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith('/'):
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0]

        if len(parts) == 1 and not text.endswith(' '):
            for name, desc in self.commands.items():
                if name.startswith(cmd):
                    yield Completion(
                        name,
                        start_position=-len(cmd),
                        display_meta=desc
                    )
            return

        if text.endswith(' ') and len(parts) == 1:
            arg = ""
        elif len(parts) == 2:
            arg = parts[1]
        else:
            return

        if cmd in ('/add', '/save', '/load'):
            sub_doc = Document(arg, cursor_position=len(arg))
            for completion in self.path_completer.get_completions(sub_doc, complete_event):
                yield completion
        elif cmd == '/model':
            models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-pro"]
            for m in models:
                if m.startswith(arg):
                    yield Completion(m, start_position=-len(arg))


def _handle_slash_command(
    raw: str,
    agent: MinimalAgent,
    history: list,
    session_manager=None,
    current_session=None,
) -> bool | str | tuple:
    """Handle a slash command.

    Returns:
      - True if consumed
      - A string to pre-fill the next prompt
      - A tuple ("session_update", new_session) to update the current session
    """
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/":
        options = list(SLASH_COMMANDS.items())
        console.print("[dim]↑/↓ to pick command, Enter to confirm, Esc to cancel[/dim]\n")
        choice = interactive_select(options)
        if choice is None:
            return True
        cmd = options[choice][0]

    if cmd == "/help":
        table = Table(title="Slash Commands", show_header=False, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()
        for name, desc in SLASH_COMMANDS.items():
            table.add_row(name, desc)
        console.print(table)
        return True

    if cmd == "/model":
        
        models_list = [
            ("gemini-3.5-flash",       "Flagship fast & smart model"),
            ("gemini-3.1-flash-lite",  "Lightweight, ultra-fast"),
            ("gemini-3.1-pro-preview", "Flagship reasoning (preview)"),
            ("gemini-2.5-pro",         "High intelligence reasoning"),
        ]
        

        known_names = [m[0] for m in models_list]
        if not arg:
            # Find which index is currently active
            cur_idx = next((i for i, m in enumerate(models_list) if m[0] == agent.model), 0)
            console.print(f"[dim]Current:[/dim] [bold]{agent.model}[/bold]")
            console.print("[dim]↑/↓ to pick, Enter to confirm, Esc to cancel[/dim]\n")
            choice = interactive_select(models_list, current=cur_idx)
            if choice is None:
                console.print("[dim]Cancelled.[/dim]")
                return True
            old = agent.model
            agent.model = models_list[choice][0]
            if agent.model != old:
                console.print(f"[green]Model changed:[/green] {old} → [bold]{agent.model}[/bold]")
            else:
                console.print(f"[dim]Model unchanged:[/dim] {agent.model}")
            return True
        # Accept a number shortcut
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(models_list):
                arg = models_list[idx][0]
            else:
                print_error(f"Pick 1–{len(models_list)}")
                return True
        old = agent.model
        agent.model = arg
        note = "" if arg in known_names else "  [dim](custom / may require paid tier)[/dim]"
        console.print(f"[green]Model changed:[/green] {old} → [bold]{agent.model}[/bold]{note}")
        return True

    if cmd == "/clear":
        history.clear()
        console.print("[green]Conversation history cleared.[/green]")
        return True

    if cmd == "/config":
        console.print(f"  [bold]model[/bold]          {agent.model}")
        console.print(f"  [bold]max_tool_calls[/bold] {agent.max_tool_calls}")
        console.print(f"  [bold]verbose[/bold]        {agent.verbose}")
        return True

    if cmd == "/verbose":
        agent.verbose = not agent.verbose
        state = "on" if agent.verbose else "off"
        console.print(f"[green]Verbose mode:[/green] {state}")
        return True

    if cmd == "/add":
        if not arg:
            return "/add "
        path = Path(arg).resolve()
        if not path.is_file():
            print_error(f"File not found: {path}")
            return True
        if path.stat().st_size > 2 * 1024 * 1024:
            print_error(f"File too large to add (> 2MB): {path.name}")
            return True
        try:
            content = path.read_text(encoding="utf-8")
            from google.genai import types
            history.append(types.Content(role="user", parts=[types.Part.from_text(text=f"[Context loaded from file: {path.name}]\n\n{content}")]))
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=f"Understood. I have read the contents of {path.name} and will keep it in mind.")]))
            console.print(f"[green]Added {path.name} to context ({len(content)} chars).[/green]")
        except Exception as e:
            print_error(f"Failed to read {path.name}: {e}")
        return True

    if cmd == "/save":
        if not arg:
            return "/save "
        path = Path(arg).resolve()
        try:
            from google.genai import types
            data = []
            for msg in history:
                if getattr(msg, "model_dump", None):
                    data.append(msg.model_dump(mode="json"))
            path.parent.mkdir(parents=True, exist_ok=True)
            import json
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print(f"[green]Conversation saved to {path}[/green]")
        except Exception as e:
            print_error(f"Failed to save conversation: {e}")
        return True

    if cmd == "/load":
        if not arg:
            return "/load "
        path = Path(arg).resolve()
        if not path.is_file():
            print_error(f"File not found: {path}")
            return True
        try:
            import json
            from google.genai import types
            data = json.loads(path.read_text(encoding="utf-8"))
            history.clear()
            for item in data:
                # Basic restoration from model_dump() dict
                # Note: Function call parts might require more careful parsing, 
                # but standard type mapping usually handles it.
                role = item.get("role")
                parts_data = item.get("parts", [])
                parts = []
                for p in parts_data:
                    if p.get("text") is not None:
                        parts.append(types.Part.from_text(text=p["text"]))
                    elif p.get("function_call") is not None:
                        # minimal restoration of function calls
                        fc = p["function_call"]
                        parts.append(types.Part.from_function_call(name=fc["name"], args=fc.get("args", {})))
                    elif p.get("function_response") is not None:
                        fr = p["function_response"]
                        parts.append(types.Part.from_function_response(name=fr["name"], response=fr.get("response", {})))
                
                if parts:
                    history.append(types.Content(role=role, parts=parts))
                    
            console.print(f"[green]Loaded {len(history)} messages from {path}[/green]")
        except Exception as e:
            print_error(f"Failed to load conversation: {e}")
        return True

    # --- Session management commands ---

    if cmd == "/sessions":
        if session_manager is None:
            print_error("Session management not available.")
            return True
        sessions = session_manager.list_sessions()
        print_sessions_table(sessions)
        return True

    if cmd == "/resume":
        if session_manager is None:
            print_error("Session management not available.")
            return True
        sessions = session_manager.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions to resume.[/dim]")
            return True

        target_session = None
        if arg:
            # Try as number first
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target_session = sessions[idx]
                else:
                    print_error(f"Pick 1–{len(sessions)}")
                    return True
            else:
                # Try as ID prefix or name match
                for s in sessions:
                    if s.id.startswith(arg) or s.name.lower() == arg.lower():
                        target_session = s
                        break
                if not target_session:
                    print_error(f"No session matching '{arg}'")
                    return True
        else:
            # Interactive picker
            options = [(f"{s.name} ({s.message_count} msgs)", s.id[:8]) for s in sessions]
            console.print("[dim]↑/↓ to pick session, Enter to confirm, Esc to cancel[/dim]\n")
            choice = interactive_select(options)
            if choice is None:
                console.print("[dim]Cancelled.[/dim]")
                return True
            target_session = sessions[choice]

        # Load the session
        try:
            loaded_history = session_manager.load_session(target_session.id)
            history.clear()
            history.extend(loaded_history)
            console.print(f"[green]Resumed session:[/green] [bold]{target_session.name}[/bold] ({len(history)} messages)")
            return ("session_update", target_session)
        except Exception as e:
            print_error(f"Failed to resume session: {e}")
            return True

    if cmd == "/rename":
        if session_manager is None or current_session is None:
            print_error("No active session to rename.")
            return True
        if not arg:
            return "/rename "
        try:
            session_manager.rename_session(current_session.id, arg)
            console.print(f"[green]Session renamed to:[/green] [bold]{arg}[/bold]")
            # Update the current session object in-place by returning the new name
            return ("session_rename", arg)
        except Exception as e:
            print_error(f"Failed to rename session: {e}")
        return True

    return False


@click.command(context_settings={"help_option_names": ["--help"]})
@click.argument("prompt", required=False)
@click.option("--model", "model_opt", "-m", default=None, help="Model to use (default: gemini-3.1-flash-lite)")
@click.option("--api-key", default=None, help="Gemini API key (overrides env/config)")
@click.option("--verbose/--no-verbose", "-v", "verbose_opt", default=None, help="Show detailed tool call info")
@click.version_option(__version__)
def main(
    prompt: str | None,
    model_opt: str | None,
    api_key: str | None,
    verbose_opt: bool | None,
) -> None:
    """A lightweight coding agent powered by Gemini."""
    ensure_utf8_stdout()
    try:
        cfg = resolve_config(cli_model=model_opt, cli_api_key=api_key)
    except ValueError as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc

    is_verbose = verbose_opt if verbose_opt is not None else cfg.verbose

    agent = MinimalAgent(
        api_key=cfg.api_key,
        model=cfg.model,
        max_tool_calls=cfg.max_tool_calls,
        command_timeout=cfg.command_timeout,
        verbose=is_verbose,
        system_prompt=cfg.system_prompt,
    )

    console.print(f"minimal-cli v{__version__} | {cfg.model} | {Path.cwd()}")

    if prompt:
        # One-shot mode — no session management
        try:
            result = agent.run_turn(prompt)
            console.print()
            print_token_usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                thoughts_tokens=result.thoughts_tokens,
                tool_calls=result.tool_calls,
                elapsed=result.elapsed_seconds,
            )
        except APIError as e:
            msg = getattr(e, "message", str(e))
            prefix = getattr(e, "status", "API ERROR")
            print_error(f"{getattr(e, 'code', '')} {prefix}: {msg}".strip())
        except Exception as e:
            print_error(f"Unexpected error: {e}")
        return

    # ---------------------------------------------------------------------------
    # Interactive mode — session lifecycle
    # ---------------------------------------------------------------------------
    from .session import SessionManager

    session_manager = SessionManager(CONFIG_DIR / "sessions")
    current_session = None

    # Auto-resume or create new session
    if cfg.auto_resume:
        last = session_manager.get_last_session()
        if last and last.project_dir == str(Path.cwd()):
            try:
                loaded = session_manager.load_session(last.id)
                current_session = last
                history: list = loaded
                console.print(f"[dim]Resumed session:[/dim] [bold]{last.name}[/bold] ({len(history)} messages)")
            except Exception:
                current_session = None

    if current_session is None:
        current_session = session_manager.create_session(
            model=cfg.model,
            project_dir=str(Path.cwd()),
        )
        history = []

    # Prune old sessions
    if cfg.auto_save_sessions:
        pruned = session_manager.prune_sessions(cfg.max_sessions)
        if pruned:
            console.print(f"[dim]Pruned {pruned} old session(s).[/dim]")

    # Track session-level cumulative stats in memory
    session_tokens = current_session.total_tokens if current_session else 0
    session_tools = current_session.total_tool_calls if current_session else 0

    session_info = f"[dim]Session:[/dim] {current_session.name}"
    console.print(f"{session_info} | [dim]Type /help for commands[/dim]")

    # Context pruning setup
    from .context import should_prune, prune_history

    def _make_summarize_fn():
        """Create a summarize function that uses the agent."""
        def _summarize(text: str) -> str:
            try:
                result = agent.run_turn(
                    f"Summarize this conversation so far in 3-5 bullet points, "
                    f"focusing on what files were discussed, what changes were made, "
                    f"and any decisions reached:\n\n{text}",
                    silent=True
                )
                return result.text
            except Exception:
                return "(Summary unavailable)"
        return _summarize

    bindings = KeyBindings()

    @bindings.add('enter')
    def handle_enter(event):
        buffer = event.current_buffer
        text = buffer.text
        if text.endswith('\\') and not text.endswith('\\\\'):
            new_text = text[:-1] + '\n'
            buffer.text = new_text
            buffer.cursor_position = len(new_text)
        else:
            buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def handle_newline(event):
        event.current_buffer.insert_text('\n')

    completer = SlashCommandCompleter(SLASH_COMMANDS)

    def prompt_continuation(width, line_number, is_soft_wrap):
        return '... '

    session_prompt = PromptSession(
        message=">: ",
        completer=completer,
        complete_while_typing=True,
        key_bindings=bindings,
        multiline=True,
        prompt_continuation=prompt_continuation
    )

    default_prompt = ""
    
    def pre_run():
        import prompt_toolkit
        app = prompt_toolkit.application.current.get_app()
        if app.current_buffer.text.startswith("/add "):
            app.current_buffer.start_completion()

    while True:
        try:
            user_in = session_prompt.prompt(default=default_prompt, pre_run=pre_run)
            default_prompt = ""
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye :)")
            break

        if user_in.strip().lower() in {"exit", "quit"}:
            console.print("bye :)")
            break
        if not user_in.strip():
            continue

        # Slash commands
        if user_in.strip().startswith("/"):
            res = _handle_slash_command(
                user_in, agent, history,
                session_manager=session_manager,
                current_session=current_session,
            )
            if res is True:
                continue
            if isinstance(res, str):
                default_prompt = res
                continue
            if isinstance(res, tuple):
                action = res[0]
                if action == "session_update":
                    current_session = res[1]
                    session_tokens = current_session.total_tokens
                    session_tools = current_session.total_tool_calls
                elif action == "session_rename":
                    # Update the name on the current session object
                    current_session = session_manager.get_session(current_session.id)
                continue
            print_error(f"Unknown command: {user_in.strip().split()[0]}  — type /help")
            continue

        # Context pruning before sending to LLM
        pruned = False
        if should_prune(history, cfg.context_max_tokens):
            console.print("[dim]Pruning conversation context...[/dim]")
            active_history = prune_history(
                history,
                cfg.context_max_tokens,
                summarize_fn=_make_summarize_fn(),
                keep_recent=cfg.context_keep_recent,
            )
            pruned = True
        else:
            active_history = history

        pre_turn_len = len(active_history)

        try:
            result = agent.run_turn(user_in, history=active_history)
            console.print()
            
            if pruned:
                # Replace history with the pruned base + new turn messages
                new_messages = active_history[pre_turn_len:]
                history.clear()
                history.extend(active_history[:pre_turn_len])
                history.extend(new_messages)
            
            tokens_added = result.total_tokens or ((result.input_tokens or 0) + (result.output_tokens or 0))
            session_tokens += tokens_added
            session_tools += result.tool_calls

            print_token_usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                thoughts_tokens=result.thoughts_tokens,
                tool_calls=result.tool_calls,
                elapsed=result.elapsed_seconds,
                session_tokens=session_tokens,
                session_tools=session_tools,
            )

            # Auto-save after each turn
            if cfg.auto_save_sessions and current_session:
                try:
                    session_manager.save_turn(
                        current_session.id,
                        history,
                        tokens_added=tokens_added,
                        tools_added=result.tool_calls,
                    )
                    # Sync current_session object with stored values
                    updated_s = session_manager.get_session(current_session.id)
                    if updated_s:
                        current_session = updated_s
                except Exception:
                    pass  # Don't crash on save failure

        except KeyboardInterrupt:
            print_error("Generation cancelled.")
        except APIError as e:
            msg = getattr(e, "message", str(e))
            prefix = getattr(e, "status", "API ERROR")
            print_error(f"{getattr(e, 'code', '')} {prefix}: {msg}".strip())
        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
