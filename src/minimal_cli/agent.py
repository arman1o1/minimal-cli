from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .display import console, print_tool_call
from rich.live import Live
from rich.markdown import Markdown
from .project import detect_project, build_project_context
from .tool_schema import TOOL_DECLARATIONS
from .tools import tool_registry


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AgentResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thoughts_tokens: int | None = None
    elapsed_seconds: float | None = None
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------
class MinimalAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_tool_calls: int = 25,
        command_timeout: int = 60,
        verbose: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

        self.model = model
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose
        self._custom_system_prompt = system_prompt
        self._executors = tool_registry(command_timeout=command_timeout)

        # Cache project context (detected once, used every turn)
        cwd = Path.cwd()
        project_info = detect_project(cwd)
        self._project_context = build_project_context(project_info, cwd) if project_info else None

    def _system_prompt(self) -> str:
        cwd = Path.cwd()
        os_info = platform.platform()
        base = (
            "You are a concise coding assistant. Write working code and explain only when asked.\n"
            "CRITICAL: If the user says 'hi', 'hello', or other greetings/casual chat, do NOT call any tools. Just respond with a friendly greeting and ask how you can help.\n"
            "Only call tools or run commands to fulfill user tasks or answer user questions; do not call tools for simple greetings or casual conversation.\n"
            f"Current working directory: {cwd}\n"
            f"OS: {os_info}\n"
        )
        if self._project_context:
            base += f"\n{self._project_context}\n"
        if self._custom_system_prompt:
            base += f"\nCustom Instructions:\n{self._custom_system_prompt}\n"
        return base

    def _execute_tool(self, name: str, args: dict[str, Any], silent: bool = False) -> str:
        executor = self._executors.get(name)
        if not executor:
            return f"Error: unknown tool '{name}'"
        summary = json.dumps(args) if self.verbose else _summarize_args(name, args)
        if not silent:
            print_tool_call(name, summary)
        try:
            return str(executor(**args))
        except Exception as exc:  # noqa: BLE001
            return f"Error running {name}: {exc}"

    def run_turn(
        self,
        user_prompt: str,
        history: list | None = None,
        silent: bool = False,
    ) -> AgentResult:
        """Run one user turn through the agentic loop.

        If *history* is provided, it is mutated in-place so the caller
        keeps multi-turn context across calls.
        """
        conversation: list = list(history) if history is not None else []
        conversation.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )

        if not self.client:
            return AgentResult("Error: Gemini model selected but no GEMINI_API_KEY found.")
        return self._run_gemini_turn(conversation, history, silent=silent)

    def _run_gemini_turn(self, conversation: list, history: list | None, silent: bool = False) -> AgentResult:
        started = time.perf_counter()
        tool_calls_count = 0
        thoughts_tokens = 0
        accumulated_input_tokens = 0
        usage_out = 0

        final_text: list[str] = []
        system_prompt = self._system_prompt()

        for _ in range(self.max_tool_calls):
            model_parts: list = []
            function_calls: list = []
            spinner_running = True
            live_renderer = None
            stream_out = 0  # Initialize to avoid UnboundLocalError
            current_in = 0

            status = None
            if not silent:
                status = console.status("Thinking...")
                status.start()

            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction=system_prompt,
                        tools=TOOL_DECLARATIONS,
                    ),
                )

                for chunk in stream:
                    # Accumulate model parts + detect function calls
                    chunk_text = ""
                    if getattr(chunk, "candidates", None):
                        for cand in chunk.candidates or []:
                            if not getattr(cand, "content", None):
                                continue
                            for part in cand.content.parts or []:
                                model_parts.append(part)
                                if getattr(part, "text", None):
                                    chunk_text += part.text
                                elif getattr(part, "function_call", None):
                                    function_calls.append(part.function_call)

                    # Stop spinner on first text chunk, then stream
                    if chunk_text:
                        if not silent:
                            if spinner_running:
                                if status:
                                    status.stop()
                                spinner_running = False
                                live_renderer = Live(console=console, auto_refresh=False)
                                live_renderer.start()
                            if live_renderer:
                                combined_so_far = "".join(final_text)
                                live_renderer.update(Markdown(combined_so_far), refresh=True)
                        final_text.append(chunk_text)

                    # Track token usage
                    if getattr(chunk, "usage_metadata", None):
                        current_in = getattr(chunk.usage_metadata, "prompt_token_count", 0)
                        stream_out = getattr(chunk.usage_metadata, "candidates_token_count", 0)
                        thoughts_tokens = (
                            getattr(chunk.usage_metadata, "thoughts_token_count", None)
                            or getattr(chunk.usage_metadata, "thinking_token_count", None)
                            or thoughts_tokens
                        )
                
                if current_in:
                    accumulated_input_tokens += current_in
            finally:
                if not silent:
                    if spinner_running and status:
                        status.stop()
                    if live_renderer:
                        live_renderer.stop()

            if stream_out:
                usage_out += stream_out

            if final_text and not "".join(final_text).endswith(("\n", " ")):
                final_text.append("\n\n")

            # Append full model response to conversation (preserves thought signatures)
            if model_parts:
                conversation.append(types.Content(role="model", parts=model_parts))

            # No function calls → done
            if not function_calls:
                combined = "".join(final_text).strip()
                _sync_history(history, conversation)
                return AgentResult(
                    text=combined,
                    input_tokens=accumulated_input_tokens or None,
                    output_tokens=usage_out or None,
                    total_tokens=(accumulated_input_tokens + usage_out) or None,
                    thoughts_tokens=thoughts_tokens or None,
                    elapsed_seconds=time.perf_counter() - started,
                    tool_calls=tool_calls_count,
                )

            # Execute each tool and send results back
            # Removed final_text.clear() to preserve model reasoning/preface between tool calls
            response_parts: list = []
            for fc in function_calls:
                args = dict(fc.args or {})
                output = self._execute_tool(fc.name, args, silent=silent)
                tool_calls_count += 1
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": output}
                    )
                )
            conversation.append(types.Content(role="tool", parts=response_parts))

        _sync_history(history, conversation)
        return AgentResult(
            text="Stopped: reached max tool call iterations.",
            input_tokens=accumulated_input_tokens or None,
            output_tokens=usage_out or None,
            total_tokens=(accumulated_input_tokens + usage_out) or None,
            thoughts_tokens=thoughts_tokens or None,
            elapsed_seconds=time.perf_counter() - started,
            tool_calls=tool_calls_count,
        )




def _sync_history(history: list | None, conversation: list) -> None:
    """Mutate *history* in place so the caller keeps context."""
    if history is not None:
        history.clear()
        history.extend(conversation)


def _summarize_args(name: str, args: dict) -> str:
    """Compact display for tool calls (non-verbose mode)."""
    if name in ("read_file", "write_file", "replace_in_file"):
        return args.get("path", "")
    if name == "list_directory":
        return args.get("path", ".")
    if name == "run_command":
        return args.get("command", "")
    if name == "grep_search":
        return f"{args.get('pattern', '')} in {args.get('path', '.')}"
    if name == "fetch_url":
        return args.get("url", "")
    return json.dumps(args)
