"""Context window management — token estimation and history pruning."""
from __future__ import annotations

import json
from collections.abc import Callable


def estimate_tokens(contents: list) -> int:
    """Estimate token count for a list of Content objects.

    Uses a rough 4-chars-per-token heuristic for text, plus overhead
    for function calls and responses.
    """
    total = 0
    for content in contents:
        for part in content.parts:
            text = getattr(part, "text", None)
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            if text is not None:
                total += len(text) // 4
            elif fc is not None:
                args_str = json.dumps(dict(fc.args))
                total += len(args_str) // 4 + 10
            elif fr is not None:
                resp_str = json.dumps(dict(fr.response))
                total += len(resp_str) // 4 + 10
    return total


def should_prune(contents: list, max_tokens: int) -> bool:
    """Return True if estimated token usage exceeds max_tokens."""
    return estimate_tokens(contents) > max_tokens


def build_summary_text(contents: list) -> str:
    """Build a plain-text representation of Content objects for summarization.

    User messages include raw text. Model messages include text and annotate
    function calls. Tool responses are truncated to 200 chars.
    """
    lines: list[str] = []
    for content in contents:
        for part in content.parts:
            text = getattr(part, "text", None)
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            if text is not None:
                lines.append(text)
            elif fc is not None:
                args_repr = ", ".join(
                    f"{k}={v!r}" for k, v in dict(fc.args).items()
                )
                lines.append(f"[Called: {fc.name}({args_repr})]")
            elif fr is not None:
                result_str = json.dumps(dict(fr.response))
                lines.append(
                    f"[Tool result for {fr.name}: {result_str[:200]}]"
                )
    return "\n".join(lines)


def prune_history(
    contents: list,
    max_tokens: int,
    summarize_fn: Callable[[str], str] | None = None,
    keep_recent: int = 4,
) -> list:
    """Prune conversation history to fit within max_tokens.

    Keeps the last *keep_recent* messages. If *summarize_fn* is provided,
    older messages are summarized into a single user/model pair prepended
    to the recent window. Otherwise older messages are simply dropped.
    """
    if estimate_tokens(contents) <= max_tokens:
        return contents

    from google.genai.types import Content, Part

    old_messages = contents[:-keep_recent]
    recent = contents[-keep_recent:]

    if summarize_fn is not None:
        text_repr = build_summary_text(old_messages)
        summary = summarize_fn(text_repr)
        summary_content = Content(
            role="user",
            parts=[Part.from_text(text=f"[Conversation summary]\n{summary}")],
        )
        ack_content = Content(
            role="model",
            parts=[
                Part.from_text(
                    text="Understood. I'll keep this context in mind."
                )
            ],
        )
        return [summary_content, ack_content] + recent

    # Fallback: drop old messages (lossy but functional)
    return recent
