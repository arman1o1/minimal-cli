from __future__ import annotations

from pathlib import Path

import pytest

from minimal_cli.agent import MinimalAgent
from minimal_cli.config import resolve_config, save_config_file
from minimal_cli.main import _handle_slash_command
from minimal_cli.tool_schema import TOOL_DECLARATIONS
from minimal_cli.tools import tool_registry


class FakePart:
    def __init__(self, text: str = "", function_call=None):
        self.text = text
        self.function_call = function_call


class FakeContent:
    def __init__(self, parts):
        self.parts = parts
        self.role = "model"


class FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeChunk:
    def __init__(self, text: str = "", usage=None, candidates=None):
        self.text = text
        self.usage_metadata = usage
        if candidates is None and text:
            part = FakePart(text=text)
            self.candidates = [FakeCandidate(FakeContent([part]))]
        else:
            self.candidates = candidates


class Usage:
    prompt_token_count = 11
    candidates_token_count = 7


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content_stream(self, **_kwargs):
        self.calls.append({**_kwargs, "contents": list(_kwargs["contents"])})
        yield FakeChunk(text="hello", usage=Usage())


class FakeClient:
    def __init__(self, **_kwargs):
        self.models = FakeModels()


def test_agent_simple_text(monkeypatch):
    monkeypatch.setattr("minimal_cli.agent.genai.Client", lambda **kwargs: FakeClient())
    agent = MinimalAgent(api_key="k")
    result = agent.run_turn("hi")
    assert result.text == "hello"
    assert result.input_tokens == 11
    assert result.output_tokens == 7


def test_agent_history_is_updated(monkeypatch):
    """Verify that run_turn mutates the history list in place."""
    monkeypatch.setattr("minimal_cli.agent.genai.Client", lambda **kwargs: FakeClient())
    agent = MinimalAgent(api_key="k")
    history = []
    agent.run_turn("hi", history=history)
    # history should now contain the user message + model response
    assert len(history) >= 2
    assert history[0].role == "user"


def test_agent_sends_function_response_as_tool_content(monkeypatch):
    class FakeFunctionCall:
        def __init__(self, name: str, args: dict):
            self.name = name
            self.args = args

    class FunctionCallModels(FakeModels):
        def generate_content_stream(self, **kwargs):
            self.calls.append({**kwargs, "contents": list(kwargs["contents"])})
            if len(self.calls) == 1:
                part = FakePart(function_call=FakeFunctionCall("demo_tool", {}))
                yield FakeChunk(candidates=[FakeCandidate(FakeContent([part]))], usage=Usage())
                return
            yield FakeChunk(text="done", usage=Usage())

    fake_client = FakeClient()
    fake_client.models = FunctionCallModels()
    monkeypatch.setattr("minimal_cli.agent.genai.Client", lambda **kwargs: fake_client)

    agent = MinimalAgent(api_key="k")
    agent._executors = {"demo_tool": lambda: "ok"}
    result = agent.run_turn("hi")

    assert result.text == "done"
    assert fake_client.models.calls[1]["contents"][-1].role == "tool"


def test_agent_unknown_tool_returns_error(monkeypatch):
    monkeypatch.setattr("minimal_cli.agent.genai.Client", lambda **kwargs: FakeClient())
    agent = MinimalAgent(api_key="k")
    assert "unknown tool" in agent._execute_tool("missing_tool", {})


def test_tool_declarations_match_registry():
    declared = {
        declaration.name
        for tool in TOOL_DECLARATIONS
        for declaration in tool.function_declarations
    }
    registered = set(tool_registry())
    assert declared <= registered
    assert "list_files_recursive" in declared


def test_add_slash_command_loads_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("minimal_cli.agent.genai.Client", lambda **kwargs: FakeClient())
    file_path = tmp_path / "notes.md"
    file_path.write_text("important context", encoding="utf-8")
    agent = MinimalAgent(api_key="k")
    history = []

    result = _handle_slash_command(f"/add {file_path}", agent, history)

    assert result is True
    assert len(history) == 2
    assert history[0].role == "user"
    assert "important context" in history[0].parts[0].text


def test_config_priority(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    save_config_file({"api_key": "file-key", "model": "x"}, cfg_path)
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    cfg = resolve_config(cli_model=None, cli_api_key=None, prompt_for_key=False, config_path=cfg_path)
    assert cfg.api_key == "env-key"
    assert cfg.model == "env-model"


def test_config_missing_key_non_interactive_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("minimal_cli.config.sys.stdin", None)
    with pytest.raises(ValueError, match="No GEMINI_API_KEY found."):
        resolve_config(prompt_for_key=True, config_path=tmp_path / "config.json")

