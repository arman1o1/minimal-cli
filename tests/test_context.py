from __future__ import annotations

import pytest
from google.genai import types

from minimal_cli.context import estimate_tokens, should_prune, build_summary_text, prune_history


def test_estimate_tokens():
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="hello world")] # 11 chars -> 2 tokens
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(name="tool_call", args={"arg1": "val1"}), # {"arg1": "val1"} is ~16 chars -> 4 + 10 = 14 tokens
            ]
        ),
        types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(name="tool_call", response={"result": "ok"}), # {"result": "ok"} is ~16 chars -> 4 + 10 = 14 tokens
            ]
        ),
    ]
    tokens = estimate_tokens(contents)
    # 2 + 14 + 14 = 30 tokens approx (estimate_tokens uses len(...) // 4 for text and args)
    # len("hello world") = 11. 11 // 4 = 2.
    # json.dumps({"arg1": "val1"}) -> '{"arg1": "val1"}' is 16 chars. 16 // 4 = 4. 4 + 10 = 14.
    # json.dumps({"result": "ok"}) -> '{"result": "ok"}' is 16 chars. 16 // 4 = 4. 4 + 10 = 14.
    assert tokens == 30


def test_should_prune():
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="a" * 100)] # 100 // 4 = 25 tokens
        )
    ]
    assert should_prune(contents, 20) is True
    assert should_prune(contents, 30) is False


def test_build_summary_text():
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="User message")]
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="Model response"),
                types.Part.from_function_call(name="my_tool", args={"param": 123}),
            ]
        ),
        types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(name="my_tool", response={"output": "success"}),
            ]
        )
    ]
    
    summary_text = build_summary_text(contents)
    assert "User message" in summary_text
    assert "Model response" in summary_text
    assert "[Called: my_tool(param=123)]" in summary_text
    assert "[Tool result for my_tool: {\"output\": \"success\"}]" in summary_text or "[Tool result for my_tool: {\"output\": \"success\"" in summary_text


def test_prune_history_under_budget():
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="hello")]
        )
    ]
    # Pruning limit is very high, so history should be returned exactly as is
    pruned = prune_history(contents, 1000)
    assert pruned == contents


def test_prune_history_over_budget_no_summary():
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="msg1")]),
        types.Content(role="model", parts=[types.Part.from_text(text="msg2")]),
        types.Content(role="user", parts=[types.Part.from_text(text="msg3")]),
        types.Content(role="model", parts=[types.Part.from_text(text="msg4")]),
        types.Content(role="user", parts=[types.Part.from_text(text="msg5")]),
    ]
    
    # Prune history with limit 1 token, meaning it will prune since "msg1" etc is longer
    # Without summarize_fn, it drops old messages keeping the last keep_recent (4)
    pruned = prune_history(contents, 1, summarize_fn=None)
    
    assert len(pruned) == 4
    # The first message (msg1) should have been dropped
    assert pruned[0].parts[0].text == "msg2"
    assert pruned[1].parts[0].text == "msg3"
    assert pruned[2].parts[0].text == "msg4"
    assert pruned[3].parts[0].text == "msg5"


def test_prune_history_over_budget_with_summary():
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="msg1")]),
        types.Content(role="model", parts=[types.Part.from_text(text="msg2")]),
        types.Content(role="user", parts=[types.Part.from_text(text="msg3")]),
        types.Content(role="model", parts=[types.Part.from_text(text="msg4")]),
        types.Content(role="user", parts=[types.Part.from_text(text="msg5")]),
    ]
    
    def mock_summarize(text):
        return f"summary of: {text.replace(chr(10), ' ')}"
        
    pruned = prune_history(contents, 1, summarize_fn=mock_summarize)
    
    # With summarize_fn, it should keep last 4, summarize the rest (msg1), and prepend:
    # 1. Summary user message
    # 2. Ack model message
    # 3. Last 4 messages
    assert len(pruned) == 6
    assert pruned[0].role == "user"
    assert "[Conversation summary]" in pruned[0].parts[0].text
    assert "summary of: msg1" in pruned[0].parts[0].text
    
    assert pruned[1].role == "model"
    assert "Understood" in pruned[1].parts[0].text
    
    # Last 4 should match the original last 4
    assert pruned[2].parts[0].text == "msg2"
    assert pruned[3].parts[0].text == "msg3"
    assert pruned[4].parts[0].text == "msg4"
    assert pruned[5].parts[0].text == "msg5"
