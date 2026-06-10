from __future__ import annotations

import json
from pathlib import Path
import pytest

from google.genai import types
from minimal_cli.session import Session, SessionManager, _session_to_dict, _dict_to_session


def test_session_serialization_helpers():
    session = Session(
        id="a1b2c3d4",
        name="test-session",
        model="gemini-2.5-flash",
        created_at="2026-06-09T12:00:00Z",
        updated_at="2026-06-09T13:00:00Z",
        message_count=10,
        project_dir="/some/path",
    )
    d = _session_to_dict(session)
    assert d["id"] == "a1b2c3d4"
    assert d["message_count"] == 10
    
    session2 = _dict_to_session(d)
    assert session2.id == "a1b2c3d4"
    assert session2.message_count == 10
    assert session2.project_dir == "/some/path"


def test_create_session(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.create_session("gemini-2.5-flash", "/project/dir")
    
    assert len(session.id) == 8
    assert session.model == "gemini-2.5-flash"
    assert session.project_dir == "/project/dir"
    assert session.message_count == 0
    
    # Check index.json
    index_path = tmp_path / "index.json"
    assert index_path.is_file()
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index_data["sessions"]) == 1
    assert index_data["sessions"][0]["id"] == session.id

    # Check empty JSONL file created
    jsonl_path = tmp_path / f"{session.id}.jsonl"
    assert jsonl_path.is_file()
    assert jsonl_path.read_text(encoding="utf-8") == ""


def test_save_and_load_turn(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.create_session("gemini-2.5-flash", "/project/dir")
    
    # Create some mock Content objects
    content1 = types.Content(
        role="user",
        parts=[types.Part.from_text(text="hello")]
    )
    content2 = types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(name="read_file", args={"path": "a.txt"})
        ]
    )
    content3 = types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(name="read_file", response={"result": "file content"})
        ]
    )
    
    # Save first two messages
    manager.save_turn(session.id, [content1, content2], tokens_added=150, tools_added=1)
    
    # Verify index updated
    updated_session = manager.get_session(session.id)
    assert updated_session is not None
    assert updated_session.message_count == 2
    assert updated_session.total_tokens == 150
    assert updated_session.total_tool_calls == 1
    
    # Save turn with the full history list (should only append the third one)
    manager.save_turn(session.id, [content1, content2, content3], tokens_added=50, tools_added=0)
    
    updated_session = manager.get_session(session.id)
    assert updated_session.message_count == 3
    assert updated_session.total_tokens == 200
    assert updated_session.total_tool_calls == 1
    
    # Load and verify
    loaded = manager.load_session(session.id)
    assert len(loaded) == 3
    assert loaded[0].role == "user"
    assert loaded[0].parts[0].text == "hello"
    assert loaded[1].role == "model"
    assert loaded[1].parts[0].function_call.name == "read_file"
    assert loaded[1].parts[0].function_call.args == {"path": "a.txt"}
    assert loaded[2].role == "tool"
    assert loaded[2].parts[0].function_response.name == "read_file"
    assert loaded[2].parts[0].function_response.response == {"result": "file content"}


def test_list_sessions(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    s1 = manager.create_session("gemini-2.5-flash", "/dir1")
    s2 = manager.create_session("gemini-2.5-flash", "/dir2")
    
    sessions = manager.list_sessions()
    assert len(sessions) == 2
    # s2 is newer, so it should be first
    assert sessions[0].id == s2.id
    assert sessions[1].id == s1.id


def test_delete_session(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    s = manager.create_session("gemini-2.5-flash", "/dir")
    
    assert (tmp_path / f"{s.id}.jsonl").is_file()
    
    manager.delete_session(s.id)
    assert not (tmp_path / f"{s.id}.jsonl").is_file()
    assert len(manager.list_sessions()) == 0


def test_rename_session(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    s = manager.create_session("gemini-2.5-flash", "/dir")
    
    manager.rename_session(s.id, "new-name")
    updated = manager.get_session(s.id)
    assert updated.name == "new-name"


def test_prune_sessions(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    ids = []
    # Create 5 sessions
    for i in range(5):
        s = manager.create_session("gemini-2.5-flash", f"/dir{i}")
        ids.append(s.id)
        # Artificially set different updated_at to ensure deterministic ordering
        # index loaded, modified, saved
        index = manager._load_index()
        entry = manager._find_entry(index, s.id)
        entry["updated_at"] = f"2026-06-09T10:0{i}:00Z"
        manager._save_index(index)
        
    # Prune to max 3
    num_deleted = manager.prune_sessions(3)
    assert num_deleted == 2
    
    remaining = manager.list_sessions()
    assert len(remaining) == 3
    # The remaining should be the 3 newest (created last, with higher timestamp indices 4, 3, 2)
    remaining_ids = {r.id for r in remaining}
    assert ids[4] in remaining_ids
    assert ids[3] in remaining_ids
    assert ids[2] in remaining_ids
    assert ids[0] not in remaining_ids
    assert ids[1] not in remaining_ids


def test_get_last_session(tmp_path: Path):
    manager = SessionManager(sessions_dir=tmp_path)
    assert manager.get_last_session() is None
    
    s1 = manager.create_session("gemini-2.5-flash", "/dir1")
    s2 = manager.create_session("gemini-2.5-flash", "/dir2")
    
    assert manager.get_last_session().id == s2.id
