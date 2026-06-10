from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Session:
    """Metadata for a single conversation session."""

    id: str               # 8-char hex (uuid4[:8])
    name: str             # user-assigned or auto-generated
    model: str            # model used in session
    created_at: str       # ISO 8601 UTC
    updated_at: str       # ISO 8601 UTC
    message_count: int    # total Content objects persisted
    project_dir: str      # cwd when session was created
    total_tokens: int = 0
    total_tool_calls: int = 0


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------
class SessionManager:
    """Manages session persistence via JSONL files and a shared index."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = sessions_dir or Path.home() / ".minimal-cli" / "sessions"

    # -- public API ---------------------------------------------------------

    def create_session(self, model: str, project_dir: str) -> Session:
        """Create a new session with an empty JSONL file."""
        self._ensure_dir()
        now = datetime.now(timezone.utc).isoformat()
        sid = uuid.uuid4().hex[:8]

        index = self._load_index()
        project_sessions = sum(1 for s in index["sessions"] if s["project_dir"] == project_dir)
        session_num = project_sessions + 1

        session = Session(
            id=sid,
            name=self._auto_name(project_dir, session_num),
            model=model,
            created_at=now,
            updated_at=now,
            message_count=0,
            project_dir=project_dir,
        )
        # Create empty JSONL file with 0o600 permissions
        jsonl_path = self.sessions_dir / f"{sid}.jsonl"
        import sys
        if sys.platform != "win32":
            try:
                import os
                fd = os.open(jsonl_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                os.close(fd)
            except OSError:
                jsonl_path.write_text("", encoding="utf-8")
                try:
                    jsonl_path.chmod(0o600)
                except OSError:
                    pass
        else:
            jsonl_path.write_text("", encoding="utf-8")
            from .config import _secure_path_win32
            _secure_path_win32(jsonl_path)

        # Add to index
        index["sessions"].append(_session_to_dict(session))
        self._save_index(index)
        return session

    def save_turn(self, session_id: str, new_contents: list, tokens_added: int = 0, tools_added: int = 0) -> None:
        """Append only new Content objects to the session JSONL file.

        *new_contents* is the full history list.  Only messages beyond the
        current ``message_count`` recorded in the index are written.
        """
        self._ensure_dir()
        index = self._load_index()
        entry = self._find_entry(index, session_id)
        if entry is None:
            logger.warning("Session %s not found in index — skipping save", session_id)
            return

        existing_count = entry["message_count"]
        to_write = new_contents[existing_count:]
        if not to_write:
            return

        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        try:
            with jsonl_path.open("a", encoding="utf-8") as fh:
                for msg in to_write:
                    fh.write(json.dumps(msg.model_dump(mode="json")) + "\n")
        except Exception:
            logger.exception("Failed to append to %s", jsonl_path)
            return

        entry["message_count"] = existing_count + len(to_write)
        entry["total_tokens"] = entry.get("total_tokens", 0) + tokens_added
        entry["total_tool_calls"] = entry.get("total_tool_calls", 0) + tools_added
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_index(index)

    def load_session(self, session_id: str) -> list:
        """Read a JSONL file and reconstruct Content objects."""
        from google.genai import types

        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        if not jsonl_path.is_file():
            logger.warning("JSONL file not found: %s", jsonl_path)
            return []

        contents: list = []
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            logger.exception("Failed to read %s", jsonl_path)
            return []

        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", lineno, jsonl_path.name)
                continue

            role = item.get("role")
            parts_data = item.get("parts", [])
            parts: list = []
            for p in parts_data:
                if p.get("text") is not None:
                    parts.append(types.Part.from_text(text=p["text"]))
                elif p.get("function_call") is not None:
                    fc = p["function_call"]
                    parts.append(types.Part.from_function_call(
                        name=fc["name"], args=fc.get("args", {}),
                    ))
                elif p.get("function_response") is not None:
                    fr = p["function_response"]
                    parts.append(types.Part.from_function_response(
                        name=fr["name"], response=fr.get("response", {}),
                    ))
            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents

    def list_sessions(self) -> list[Session]:
        """Return all sessions sorted by updated_at descending."""
        index = self._load_index()
        sessions = [_dict_to_session(d) for d in index["sessions"]]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> None:
        """Remove a session's JSONL file and its index entry."""
        index = self._load_index()
        index["sessions"] = [s for s in index["sessions"] if s["id"] != session_id]
        self._save_index(index)

        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to delete %s", jsonl_path)

    def rename_session(self, session_id: str, name: str) -> None:
        """Update a session's name in the index."""
        index = self._load_index()
        entry = self._find_entry(index, session_id)
        if entry is None:
            logger.warning("Session %s not found — cannot rename", session_id)
            return
        entry["name"] = name
        self._save_index(index)

    def prune_sessions(self, max_count: int) -> int:
        """Delete oldest sessions beyond *max_count*. Returns number deleted."""
        index = self._load_index()
        sessions = index["sessions"]
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)

        if len(sessions) <= max_count:
            return 0

        to_remove = sessions[max_count:]
        index["sessions"] = sessions[:max_count]
        self._save_index(index)

        for entry in to_remove:
            jsonl_path = self.sessions_dir / f"{entry['id']}.jsonl"
            try:
                jsonl_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to delete %s during prune", jsonl_path)

        return len(to_remove)

    def get_last_session(self) -> Session | None:
        """Return the most recently updated session, or None."""
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def get_session(self, session_id: str) -> Session | None:
        """Look up a session by ID."""
        index = self._load_index()
        entry = self._find_entry(index, session_id)
        return _dict_to_session(entry) if entry else None

    # -- internal -----------------------------------------------------------

    def _load_index(self) -> dict:
        """Load index.json, returning an empty structure if missing or corrupt."""
        index_path = self.sessions_dir / "index.json"
        if not index_path.is_file():
            return {"sessions": []}
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(data.get("sessions"), list):
                return {"sessions": []}
            return data
        except Exception:
            logger.exception("Corrupt index.json — starting fresh")
            return {"sessions": []}

    def _save_index(self, data: dict) -> None:
        """Write index.json atomically via temp-file + rename."""
        self._ensure_dir()
        index_path = self.sessions_dir / "index.json"
        tmp_path = index_path.with_suffix(".tmp")
        content = json.dumps(data, indent=2) + "\n"
        try:
            import sys
            if sys.platform != "win32":
                try:
                    import os
                    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with open(fd, "w", encoding="utf-8") as f:
                        f.write(content)
                except OSError:
                    tmp_path.write_text(content, encoding="utf-8")
                    try:
                        tmp_path.chmod(0o600)
                    except OSError:
                        pass
            else:
                tmp_path.write_text(content, encoding="utf-8")
                from .config import _secure_path_win32
                _secure_path_win32(tmp_path)
            tmp_path.replace(index_path)
            if sys.platform == "win32":
                from .config import _secure_path_win32
                _secure_path_win32(index_path)
        except Exception:
            logger.exception("Failed to write %s", index_path)
            tmp_path.unlink(missing_ok=True)

    def _ensure_dir(self) -> None:
        """Lazily create the sessions directory on first write."""
        if self.sessions_dir.exists():
            return
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        import sys
        if sys.platform != "win32":
            try:
                self.sessions_dir.chmod(0o700)
            except OSError:
                pass
        else:
            from .config import _secure_path_win32
            _secure_path_win32(self.sessions_dir)

    @staticmethod
    def _auto_name(project_dir: str, number: int) -> str:
        """Generate a name like 'Session 1 (Jun 09, 22:45)'."""
        now_str = datetime.now().strftime("%b %d, %H:%M")
        return f"Session {number} ({now_str})"

    @staticmethod
    def _find_entry(index: dict, session_id: str) -> dict | None:
        """Find a session entry dict by ID, or None."""
        for entry in index["sessions"]:
            if entry["id"] == session_id:
                return entry
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_to_dict(session: Session) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "model": session.model,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": session.message_count,
        "project_dir": session.project_dir,
        "total_tokens": session.total_tokens,
        "total_tool_calls": session.total_tool_calls,
    }


def _dict_to_session(d: dict) -> Session:
    return Session(
        id=d["id"],
        name=d["name"],
        model=d["model"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        message_count=d["message_count"],
        project_dir=d["project_dir"],
        total_tokens=d.get("total_tokens", 0),
        total_tool_calls=d.get("total_tool_calls", 0),
    )
