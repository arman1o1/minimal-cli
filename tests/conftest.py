from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:  # noqa: PT004
    # Override pytest's built-in tmp_path to keep temp dirs inside
    # .test-workspaces/ (under CWD) so that tool sandbox tests work
    # correctly with workspace-relative path validation.
    base = Path.cwd() / ".test-workspaces"
    base.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    path = base / f"{safe_name}-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

