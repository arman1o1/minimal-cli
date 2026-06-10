from __future__ import annotations

import json
from pathlib import Path
import pytest

from minimal_cli.project import detect_project, build_project_context, ProjectInfo


def test_detect_no_project(tmp_path: Path):
    assert detect_project(tmp_path) is None


def test_detect_python_project_pyproject_toml(tmp_path: Path):
    # Pyproject.toml with metadata
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "my-awesome-python-app"
description = "A description of the app"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "fastapi",
]
""",
        encoding="utf-8",
    )
    
    info = detect_project(tmp_path)
    assert info is not None
    assert info.type == "python"
    assert info.name == "my-awesome-python-app"
    assert "Python >=3.11" in info.language or "Python" in info.language
    assert info.framework in ("click", "fastapi") # depending on parser order/framework dict
    assert info.summary == "A description of the app"
    assert "pyproject.toml" in info.key_files


def test_detect_node_project(tmp_path: Path):
    package_json = tmp_path / "package.json"
    package_json.write_text(
        json.dumps({
            "name": "my-node-app",
            "description": "Node JS app description",
            "dependencies": {
                "next": "^14.0.0",
                "react": "^18.2.0"
            }
        }),
        encoding="utf-8"
    )
    
    # Let's create a tsconfig.json to check TypeScript language detection
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    
    info = detect_project(tmp_path)
    assert info is not None
    assert info.type == "node"
    assert info.name == "my-node-app"
    assert info.language == "TypeScript"
    assert info.framework == "next.js"
    assert info.summary == "Node JS app description"


def test_detect_rust_project(tmp_path: Path):
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text(
        """
[package]
name = "my-rust-crate"
description = "Rust crate description"
edition = "2021"
""",
        encoding="utf-8"
    )
    
    info = detect_project(tmp_path)
    assert info is not None
    assert info.type == "rust"
    assert info.name == "my-rust-crate"
    assert info.language == "Rust 2021"
    assert info.summary == "Rust crate description"


def test_detect_go_project(tmp_path: Path):
    go_mod = tmp_path / "go.mod"
    go_mod.write_text(
        """
module github.com/user/my-go-module

go 1.21.0
""",
        encoding="utf-8"
    )
    
    info = detect_project(tmp_path)
    assert info is not None
    assert info.type == "go"
    assert info.name == "github.com/user/my-go-module"
    assert info.language == "Go 1.21.0"


def test_multiple_markers_priority(tmp_path: Path):
    # Pyproject.toml and Makefile. pyproject.toml is checked first.
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = \"primary-python\"\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all: build\n", encoding="utf-8")
    
    info = detect_project(tmp_path)
    assert info is not None
    assert info.type == "python"
    assert info.name == "primary-python"


def test_build_project_context(tmp_path: Path):
    info = ProjectInfo(
        type="python",
        name="test-project",
        language="Python 3.11",
        framework="fastapi",
        key_files=["pyproject.toml", "src/"],
        summary="A web app."
    )
    
    # Create some file structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    
    context = build_project_context(info, tmp_path)
    
    assert "Type: Python 3.11 (fastapi)" in context
    assert "Name: test-project" in context
    assert "Description: A web app." in context
    assert "Key files: pyproject.toml, src/" in context
    assert "Structure:" in context
    assert "src/" in context
    assert "tests/" in context
    assert "pyproject.toml" in context
    # Should skip hidden dirs
    assert ".git" not in context
    assert "node_modules" not in context
