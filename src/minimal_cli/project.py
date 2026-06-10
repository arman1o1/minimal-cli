from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectInfo:
    """Detected project metadata."""

    type: str  # "python", "node", "rust", "go", "java", "ruby", "php", "c_cpp", "generic"
    name: str  # project name extracted from config file
    language: str  # "Python 3.10+", "TypeScript", "Rust", etc.
    framework: str  # "click", "fastapi", "react", etc. or ""
    key_files: list[str] = field(default_factory=list)  # ["pyproject.toml", "src/", "tests/"]
    summary: str = ""  # one-line description from config or README


MARKER_FILES: dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "requirements.txt": "python",
    "package.json": "node",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "Gemfile": "ruby",
    "composer.json": "php",
    "CMakeLists.txt": "c_cpp",
    "Makefile": "generic",
}

_HIDDEN_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", ".env", "env",
    "__pycache__", "node_modules", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache", ".tox", ".nox",
    "dist", "build", ".eggs",
})

_KEY_DIRS = ("src/", "tests/", "test/", "lib/", "docs/", ".github/")
_KEY_FILES = ("README.md", "README.rst", "README.txt")

_PYTHON_FRAMEWORKS: dict[str, str] = {
    "click": "click",
    "typer": "typer",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "pytest": "pytest",
    "starlette": "starlette",
    "httpx": "httpx",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
}


def _read_text(path: Path) -> str | None:
    """Read file text, returning None on any error."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _scan_key_files(cwd: Path) -> list[str]:
    """Find notable files and directories at the project root."""
    found: list[str] = []
    for d in _KEY_DIRS:
        if (cwd / d.rstrip("/")).is_dir():
            found.append(d)
    for f in _KEY_FILES:
        if (cwd / f).is_file():
            found.append(f)
    # Also include marker files that exist
    for marker in MARKER_FILES:
        if (cwd / marker).is_file() and marker not in found:
            found.append(marker)
    return found


def _detect_python_framework(deps: list[str]) -> str:
    """Find the first known framework in a dependency list."""
    dep_names = set()
    for dep in deps:
        # Strip version specifiers: "click>=8.0" -> "click"
        name = re.split(r"[><=!~\[;]", dep)[0].strip().lower().replace("-", "_")
        dep_names.add(name)
    for pkg, label in _PYTHON_FRAMEWORKS.items():
        if pkg in dep_names:
            return label
    return ""


def _parse_toml_regex(text: str, key: str) -> str:
    """Extract a simple string value from TOML using regex fallback."""
    pattern = rf'^\s*{re.escape(key)}\s*=\s*"([^"]*)"'
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else ""


def _parse_toml_array_regex(text: str, key: str) -> list[str]:
    """Extract a simple string array from TOML using regex fallback."""
    pattern = rf"^\s*{re.escape(key)}\s*=\s*\[([^\]]*)\]"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]*)"', m.group(1))


def _parse_python(cwd: Path, marker_path: Path) -> ProjectInfo:
    """Parse a Python project from pyproject.toml or fallback markers."""
    name = cwd.name
    summary = ""
    requires_python = ""
    framework = ""
    deps: list[str] = []

    if marker_path.name == "pyproject.toml":
        text = _read_text(marker_path)
        if text:
            # Try tomllib first (Python 3.11+)
            parsed = False
            try:
                import tomllib
                data = tomllib.loads(text)
                project = data.get("project", {})
                name = project.get("name", name)
                summary = project.get("description", "")
                requires_python = project.get("requires-python", "")
                deps = project.get("dependencies", [])
                parsed = True
            except ImportError:
                pass
            except Exception:
                pass

            if not parsed:
                # Regex fallback
                name = _parse_toml_regex(text, "name") or name
                summary = _parse_toml_regex(text, "description") or summary
                requires_python = _parse_toml_regex(text, "requires-python") or requires_python
                deps = _parse_toml_array_regex(text, "dependencies")

            framework = _detect_python_framework(deps)

    language = "Python"
    if requires_python:
        language = f"Python {requires_python}"

    return ProjectInfo(
        type="python",
        name=name,
        language=language,
        framework=framework,
        key_files=_scan_key_files(cwd),
        summary=summary,
    )


def _parse_node(cwd: Path, marker_path: Path) -> ProjectInfo:
    """Parse a Node.js/TypeScript project from package.json."""
    name = cwd.name
    summary = ""
    framework = ""
    language = "JavaScript"

    text = _read_text(marker_path)
    if text:
        try:
            data = json.loads(text)
            name = data.get("name", name)
            summary = data.get("description", "")

            # Check for TypeScript
            if (cwd / "tsconfig.json").is_file():
                language = "TypeScript"

            # Detect framework from dependencies + devDependencies
            all_deps: dict[str, str] = {}
            all_deps.update(data.get("dependencies", {}))
            all_deps.update(data.get("devDependencies", {}))

            node_frameworks = {
                "next": "next.js",
                "nuxt": "nuxt",
                "nest": "nestjs",
                "@nestjs/core": "nestjs",
                "react": "react",
                "vue": "vue",
                "svelte": "svelte",
                "@angular/core": "angular",
                "express": "express",
                "fastify": "fastify",
                "electron": "electron",
            }
            for pkg, label in node_frameworks.items():
                if pkg in all_deps:
                    framework = label
                    break
        except (json.JSONDecodeError, AttributeError):
            pass

    return ProjectInfo(
        type="node",
        name=name,
        language=language,
        framework=framework,
        key_files=_scan_key_files(cwd),
        summary=summary,
    )


def _parse_rust(cwd: Path, marker_path: Path) -> ProjectInfo:
    """Parse a Rust project from Cargo.toml."""
    name = cwd.name
    summary = ""
    edition = ""

    text = _read_text(marker_path)
    if text:
        try:
            import tomllib
            data = tomllib.loads(text)
            package = data.get("package", {})
            name = package.get("name", name)
            summary = package.get("description", "")
            edition = package.get("edition", "")
        except ImportError:
            name = _parse_toml_regex(text, "name") or name
            summary = _parse_toml_regex(text, "description") or summary
            edition = _parse_toml_regex(text, "edition") or edition
        except Exception:
            pass

    language = f"Rust {edition}" if edition else "Rust"

    return ProjectInfo(
        type="rust",
        name=name,
        language=language,
        framework="",
        key_files=_scan_key_files(cwd),
        summary=summary,
    )


def _parse_go(cwd: Path, marker_path: Path) -> ProjectInfo:
    """Parse a Go project from go.mod."""
    name = cwd.name
    go_version = ""

    text = _read_text(marker_path)
    if text:
        # Extract module name: "module github.com/user/repo"
        m = re.search(r"^\s*module\s+(\S+)", text, re.MULTILINE)
        if m:
            name = m.group(1)
        # Extract go version: "go 1.21"
        m = re.search(r"^\s*go\s+(\S+)", text, re.MULTILINE)
        if m:
            go_version = m.group(1)

    language = f"Go {go_version}" if go_version else "Go"

    return ProjectInfo(
        type="go",
        name=name,
        language=language,
        framework="",
        key_files=_scan_key_files(cwd),
    )


_GENERIC_LANGUAGES: dict[str, str] = {
    "java": "Java",
    "ruby": "Ruby",
    "php": "PHP",
    "c_cpp": "C/C++",
    "generic": "Unknown",
}


def _parse_generic(cwd: Path, project_type: str) -> ProjectInfo:
    """Fallback parser for project types without specialized parsing."""
    return ProjectInfo(
        type=project_type,
        name=cwd.name,
        language=_GENERIC_LANGUAGES.get(project_type, "Unknown"),
        framework="",
        key_files=_scan_key_files(cwd),
    )


_PARSERS: dict[str, object] = {
    "python": _parse_python,
    "node": _parse_node,
    "rust": _parse_rust,
    "go": _parse_go,
}


def detect_project(cwd: Path) -> ProjectInfo | None:
    """Detect project type by scanning for marker files.

    Returns ProjectInfo if a known project type is detected, None otherwise.
    """
    try:
        for marker, project_type in MARKER_FILES.items():
            marker_path = cwd / marker
            if marker_path.exists():
                parser = _PARSERS.get(project_type)
                if parser:
                    return parser(cwd, marker_path)  # type: ignore[operator]
                return _parse_generic(cwd, project_type)
    except Exception:
        return None
    return None


def build_project_context(info: ProjectInfo, cwd: Path) -> str:
    """Generate a compact context string for the system prompt.

    Includes project metadata and a depth-1 directory listing, skipping
    hidden/generated directories. Output is kept under ~500 tokens.
    """
    # Build type line: "Python (click CLI)" or just "Python"
    type_label = info.language
    if info.framework:
        type_label = f"{info.language} ({info.framework})"

    lines = ["Project Context:"]
    lines.append(f"  Type: {type_label}")
    lines.append(f"  Name: {info.name}")
    if info.summary:
        lines.append(f"  Description: {info.summary}")
    if info.key_files:
        lines.append(f"  Key files: {', '.join(info.key_files)}")

    # Depth-1 directory listing, skip hidden/generated dirs
    try:
        entries: list[str] = []
        for child in sorted(cwd.iterdir()):
            child_name = child.name
            # Skip hidden dirs and known noise
            if child_name.startswith(".") or child_name in _HIDDEN_DIRS:
                continue
            if child_name.endswith(".egg-info"):
                continue
            if child.is_dir():
                entries.append(f"    {child_name}/")
            else:
                entries.append(f"    {child_name}")
        if entries:
            lines.append("  Structure:")
            lines.extend(entries)
    except Exception:
        pass

    return "\n".join(lines)
