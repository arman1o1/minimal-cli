from __future__ import annotations

from google.genai import types


TOOL_DECLARATIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="read_file",
                description="Read file contents with optional line range. Returns line-numbered content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "start_line": {"type": "integer", "description": "Start line (1-indexed, inclusive)"},
                        "end_line": {"type": "integer", "description": "End line (1-indexed, inclusive)"},
                    },
                    "required": ["path"],
                },
            ),
            types.FunctionDeclaration(
                name="write_file",
                description="Create or overwrite a file with the given content. Creates parent directories if needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write to"},
                        "content": {"type": "string", "description": "Full file content to write"},
                    },
                    "required": ["path", "content"],
                },
            ),
            types.FunctionDeclaration(
                name="list_directory",
                description="List directory contents with file sizes. Respects .gitignore patterns.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: current directory)"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="list_files_recursive",
                description="List files and directories recursively up to a maximum depth.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: current directory)"},
                        "max_depth": {"type": "integer", "description": "Maximum directory depth to list (default: 3)"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="run_command",
                description="Execute a shell command and return stdout+stderr. Requires user confirmation before running.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            ),
            types.FunctionDeclaration(
                name="grep_search",
                description="Search for a text pattern in files. Returns matching lines with file:line:content format.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Text pattern to search for"},
                        "path": {"type": "string", "description": "Directory or file to search (default: '.')"},
                        "include": {"type": "string", "description": "Glob to filter files, e.g. '*.py'"},
                    },
                    "required": ["pattern"],
                },
            ),
            types.FunctionDeclaration(
                name="replace_in_file",
                description=(
                    "Find and replace exact text in a file. "
                    "Fails if text not found or found multiple times unless allow_multiple is true."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to modify"},
                        "old_text": {"type": "string", "description": "Exact text to find"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                        "allow_multiple": {"type": "boolean", "description": "Replace all occurrences if true"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            ),
            types.FunctionDeclaration(
                name="fetch_url",
                description="Fetch a URL and return content as clean markdown. Strips HTML nav/script/style tags.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "max_length": {"type": "integer", "description": "Max chars to return (default: 20000)"},
                    },
                    "required": ["url"],
                },
            ),
        ]
    )
]

