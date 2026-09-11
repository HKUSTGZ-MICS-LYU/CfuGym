"""Read-only tools the agent can call during a bounded tool loop.

Every path goes through the stage access policy (``RepoToolbox``), so a tool
call can never read more than the surrounding node is allowed to read. Outputs
are plain text so they can be fed back to the model verbatim, and every call is
returned in the loop transcript for auditability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .policy import PolicyError
from .tools import RepoToolbox


READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a text file this stage is allowed to read. Returns numbered "
            "lines. Use start_line/end_line to read a slice of a large file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path."},
                "start_line": {"type": "integer", "description": "First line (1-based)."},
                "end_line": {"type": "integer", "description": "Last line (inclusive)."},
            },
            "required": ["path"],
        },
    },
}

LIST_DIR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List visible files/directories under an allowed directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative directory (default '.')."},
                "glob": {"type": "string", "description": "Glob pattern (default '*')."},
            },
            "required": [],
        },
    },
}

SEARCH_TEXT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_text",
        "description": (
            "Search visible files for a regular expression; returns path:line matches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Repo-relative root (default '.')."},
                "glob": {"type": "string", "description": "File glob under root (default '**/*')."},
            },
            "required": ["pattern"],
        },
    },
}

READ_ONLY_TOOL_SCHEMAS = (READ_FILE_SCHEMA, LIST_DIR_SCHEMA, SEARCH_TEXT_SCHEMA)


@dataclass
class ReadOnlyToolset:
    """Harness-side executor for the read-only tools."""

    toolbox: RepoToolbox
    max_read_chars: int = 20000
    max_hits: int = 40
    max_files: int = 300
    max_file_bytes: int = 400_000
    calls: list[dict[str, Any]] = field(default_factory=list)

    def schemas(self) -> list[dict[str, Any]]:
        # Deep-ish copy so callers cannot mutate the module-level schemas.
        return [
            {**schema, "function": {**schema["function"], "parameters": {
                **schema["function"]["parameters"],
                "properties": dict(schema["function"]["parameters"].get("properties", {})),
            }}}
            for schema in READ_ONLY_TOOL_SCHEMAS
        ]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "search_text": self._search_text,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name!r}"
        try:
            return handler(arguments or {})
        except PolicyError as exc:
            return f"denied by access policy: {exc}"
        except (OSError, ValueError) as exc:
            return f"error: {type(exc).__name__}: {exc}"

    # --- internals ---------------------------------------------------------
    def _root(self, raw: str) -> Path:
        raw = (raw or ".").strip() or "."
        root = self.toolbox.workdir if raw in {".", "./"} else self.toolbox.resolve(raw)
        if not root.exists() or not root.is_dir():
            raise PolicyError(f"not a visible directory: {raw}")
        return root

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.toolbox.workdir).as_posix()

    def _check_glob(self, glob: str) -> str:
        glob = (glob or "").strip() or "*"
        if glob.startswith("/") or glob.startswith("\\") or ".." in Path(glob).parts:
            raise PolicyError(f"glob must be relative and stay inside the directory: {glob!r}")
        return glob

    def _visible(self, root: Path, glob: str) -> Iterator[Path]:
        seen = 0
        for path in sorted(root.glob(glob)):
            if seen >= self.max_files:
                return
            try:
                rel = self._rel(path)
            except ValueError:
                continue
            if not self.toolbox.policy.can_read(self.toolbox.stage, rel):
                continue
            seen += 1
            yield path

    def _read_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", "") or "").strip()
        if not path:
            return "error: path is required"
        text = self.toolbox.read_text(path, limit=self.max_file_bytes)
        if not text:
            return "(empty file, missing file, or not visible to this stage)"
        lines = text.split("\n")
        offset = 1
        start = args.get("start_line")
        end = args.get("end_line")
        if isinstance(start, int) or isinstance(end, int):
            first = max(1, int(start)) if isinstance(start, int) else 1
            last = int(end) if isinstance(end, int) else len(lines)
            last = max(first, min(last, len(lines)))
            lines = lines[first - 1:last]
            offset = first
        body = "\n".join(f"{offset + index:>6}\t{line}" for index, line in enumerate(lines))
        if len(body) > self.max_read_chars:
            body = body[: self.max_read_chars] + f"\n...[truncated at {self.max_read_chars} chars]"
        return body or "(empty file range)"

    def _list_dir(self, args: dict[str, Any]) -> str:
        root = self._root(str(args.get("path", ".") or "."))
        glob = self._check_glob(str(args.get("glob", "*") or "*"))
        entries: list[str] = []
        for path in self._visible(root, glob):
            entries.append(f"{self._rel(path)}{'/' if path.is_dir() else ''}")
            if len(entries) >= self.max_files:
                entries.append("...[truncated]")
                break
        return "\n".join(entries) or "(no visible entries)"

    def _search_text(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "") or "")
        if not pattern:
            return "error: pattern is required"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"error: invalid regex: {exc}"
        root = self._root(str(args.get("path", ".") or "."))
        glob = self._check_glob(str(args.get("glob", "**/*") or "**/*"))
        hits: list[str] = []
        for path in self._visible(root, glob):
            if not path.is_file():
                continue
            rel = self._rel(path)
            try:
                text = self.toolbox.read_text(rel, limit=self.max_file_bytes)
            except PolicyError:
                continue
            if not text:
                continue
            for lineno, line in enumerate(text.split("\n"), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= self.max_hits:
                        hits.append("...[truncated]")
                        return "\n".join(hits)
        return "\n".join(hits) or "(no matches)"
