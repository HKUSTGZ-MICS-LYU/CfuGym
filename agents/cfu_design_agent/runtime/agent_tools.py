"""Read-only tools the agent can call during a bounded tool loop.

Every path goes through the stage access policy (``RepoToolbox``), so a tool
call can never read more than the surrounding node is allowed to read. Outputs
are plain text so they can be fed back to the model verbatim, and every call is
returned in the loop transcript for auditability.
"""

from __future__ import annotations

import re
import os
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .policy import PolicyError
from .tools import RepoToolbox, redirect_target


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

def _validate_search_pattern(pattern: str) -> None:
    """Conservative linear-search subset: no repeated groups or backreferences."""
    if len(pattern) > 512:
        raise ValueError("regex pattern budget exceeded")
    # Parse rather than scan textual parentheses (escaped literals are harmless).
    try:
        from re import _parser, _constants
    except ImportError:  # Python 3.10
        import sre_parse as _parser
        import sre_constants as _constants
    repeats = {_constants.MAX_REPEAT, _constants.MIN_REPEAT}
    if hasattr(_constants, "POSSESSIVE_REPEAT"):
        repeats.add(_constants.POSSESSIVE_REPEAT)
    count = 0
    def walk(nodes):
        nonlocal count
        for op, arg in nodes:
            if op in repeats:
                count += 1
                if count > 1 or any(child not in {_constants.LITERAL, _constants.NOT_LITERAL,
                                                   _constants.IN, _constants.ANY, _constants.CATEGORY}
                                     for child, _ in arg[2]):
                    raise ValueError("unsafe regex repetition; simplify the expression")
            elif op == _constants.SUBPATTERN:
                walk(arg[-1])
            elif op == _constants.BRANCH:
                for branch in arg[1]:
                    walk(branch)
            elif op in {_constants.GROUPREF, _constants.GROUPREF_EXISTS,
                        _constants.ASSERT, _constants.ASSERT_NOT}:
                raise ValueError("regex backreferences/lookarounds are unsupported")
    walk(_parser.parse(pattern, 0))


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
        if not self.toolbox.is_current_run_path(self._rel(root)):
            raise PolicyError(f"directory belongs to another run: {raw}")
        if not root.exists() or not root.is_dir():
            raise PolicyError(f"not a visible directory: {raw}")
        return root

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.toolbox.workdir).as_posix()

    def _live_path(self, rel_path: str) -> str:
        """Route a repo path to its live run-workspace copy when one exists.

        The hardware prompt shows, and an edit targets, the run-workspace
        overlay under the tracked repo path. The tool must read the same bytes:
        otherwise a repair reads the pristine scaffold instead of the model's
        previous change and "corrects" a file it never sees.
        """
        try:
            normalized = self.toolbox.normalize(rel_path)
            if not self.toolbox.is_current_run_path(normalized):
                return rel_path
        except PolicyError:
            return rel_path
        if not self.toolbox.run_root:
            return rel_path
        redirected = redirect_target(
            normalized,
            self.toolbox.run_root,
            self.toolbox.embench_benchmark,
        )
        # Only follow a stage that may read the overlay: some stages (for
        # example analysis) are allowed the tracked scaffold but not the design
        # workspace, and must keep reading the tracked path there.
        if (
            redirected
            and (self.toolbox.workdir / redirected).is_file()
            and self.toolbox.is_read_allowed(redirected)
        ):
            return redirected
        return rel_path

    def _check_glob(self, glob: str) -> str:
        glob = (glob or "").strip() or "*"
        if glob.startswith("/") or glob.startswith("\\") or ".." in Path(glob).parts:
            raise PolicyError(f"glob must be relative and stay inside the directory: {glob!r}")
        return glob

    def _visible(self, root: Path, glob: str) -> Iterator[Path]:
        seen = visited = 0
        stack = [root]
        recursive = "/" in glob
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > self.max_files * 20:
                        raise ValueError("directory enumeration budget exceeded; narrow path/glob")
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    rel = self._rel(path)
                    if not self.toolbox.is_current_run_path(rel):
                        continue
                    if self.toolbox.policy.is_denied(rel) or self.toolbox.policy.is_denied(rel + "/"):
                        continue
                    if recursive and entry.is_dir(follow_symlinks=False):
                        stack.append(path)
                    local = path.relative_to(root).as_posix()
                    matches = fnmatch.fnmatchcase(local, glob) or (
                        glob.startswith("**/") and fnmatch.fnmatchcase(local, glob[3:]))
                    if not matches or not self.toolbox.is_read_allowed(rel):
                        continue
                    if seen >= self.max_files:
                        raise ValueError("visible file budget exceeded; narrow path/glob")
                    seen += 1
                    yield path

    def _read_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", "") or "").strip()
        if not path:
            return "error: path is required"
        text = self.toolbox.read_text(self._live_path(path), limit=self.max_file_bytes)
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
            _validate_search_pattern(pattern)
            regex = re.compile(pattern)
        except (re.error, ValueError) as exc:
            return f"error: invalid regex: {exc}"
        root = self._root(str(args.get("path", ".") or "."))
        glob = self._check_glob(str(args.get("glob", "**/*") or "**/*"))
        hits: list[str] = []
        for path in self._visible(root, glob):
            if not path.is_file():
                continue
            rel = self._rel(path)
            try:
                text = self.toolbox.read_text(self._live_path(rel), limit=self.max_file_bytes)
            except PolicyError:
                continue
            if not text:
                continue
            for lineno, line in enumerate(text.split("\n"), start=1):
                if len(line) > 20000:
                    raise ValueError("search line budget exceeded; use a narrower file/range")
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= self.max_hits:
                        hits.append("...[truncated]")
                        return "\n".join(hits)
        return "\n".join(hits) or "(no matches)"
