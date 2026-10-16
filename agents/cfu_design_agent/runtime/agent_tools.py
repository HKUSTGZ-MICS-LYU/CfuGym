"""Read-only tools the agent can call during a bounded tool loop.

Every path goes through the flow's shared read tree (``RepoToolbox``), so a
tool call can never read a path the flow does not expose; only writes are
stage-scoped. Outputs are plain text so they can be fed back to the model
verbatim, and every call is returned in the loop transcript for auditability.
"""

from __future__ import annotations

import re
import os
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import flow_memory, source_visibility
from . import tools as tools_mod
from .policy import PolicyError
from .tools import RepoToolbox, redirect_target
from ..ir.observation import inspect_observation


READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a text file the flow can see. Returns numbered "
            "lines. Use start_line/end_line to read a slice of a large file. "
            "Missing, empty and not-visible files produce distinct messages."
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
        "description": (
            "List the visible entries of a directory, including sub-directories. "
            "Use it to find where a path lives before reading it."
        ),
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
            "Search visible files for a regular expression; returns path:line "
            "matches. 'path' may be a directory (walked per glob) or a single "
            "file (searched alone). Invisible subtrees are skipped and lines "
            "over 20000 chars are reported as skipped, so a search over a whole "
            "tree still returns its matches. If hits are truncated, page with "
            "'skip'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Repo-relative root directory or a single file (default '.')."},
                "glob": {"type": "string", "description": "File glob under root (default '**/*')."},
                "skip": {"type": "integer", "description": "Skip the first N hits before applying the hit limit (paging, default 0)."},
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
                    raise ValueError(
                        "unsafe regex repetition; use a simpler pattern or several "
                        "separate search_text calls (repetition over groups/alternation "
                        "is rejected)"
                    )
            elif op == _constants.SUBPATTERN:
                walk(arg[-1])
            elif op == _constants.BRANCH:
                for branch in arg[1]:
                    walk(branch)
            elif op in {_constants.GROUPREF, _constants.GROUPREF_EXISTS,
                        _constants.ASSERT, _constants.ASSERT_NOT}:
                raise ValueError("regex backreferences/lookarounds are unsupported")
    walk(_parser.parse(pattern, 0))


FAILURE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_failure",
        "description": (
            "Inspect one deterministic compiler/elaboration/conformance/simulation/"
            "synthesis failure by its fail_... ID. Returns the typed summary and, "
            "when requested, a bounded excerpt of its persisted full command log."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Exact fail_... ID from repair evidence."},
                "view": {"type": "string", "enum": ["summary", "log"], "description": "Evidence to inspect."},
                "max_lines": {"type": "integer", "description": "Maximum log excerpt lines, 8–200."},
            },
            "required": ["id"],
        },
    },
}


OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_observation",
        "description": (
            "Inspect a measured Embench observation ID: bounded source-level "
            "LLVM IR candidates, reference-build function-level ASM candidates, "
            "or the static feature XML (data movement, typed op counts, dependencies). "
            "Returns confidence/ambiguity and never estimates cycles from static code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Exact obs_loop_... or obs_fn_... ID from the measured profile."},
                "view": {"type": "string", "enum": ["summary", "asm", "ir", "features"], "description": "Evidence to inspect."},
                "max_lines": {"type": "integer", "description": "Maximum excerpt lines, 8–100."},
            },
            "required": ["id"],
        },
    },
}

LIST_COMMAND_LOGS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_command_logs",
        "description": (
            "List this run's persisted command logs (builds, simulations, synthesis, "
            "isolated elaboration/compile checks, conformance runs), newest first, with "
            "rung, return code, size and command line. Read one with read_file to get the "
            "complete stdout/stderr of a command the harness already ran."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rung": {
                    "type": "string",
                    "description": "Optional exact rung filter, for example soc_sim, hw_build, "
                                   "soc_generate_agent, hw_check, sw_build, cost, embench_build.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows, newest first (default 20, maximum 50).",
                },
            },
            "required": [],
        },
    },
}

READ_ONLY_TOOL_SCHEMAS = (READ_FILE_SCHEMA, LIST_DIR_SCHEMA, SEARCH_TEXT_SCHEMA,
                          LIST_COMMAND_LOGS_SCHEMA)


class ToolUsageError(ValueError):
    """A fixable tool-argument problem (bad path kind, bad paging, budget hit).

    Rendered as a plain "usage error:" so it is never confused with a real
    access-policy denial ("denied by access policy:"), which the model cannot
    fix by rephrasing the call.
    """


class ScanBudgetExceeded(ToolUsageError):
    """The walk hit its own enumeration limit before the query completed.

    A distinct type because a partial answer is still a valid answer here: the
    caller keeps the entries and hits it already collected and appends the
    reason, instead of discarding real evidence.
    """


@dataclass
class ReadOnlyToolset:
    """Harness-side executor for the read-only tools."""

    toolbox: RepoToolbox
    observation_index: dict[str, Any] | None = None
    failure_index: dict[str, Any] | None = None
    max_read_chars: int = 20000
    max_hits: int = 40
    max_files: int = 300
    # Readable candidates one call may examine. Counted after pruning, so a
    # denied or gated tree can never consume the budget.
    max_entries: int = 6000
    # A single generated data line (20k+ chars) must not abort a multi-file
    # search; those lines are skipped and reported instead.
    max_line_chars: int = 20000
    max_file_bytes: int = 400_000
    # Ablation (--no-llvm-ir): the observation tool keeps only asm/summary; the
    # ir/features views and their schema are removed with the IR evidence.
    ir_views: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _observation_schema(self) -> dict[str, Any]:
        if self.ir_views:
            return OBSERVATION_SCHEMA
        schema = {**OBSERVATION_SCHEMA, "function": dict(OBSERVATION_SCHEMA["function"])}
        function = schema["function"]
        function["description"] = (
            "Inspect a measured Embench observation ID: reference-build "
            "function-level ASM candidates or the summary provenance. "
            "Returns confidence/ambiguity and never estimates cycles from static code."
        )
        properties = dict(function["parameters"]["properties"])
        properties["view"] = {**properties["view"], "enum": ["summary", "asm"]}
        function["parameters"] = {**function["parameters"], "properties": properties}
        return schema

    def schemas(self) -> list[dict[str, Any]]:
        # Deep-ish copy so callers cannot mutate the module-level schemas.
        offered = list(READ_ONLY_TOOL_SCHEMAS)
        if self.toolbox.stage == "analysis" and (self.observation_index or {}).get("observations"):
            offered.append(self._observation_schema())
        if self.toolbox.stage == "implementation" and (self.failure_index or {}).get("failures"):
            offered.append(FAILURE_SCHEMA)
        return [
            {**schema, "function": {**schema["function"], "parameters": {
                **schema["function"]["parameters"],
                "properties": dict(schema["function"]["parameters"].get("properties", {})),
            }}}
            for schema in offered
        ]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "search_text": self._search_text,
            "list_command_logs": self._list_command_logs,
            **({"inspect_observation": self._inspect_observation} if self.toolbox.stage == "analysis" and (self.observation_index or {}).get("observations") else {}),
            **({"inspect_failure": self._inspect_failure} if self.toolbox.stage == "implementation" else {}),
        }
        handler = handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name!r}"
        try:
            return self._render(handler(arguments or {}))
        except PolicyError as exc:
            return f"denied by access policy: {exc}"
        except ToolUsageError as exc:
            return f"usage error: {exc}"
        except (OSError, ValueError) as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def _render(self, text: str) -> str:
        """Show this run's real root as ``{{run-dir}}``, matching the prompt."""
        return tools_mod.virtualize_run_paths(
            source_visibility.mask_model_payload(text), str(self.toolbox.run_root or ""))

    def _list_command_logs(self, args: dict[str, Any]) -> str:
        """Enumerate the run's persisted command logs so the model can read them.

        Every executed command writes its complete stdout/stderr under its
        stage's ``logs/`` directory (``10_validation/logs`` for builds,
        simulations and isolated checks, ``11_cost/logs`` for synthesis,
        ``05_profile/logs`` for the profile-stage builds). Failures report one
        log path in their evidence; this tool is how the model finds the rest
        without guessing numbered filenames.
        """
        run_root = str(self.toolbox.run_root or "").strip("/")
        if not run_root:
            raise ToolUsageError("command logs are available only inside a run")
        wanted = str(args.get("rung", "") or "").strip()
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            raise ToolUsageError("limit must be an integer in 1..50")
        limit = max(1, min(50, limit))
        root = self.toolbox.resolve(run_root)
        if not root.is_dir():
            raise ToolUsageError("no such run directory: " + run_root)
        rows: list[tuple[float, str]] = []
        for path in sorted(root.glob("*/logs/*.log")):
            if not path.is_file():
                continue
            rel = self._rel(path)
            if not self.toolbox.is_read_allowed(rel):
                continue
            rung = path.name.rsplit("_", 1)[0]
            if wanted and rung != wanted:
                continue
            rows.append((path.stat().st_mtime, self._log_row(rel, path, rung)))
        if not rows:
            scope = f" for rung {wanted!r}" if wanted else ""
            return (f"no command log{scope} under {run_root}/<stage>/logs; the harness "
                    "writes one for every command it executes")
        rows.sort(key=lambda item: item[0], reverse=True)
        shown = rows[:limit]
        return "\n".join(
            [f"{len(rows)} command log(s) under {run_root} (newest first, "
             f"{len(shown)} shown)"] + [row for _, row in shown])

    def _log_row(self, rel: str, path: Path, rung: str) -> str:
        """One log line: path, rung, return code, size and the command itself."""
        header: dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for _ in range(8):
                    line = handle.readline()
                    if not line:
                        break
                    key, separator, value = line.partition(":")
                    if separator and key.strip() in {"command", "command_id", "returncode"}:
                        header[key.strip()] = value.strip()
            size = path.stat().st_size
        except OSError as exc:
            return f"- {rel} (unreadable: {exc})"
        command = header.get("command", "")
        if len(command) > 120:
            command = command[:117] + "..."
        return (f"- {rel} rung={rung} rc={header.get('returncode', '?')} size={size}B\n"
                f"  command: {command}")

    def _inspect_observation(self, args: dict[str, Any]) -> str:
        if not (self.observation_index or {}).get("observations") or self.toolbox.stage != "analysis":
            raise PolicyError("observation inspection is available only in analysis")
        return inspect_observation(
            self.observation_index, self.toolbox,
            str(args.get("id", "") or ""), str(args.get("view", "summary") or "summary"),
            args.get("max_lines", 60), allow_ir=self.ir_views,
        )

    def _inspect_failure(self, args: dict[str, Any]) -> str:
        if self.toolbox.stage != "implementation":
            raise PolicyError("failure inspection is available only during implementation repair")
        ident = str(args.get("id", "") or "").strip()
        item = (self.failure_index or {}).get("failures", {}).get(ident)
        if not isinstance(item, dict):
            raise ToolUsageError(f"unknown failure ID: {ident or '(empty)'}")
        view = str(args.get("view", "summary") or "summary")
        if view not in {"summary", "log"}:
            raise ToolUsageError("view must be summary or log")
        lines = [
            f"Failure {ident}",
            f"status: {item.get('status', 'current')}",
            f"rung: {item.get('rung', '')}",
            f"category: {item.get('category', '')}",
            f"returncode: {item.get('returncode', '')}",
            f"signature: {item.get('signature', '')}",
            f"command: {item.get('command', '')}",
            "source_locations: " + ", ".join(item.get("source_locations") or []),
            "summary:",
            str(item.get("summary", "")),
        ]
        if view == "log":
            path = str(item.get("log_path", "") or "")
            if not path:
                lines.append("log: unavailable")
            else:
                maximum = args.get("max_lines", 80)
                try:
                    if isinstance(maximum, bool):
                        raise ValueError
                    maximum = max(8, min(200, int(maximum)))
                except (TypeError, ValueError):
                    raise ToolUsageError("max_lines must be an integer in 8..200")
                text = source_visibility.mask_verifier_bodies(
                    self.toolbox.read_text(path, limit=self.max_file_bytes))
                raw_lines = text.splitlines()
                excerpt = raw_lines[-maximum:]
                lines.extend([
                    f"log_path: {path}",
                    f"log_excerpt: last {len(excerpt)} of {len(raw_lines)} lines",
                    *excerpt,
                ])
        return "\n".join(lines)

    # --- internals ---------------------------------------------------------
    def _root(self, raw: str) -> Path:
        raw = (raw or ".").strip() or "."
        root = self.toolbox.workdir if raw in {".", "./"} else self.toolbox.resolve(raw)
        rel = self._rel(root)
        if not self.toolbox.is_current_run_path(rel):
            raise PolicyError(f"directory belongs to another run: {raw}")
        # Missing/file paths are usage problems, not policy denials: reporting
        # them as "denied by access policy" made the model retry the same call
        # against a policy it cannot change (observed 2026-09-28 run).
        if not root.exists():
            raise ToolUsageError(self._missing(f"no such directory: {raw}", raw))
        if not root.is_dir():
            raise ToolUsageError(f"not a directory: {raw}")
        if not self.toolbox.visible_subtree(rel):
            # An invisible directory must say so: answering "(no visible
            # entries)" reads as "this directory is empty" and hides the fact
            # that no amount of rephrasing will make it listable.
            raise PolicyError(f"{self.toolbox.policy.subtree_exclusion(rel)}: {raw}")
        return root

    def _search_base(self, raw: str) -> tuple[Path, bool]:
        """Resolve a search path to (file, True) or (directory, False).

        A file path searches just that file: forcing the model to express a
        file-scoped search as a directory+glob walk is what produced five
        wasted "not a visible directory" retries on a single file (observed
        2026-09-28 run).
        """
        raw = (raw or ".").strip() or "."
        root = self.toolbox.workdir if raw in {".", "./"} else self.toolbox.resolve(raw)
        if not self.toolbox.is_current_run_path(self._rel(root)):
            raise PolicyError(f"path belongs to another run: {raw}")
        if not root.exists():
            raise ToolUsageError(self._missing(f"no such path: {raw}", raw))
        if root.is_file():
            if not self.toolbox.is_read_allowed(self._rel(root)):
                raise PolicyError(f"path is not visible to this flow: {raw}")
            return root, True
        if not root.is_dir():
            raise ToolUsageError(f"not a regular file or directory: {raw}")
        rel = self._rel(root)
        if not self.toolbox.visible_subtree(rel):
            # Same reasoning as _root: an invisible tree must not answer
            # "(no matches)", which the model reads as "the symbol is absent".
            raise PolicyError(f"{self.toolbox.policy.subtree_exclusion(rel)}: {raw}")
        return root, False

    def _missing(self, message: str, raw: str) -> str:
        """Append a bounded "did you mean" to a missing-path message."""
        matches = self._nearest(raw)
        if not matches:
            return message
        return f"{message}; did you mean: {', '.join(matches)}"

    def _nearest(self, raw: str) -> list[str]:
        """Up to three visible paths whose name matches the missing component.

        A bare wrong path (``vexiiriscv`` for ``src/main/scala/vexiiriscv``) is
        indistinguishable from a hidden one and cost the observed run its whole
        exploration budget.
        """
        target = self._missing_component(raw)
        if len(target) < 3:
            return []
        matches: list[str] = []
        try:
            for path in self._visible(self.toolbox.workdir, "**/*", include_dirs=True):
                if path.name != target:
                    continue
                matches.append(f"{self._rel(path)}{'/' if path.is_dir() else ''}")
                if len(matches) >= 3:
                    break
        except ScanBudgetExceeded:
            # Partial suggestions are still suggestions.
            pass
        return matches

    def _missing_component(self, raw: str) -> str:
        """The first path component that does not exist ("" when all do)."""
        parts = [part for part in Path(raw.strip().strip("/")).parts if part != "."]
        for index, part in enumerate(parts):
            candidate = "/".join(parts[: index + 1])
            if not (self.toolbox.workdir / candidate).exists():
                return part
        return ""

    @staticmethod
    def _skip_count(value: Any) -> int:
        """Validate the search paging argument."""
        if value is None or value == "":
            return 0
        try:
            if isinstance(value, bool):
                raise ValueError
            skip = int(value)
        except (TypeError, ValueError):
            raise ToolUsageError("skip must be a non-negative integer")
        if skip < 0:
            raise ToolUsageError("skip must be a non-negative integer")
        return skip

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

    def _visible(self, root: Path, glob: str, *, include_dirs: bool = False) -> Iterator[Path]:
        """Yield entries under ``root`` that match ``glob`` and are readable.

        Directories are yielded (when asked) because a directory is never itself
        in the read tree: filtering them through a read check reported "(no
        visible entries)" for a tree that only contains sub-directories, which
        is what sent the observed run's exploration into a dead end.

        Subtrees that cannot contain a readable file are pruned *before* they
        are counted, so the enumeration budget is spent on candidates rather
        than on denied or gated trees. Traversal is sorted so the same query
        produces the same walk and the same narrowing hint every time.

        Only the walk itself is bounded here; a caller that wants a shorter
        answer (``list_dir``) truncates its own list.
        """
        root_rel = self._rel(root)
        recursive = "/" in glob
        visited = 0
        hint = root_rel
        stack = [root]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as scanned:
                # Decide inside the scandir context: a DirEntry is not a stable
                # handle once its iterator is closed.
                entries = sorted(
                    (
                        (item.name, Path(item.path), item.is_symlink(),
                         item.is_dir(follow_symlinks=False))
                        for item in scanned
                    ),
                    key=lambda item: item[0],
                )
            child_dirs: list[Path] = []
            for _, path, is_symlink, is_dir in entries:
                if is_symlink:
                    continue
                rel = self._rel(path)
                if not self.toolbox.is_current_run_path(rel):
                    continue
                if is_dir:
                    if not self.toolbox.visible_subtree(rel):
                        continue
                    if hint == root_rel or (path / "src").is_dir():
                        hint = rel
                    if recursive:
                        child_dirs.append(path)
                    if not include_dirs:
                        continue
                elif not self.toolbox.is_read_allowed(rel):
                    continue
                visited += 1
                if visited > self.max_entries:
                    raise ScanBudgetExceeded(
                        f"enumeration budget exceeded: {visited}+ readable candidates "
                        f"visited (limit {self.max_entries}) under '{root_rel}'; narrow "
                        f"path (e.g. path={hint}) or tighten glob"
                    )
                local = path.relative_to(root).as_posix()
                matches = fnmatch.fnmatchcase(local, glob) or (
                    glob.startswith("**/") and fnmatch.fnmatchcase(local, glob[3:]))
                if not matches:
                    continue
                yield path
            # Reversed so the stack pops children in sorted order.
            stack.extend(reversed(child_dirs))

    def _read_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", "") or "").strip()
        if not path:
            raise ToolUsageError("path is required")
        rel = self._live_path(path)
        try:
            text = source_visibility.mask_file_view(
                self.toolbox.read_text(rel, limit=self.max_file_bytes), rel)
        except PolicyError:
            # A path that does not exist is a typo, not a policy denial: saying
            # "denied" for a mistyped path left the model retrying a policy it
            # cannot change. An existing-but-invisible path keeps the denial.
            if not (self.toolbox.workdir / self.toolbox.normalize(rel)).exists():
                raise ToolUsageError(self._missing(f"no such file: {path}", path))
            raise
        if not text:
            # Missing, empty and not-visible are three different situations for
            # the model; a single conflated message cost needless retries.
            # Not-visible already raised PolicyError above ("denied by access
            # policy: ..."); distinguish the other two here.
            full = self.toolbox.workdir / self.toolbox.normalize(rel)
            if full.is_dir():
                raise ToolUsageError(f"{path} is a directory (use list_dir or search_text)")
            if not full.is_file():
                raise ToolUsageError(self._missing(f"no such file: {path}", path))
            return "(empty file)"
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
        result = body or "(empty file range)"
        # Store canonical live paths, not the model's {{run-dir}} spelling.
        # Policy checks above still govern the read; memory must not relax them.
        flow_memory.capture_read(self.toolbox, self.toolbox.normalize(rel), args, result)
        return result

    def _list_dir(self, args: dict[str, Any]) -> str:
        root = self._root(str(args.get("path", ".") or "."))
        glob = self._check_glob(str(args.get("glob", "*") or "*"))
        entries: list[str] = []
        try:
            for path in self._visible(root, glob, include_dirs=True):
                if len(entries) >= self.max_files:
                    entries.append(f"...[truncated at {self.max_files} entries]")
                    break
                entries.append(f"{self._rel(path)}{'/' if path.is_dir() else ''}")
        except ScanBudgetExceeded as exc:
            # A partial listing is still useful: keep what was found and say why.
            entries.append(f"...[truncated: {exc}]")
        return "\n".join(entries) or "(no visible entries)"

    def _search_text(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "") or "")
        if not pattern:
            raise ToolUsageError("pattern is required")
        try:
            _validate_search_pattern(pattern)
            regex = re.compile(pattern)
        except (re.error, ValueError) as exc:
            return f"error: invalid regex: {exc}"
        root, single_file = self._search_base(str(args.get("path", ".") or "."))
        glob = self._check_glob(str(args.get("glob", "**/*") or "**/*"))
        skip = self._skip_count(args.get("skip"))
        hits: list[str] = []
        matched = 0
        long_lines = 0
        first_long = ""
        budget = ""
        paths: Iterable[Path] = [root] if single_file else self._visible(root, glob)
        try:
            for path in paths:
                if not path.is_file():
                    continue
                rel = self._rel(path)
                try:
                    text = source_visibility.mask_file_view(
                        self.toolbox.read_text(self._live_path(rel), limit=self.max_file_bytes), rel)
                except PolicyError:
                    continue
                if not text:
                    continue
                for lineno, line in enumerate(text.split("\n"), start=1):
                    if len(line) > self.max_line_chars:
                        # One generated data blob (xgboost.c) used to abort the
                        # whole multi-file search; skip the line and keep going.
                        long_lines += 1
                        if not first_long:
                            first_long = rel
                        continue
                    if regex.search(line):
                        matched += 1
                        if matched <= skip:
                            continue
                        hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                        if len(hits) >= self.max_hits:
                            hits.append(
                                f"...[truncated at {self.max_hits} hits; "
                                f"use skip={skip + self.max_hits} to see more]"
                            )
                            return "\n".join(hits)
        except ScanBudgetExceeded as exc:
            budget = str(exc)
        notes = []
        if long_lines:
            notes.append(
                f"...[skipped {long_lines} line(s) longer than {self.max_line_chars} "
                f"chars (first in {first_long})]"
            )
        if budget:
            notes.append(f"...[search stopped early: {budget}]")
        if not hits:
            outcome = []
            if matched:
                outcome.append(
                    f"(no new matches after skip={skip}; {matched} hit(s) were skipped)"
                )
            elif not budget:
                outcome.append("(no matches)")
            return "\n".join([*outcome, *notes]) or "(no matches)"
        return "\n".join([*hits, *notes])
