"""Narrow software coding capabilities for the implementation-stage pilot.

This facade deliberately does not expose arbitrary paths, patches, argv, cwd, shell,
simulation, gate state, or synthesis.  The graph binds its target set and injects a
host-owned compile callback; this module only enforces budgets and presents bounded
observations to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent_tools import FAILURE_SCHEMA, READ_ONLY_TOOL_SCHEMAS, ReadOnlyToolset
from .coding_session import CodingSession
from .coding_tools import DIAGNOSIS_SCHEMA, render_check_result
from .tools import RepoToolbox


EDIT_SOFTWARE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit_software",
        "description": (
            "Apply one exact SEARCH/REPLACE edit to one graph-approved software target. "
            "SEARCH must be non-empty and match exactly once. No whole-file or patch edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Exact approved run-workspace path."},
                "search": {"type": "string", "description": "Exact existing text; must match once."},
                "replace": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "search", "replace"],
        },
    },
}

CHECK_SOFTWARE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_software_compile",
        "description": (
            "Run the host-owned isolated software compile check after an edit. "
            "Takes no command, argv, cwd, target, or gate parameters; this is evidence only."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@dataclass
class SoftwareCodingToolset:
    """Read tools plus tightly bound edit/check capabilities for one SW attempt."""

    toolbox: RepoToolbox
    allowed_targets: frozenset[str]
    check_compile: Callable[[], dict[str, Any]]
    observation_index: dict[str, Any] | None = None  # executor compatibility; intentionally unused
    failure_index: dict[str, Any] | None = None
    diagnosis_required: bool = False
    max_edits: int = 10
    max_checks: int = 10
    max_edit_chars: int = 60_000
    edits_used: int = 0
    checks_used: int = 0
    edited_since_check: bool = False
    readonly: ReadOnlyToolset = field(init=False)
    session: CodingSession = field(init=False)

    @property
    def touched_paths(self) -> list[str]:
        return self.session.touched_paths

    def __post_init__(self) -> None:
        self.readonly = ReadOnlyToolset(self.toolbox, failure_index=self.failure_index)
        self.session = CodingSession(
            self.toolbox,
            self.allowed_targets,
            failure_index=self.failure_index,
            diagnosis_required=self.diagnosis_required,
        )

    def schemas(self) -> list[dict[str, Any]]:
        offered = list(READ_ONLY_TOOL_SCHEMAS)
        if (self.failure_index or {}).get("failures"):
            offered.append(FAILURE_SCHEMA)
        if self.diagnosis_required:
            offered.append(DIAGNOSIS_SCHEMA)
        offered.extend([EDIT_SOFTWARE_SCHEMA, CHECK_SOFTWARE_SCHEMA])
        return [{**item, "function": {**item["function"], "parameters": {
            **item["function"]["parameters"],
            "properties": dict(item["function"]["parameters"].get("properties", {})),
        }}} for item in offered]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "submit_repair_diagnosis" and self.diagnosis_required:
            return self.session.submit_diagnosis(arguments or {})
        if name == "edit_software":
            return self._edit(arguments or {})
        if name == "check_software_compile":
            return self._check(arguments or {})
        return self.readonly.run(name, arguments or {})

    def _edit(self, args: dict[str, Any]) -> str:
        if self.edits_used >= self.max_edits:
            return f"error: software edit budget exhausted ({self.max_edits})"
        path = str(args.get("path", "") or "")
        search = args.get("search")
        replace = args.get("replace")
        resolved = self.session.resolve_target(path)
        if resolved is None:
            return (
                "denied by software capability: path is not an approved software "
                "target; edit only: " + ", ".join(sorted(self.allowed_targets))
            )
        path = resolved
        if not isinstance(search, str) or not search:
            return "usage error: search must be a non-empty string"
        if not isinstance(replace, str):
            return "usage error: replace must be a string"
        if len(search) + len(replace) > self.max_edit_chars:
            return f"usage error: edit character budget exceeded ({self.max_edit_chars})"
        result = self.session.apply_edits([(path, [(search, replace)])])
        if result.get("returncode") != 0:
            return "edit rejected: " + str(result.get("stderr_tail", ""))[:2000]
        touched = list((result.get("parsed") or {}).get("touched_paths") or [])
        self.edits_used += 1
        self.edited_since_check = self.edited_since_check or bool(touched)
        return str(result.get("stdout_tail", "") or "edit applied")[:6000]

    def _check(self, args: dict[str, Any]) -> str:
        if args:
            return "usage error: check_software_compile accepts no arguments"
        if self.checks_used >= self.max_checks:
            return f"error: software check budget exhausted ({self.max_checks})"
        if not self.edited_since_check:
            return "usage error: edit software before requesting a compile check"
        self.checks_used += 1
        result = self.check_compile()
        self.session.record_check(result)
        # A failed check still consumes the check budget but may be followed by
        # another check after a new edit; a successful check freezes this edit.
        self.edited_since_check = False
        return render_check_result(result, label="software compile", final_gate="final sw_build gate")
