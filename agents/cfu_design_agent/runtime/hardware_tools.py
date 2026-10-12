"""Bounded transactional coding tools for AgentCfu hardware overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent_tools import FAILURE_SCHEMA, READ_ONLY_TOOL_SCHEMAS, ReadOnlyToolset
from .coding_session import CodingSession
from . import coding_tools
from .tools import RepoToolbox


EDIT_HARDWARE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit_hardware",
        "description": (
            "Atomically apply one or more exact SEARCH/REPLACE edits to the two "
            "graph-approved AgentCfu hardware overlay files. No new files or patches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "search": {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["path", "search", "replace"],
                    },
                },
            },
            "required": ["edits"],
        },
    },
}

CHECK_HARDWARE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_hardware_elaboration",
        "description": (
            "Run the host-owned isolated GenAgentCfu elaboration after an edit. "
            "Takes no argv/cwd/options and cannot update a graph Gate."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@dataclass
class HardwareCodingToolset:
    toolbox: RepoToolbox
    allowed_targets: frozenset[str]
    check_elaboration: Callable[[], dict[str, Any]]
    observation_index: dict[str, Any] | None = None
    failure_index: dict[str, Any] | None = None
    diagnosis_required: bool = False
    max_edits: int = 10
    max_checks: int | None = None
    max_edit_chars: int = 100_000
    edits_used: int = 0
    checks_used: int = 0
    edited_since_check: bool = False
    readonly: ReadOnlyToolset = field(init=False)
    session: CodingSession = field(init=False)

    def __post_init__(self) -> None:
        self.readonly = ReadOnlyToolset(self.toolbox, failure_index=self.failure_index)
        self.session = CodingSession(
            self.toolbox,
            self.allowed_targets,
            failure_index=self.failure_index,
            diagnosis_required=self.diagnosis_required,
        )

    @property
    def touched_paths(self) -> list[str]:
        return self.session.touched_paths

    def schemas(self) -> list[dict[str, Any]]:
        offered = list(READ_ONLY_TOOL_SCHEMAS)
        if (self.failure_index or {}).get("failures"):
            offered.append(FAILURE_SCHEMA)
        if self.diagnosis_required:
            offered.append(coding_tools.DIAGNOSIS_SCHEMA)
        offered.extend([EDIT_HARDWARE_SCHEMA, CHECK_HARDWARE_SCHEMA])
        return [{**item, "function": {**item["function"], "parameters": {
            **item["function"]["parameters"],
            "properties": dict(item["function"]["parameters"].get("properties", {})),
        }}} for item in offered]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "submit_repair_diagnosis" and self.diagnosis_required:
            return self.session.submit_diagnosis(arguments or {})
        if name == "edit_hardware":
            return self._edit(arguments or {})
        if name == "check_hardware_elaboration":
            return self._check(arguments or {})
        return self.readonly.run(name, arguments or {})

    def _edit(self, args: dict[str, Any]) -> str:
        if self.edits_used >= self.max_edits:
            return f"error: hardware edit budget exhausted ({self.max_edits})"
        raw = args.get("edits")
        if not isinstance(raw, list) or not raw or len(raw) > 8:
            return "usage error: edits must contain 1..8 SEARCH/REPLACE objects"
        total_chars = 0
        grouped: list[tuple[str, list[tuple[str, str]]]] = []
        by_path: dict[str, list[tuple[str, str]]] = {}
        order: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                return "usage error: every edit must be an object"
            path = str(item.get("path", "") or "")
            search = item.get("search")
            replace = item.get("replace")
            resolved = self.session.resolve_target(path)
            if resolved is None:
                return (
                    "denied by hardware capability: path is not an approved AgentCfu "
                    "target; edit only: " + ", ".join(sorted(self.allowed_targets))
                )
            path = resolved
            if not isinstance(search, str) or not search:
                return "usage error: search must be a non-empty string"
            if not isinstance(replace, str):
                return "usage error: replace must be a string"
            total_chars += len(search) + len(replace)
            if path not in by_path:
                by_path[path] = []
                order.append(path)
            by_path[path].append((search, replace))
        if total_chars > self.max_edit_chars:
            return f"usage error: edit character budget exceeded ({self.max_edit_chars})"
        grouped = [(path, by_path[path]) for path in order]
        result = self.session.apply_edits(grouped)
        if result.get("returncode") != 0:
            return "edit rejected: " + str(result.get("stderr_tail", ""))[:3000]
        touched = list((result.get("parsed") or {}).get("touched_paths") or [])
        self.edits_used += 1
        self.edited_since_check = self.edited_since_check or bool(touched)
        return str(result.get("stdout_tail", "") or "hardware edits applied atomically")[:8000]

    def _check(self, args: dict[str, Any]) -> str:
        if args:
            return "usage error: check_hardware_elaboration accepts no arguments"
        if self.max_checks is not None and self.checks_used >= self.max_checks:
            return f"error: hardware check budget exhausted ({self.max_checks})"
        if not self.edited_since_check:
            return "usage error: edit hardware before requesting elaboration"
        self.checks_used += 1
        result = self.check_elaboration()
        self.session.record_check(result)
        self.edited_since_check = False
        return coding_tools.render_check_result(
            result, label="hardware elaboration", final_gate="final hw_build gate"
        )
