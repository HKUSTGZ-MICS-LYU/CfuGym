"""Run-scoped ISA draft editing; the stage, not a tool, publishes the contract.

Only draft artifacts are written. Syntax, schema and extension compatibility are
checked independently on submission; no hardware/software or shell capability.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .. import isa
from . import agent_tools, parsing, stage_services, tools as tools_mod
from .policy import PolicyError

MAX_DRAFT_CHARS = 2_000_000


def _schema(name: str, description: str, properties: dict, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(required), "additionalProperties": False}}}


ISA_TOOL_SCHEMAS = (
    _schema("read_isa", "Read the current editable ISA draft with numbered lines; not the accepted contract.",
            {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}}),
    _schema("edit_isa", "Apply an atomic batch of exact SEARCH/REPLACE edits to the ISA draft only. Each nonempty search must match exactly once. Read/check the draft, then submit_isa; no other path can be written.",
            {"edits": {"type": "array", "items": {"type": "object", "properties": {
                "search": {"type": "string"}, "replace": {"type": "string"}},
                "required": ["search", "replace"], "additionalProperties": False}}}, ("edits",)),
    _schema("check_isa", "Check the current draft's YAML/JSON syntax, ISA schema, encodings and append-only ABI compatibility. This is lint, not hardware correctness or performance proof.", {}),
)


class IsaDraftSession:
    def __init__(self, state: dict[str, Any], toolbox: Any, *, seed: dict, stage: str):
        if stage not in {"design_contract", "extend_isa"}:
            raise ValueError("invalid ISA draft stage")
        self.state, self.toolbox, self.stage = state, toolbox, stage
        self.run_root = str(state.get("run_root") or "").rstrip("/")
        if not self.run_root or self.run_root != str(toolbox.run_root or "").rstrip("/"):
            raise PolicyError("ISA draft requires a matching current run")
        self.relative = f"08_isa/drafts/{stage}.yaml"
        self.path = f"{self.run_root}/{self.relative}"
        self._safe_path(self.path, write=True)
        self.base = deepcopy(seed)
        self.seed_text = parsing._yaml_str(self.base)
        self.base_sha256 = self.digest(self.seed_text)
        self.text = self.seed_text
        self.record: dict[str, Any] = {}
        self.submitted_sha256 = ""
        self.accepted: dict[str, Any] | None = None
        self.edits = 0
        self._restore(state.get("isa_draft") or {})

    def _safe_path(self, path: str, *, write: bool = False) -> Path:
        if self.toolbox.normalize(path) != path or not self.toolbox.is_current_run_path(path):
            raise PolicyError("ISA draft path is not canonical in the current run")
        if not self.toolbox.is_read_allowed(path) or (write and not self.toolbox.is_write_allowed(path)):
            raise PolicyError("ISA draft path is outside stage policy")
        full = self.toolbox.workdir / path
        for part in (full, *full.parents):
            if part == self.toolbox.workdir:
                break
            if part.is_symlink():
                raise PolicyError("ISA draft path contains a symlink")
        return full

    @staticmethod
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _restore(self, record: dict) -> None:
        if (record.get("stage") != self.stage or record.get("base_sha256") != self.base_sha256):
            return  # Different contract/rewind lineage: never adopt an old draft.
        path = record.get("immutable_path")
        expected_prefix = f"{self.run_root}/08_isa/drafts/versions/{self.stage}.yaml."
        if not isinstance(path, str) or not path.startswith(expected_prefix):
            raise PolicyError("ISA draft snapshot is outside this stage's current-run draft")
        self._safe_path(path)
        text = self.toolbox.read_text(path, limit=MAX_DRAFT_CHARS + 1)
        if len(text) > MAX_DRAFT_CHARS or self.digest(text) != record.get("sha256"):
            raise ValueError("ISA draft snapshot hash is mismatched")
        self.text, self.record = text, dict(record)

    def _save(self, text: str) -> None:
        if not isinstance(text, str) or len(text) > MAX_DRAFT_CHARS:
            raise ValueError("ISA draft exceeds the text budget")
        self._safe_path(self.path, write=True)
        self._safe_path(f"{self.run_root}/08_isa/drafts/versions/{self.stage}.yaml.{self.digest(text)}", write=True)
        artifacts = stage_services._artifacts_for_stage(self.state, "isa")
        if artifacts.root_rel != self.run_root:
            raise PolicyError("ISA draft artifact writer belongs to another run")
        ref = artifacts.write_text("isa", self.relative, text, kind="isa_draft", metadata={
            "draft_stage": self.stage, "base_sha256": self.base_sha256})
        self.text = text
        self.record = {"stage": self.stage, "base_sha256": self.base_sha256,
                       "path": ref["path"], "immutable_path": ref["immutable_path"], "sha256": ref["sha256"]}
        self.submitted_sha256, self.accepted = "", None

    def read(self, start_line=1, end_line=None) -> str:
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            raise ValueError("start_line must be a positive integer")
        if end_line is None:
            end_line = start_line + 159
        if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < start_line:
            raise ValueError("end_line must be >= start_line")
        lines = self.text.splitlines()
        end_line = min(end_line, start_line + 199, len(lines))
        excerpt = "\n".join(f"{index + 1}: {lines[index]}" for index in range(start_line - 1, end_line))
        if len(excerpt) > 20000:
            excerpt = excerpt[:20000] + "\n[character budget reached; request a smaller range]"
        return (f"ISA draft {self.path}; sha256={self.digest(self.text)}; {len(lines)} lines; "
                f"showing {start_line}-{end_line}\n{excerpt}")

    def edit(self, edits: Any) -> str:
        if not isinstance(edits, list) or not edits or len(edits) > 32:
            raise ValueError("edits must contain 1..32 search/replace mappings")
        changed = self.text
        for index, edit in enumerate(edits, 1):
            if not isinstance(edit, dict) or set(edit) != {"search", "replace"}:
                raise ValueError(f"edit {index}: requires only search and replace")
            search, replace = edit["search"], edit["replace"]
            if not isinstance(search, str) or not search or not isinstance(replace, str):
                raise ValueError(f"edit {index}: search must be nonempty and replace must be text")
            if tools_mod.RUN_DIR_TOKEN in search:
                visible = tools_mod.virtualize_run_paths(changed, self.run_root)
                if visible.count(search) != 1:
                    raise ValueError(f"edit {index}: virtual SEARCH must match exactly once; include more context")
                if not changed.count(search):
                    search = search.replace(tools_mod.RUN_DIR_TOKEN, self.run_root)
                    replace = replace.replace(tools_mod.RUN_DIR_TOKEN, self.run_root)
            count = changed.count(search)
            if count != 1:
                raise ValueError(f"edit {index}: SEARCH matched {count} times (must be exactly one); read_isa for exact text")
            changed = changed.replace(search, replace, 1)
            if len(changed) > MAX_DRAFT_CHARS:
                raise ValueError("ISA draft exceeds the text budget")
        if changed == self.text:
            raise ValueError("edit batch made no change")
        self._save(changed)
        self.edits += len(edits)
        return f"applied {len(edits)} ISA draft edit(s); sha256={self.digest(self.text)}; use check_isa before submit_isa"

    def candidate(self) -> tuple[dict, list[str]]:
        try:
            value, problems = isa.validate_isa_text(self.text)
        except (isa.IsaLintError, TypeError, AttributeError, ValueError) as exc:
            return {}, [f"ISA draft syntax: {exc}"]
        if not problems and self.stage == "extend_isa":
            problems = isa.validate_isa_extension(self.base, value)
            if not problems:
                old_pairs = {(c["function_id"], c.get("subfunction_id", 0)) for c in self.base.get("commands") or []}
                if not any((c["function_id"], c.get("subfunction_id", 0)) not in old_pairs for c in value.get("commands") or []):
                    problems = ["ISA extension must append at least one new command"]
        return value, problems

    def check(self) -> str:
        candidate, problems = self.candidate()
        label = "ISA check failed" if problems else "ISA check passed (schema/compatibility only)"
        commands = candidate.get("commands")
        return label + "\n" + json.dumps({"sha256": self.digest(self.text), "problems": problems,
                                          "commands": len(commands) if isinstance(commands, list) else None}, ensure_ascii=True)

    def submit(self, delivery: str) -> tuple[bool, str]:
        if not isinstance(delivery, str) or not delivery.strip():
            return False, "submit_isa requires a factual summary; edit the draft first"
        candidate, problems = self.candidate()
        if problems:
            return False, "ISA check failed:\n" + "\n".join(problems)
        if candidate == self.base:
            return False, "ISA draft is unchanged; use edit_isa to author the contract/extension"
        self.accepted = deepcopy(candidate)
        self.submitted_sha256 = self.digest(self.text)
        return True, ""

    def rollback(self) -> None:
        # Drafts remain audit evidence, but cannot be published after a failed
        # execution/recording step. The accepted contract was never touched.
        self.accepted, self.submitted_sha256 = None, ""

    def accepted_text(self) -> str:
        if self.accepted is None or self.submitted_sha256 != self.digest(self.text):
            raise ValueError("ISA draft was not explicitly submitted, or changed after submission")
        return parsing._yaml_str(self.accepted)

    def apply_delivery(self, text: str) -> None:
        """Optional plain EDIT protocol; complete YAML/JSON remains supported."""
        blocks = tools_mod.extract_edit_blocks(text)
        if not blocks:
            self._save(text)
            return
        edits = []
        for target, pairs in blocks:
            if self.toolbox.normalize(target) != self.path:
                raise PolicyError("ISA EDIT delivery may target only the current draft")
            edits.extend({"search": search, "replace": replace} for search, replace in pairs)
        self.edit(edits)


class IsaToolset(agent_tools.ReadOnlyToolset):
    def __init__(self, toolbox, *, session: IsaDraftSession, **kwargs):
        super().__init__(toolbox, **kwargs)
        if toolbox is not session.toolbox:
            raise ValueError("ISA toolset and draft session must share a toolbox")
        self.session = session

    def schemas(self):
        return [*super().schemas(), *deepcopy(ISA_TOOL_SCHEMAS)]

    def rollback(self):
        self.session.rollback()

    def run(self, name, arguments):
        if name not in {"read_isa", "edit_isa", "check_isa"}:
            return super().run(name, arguments)
        try:
            args = arguments or {}
            allowed = {"read_isa": {"start_line", "end_line"}, "edit_isa": {"edits"}, "check_isa": set()}[name]
            if not isinstance(args, dict) or set(args) - allowed:
                raise ValueError(f"{name} has unsupported arguments")
            if name == "read_isa":
                result = self.session.read(**args)
            elif name == "edit_isa":
                result = self.session.edit(args.get("edits"))
            else:
                result = self.session.check()
            return self._render(result)
        except PolicyError as exc:
            return f"denied by access policy: {exc}"
        except (ValueError, OSError) as exc:
            return f"error: {exc}"
