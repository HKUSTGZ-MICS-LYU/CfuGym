"""Stage 3's single, bounded HW/SW coding capability (not a validation Gate).

The graph owns target discovery, command construction, and final acceptance. Neither
model arguments nor this facade can choose a command, cwd or gate verdict. Oracle
content is delivered only through the dedicated validated write_conformance tool.
Call ``commit`` only after accepting the whole Stage 3 delivery; otherwise call
``rollback`` (also on provider/tool failures). Checks may precede *either* edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable

from . import observations
from .agent_tools import FAILURE_SCHEMA, READ_ONLY_TOOL_SCHEMAS, ReadOnlyToolset
from .coding_session import CodingSession
from .coding_tools import DIAGNOSIS_SCHEMA, render_check_result
from .hardware_tools import EDIT_HARDWARE_SCHEMA, CHECK_HARDWARE_SCHEMA
from .software_tools import EDIT_SOFTWARE_SCHEMA, CHECK_SOFTWARE_SCHEMA
from .tools import RepoToolbox, design_workspace_paths




def verification_schema(name: str, description: str,
                        parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Schema for a verification tool.

    A verification never accepts argv, cwd, options or an output directory: the
    framework owns the command. The only optional argument is *content* the
    harness consumes, such as a conformance oracle the agent authors.
    """
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": parameters or {"type": "object", "properties": {}, "required": []}}}


@dataclass(frozen=True)
class Verification:
    """One stage-local verification the agent may run in whatever order it likes.

    It executes a framework-built command, records the outcome as *evidence*, and
    never writes a Gate: the ladder reruns the authoritative rung afterwards and
    decides. `budget` bounds how often an expensive one may be repeated.
    """

    name: str
    description: str
    label: str
    final_gate: str
    budget: int
    callback: Callable[[dict[str, Any]], dict[str, Any]]
    #: Optional argument schema. Omitted means "takes no arguments at all".
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name or not all(ch.isalnum() or ch == "_" for ch in self.name):
            raise ValueError(f"invalid verification name: {self.name!r}")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 1:
            raise ValueError(f"verification {self.name!r} needs a positive budget")


def _schema_copy(schema: dict[str, Any]) -> dict[str, Any]:
    function = schema["function"]
    parameters = function["parameters"]
    return {**schema, "function": {**function, "parameters": {
        **parameters, "properties": dict(parameters.get("properties", {})),
    }}}


@dataclass
class ImplementationToolset:
    """One transaction across independent HW and SW targets in either order.

    Target paths are repo-relative and supplied by the graph: an explicit
    allowlist, not a model-discovered glob. A check is an
    optional, zero-argument graph callback and is allowed before any edit.
    """

    toolbox: RepoToolbox
    hardware_targets: frozenset[str]
    software_targets: frozenset[str]
    check_elaboration: Callable[[], dict[str, Any]] | None = None
    check_compile: Callable[[], dict[str, Any]] | None = None
    conformance_session: Any = None
    check_conformance_syntax: Callable[[], dict[str, Any]] | None = None
    #: Verifications the agent schedules itself (build the SoC, build the software,
    #: simulate the workload). Unlike the post-edit checks below, these do not
    #: require an edit in between: re-measuring after an unrelated fix is normal.
    verifications: tuple[Verification, ...] = ()
    failure_index: dict[str, Any] | None = None
    diagnosis_required: bool = False
    #: Per-domain budgets bound one stage's thrash, not its work: a correct
    #: delivery commonly needs an include guard, the kernel body and a boundary
    #: fix. `2` (the original value) rejected the third edit of a legitimate
    #: patch in run-0003 and left no way to answer the gate.
    max_hardware_edits: int = 6
    max_software_edits: int = 6
    max_hardware_checks: int = 2
    max_software_checks: int = 2
    max_edit_chars: int = 100_000
    edits_used: dict[str, int] = field(default_factory=lambda: {"hardware": 0, "software": 0})
    checks_used: dict[str, int] = field(default_factory=lambda: {"hardware": 0, "software": 0})
    verifications_used: dict[str, int] = field(default_factory=dict)
    verification_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    readonly: ReadOnlyToolset = field(init=False)
    hardware_session: CodingSession = field(init=False)
    software_session: CodingSession = field(init=False)

    def __post_init__(self) -> None:
        run_root = str(self.toolbox.run_root or "").strip("/")
        if not run_root or not run_root.startswith("agents/generated/runs/"):
            raise ValueError("Stage 3 coding requires a run-scoped workspace")
        design = set(design_workspace_paths(run_root).values())
        if not self.hardware_targets.issubset(design):
            raise ValueError("hardware targets must be AgentCfu design overlays")
        prefix = run_root + "/workspace/"
        if any(not p.startswith(prefix) or p in design
               or PurePosixPath(p).suffix not in {".c", ".h"}
               for p in self.software_targets):
            raise ValueError("software targets must be graph-approved workspace sources")
        for name, budget in (("hardware edits", self.max_hardware_edits),
                             ("software edits", self.max_software_edits),
                             ("hardware checks", self.max_hardware_checks),
                             ("software checks", self.max_software_checks),
                             ("edit characters", self.max_edit_chars)):
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise ValueError(f"{name} budget must be a non-negative integer")
        groups = (self.hardware_targets, self.software_targets)
        if sum(map(len, groups)) != len(set().union(*groups)):
            raise ValueError("Stage 3 target groups must not overlap")
        for path in set().union(*groups):
            # Normalize before snapshotting: reject traversal, absolute paths and
            # symlinks, including a symlink in an ancestor of a yet-uncreated file.
            if self.toolbox.normalize(path) != path or not self.toolbox.is_write_allowed(path):
                raise ValueError(f"target is not a writable run-workspace file: {path}")
        names = [v.name for v in self.verifications]
        if len(names) != len(set(names)):
            raise ValueError("verification names must be unique")
        reserved = {"edit_hardware", "edit_software",
                    "check_hardware_elaboration", "check_software_compile",
                    "submit_repair_diagnosis", "submit_stage", "write_conformance", "check_conformance_syntax"}
        clash = reserved.intersection(names)
        if clash:
            raise ValueError(f"verification name shadows a reserved tool: {sorted(clash)}")
        if self.failure_index is None:
            self.failure_index = {"failures": {}}
        self.readonly = ReadOnlyToolset(self.toolbox, failure_index=self.failure_index)
        self.hardware_session = CodingSession(
            self.toolbox, self.hardware_targets, failure_index=self.failure_index,
            diagnosis_required=self.diagnosis_required,
        )
        self.software_session = CodingSession(
            self.toolbox, self.software_targets,
            failure_index=self.failure_index, diagnosis_required=self.diagnosis_required,
        )

    @property
    def sessions(self) -> tuple[CodingSession, CodingSession]:
        return self.hardware_session, self.software_session

    @property
    def touched_paths(self) -> list[str]:
        return self.hardware_session.touched_paths + self.software_session.touched_paths

    @property
    def checks(self) -> dict[str, list[dict[str, Any]]]:
        trail = {"hardware": list(self.hardware_session.checks),
                 "software": list(self.software_session.checks)}
        for name, results in self.verification_results.items():
            trail[name] = list(results)
        return trail

    def commit(self) -> None:
        if (any(session.rolled_back for session in self.sessions)
                or (self.conformance_session is not None and self.conformance_session.rolled_back)):
            raise RuntimeError("cannot commit a rolled-back Stage 3 attempt")
        if self.conformance_session is not None:
            self.conformance_session.commit()
        for session in self.sessions:
            session.commit()

    def rollback(self) -> None:
        if (any(session.committed for session in self.sessions)
                or (self.conformance_session is not None and self.conformance_session.committed)):
            raise RuntimeError("cannot roll back a committed Stage 3 attempt")
        errors = []
        participants = (*self.sessions, *((self.conformance_session,) if self.conformance_session is not None else ()))
        for session in participants:
            try:
                session.rollback()
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            raise OSError("Stage 3 rollback failed: " + "; ".join(errors))

    def schemas(self) -> list[dict[str, Any]]:
        schemas = list(READ_ONLY_TOOL_SCHEMAS)
        schemas.append(FAILURE_SCHEMA)
        if self.conformance_session is not None:
            schemas.append(verification_schema("write_conformance",
                "Deliver an oracle: validate and render fixed host-owned test source. No build or simulation. "
                "Choose small counts/data for quick functional tests; preserve independent scalar reference.",
                {"type": "object", "properties": {"oracle": {"type": "string"}}, "required": ["oracle"]}))
            if self.check_conformance_syntax is not None:
                schemas.append(verification_schema("check_conformance_syntax",
                    "Optional syntax-only compiler check of the delivered test. No link, ELF or simulation."))
        if self.diagnosis_required:
            schemas.append(DIAGNOSIS_SCHEMA)
        if self.hardware_targets:
            schemas.append(EDIT_HARDWARE_SCHEMA)
        if self.software_targets:
            schemas.append(EDIT_SOFTWARE_SCHEMA)
        if self.check_elaboration is not None and self.max_hardware_checks > 0:
            schemas.append(CHECK_HARDWARE_SCHEMA)
        if self.check_compile is not None and self.max_software_checks > 0:
            schemas.append(CHECK_SOFTWARE_SCHEMA)
        for verification in self.verifications:
            if self.verifications_used.get(verification.name, 0) < verification.budget:
                schemas.append(verification_schema(verification.name, verification.description,
                                                   verification.parameters))
        return [_schema_copy(item) for item in schemas]

    def grant_correction(self) -> None:
        """Re-open the local edit/check budgets after a stage-gate rejection.

        The gate rejected the bytes as they are, so the model must be able to
        change and re-check them; refusing every later edit makes the demanded
        correction impossible (observed run-0003: the CFU-usage check rejected
        the kernel, and every following `edit_software` returned "budget
        exhausted"). The corrective round stays bounded by the loop's own
        turn/call caps and by the single allowed submission retry.
        """
        self.edits_used["hardware"] = 0
        self.edits_used["software"] = 0
        self.checks_used["hardware"] = 0
        self.checks_used["software"] = 0

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        args = arguments or {}
        if any(session.committed or session.rolled_back for session in self.sessions):
            return "error: Stage 3 coding transaction is closed"
        if name == "submit_repair_diagnosis" and self.diagnosis_required:
            answer = self.hardware_session.submit_diagnosis(args)
            if answer.startswith("diagnosis accepted"):
                self.software_session.diagnosis = dict(self.hardware_session.diagnosis or {})
                if self.conformance_session is not None:
                    self.conformance_session.diagnosis = dict(self.hardware_session.diagnosis or {})
            return answer
        if name in {"run_conformance", "run_simulation"}:
            return "error: simulation is available only in Verify"
        if name == "write_conformance" and self.conformance_session is not None:
            if set(args) != {"oracle"} or not isinstance(args.get("oracle"), str):
                return "usage error: write_conformance requires only an oracle string"
            delivery, problems = self.conformance_session.write(args["oracle"])
            if problems:
                return "error: oracle delivery rejected: " + "; ".join(problems)
            # Host-owned compile preflight: a delivered test that cannot even be
            # parsed must not reach the ladder (run 20261003-130005 spent the
            # last answer-repair round on `redefinition of 'md5_oracle_mem'`).
            # Rejecting here keeps the previous delivery and hands the compiler
            # error straight back to the model.
            syntax = self._conformance_syntax_problems()
            if syntax:
                self.conformance_session.revert_pending()
                return "error: oracle delivery rejected: the delivered C does not compile: " + syntax
            return "conformance oracle delivered and source compiled (syntax only); no simulation. " + delivery["source_path"]
        if name == "check_conformance_syntax" and self.check_conformance_syntax is not None:
            if args:
                return "usage error: check_conformance_syntax takes no arguments"
            result = self.check_conformance_syntax()
            result = observations.record_failure_result(self.failure_index, result,
                rung="conformance_syntax", workdir=str(self.toolbox.workdir))
            return render_check_result(result,
                label="conformance syntax", final_gate="hw_conformance (simulation in Verify only)")
        if name == "edit_hardware" and self.hardware_targets:
            return self._edit_hardware(args)
        if name == "edit_software" and self.software_targets:
            return self._edit_software(args)
        if name == "check_hardware_elaboration" and self.check_elaboration is not None and self.max_hardware_checks > 0:
            return self._check(args, "hardware", self.check_elaboration, self.hardware_session,
                               self.max_hardware_checks, "hardware elaboration", "hw_build")
        if name == "check_software_compile" and self.check_compile is not None and self.max_software_checks > 0:
            return self._check(args, "software", self.check_compile, self.software_session,
                               self.max_software_checks, "software compile", "sw_build")
        for verification in self.verifications:
            if name == verification.name:
                return self._verify(args, verification)
        return self.readonly.run(name, args)

    def _verify(self, args: dict[str, Any], verification: Verification) -> str:
        if args and verification.parameters is None:
            return f"usage error: {verification.name} accepts no arguments"
        used = self.verifications_used.get(verification.name, 0)
        if used >= verification.budget:
            return f"error: {verification.label} budget exhausted ({verification.budget})"
        self.verifications_used[verification.name] = used + 1
        result = verification.callback(dict(args))
        result = observations.record_failure_result(
            self.failure_index, result, rung=verification.final_gate,
            workdir=str(self.toolbox.workdir),
        )
        self.verification_results.setdefault(verification.name, []).append(dict(result))
        return render_check_result(result, label=verification.label,
                                   final_gate=verification.final_gate)

    def _edit_hardware(self, args: dict[str, Any]) -> str:
        if self.edits_used["hardware"] >= self.max_hardware_edits:
            return "error: hardware edit budget exhausted"
        raw = args.get("edits")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
            return "usage error: edits must contain 1..8 SEARCH/REPLACE objects"
        grouped: dict[str, list[tuple[str, str]]] = {}
        count = 0
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                return "usage error: each edit requires a path"
            path = self.hardware_session.resolve_target(item["path"])
            if path not in self.hardware_targets:
                return "denied by hardware capability: edit only " + ", ".join(sorted(self.hardware_targets))
            search, replace = item.get("search"), item.get("replace")
            if not isinstance(search, str) or not search or not isinstance(replace, str):
                return "usage error: search must be non-empty and replace must be a string"
            count += len(search) + len(replace)
            grouped.setdefault(path, []).append((search, replace))
        if count > self.max_edit_chars:
            return "usage error: edit character budget exceeded"
        return self._apply(self.hardware_session, [(p, pairs) for p, pairs in grouped.items()], "hardware")

    def _edit_software(self, args: dict[str, Any]) -> str:
        if self.edits_used["software"] >= self.max_software_edits:
            return "error: software edit budget exhausted"
        path = args.get("path")
        if not isinstance(path, str) or self.software_session.resolve_target(path) not in self.software_targets:
            return "denied by software capability: edit only " + ", ".join(sorted(self.software_targets))
        path = self.software_session.resolve_target(path)
        search, replace = args.get("search"), args.get("replace")
        if not isinstance(search, str) or not search or not isinstance(replace, str):
            return "usage error: search must be non-empty and replace must be a string"
        if len(search) + len(replace) > self.max_edit_chars:
            return "usage error: edit character budget exceeded"
        return self._apply(self.software_session, [(path, [(search, replace)])], "software")

    def _apply(self, session: CodingSession, edits: list[tuple[str, list[tuple[str, str]]]], kind: str) -> str:
        result = session.apply_edits(edits)
        if result.get("returncode") != 0:
            return "edit rejected: " + str(result.get("stderr_tail", ""))[:2000]
        self.edits_used[kind] += 1
        return str(result.get("stdout_tail", "") or "edits applied atomically")[:6000]

    def _conformance_syntax_problems(self) -> str:
        """Compiler diagnostics for the just-delivered test, or "" when it compiles.

        The parent owns the command; this only consumes its result. A negative
        status (policy denial, missing toolchain, timeout) is *not* a delivery
        rejection -- the model cannot fix that, and the final ``hw_conformance``
        rung still owns the verdict -- so only a real compiler error (positive
        return code) blocks the delivery.
        """
        if self.check_conformance_syntax is None:
            return ""
        try:
            result = self.check_conformance_syntax()
        except Exception:
            return ""
        if not isinstance(result, dict):
            return ""
        returncode = result.get("returncode")
        if returncode is None or not isinstance(returncode, int) or returncode <= 0:
            return ""
        text = str(result.get("stderr_tail") or result.get("stdout_tail") or "").strip()
        return " ".join(text.split())[-1200:]

    def _check(self, args: dict[str, Any], kind: str, callback: Callable[[], dict[str, Any]],
               session: CodingSession, budget: int, label: str, gate: str) -> str:
        if args:
            return f"usage error: check_{kind} accepts no arguments (no argv/cwd/gate)"
        if self.checks_used[kind] >= budget:
            return f"error: {kind} check budget exhausted"
        self.checks_used[kind] += 1
        try:
            result = callback()
            if not isinstance(result, dict):
                raise TypeError("check callback must return a result mapping")
        except Exception as exc:
            result = {"returncode": 1, "stderr_tail": f"check callback failed: {type(exc).__name__}: {exc}"}
        result = observations.record_failure_result(
            self.failure_index, result, rung="hw_check" if kind == "hardware" else "sw_check",
            workdir=str(self.toolbox.workdir),
        )
        session.record_check(result)
        return render_check_result(result, label=label, final_gate=f"final {gate} gate")
