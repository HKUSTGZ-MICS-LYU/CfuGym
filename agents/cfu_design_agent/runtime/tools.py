"""Repo-safe tools used by the CFU design agent."""

from __future__ import annotations

import difflib
import contextlib
import contextvars
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .persistence import RunPersistence
from .policy import AccessPolicy, PolicyError, load_policy, normalize_repo_path
from ..state import CommandResult, RepoContext


MAX_FILE_CHARS = 20000
#: Virtual path token standing for the executing run's artifact root. Prompts and
#: tool results render it instead of ``agents/generated/runs/<run-id>``, so the
#: task text is byte-identical across runs (provider prefix caching) and a run id
#: never reaches the model; every path argument the model sends back is expanded
#: to the real root before any access-policy check.
RUN_DIR_TOKEN = "{{run-dir}}"


def expand_run_dir(value: str, run_root: str) -> str:
    """Server-side translation of the ``{{run-dir}}`` token, or the input.

    Only a leading token is a path prefix; anything else is returned unchanged so
    the caller's normal path validation reports it.
    """
    text = str(value or "")
    root = str(run_root or "").strip("/")
    if not root or RUN_DIR_TOKEN not in text:
        return text
    if text == RUN_DIR_TOKEN:
        return root
    if text.startswith(RUN_DIR_TOKEN + "/"):
        return root + text[len(RUN_DIR_TOKEN):]
    return text


#: Result prefixes a toolset uses for a rejected call rather than an outcome.
_TOOL_FAILURE_PREFIXES = ("error:", "tool error:", "denied by", "skipped:", "usage error:")


def tool_result_ok(text: str) -> bool:
    """Whether one tool result is an accepted outcome, for the run ledger.

    A *failed* isolated check renders as prose (``isolated hardware elaboration
    check failed``), so a prefix-only test journaled it as ``ok`` and the
    transcript, the live CLI line and the report all claimed the agent's checks
    had passed (run 20261003-130005). Only the first line is inspected, so a log
    excerpt that merely mentions a failure cannot flip a real result.
    """
    value = str(text or "").lstrip()
    if value.startswith(_TOOL_FAILURE_PREFIXES):
        return False
    head = value.splitlines()[0] if value.strip() else ""
    return "check failed" not in head


def virtualize_run_paths(text: str, run_root: str) -> str:
    """Render this run's root as ``{{run-dir}}`` in text shown to the model."""
    root = str(run_root or "").strip("/")
    if not root or root not in str(text or ""):
        return str(text or "")
    return str(text).replace(root, RUN_DIR_TOKEN)


_ACTIVE_TOOL_PERSISTENCE: contextvars.ContextVar[RunPersistence | None] = contextvars.ContextVar(
    "cfu_tool_persistence", default=None
)


def set_tool_persistence(value: RunPersistence | None):
    """Scope tool-level mutation journaling to one graph invocation."""
    return _ACTIVE_TOOL_PERSISTENCE.set(value)


def reset_tool_persistence(token) -> None:
    _ACTIVE_TOOL_PERSISTENCE.reset(token)


@contextlib.contextmanager
def _tool_operation(toolbox: "RepoToolbox", action: str, target: str):
    persisted = _ACTIVE_TOOL_PERSISTENCE.get()
    if persisted is None:
        yield
        return
    if not toolbox.run_root or toolbox.workdir / toolbox.run_root != persisted.run_root:
        raise RuntimeError("tool mutation does not belong to the active persisted run")
    digest = hashlib.sha256(target.encode()).hexdigest()[:16]
    attempt = len(persisted.journal._load()["operations"]) + 1
    with persisted.journal.external_operation(
        action, f"{action}-{attempt:08d}-{digest}",
        {"stage": toolbox.stage, "target": target[:500]},
    ):
        yield

# ANSI CSI/OSC sequences and terminal control bytes are presentation metadata,
# not useful model input. Keep tabs/newlines while removing cursor/color noise.
ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)


# Kept as a public compatibility constant for callers that inspect the old API.
# Hardware design writes live in the per-run workspace only; the tracked
# AgentCfu/Fiber scaffolds are read-only defaults.
ALLOW_WRITE_ROOTS = (
    "agents/generated",
    "sw/tests",
)

# The only files a design agent may write for hardware. They are seeded from the
# tracked scaffolds and compiled over them as a same-package overlay. The shared
# AgentCfuLib.scala platform is deliberately absent: it is compiled from the
# repository copy and cannot be overlaid, replaced or deleted by a run.
DESIGN_SOURCE_FILES = (
    "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
)
DESIGN_SCAFFOLD_ROOT = "src/main/scala/vexiiriscv/soc/mico"
DESIGN_WORKSPACE_SUBDIR = "workspace/design"
SOC_WORKSPACE_SUBDIR = "workspace/soc"
RUN_ARTIFACT_ROOT = "agents/generated/runs/"

DEFAULT_CONTEXT_FILES = (
    "skills/cfu-designer/SKILL.md",
    "skills/cfu-designer/references/acceleration-patterns.md",
    "skills/cfu-designer/references/roofline-analysis.md",
    "skills/cfu-designer/references/validation-flow.md",
    "skills/cfu-designer/references/yosys-cost-flow.md",
    "skills/cfu-designer/references/vector-rf-vpu-pattern.md",
    "src/main/scala/vexiiriscv/soc/cfu/CfuLib.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
    "src/main/scala/vexiiriscv/soc/mico/MiCoSocParam.scala",
    "src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
)

PROFILE_RE = re.compile(r"(?P<name>[A-Z0-9_]+_PROFILE)\s+(?P<body>.*)")
# Summary line printed by skills/cfu-designer/scripts/yosys_cost_report.py.
COST_RE = re.compile(
    r"cost top=(?P<top>\S+)\s+cells=(?P<cells>\d+)\s+wire_bits=(?P<wire_bits>\d+)"
    r"\s+memories=(?P<memories>\d+)\s+memory_bits=(?P<memory_bits>\d+)"
)
# Verify diagnostic printed by the generated embench_main.c (see embench.py).
VERIFY_RE = re.compile(r"EMBENCH_VERIFY_(?P<status>PASS|FAIL)\s+(?P<body>.*)")


@dataclass
class RepoToolbox:
    workdir: Path
    dry_run: bool = False
    policy: AccessPolicy | None = None
    stage: str = "load_context"
    embench_benchmark: str = ""
    run_root: str = ""
    apply_patch: bool = False

    def __post_init__(self) -> None:
        self.workdir = self.workdir.resolve()
        if self.policy is None:
            self.policy = load_policy(self.workdir)

    def expand_run_dir(self, rel_path: str | Path) -> str:
        """Translate a ``{{run-dir}}`` prefix to this run's real root.

        Raised before any policy check, so the access decision is always made on
        the real path; a token outside a run (or used mid-path) is a usage error
        the model cannot fix by rephrasing around the policy.
        """
        text = str(rel_path)
        if RUN_DIR_TOKEN not in text:
            return text
        if not self.run_root:
            raise PolicyError(
                f"{RUN_DIR_TOKEN} is only valid inside a run; use a repository-relative path"
            )
        expanded = expand_run_dir(text, self.run_root)
        if expanded == text:
            raise PolicyError(
                f"{RUN_DIR_TOKEN} must be a path prefix, for example "
                f"{RUN_DIR_TOKEN}/08_isa/cfu_isa.yaml"
            )
        return expanded

    def normalize(self, rel_path: str | Path) -> str:
        return normalize_repo_path(self.workdir, self.expand_run_dir(rel_path))

    def is_read_allowed(self, rel_path: str | Path) -> bool:
        try:
            normalized = self.normalize(rel_path)
        except PolicyError:
            return False
        return self.is_current_run_path(normalized) and self.policy.can_read(normalized)

    def is_current_run_path(self, normalized: str) -> bool:
        """Run artifacts belong to the executing run, and this fails closed.

        The shared read tree exposes the whole `agents/generated/runs/**` area,
        so a toolbox without a run root would otherwise see every historical
        run. Everything outside the runs area is unaffected.
        """
        if not normalized.startswith(RUN_ARTIFACT_ROOT):
            return True
        if not self.run_root:
            return False
        root = self.run_root.strip("/")
        return normalized == root or normalized.startswith(root + "/")

    def visible_subtree(self, rel_path: str) -> bool:
        """True when a readable file may live at or below ``rel_path``.

        Directory navigation uses this instead of ``is_read_allowed``: a
        directory is never itself in the read tree, so asking whether it is
        readable would hide every directory-only tree.
        """
        raw = str(rel_path).strip()
        if raw in ("", ".", "./"):
            return True
        try:
            normalized = self.normalize(raw)
        except PolicyError:
            return False
        if not self.is_current_run_path(normalized):
            return False
        return self.policy.may_contain_readable(normalized)

    def resolve(self, rel_path: str | Path) -> Path:
        return (self.workdir / self.normalize(rel_path)).resolve()

    def is_write_allowed(self, rel_path: str | Path) -> bool:
        try:
            normalized = self.normalize(rel_path)
            self.policy.check_write(self.stage, normalized)
        except PolicyError:
            return False
        return True

    def read_text(self, rel_path: str, *, limit: int = MAX_FILE_CHARS) -> str:
        normalized = self.normalize(rel_path)
        if not self.is_current_run_path(normalized):
            raise PolicyError(f"read path belongs to another run: {normalized}")
        self.policy.check_read(normalized)
        path = self.workdir / normalized
        if not path.exists() or not path.is_file():
            return ""
        if limit < 0:
            raise ValueError("read limit must be non-negative")
        with path.open(encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)

    def write_text(self, rel_path: str, text: str) -> str:
        normalized = self.normalize(rel_path)
        self.policy.check_write(self.stage, normalized)
        path = self.workdir / normalized
        if self.dry_run and not is_dry_run_artifact(normalized):
            return f"dry-run: would write {normalized}"
        with _tool_operation(self, "write_text", normalized):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return f"wrote {normalized}"

    def discover_context(self, extra_files: Iterable[str] = ()) -> RepoContext:
        files: dict[str, str] = {}
        for rel_path in [*DEFAULT_CONTEXT_FILES, *extra_files]:
            text = self.read_text(rel_path)
            if text:
                files[rel_path] = text

        return {
            "files": files,
            "cfu_entrypoints": self.find_paths("src/main/scala/vexiiriscv/soc/mico", "*Cfu.scala"),
            "software_tests": self.find_paths("sw/tests", "*cfu*"),
        }

    def find_paths(self, rel_root: str, pattern: str) -> list[str]:
        root = self.resolve(rel_root)
        if not root.exists():
            return []
        paths: list[str] = []
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(self.workdir))
            if self.is_read_allowed(rel_path):
                paths.append(rel_path)
        return sorted(paths)

    def command_allowed(self, args: list[str], *, command_id: str | None = None) -> bool:
        if not args:
            return False
        try:
            selected = command_id or infer_command_id(args, self.policy)
            self.policy.check_command(self.stage, selected, args, self.workdir)
        except (PolicyError, ValueError):
            return False
        return True

    def run_command(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        timeout: int = 1800,
        mutate: bool = False,
        command_id: str | None = None,
        execute: bool = True,
        capture_full: bool = False,
    ) -> CommandResult:
        selected = command_id or infer_command_id(args, self.policy)
        self.policy.check_command(self.stage, selected, args, self.workdir)
        if (mutate and self.dry_run) or not execute:
            return {
                "command": " ".join(args),
                "cwd": cwd or str(self.workdir),
                "returncode": None,
                "skipped": True,
                "reason": "dry-run" if self.dry_run else "execution-disabled",
            }

        run_cwd = self.resolve(cwd) if cwd else self.workdir
        # Legacy policy descriptors include --unsafe-paths; validate that
        # descriptor unchanged, but never grant Git the unsafe execution flag.
        execution_args = [arg for arg in args if arg != "--unsafe-paths"] if selected in {
            "git_apply", "git_apply_check"
        } else args
        with _tool_operation(self, "command", " ".join(execution_args)):
            result = subprocess.run(
                execution_args,
                cwd=run_cwd,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        stdout = filter_output(result.stdout)
        stderr = filter_output(result.stderr)
        command_result: CommandResult = {
            "command": " ".join(args),
            "cwd": str(run_cwd),
            "returncode": result.returncode,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "parsed": parse_command_output(stdout + "\n" + stderr),
        }
        if capture_full:
            # Machine-readable domain verdicts are extracted before output tails
            # are truncated, so a long simulator epilogue cannot hide them.
            from ..benchmark.embench import parse_profile
            from ..conformance import parse_conformance

            command_result["parsed"]["embench_profile"] = parse_profile(stdout + "\n" + stderr)
            command_result["parsed"]["conformance"] = parse_conformance(stdout + "\n" + stderr)
            # Private transport fields: graph._run_check persists these as a
            # run-scoped artifact and removes them before state serialization.
            command_result["_stdout_full"] = stdout  # type: ignore[typeddict-unknown-key]
            command_result["_stderr_full"] = stderr  # type: ignore[typeddict-unknown-key]
        return command_result

    def apply_patch_text(self, patch_text: str) -> CommandResult:
        # Redirect tracked scaffold targets into the run workspace before any
        # git operation, so the repo tree is never touched.
        targets = validate_text_patch(patch_text)
        redirects = [self.resolve_target(path) for path in targets]
        mapping = dict(zip(targets, redirects))
        patch_text = rewrite_text_patch(patch_text, mapping)
        rejected: list[str] = []
        if rejected:
            allowed = "AgentCfu.scala, AgentCfuFiber.scala"
            if self.embench_benchmark:
                allowed += f", benchmarks/embench-iot/src/{self.embench_benchmark}/*.c"
            raise ValueError(
                "patch touches source paths outside the writable workspace: "
                + ", ".join(sorted(set(rejected)))
                + f"; only {allowed} may be designed"
            )

        touched = patch_paths(patch_text)
        normalized: list[str] = []
        for path in touched:
            normalized.append(self.normalize(path))
            if not self.is_write_allowed(path):
                raise ValueError(f"patch touches non-allowlisted path: {path}")
        check = self.run_command(
            ["git", "apply", "--check", "-", "--unsafe-paths"],
            input_text=patch_text,
            command_id="git_apply_check",
            mutate=False,
        )
        if check["returncode"] != 0 or not self.apply_patch:
            if not self.apply_patch and check["returncode"] == 0:
                check["skipped"] = True
                check["reason"] = "patch-application-disabled"
            check["parsed"] = {
                **check.get("parsed", {}),
                "touched_paths": normalized,
                "redirected_paths": redirects,
            }
            return check
        result = self.run_command(
            ["git", "apply", "-", "--unsafe-paths"],
            input_text=patch_text,
            command_id="git_apply",
            mutate=True,
        )
        result["parsed"] = {
            **result.get("parsed", {}),
            "touched_paths": normalized,
            "redirected_paths": redirects,
        }
        return result

    def seed_design_workspace(self) -> list[str]:
        """Copy the tracked scaffolds into this run's design workspace.

        Returns the repo-relative overlay paths that were seeded (or would be
        seeded in dry-run mode).
        """
        if not self.run_root:
            raise PolicyError("cannot seed a design workspace without a run root")
        seeded: list[str] = []
        for source, overlay in design_workspace_paths(self.run_root).items():
            text = self.read_text(source, limit=1_000_000)
            if self.dry_run:
                seeded.append(overlay)
                continue
            self.write_text(overlay, text)
            seeded.append(overlay)
        if not self.dry_run:
            # MiCoSocGen runs with this directory as its working directory, so
            # it must exist before the forked JVM starts.
            soc_dir = self.workdir / self.run_root / SOC_WORKSPACE_SUBDIR
            soc_dir.mkdir(parents=True, exist_ok=True)
        return seeded

    def resolve_target(self, target: str) -> str:
        """Writable workspace path for an LLM-named repo path, or raise."""
        if not self.run_root:
            raise ValueError("cannot resolve a target without a run root")
        target = self.expand_run_dir(target)
        resolved = redirect_target(target, self.run_root, self.embench_benchmark)
        if not resolved and self._is_run_workspace_target(target):
            resolved = self.normalize(target)
        if not resolved:
            raise ValueError(f"path is not a writable design target: {target}")
        resolved = self.normalize(resolved)
        if not resolved.startswith(f"{self.run_root.strip('/')}/workspace/") or not self.is_write_allowed(resolved):
            raise ValueError(f"path is not a writable design target: {target}")
        return resolved

    def _is_run_workspace_target(self, target: str) -> bool:
        """True when target already names a writable file in this run."""
        prefix = f"{self.run_root.strip('/')}/"
        if not self.run_root or not target.startswith(prefix):
            return False
        try:
            return self.is_write_allowed(target)
        except PolicyError:
            return False

    def apply_edit_blocks(
        self, edits: list[tuple[str, list[tuple[str, str]]]]
    ) -> CommandResult:
        """Apply SEARCH/REPLACE edits to run-workspace files.

        Several ``EDIT_BEGIN`` sections may target the same file; they are
        applied in response order to the evolving content, so a response can
        split a large change into consecutive sections. Every section goes
        through the same redirect and write policy as a whole-file replacement,
        and all sections are validated before any file is written, so one
        rejected section cannot leave the workspace half-edited.

        On success the result carries a compact unified diff of every touched
        file (capped at EDIT_DIFF_MAX_CHARS) so the model sees what changed
        without re-reading. A response whose sections only re-apply already
        present content succeeds with a note and touched_paths == [] instead of
        erroring, so replayed deliveries do not churn the repair loop.
        """
        # Resolve targets first so diagnostics can report "section i/n".
        resolved_targets: list[str | None] = []
        section_totals: dict[str, int] = {}
        for target, _pairs in edits:
            try:
                resolved = self.resolve_target(target)
            except ValueError:
                resolved_targets.append(None)
                continue
            resolved_targets.append(resolved)
            section_totals[resolved] = section_totals.get(resolved, 0) + 1

        pending: dict[str, str] = {}
        originals: dict[str, str] = {}
        section_seen: dict[str, int] = {}
        notes: list[str] = []
        errors: list[str] = []
        for (target, pairs), resolved in zip(edits, resolved_targets):
            if resolved is None:
                try:
                    self.resolve_target(target)
                except ValueError as exc:
                    errors.append(f"{target}: {exc}")
                continue
            section_seen[resolved] = section_seen.get(resolved, 0) + 1
            section = section_seen[resolved]
            total = section_totals[resolved]
            label = f"section {section}/{total}" if total > 1 else "edit"
            if not self.is_write_allowed(resolved):
                errors.append(f"{target}: path is not writable")
                continue
            try:
                content = pending.get(resolved)
                if content is None:
                    content = self.read_text(resolved, limit=2_000_000)
                    originals[resolved] = content
            except PolicyError as exc:
                errors.append(f"{target}: cannot read the current content ({exc})")
                continue
            if not content:
                errors.append(
                    f"{target}: no existing content to edit; use FILE_BEGIN for a new file"
                )
                continue
            updated, problems = apply_edit_pairs(content, pairs)
            if problems:
                errors.extend(f"{resolved}: {label}: {problem}" for problem in problems)
                continue
            if updated == content:
                # A no-op pair (SEARCH == REPLACE, or an already-applied replay)
                # is success-with-note, not an error: echo/replayed deliveries
                # must not churn the repair loop. Unmatched SEARCH is still an
                # error above: there the model clearly intended a change.
                notes.append(f"{resolved}: {label}: edit was already applied (no content change)")
                continue
            pending[resolved] = updated

        if errors:
            return CommandResult(
                command="edit",
                cwd=str(self.workdir),
                returncode=1,
                stdout_tail="",
                stderr_tail="; ".join(errors)[:2000],
                parsed={"touched_paths": []},
            )
        if not edits:
            return CommandResult(
                command="edit",
                cwd=str(self.workdir),
                returncode=1,
                stdout_tail="",
                stderr_tail="no edit blocks were supplied",
                parsed={"touched_paths": []},
            )
        if not pending:
            # Every section was a no-op: report success with the note.
            summary_note = "edit was already applied (no content change)"
            return CommandResult(
                command="edit",
                cwd=str(self.workdir),
                returncode=0,
                stdout_tail="\n".join([summary_note, *notes]),
                stderr_tail="",
                parsed={"touched_paths": []},
            )

        try:
            with _tool_operation(self, "edit_files", ",".join(sorted(pending))):
                self._commit_files(pending)
        except OSError as exc:
            return CommandResult(command="edit", cwd=str(self.workdir), returncode=1,
                                 stdout_tail="", stderr_tail=str(exc), parsed={"touched_paths": []})
        applied = list(pending)
        diff_text = cap_change_diffs("".join(
            render_change_diff(originals.get(name, ""), pending[name], name) for name in applied
        ))
        return CommandResult(
            command="edit",
            cwd=str(self.workdir),
            returncode=0,
            stdout_tail="\n".join(part for part in [*notes, diff_text] if part),
            stderr_tail="",
            parsed={"touched_paths": applied},
        )

    def _commit_files(self, pending: dict[str, str]) -> None:
        """Rollback on in-process I/O failure; not a crash-atomic transaction."""
        before = {name: ((self.workdir / name).read_bytes() if (self.workdir / name).exists() else None)
                  for name in pending}
        attempted: list[str] = []
        try:
            for name, text in pending.items():
                attempted.append(name)
                self.write_text(name, text)
        except OSError:
            for name in reversed(attempted):
                path = self.workdir / name
                if before[name] is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(before[name])
            raise

    def apply_file_blocks(self, blocks: list[tuple[str, str]]) -> CommandResult:
        """Prevalidate every FILE target, then commit with in-process rollback.

        Like apply_edit_blocks, the result echoes a compact unified diff of the
        touched files so the model can see the replacement it delivered.
        """
        pending: dict[str, str] = {}
        originals: dict[str, str | None] = {}
        try:
            for target, content in blocks:
                resolved = self.resolve_target(target)
                self.policy.check_write(self.stage, resolved)
                if resolved in pending:
                    raise ValueError(f"duplicate FILE target: {target}")
                existing: str | None = ""
                if (self.workdir / resolved).is_file():
                    try:
                        existing = self.read_text(resolved, limit=2_000_000)
                    except PolicyError:
                        existing = None  # unreadable: skip the diff, do not fake one
                originals[resolved] = existing
                pending[resolved] = content
            if not pending:
                raise ValueError("no file blocks supplied")
            with _tool_operation(self, "file_blocks", ",".join(sorted(pending))):
                self._commit_files(pending)
        except (ValueError, PolicyError, OSError) as exc:
            return CommandResult(command="file", cwd=str(self.workdir), returncode=1,
                                 stdout_tail="", stderr_tail=str(exc), parsed={"touched_paths": []})
        diff_text = cap_change_diffs("".join(
            render_change_diff(before or "", pending[name], name)
            for name, before in originals.items()
            if before is not None
        ))
        return CommandResult(command="file", cwd=str(self.workdir), returncode=0,
                             stdout_tail=diff_text, stderr_tail="",
                             parsed={"touched_paths": list(pending)})

    def stage_bucket(self, stage: str) -> "RepoToolbox":
        return RepoToolbox(
            self.workdir,
            dry_run=self.dry_run,
            policy=self.policy,
            stage=stage,
            run_root=self.run_root,
            apply_patch=self.apply_patch,
        )

    def copy_tree(self, src_rel: str, dst_rel: str) -> str:
        for raw in (src_rel, dst_rel):
            candidate = self.workdir / raw
            if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents
                                              if parent != self.workdir):
                raise PolicyError("copy does not permit symbolic links")
        src_normalized = self.normalize(src_rel)
        dst_normalized = self.normalize(dst_rel)
        self.policy.check_read(src_normalized)
        self.policy.check_write(self.stage, dst_normalized)
        src = self.workdir / src_normalized
        dst = self.workdir / dst_normalized
        if self.dry_run:
            return f"dry-run: would copy {src_rel} to {dst_rel}"
        ignore = shutil.ignore_patterns(
            "build*",
            "target",
            ".git",
            "__pycache__",
            "*.elf",
            "*.asm",
            "*.o",
            "*.vcd",
            "*.fst",
        )
        # Validate *every* source before touching the destination. Never follow
        # aliases: normal-looking symlink names could launder denied contents.
        files: list[tuple[Path, Path]] = []
        stack = [(src, dst)]
        visited = 0
        while stack:
            source, target = stack.pop()
            visited += 1
            if visited > 10000:
                raise PolicyError("copy enumeration budget exceeded")
            if source.is_symlink() or target.is_symlink():
                raise PolicyError("copy does not permit symbolic links")
            source_rel = self.normalize(source.relative_to(self.workdir))
            target_rel = self.normalize(target.relative_to(self.workdir))
            self.policy.check_read(source_rel)
            self.policy.check_write(self.stage, target_rel)
            if source.is_dir():
                with os.scandir(source) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            raise PolicyError("copy does not permit symbolic links")
                        if ignore(str(source), [entry.name]):
                            continue
                        stack.append((Path(entry.path), target / entry.name))
                        if len(stack) + visited > 10000:
                            raise PolicyError("copy enumeration budget exceeded")
            elif source.is_file():
                files.append((source, target))
            else:
                raise PolicyError("copy source is not a regular file or directory")
        with _tool_operation(self, "copy_tree", dst_rel):
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                dst.mkdir(parents=True, exist_ok=True)
            for source, target in files:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return f"copied {src_rel} to {dst_rel}"


def design_workspace_rel(run_root: str, name: str) -> str:
    """Repo-relative path of a design file inside the run workspace."""
    return f"{run_root.strip('/')}/{DESIGN_WORKSPACE_SUBDIR}/{name}"


def design_workspace_paths(run_root: str) -> dict[str, str]:
    """Map each tracked scaffold to its run-workspace overlay path."""
    return {
        source: design_workspace_rel(run_root, Path(source).name)
        for source in DESIGN_SOURCE_FILES
    }


def design_scaffold_name(rel_path: str) -> str:
    """Return the AgentCfu*.scala basename if rel_path points at a design file."""
    name = Path(rel_path).name
    if name in {Path(source).name for source in DESIGN_SOURCE_FILES}:
        return name
    return ""


EMBENCH_SOURCE_ROOT = "benchmarks/embench-iot/src"


def embench_cfu_rel(run_root: str, benchmark: str, filename: str) -> str:
    """Repo-relative path of an Embench source inside the CFU configuration."""
    return f"{run_root.strip('/')}/workspace/embench_{benchmark}/cfu/src/{filename}"


def embench_source_target(target: str, benchmark: str) -> str:
    """Basename if target names a source of this run's Embench benchmark."""
    prefix = f"{EMBENCH_SOURCE_ROOT}/{benchmark}/"
    if not target.startswith(prefix):
        return ""
    return Path(target).name


def redirect_target(target: str, run_root: str, embench_benchmark: str = "") -> str:
    """Map an LLM-named repo path to its writable workspace path.

    Returns "" when the path is not a redirectable source file. Callers treat
    that as "not allowed to write here".
    """
    embench_file = embench_source_target(target, embench_benchmark)
    if embench_file:
        return embench_cfu_rel(run_root, embench_benchmark, embench_file)
    name = design_scaffold_name(target)
    if name and target in DESIGN_SOURCE_FILES:
        return design_workspace_rel(run_root, name)
    return ""


def strip_code_fences(text: str) -> str:
    """Drop a markdown code fence wrapping an LLM-emitted file body.

    Models routinely wrap whole-file replacements in ```scala / ``` fences even
    when told not to. Those fences are the first lines of the file and make the
    compiler fail (``empty quoted identifier`` for Scala), so they are removed
    before the content is written.
    """
    lines = text.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip() in {"```", "~~~"}:
        lines.pop()
    return "\n".join(lines) + "\n"


# A degenerate sampler occasionally restarts its answer template inside the same
# completion, so the whole response appears twice with the second copy beginning
# immediately after the terminator (``EDIT_ENDREPAIR_DIAGNOSIS ...``, observed
# 2026-09-26). A terminator is only recognized on a line of its own, so the
# glued copy made the parser report "missing EDIT_END" and drop an otherwise
# correct edit. Split the glue before parsing; ``_`` continuations stay content
# (``EDIT_END_foo`` reads like an identifier, not a terminator run-over).
_GLUED_TERMINATOR = re.compile(r"(?m)^(EDIT_END|FILE_END)(?=[^\s_])")


def normalize_delivery_terminators(text: str) -> str:
    """Split a delivery terminator glued to the next token onto its own line.

    Applied only where delivery blocks are parsed, so the audit copy of the
    model's bytes stays untouched. ``PATCH_END`` needs no handling: the PATCH
    parser matches its marker anywhere in the text.
    """
    return _GLUED_TERMINATOR.sub(r"\1\n", text or "")


def extract_file_blocks(text: str) -> list[tuple[str, str]]:
    """Parse whole-file replacements from an LLM response.

    LLM-generated unified diffs are fragile (a malformed hunk header makes
    git apply reject the whole patch). For small target files it is far more
    reliable to ask for the complete new content:

        FILE_BEGIN <repo-relative path>
        ...content...
        FILE_END

    Returns [(path, content)] in response order. Byte-identical duplicates (a
    restarted answer template emitted the same block twice) collapse to the
    first copy, so the second copy cannot fail on already-changed content.
    """
    blocks: list[tuple[str, str]] = []
    current_path: str | None = None
    current: list[str] = []
    for line in normalize_delivery_terminators(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("FILE_BEGIN"):
            header = stripped[len("FILE_BEGIN"):].strip()
            current_path = header
            current = []
            continue
        if stripped == "FILE_END":
            if current_path:
                block = (current_path, strip_code_fences("\n".join(current)))
                if block not in blocks:
                    blocks.append(block)
            current_path = None
            current = []
            continue
        if current_path is not None:
            current.append(line)
    return blocks


SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"

# A parse failure inside one EDIT_BEGIN section is carried as this sentinel pair
# (message in the replace slot) instead of raising out of extract_edit_blocks:
# graph.py calls the parser outside its try/except, so an escaping exception
# would kill the run instead of becoming repairable feedback. apply_edit_pairs
# turns the sentinel back into the stored error message.
EDIT_PARSE_ERROR = "<<<<<<< PARSE_ERROR"

# Delivery/conflict-marker shape: 5+ '<', '>' or '=' at line start. Real content
# with these shapes (git conflict regions, comment dividers) collides with the
# edit protocol: `=======` inside a SEARCH region silently flips the parser into
# replace mode (verified 2026-09-28 audit). Such lines are rejected with an
# actionable message; the model must edit around them.
_MARKER_LINE = re.compile(r"^(?:<{5,}|>{5,}|={5,})")


def _collision_message(stripped: str) -> str:
    return (
        f"conflict-marker-like line {stripped!r} inside a SEARCH/REPLACE block "
        "collides with the edit protocol markers; lines starting with <<<<<<< / "
        ">>>>>>> or made of ======= cannot be delivered as content. Shrink the "
        "SEARCH/REPLACE block so the marker-like line is not part of it and edit "
        "the lines around it instead"
    )


def _trim_blank_edges(lines: list[str]) -> str:
    """Join lines, dropping blank lines that only pad the block markers."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _parse_edit_pairs(block_lines: list[str], *, target: str) -> list[tuple[str, str]]:
    """Parse one EDIT_BEGIN body into [(search, replace)] pairs.

    Lines shaped like git conflict markers (``<<<<<<<``/``>>>>>>>`` heads or a
    ``=======`` run) cannot appear as SEARCH/REPLACE content: they are either
    protocol markers or would silently flip the parser into the wrong mode, so
    they are rejected with an actionable message instead of mis-parsing.
    """
    pairs: list[tuple[str, str]] = []
    mode: str | None = None
    search: list[str] = []
    replace: list[str] = []
    for line in block_lines:
        stripped = line.strip()
        marker_like = bool(_MARKER_LINE.match(stripped))
        if stripped.startswith(SEARCH_MARKER) and mode is not None:
            raise ValueError(f"{target}: nested SEARCH marker")
        if mode is None:
            if stripped.startswith(SEARCH_MARKER):
                mode = "search"
                search = []
                replace = []
            elif marker_like:
                raise ValueError(f"{target}: {_collision_message(stripped)}")
        elif mode == "search":
            if stripped.startswith(DIVIDER_MARKER):
                mode = "replace"
            elif marker_like:
                raise ValueError(f"{target}: {_collision_message(stripped)}")
            else:
                search.append(line)
        else:
            if stripped.startswith(REPLACE_MARKER):
                pairs.append((_trim_blank_edges(search), _trim_blank_edges(replace)))
                mode = None
            elif marker_like:
                raise ValueError(f"{target}: {_collision_message(stripped)}")
            else:
                replace.append(line)
    if mode is not None:
        raise ValueError(f"{target}: unterminated SEARCH/REPLACE block")
    return pairs


def _is_edit_end(stripped: str) -> bool:
    """True for a section terminator, tolerating trailing junk.

    Samplers occasionally append a stray token to the terminator (observed:
    ``EDIT_END સલ``). Requiring an exact match discarded an otherwise valid edit
    and the rung reported only "no deliverable", so the model repeated it.
    """
    return bool(re.match(r"^EDIT_END\b", stripped))


def extract_edit_blocks(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Parse targeted SEARCH/REPLACE edit sections from an LLM response.

    Whole-file replacement is reliable but forces the model to re-emit every
    line, which is expensive and invites unrelated rewrites. Unified diffs are
    cheap but need exact line numbers. A search/replace edit needs neither:

        EDIT_BEGIN <repo-relative path>
        <<<<<<< SEARCH
        ...existing lines...
        =======
        ...replacement lines...
        >>>>>>> REPLACE
        EDIT_END

    One section may hold several pairs, and a response may hold several
    sections. Returns [(path, [(search, replace), ...])] in response order.
    Byte-identical duplicate sections (a restarted answer template emitted the
    same edit twice) collapse to the first copy: after the first applies, the
    duplicate's SEARCH no longer matches and would reject the whole response.
    A malformed section (nested marker, conflict-marker content, unterminated
    SEARCH/REPLACE) becomes one ``EDIT_PARSE_ERROR`` sentinel pair so the
    parser never raises into the run: the apply step reports the stored message
    as ordinary repairable feedback.
    """
    blocks: list[tuple[str, list[tuple[str, str]]]] = []
    current_path: str | None = None
    current: list[str] = []
    for line in normalize_delivery_terminators(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("EDIT_BEGIN"):
            current_path = stripped[len("EDIT_BEGIN"):].strip()
            current = []
            continue
        if _is_edit_end(stripped):
            if current_path:
                try:
                    pairs = _parse_edit_pairs(current, target=current_path)
                except ValueError as exc:
                    pairs = [(EDIT_PARSE_ERROR, str(exc))]
                block = (current_path, pairs)
                if block not in blocks:
                    blocks.append(block)
            current_path = None
            current = []
            continue
        if current_path is not None:
            current.append(line)
    return blocks


def unterminated_edit_blocks(text: str) -> int:
    """Number of EDIT_BEGIN sections that never reached a terminator.

    Used to tell "the model sent no edit at all" apart from "the edit was
    malformed", which need different repair feedback. Counted from the markers
    rather than the parsed blocks, so a legitimately duplicated section (which
    extract_edit_blocks collapses) is not misreported as unterminated.
    """
    text = normalize_delivery_terminators(text)
    lines = [line.strip() for line in (text or "").splitlines()]
    begins = sum(1 for line in lines if line.startswith("EDIT_BEGIN"))
    ends = sum(1 for line in lines if _is_edit_end(line))
    return max(0, begins - ends)


def _match_window(content_lines: list[str], search_lines: list[str]) -> list[int]:
    """Start indices where search_lines matches, ignoring trailing whitespace."""
    target = [line.rstrip() for line in search_lines]
    width = len(search_lines)
    hits: list[int] = []
    for index in range(len(content_lines) - width + 1):
        window = [line.rstrip() for line in content_lines[index:index + width]]
        if window == target:
            hits.append(index)
            if len(hits) > 1:
                break
    return hits


def _locate_closest(
    content_lines: list[str], search_lines: list[str]
) -> tuple[int, int, str, str] | None:
    """Closest existing region and its first differing line.

    Returns ``(search_line_no, content_line_no, expected, actual)`` (1-based) or
    None when the SEARCH anchor does not appear at all. Used to tell the model
    exactly which line it mis-copied instead of only "not found".
    """
    stripped_search = [line.rstrip() for line in search_lines]
    stripped_content = [line.rstrip() for line in content_lines]
    anchor_index = next((i for i, line in enumerate(search_lines) if line.strip()), 0)
    anchor = search_lines[anchor_index]
    candidates = [i for i, line in enumerate(stripped_content) if line == anchor]
    if not candidates and anchor.strip():
        candidates = [
            i for i, line in enumerate(stripped_content) if line.strip() == anchor.strip()
        ]
    if not candidates:
        return None
    best: tuple[int, int, str, str] | None = None
    for start in candidates:
        offset = start - anchor_index
        for index, expected in enumerate(search_lines):
            content_index = offset + index
            if content_index < 0 or content_index >= len(content_lines):
                actual = "<past end of file>"
                difference = (index + 1, content_index + 1, expected, actual)
                break
            if stripped_content[content_index] != stripped_search[index]:
                difference = (index + 1, content_index + 1, expected, content_lines[content_index])
                break
        else:
            continue
        # Prefer the region that matches the longest prefix of the SEARCH block.
        if best is None or difference[0] > best[0]:
            best = difference
    return best


def _describe_mismatch(content_lines: list[str], search_lines: list[str]) -> str:
    """Actionable description of why a SEARCH block did not match."""
    located = _locate_closest(content_lines, search_lines)
    if located is None:
        return (
            "The first SEARCH line does not appear in the current file either; re-read the "
            "target content shown in the prompt and rebuild the edit from it."
        )
    search_no, content_no, expected, actual = located
    return (
        f"First difference: SEARCH line {search_no} of {len(search_lines)} vs current file "
        f"line {content_no}.\n"
        f"  your SEARCH:  {expected!r}\n"
        f"  current file: {actual!r}\n"
        "Copy the current file text verbatim (comments included), or shrink the SEARCH to the "
        "few unique code lines you actually change."
    )


def apply_edit_pairs(content: str, pairs: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Apply search/replace pairs to ``content``; return (new_content, errors).

    Each SEARCH block must match exactly one location. Zero matches means the
    model's copied context was wrong; several matches means the block is not
    specific enough. Either way the file is left unchanged, so a partial edit
    can never be written.
    """
    errors: list[str] = []
    total = len(pairs)
    original_lines = content.split("\n")
    for index, (search, replace) in enumerate(pairs, start=1):
        if search == EDIT_PARSE_ERROR:
            # Carried parse failure from extract_edit_blocks: surface the stored
            # message as ordinary repairable feedback instead of mis-applying.
            errors.append(replace)
            break
        if not search.strip():
            errors.append(f"edit {index}/{total}: SEARCH block is empty")
            break
        search_lines = search.split("\n")
        content_lines = content.split("\n")
        hits = _match_window(content_lines, search_lines)
        if not hits:
            if _match_window(original_lines, search_lines):
                errors.append(
                    f"edit {index}/{total}: SEARCH block no longer matches because an earlier "
                    "section or pair changed that region; pairs apply in order, so search for the "
                    "text as it is now"
                )
            else:
                errors.append(
                    f"edit {index}/{total}: SEARCH block not found. "
                    + _describe_mismatch(content_lines, search_lines)
                )
            break
        if len(hits) > 1:
            errors.append(
                f"edit {index}/{total}: SEARCH block is ambiguous (matched more than once); "
                "add surrounding context"
            )
            break
        start = hits[0]
        # An empty replacement deletes the matched lines; splitting "" would
        # yield [""] and leave a stray blank line behind.
        replacement = replace.split("\n") if replace else []
        content_lines[start:start + len(search_lines)] = replacement
        content = "\n".join(content_lines)
    return content, errors


def repair_patch_paths(
    patch_text: str,
    run_root: str,
    *,
    embench_benchmark: str = "",
) -> tuple[str, list[str], list[str]]:
    """Redirect patch targets into the run workspace.

    An LLM naturally writes repository-relative paths for tracked files. Those
    must never be applied to the repo, so each such target is rewritten to its
    workspace copy:

    * AgentCfu.scala / AgentCfuFiber.scala -> the design workspace overlay
    * benchmarks/embench-iot/src/<bench>/<file>.c -> that run's cfu config tree

    Any other source path is rejected.

    Returns (rewritten patch, redirected targets, rejected targets).
    """
    redirected: list[str] = []
    rejected: list[str] = []
    lines: list[str] = []
    for line in patch_text.splitlines(keepends=True):
        body = line.rstrip("\n")
        newline = line[len(body):]
        replaced = body
        for prefix in ("--- a/", "+++ b/"):
            if body.startswith(prefix):
                target = expand_run_dir(body[len(prefix):], run_root)
                replacement = redirect_target(target, run_root, embench_benchmark)
                if replacement:
                    replaced = f"{prefix}{replacement}"
                    if replacement not in redirected:
                        redirected.append(replacement)
                elif target.startswith("src/") or target.startswith(EMBENCH_SOURCE_ROOT):
                    rejected.append(target)
                elif design_scaffold_name(target):
                    # An absolute/out-of-tree path naming a design file is never valid.
                    rejected.append(target)
        lines.append(replaced + newline)
    return "".join(lines), redirected, rejected


def infer_command_id(args: list[str], policy: AccessPolicy) -> str:
    """Return the best-matching declared command id for argv.

    Every matching declaration is considered and the most specific one wins
    (highest priority, then declaration order). Without this, a looser variant
    declared earlier would silently shadow a stricter one.
    """
    matches: list[tuple[int, int, str]] = []
    for index, (name, descriptor) in enumerate(policy.commands.items()):
        if not args or args[0] != descriptor.executable:
            continue
        if descriptor.argv_prefix and tuple(args[: len(descriptor.argv_prefix)]) != descriptor.argv_prefix:
            continue
        if descriptor.required_substring and descriptor.required_substring not in " ".join(args):
            continue
        if any(token not in args for token in descriptor.required_tokens):
            continue
        if any(needle not in " ".join(args) for needle in descriptor.required_substrings):
            continue
        matches.append((-descriptor.priority, index, name))
    if not matches:
        raise PolicyError(f"no declared command matches argv: {args}")
    matches.sort()
    return matches[0][2]


def filter_output(text: str) -> str:
    """Remove terminal formatting/control noise before output reaches the agent."""
    if not text:
        return ""
    text = ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\b", "")
    return "".join(
        char for char in text
        if char in "\n\t" or (ord(char) >= 0x20 and ord(char) != 0x7F)
    )


def tail(text: str, *, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


# The applied change is echoed back with every successful delivery so the model
# sees what its edit actually did instead of re-reading the whole file (run
# 2026-09-28: "Edit result: {'returncode': 0 ...}" forced a full re-read before
# the next repair could trust its own earlier change).
EDIT_DIFF_MAX_CHARS = 4000


def render_change_diff(before: str, after: str, rel_path: str) -> str:
    """Compact unified diff of one before -> after file change ("" when equal)."""
    if before == after:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=2,
        )
    )


def cap_change_diffs(diff_text: str, max_chars: int = EDIT_DIFF_MAX_CHARS) -> str:
    """Cap the echoed diff, cutting at a line boundary and saying so."""
    if len(diff_text) <= max_chars:
        return diff_text
    cut = diff_text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars // 2:
        cut = cut[: newline + 1]
    return cut + f"...[diff truncated at {max_chars} chars]"


def parse_command_output(text: str) -> dict[str, object]:
    text = filter_output(text)
    parsed: dict[str, object] = {}
    profiles: list[dict[str, object]] = []
    for match in PROFILE_RE.finditer(text):
        entry: dict[str, object] = {"name": match.group("name")}
        for item in match.group("body").split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                entry[key] = int(value)
            except ValueError:
                entry[key] = value
        profiles.append(entry)
    if profiles:
        parsed["profiles"] = profiles
    cost = COST_RE.search(text)
    if cost:
        parsed["cost"] = {
            "top": cost.group("top"),
            "cells": int(cost.group("cells")),
            "wire_bits": int(cost.group("wire_bits")),
            "memories": int(cost.group("memories")),
            "memory_bits": int(cost.group("memory_bits")),
        }
    verify = VERIFY_RE.search(text)
    if verify:
        entry: dict[str, object] = {"status": verify.group("status").lower()}
        for item in verify.group("body").split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                entry[key] = int(value)
            except ValueError:
                entry[key] = value
        parsed["verify"] = entry
    if " PASS " in text or text.strip().endswith(" PASS"):
        parsed["pass"] = True
    if "FAILURE" in text or " FAIL" in text:
        parsed["fail"] = True
    return parsed


def validate_text_patch(text: str) -> list[str]:
    """Accept only unambiguous regular-file unified diffs (no Git side effects).

    Each hunk header's declared line counts are checked against the hunk body
    (context+deleted = old count, context+added = new count) in the one
    direction git itself rejects: a body SHORTER than the header declares dies
    later inside ``git apply`` with "corrupt patch at line N" and wastes a whole
    turn, so it is rejected here with a precise message. A body longer than the
    header declares is accepted, matching git (verified empirically: ``git
    apply --check`` returns 0 for such a patch).
    """
    targets: list[str] = []
    git_pair = None
    old = None
    new = None
    has_hunk = False
    in_hunk = False
    hunk_index = 0
    declared_old = None
    declared_new = None
    seen_old = 0
    seen_new = 0

    def path(value: str, prefix: str) -> str:
        if value == "/dev/null":
            return value
        if not value.startswith(prefix) or re.search(r'[\s"\\\\]', value):
            raise ValueError("unsupported quoted or ambiguous patch path")
        value = value[len(prefix):]
        if not value or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/")):
            raise ValueError("unsafe patch path")
        return value

    def close_hunk() -> None:
        nonlocal declared_old, declared_new
        if declared_old is None:
            return
        target = new if old == "/dev/null" else old
        if seen_old < declared_old:
            raise ValueError(
                f"{target}: hunk {hunk_index}: header says {declared_old} old lines "
                f"but the body has {seen_old} (context+deleted)"
            )
        if seen_new < declared_new:
            raise ValueError(
                f"{target}: hunk {hunk_index}: header says {declared_new} new lines "
                f"but the body has {seen_new} (context+added)"
            )
        declared_old = declared_new = None

    def finish() -> None:
        nonlocal hunk_index
        close_hunk()
        hunk_index = 0
        if old is None and new is None and git_pair is None:
            return
        if old is None or new is None or not has_hunk:
            raise ValueError("patch requires paired file headers and a text hunk")
        target = new if old == "/dev/null" else old
        if target == "/dev/null" or (new != "/dev/null" and old != "/dev/null" and old != new):
            raise ValueError("rename or ambiguous patch headers are unsupported")
        if git_pair is not None and git_pair != (target, target):
            raise ValueError("diff --git and file headers disagree")
        if target in targets:
            raise ValueError("duplicate patch target")
        targets.append(target)

    for line in text.splitlines():
        if line.startswith("diff --git "):
            finish()
            old = new = None
            has_hunk = in_hunk = False
            fields = line.split(" ")
            if len(fields) != 4:
                raise ValueError("ambiguous diff --git header")
            git_pair = (path(fields[2], "a/"), path(fields[3], "b/"))
        elif line.startswith("--- "):
            if old is not None:
                finish()
                git_pair = None
                new = None
                has_hunk = False
            old = path(line[4:], "a/")
            in_hunk = False
        elif line.startswith("+++ ") and not in_hunk:
            if old is None or new is not None:
                raise ValueError("unpaired or duplicate patch header")
            new = path(line[4:], "b/")
        elif line.startswith("@@ "):
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if old is None or new is None or not match:
                raise ValueError("invalid text hunk header")
            close_hunk()
            hunk_index += 1
            declared_old = int(match.group(2)) if match.group(2) is not None else 1
            declared_new = int(match.group(4)) if match.group(4) is not None else 1
            seen_old = 0
            seen_new = 0
            has_hunk = in_hunk = True
        elif in_hunk and (line.startswith((" ", "+", "-")) or line == "\\ No newline at end of file"
                          or line == ""):
            if line.startswith("-"):
                seen_old += 1
            elif line.startswith("+"):
                seen_new += 1
            else:
                # Context line, "\ No newline ..." marker (no counts) and a bare
                # blank line, which git reads as a context line.
                if line != "\\ No newline at end of file":
                    seen_old += 1
                    seen_new += 1
        elif re.fullmatch(r"index [0-9a-fA-F]+\.\.[0-9a-fA-F]+(?: 100644)?", line) and git_pair and old is None:
            continue
        elif line:
            raise ValueError("unsupported patch metadata or content: " + line[:100])
    finish()
    if not targets:
        raise ValueError("patch has no text targets")
    return targets


def rewrite_text_patch(text: str, mapping: dict[str, str]) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if body.startswith("diff --git "):
            _, _, a, b = body.split(" ")
            body = f"diff --git a/{mapping[a[2:]]} b/{mapping[b[2:]]}"
        elif body.startswith(("--- a/", "+++ b/")):
            body = body[:6] + mapping[body[6:]]
        lines.append(body + ending)
    return "".join(lines)


def patch_paths(patch_text: str) -> list[str]:
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            if path != "/dev/null":
                paths.add(path)
    return sorted(paths)


def extract_patch(text: str) -> str:
    match = re.search(r"PATCH_BEGIN\s*(.*?)\s*PATCH_END", text, re.DOTALL)
    return strip_code_fences(match.group(1)) if match else ""


def is_dry_run_artifact(rel_path: str) -> bool:
    return rel_path.startswith("agents/generated/runs/") or rel_path.startswith("agents/generated/graph/")
