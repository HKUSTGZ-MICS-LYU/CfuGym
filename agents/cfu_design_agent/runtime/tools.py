"""Repo-safe tools used by the CFU design agent."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .policy import AccessPolicy, PolicyError, load_policy, normalize_repo_path
from ..state import CommandResult, RepoContext


MAX_FILE_CHARS = 20000

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

    def normalize(self, rel_path: str | Path) -> str:
        return normalize_repo_path(self.workdir, rel_path)

    def is_read_allowed(self, rel_path: str | Path) -> bool:
        try:
            normalized = self.normalize(rel_path)
        except PolicyError:
            return False
        return self.is_current_run_path(normalized) and self.policy.can_read(self.stage, normalized)

    def is_current_run_path(self, normalized: str) -> bool:
        """Only run artifacts are scoped; tracked sources/cache remain unchanged."""
        if not self.run_root or not normalized.startswith("agents/generated/runs/"):
            return True
        root = self.run_root.strip("/")
        return normalized == root or normalized.startswith(root + "/")

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
        self.policy.check_read(self.stage, normalized)
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
            "git_status": self.git_status(),
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

    def git_status(self) -> str:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        visible: list[str] = []
        for line in result.stdout.splitlines():
            candidate = line[3:].strip() if len(line) > 3 else ""
            try:
                normalized = self.normalize(candidate)
            except PolicyError:
                continue
            if self.policy.can_read(self.stage, normalized):
                visible.append(f"{line[:3]}{normalized}")
        return "\n".join(visible)

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
        return {
            "command": " ".join(args),
            "cwd": str(run_cwd),
            "returncode": result.returncode,
            "stdout_tail": tail(filter_output(result.stdout)),
            "stderr_tail": tail(filter_output(result.stderr)),
            "parsed": parse_command_output(result.stdout + "\n" + result.stderr),
        }

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
        section_seen: dict[str, int] = {}
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
                errors.append(f"{resolved}: {label}: edits produced no change")
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

        try:
            self._commit_files(pending)
        except OSError as exc:
            return CommandResult(command="edit", cwd=str(self.workdir), returncode=1,
                                 stdout_tail="", stderr_tail=str(exc), parsed={"touched_paths": []})
        applied = list(pending)
        return CommandResult(
            command="edit",
            cwd=str(self.workdir),
            returncode=0 if applied else 1,
            stdout_tail="",
            stderr_tail="" if applied else "no edit blocks were supplied",
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
        """Prevalidate every FILE target, then commit with in-process rollback."""
        pending: dict[str, str] = {}
        try:
            for target, content in blocks:
                resolved = self.resolve_target(target)
                self.policy.check_write(self.stage, resolved)
                if resolved in pending:
                    raise ValueError(f"duplicate FILE target: {target}")
                pending[resolved] = content
            if not pending:
                raise ValueError("no file blocks supplied")
            self._commit_files(pending)
        except (ValueError, PolicyError, OSError) as exc:
            return CommandResult(command="file", cwd=str(self.workdir), returncode=1,
                                 stdout_tail="", stderr_tail=str(exc), parsed={"touched_paths": []})
        return CommandResult(command="file", cwd=str(self.workdir), returncode=0,
                             stdout_tail="", stderr_tail="", parsed={"touched_paths": list(pending)})

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
        self.policy.check_read(self.stage, src_normalized)
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
            self.policy.check_read(self.stage, source_rel)
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
    """Parse one EDIT_BEGIN body into [(search, replace)] pairs."""
    pairs: list[tuple[str, str]] = []
    mode: str | None = None
    search: list[str] = []
    replace: list[str] = []
    for line in block_lines:
        stripped = line.strip()
        if stripped.startswith(SEARCH_MARKER):
            if mode is not None:
                raise ValueError(f"{target}: nested SEARCH marker")
            mode = "search"
            search = []
            replace = []
        elif mode == "search" and stripped.startswith(DIVIDER_MARKER):
            mode = "replace"
        elif mode == "replace" and stripped.startswith(REPLACE_MARKER):
            pairs.append((_trim_blank_edges(search), _trim_blank_edges(replace)))
            mode = None
        elif mode == "search":
            search.append(line)
        elif mode == "replace":
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
                block = (current_path, _parse_edit_pairs(current, target=current_path))
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
        content_lines[start:start + len(search_lines)] = replace.split("\n")
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
                target = body[len(prefix):]
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
    """Accept only unambiguous regular-file unified diffs (no Git side effects)."""
    targets: list[str] = []
    git_pair = None
    old = None
    new = None
    has_hunk = False
    in_hunk = False

    def path(value: str, prefix: str) -> str:
        if value == "/dev/null":
            return value
        if not value.startswith(prefix) or re.search(r'[\s"\\\\]', value):
            raise ValueError("unsupported quoted or ambiguous patch path")
        value = value[len(prefix):]
        if not value or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/")):
            raise ValueError("unsafe patch path")
        return value

    def finish() -> None:
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
            if old is None or new is None or not re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line):
                raise ValueError("invalid text hunk header")
            has_hunk = in_hunk = True
        elif in_hunk and (line.startswith((" ", "+", "-")) or line == "\\ No newline at end of file"):
            continue
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


def read_yosys_summary(path: Path) -> dict[str, object]:
    json_path = path / "summary.json"
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text())
