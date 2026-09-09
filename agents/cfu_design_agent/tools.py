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
from .state import CommandResult, RepoContext


MAX_FILE_CHARS = 20000

# Kept as a public compatibility constant for callers that inspect the old API.
# Hardware design writes live in the per-run workspace only; the tracked
# AgentCfu/Fiber scaffolds are read-only defaults.
ALLOW_WRITE_ROOTS = (
    "agents/generated",
    "sw/tests",
)

# The only files a design agent may write for hardware. They are seeded from the
# tracked scaffolds and compiled over them as a same-package overlay.
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


@dataclass
class RepoToolbox:
    workdir: Path
    dry_run: bool = False
    policy: AccessPolicy | None = None
    stage: str = "load_context"
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
        return self.policy.can_read(self.stage, normalized)

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
        self.policy.check_read(self.stage, normalized)
        path = self.workdir / normalized
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(errors="replace")
        return text[:limit]

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
        result = subprocess.run(
            args,
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
            "stdout_tail": tail(result.stdout),
            "stderr_tail": tail(result.stderr),
            "parsed": parse_command_output(result.stdout + "\n" + result.stderr),
        }

    def apply_patch_text(self, patch_text: str) -> CommandResult:
        # Redirect tracked scaffold targets into the run workspace before any
        # git operation, so the repo tree is never touched.
        redirects: list[str] = []
        rejected: list[str] = []
        if self.run_root:
            patch_text, redirects, rejected = repair_patch_paths(patch_text, self.run_root)
        if rejected:
            raise ValueError(
                "patch touches source paths outside the design workspace: "
                + ", ".join(sorted(set(rejected)))
                + "; only AgentCfu.scala and AgentCfuFiber.scala may be designed,"
                + f" and only under {self.run_root}/{DESIGN_WORKSPACE_SUBDIR}"
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
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=ignore)
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


def repair_patch_paths(patch_text: str, run_root: str) -> tuple[str, list[str], list[str]]:
    """Redirect scaffold patch targets into the run workspace.

    An LLM naturally writes repository-relative paths for the tracked
    scaffolds. Those must never be applied to the repo, so each such target is
    rewritten to its workspace overlay path. Any other source path is rejected.

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
                name = design_scaffold_name(target)
                if name and target in DESIGN_SOURCE_FILES:
                    replacement = design_workspace_rel(run_root, name)
                    replaced = f"{prefix}{replacement}"
                    if replacement not in redirected:
                        redirected.append(replacement)
                elif target.startswith("src/"):
                    rejected.append(target)
                elif name:
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


def tail(text: str, *, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def parse_command_output(text: str) -> dict[str, object]:
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
    if " PASS " in text or text.strip().endswith(" PASS"):
        parsed["pass"] = True
    if "FAILURE" in text or " FAIL" in text:
        parsed["fail"] = True
    return parsed


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
    return match.group(1).strip() + "\n" if match else ""


def is_dry_run_artifact(rel_path: str) -> bool:
    return rel_path.startswith("agents/generated/runs/") or rel_path.startswith("agents/generated/graph/")


def read_yosys_summary(path: Path) -> dict[str, object]:
    json_path = path / "summary.json"
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text())
