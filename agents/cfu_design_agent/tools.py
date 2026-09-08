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
ALLOW_WRITE_ROOTS = (
    "agents/generated",
    "src/main/scala/vexiiriscv/soc/mico",
    "src/main/scala/vexiiriscv/soc/cfu",
    "sw/tests",
)

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
        touched = patch_paths(patch_text)
        normalized: list[str] = []
        for path in touched:
            normalized.append(self.normalize(path))
            if not self.is_write_allowed(path):
                raise ValueError(f"patch touches non-allowlisted path: {path}")
        check = self.run_command(
            ["git", "apply", "--check", "-"],
            input_text=patch_text,
            command_id="git_apply_check",
            mutate=False,
        )
        if check["returncode"] != 0 or not self.apply_patch:
            if not self.apply_patch and check["returncode"] == 0:
                check["skipped"] = True
                check["reason"] = "patch-application-disabled"
            check["parsed"] = {**check.get("parsed", {}), "touched_paths": normalized}
            return check
        result = self.run_command(
            ["git", "apply", "-"],
            input_text=patch_text,
            command_id="git_apply",
            mutate=True,
        )
        result["parsed"] = {**result.get("parsed", {}), "touched_paths": normalized}
        return result

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


def infer_command_id(args: list[str], policy: AccessPolicy) -> str:
    for name, descriptor in policy.commands.items():
        if not args or args[0] != descriptor.executable:
            continue
        if descriptor.argv_prefix and tuple(args[: len(descriptor.argv_prefix)]) != descriptor.argv_prefix:
            continue
        if descriptor.required_substring and descriptor.required_substring not in " ".join(args):
            continue
        if any(token not in args for token in descriptor.required_tokens):
            continue
        return name
    raise PolicyError(f"no declared command matches argv: {args}")


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
