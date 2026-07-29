"""Repo-safe tools used by the CFU design agent."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .state import CommandResult, RepoContext


MAX_FILE_CHARS = 20000


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
    "src/main/scala/vexiiriscv/soc/mico/MiCoSocParam.scala",
    "src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
)


PROFILE_RE = re.compile(
    r"(?P<name>[A-Z0-9_]+_PROFILE)\s+(?P<body>.*)"
)


@dataclass
class RepoToolbox:
    workdir: Path
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.workdir = self.workdir.resolve()

    def resolve(self, rel_path: str | Path) -> Path:
        path = (self.workdir / rel_path).resolve()
        try:
            path.relative_to(self.workdir)
        except ValueError as exc:
            raise ValueError(f"path escapes workdir: {rel_path}") from exc
        return path

    def is_write_allowed(self, rel_path: str | Path) -> bool:
        normalized = Path(rel_path).as_posix()
        return any(
            normalized == root or normalized.startswith(root + "/")
            for root in ALLOW_WRITE_ROOTS
        )

    def read_text(self, rel_path: str, *, limit: int = MAX_FILE_CHARS) -> str:
        path = self.resolve(rel_path)
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(errors="replace")
        return text[:limit]

    def write_text(self, rel_path: str, text: str) -> str:
        if not self.is_write_allowed(rel_path):
            raise ValueError(f"write path is not allowlisted: {rel_path}")
        normalized = Path(rel_path).as_posix()
        path = self.resolve(rel_path)
        if self.dry_run and not normalized.startswith("agents/generated/"):
            return f"dry-run: would write {rel_path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return f"wrote {rel_path}"

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
        return sorted(
            str(path.relative_to(self.workdir))
            for path in root.glob(pattern)
            if path.is_file()
        )

    def git_status(self) -> str:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout.strip()

    def command_allowed(self, args: list[str]) -> bool:
        if not args:
            return False
        if args[0] == "git" and args[1:] in (["status", "--short"], ["diff", "--stat"]):
            return True
        if args[0] in {"rg", "sed"}:
            return True
        if args[0] == "sbt" and len(args) >= 2:
            return (
                "runMain vexiiriscv.soc.mico.MiCoSocGen" in args[1]
                or "runMain vexiiriscv.soc.mico.MiCoSocSim" in args[1]
            )
        if args[0] == "make":
            return any(arg == "TARGET=vexii_soc" for arg in args)
        if args[:2] == ["python3", "skills/cfu-designer/scripts/yosys_cost_report.py"]:
            return True
        if args[0] == "git" and args[1:] in (["apply", "--check", "-"], ["apply", "-"]):
            return True
        return False

    def run_command(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        timeout: int = 1800,
        mutate: bool = False,
    ) -> CommandResult:
        if not self.command_allowed(args):
            raise ValueError(f"command is not allowlisted: {args}")
        if mutate and self.dry_run:
            return {
                "command": " ".join(args),
                "cwd": cwd or str(self.workdir),
                "returncode": 0,
                "skipped": True,
                "reason": "dry-run",
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
        blocked = [path for path in touched if not self.is_write_allowed(path)]
        if blocked:
            raise ValueError(f"patch touches non-allowlisted paths: {blocked}")
        check = self.run_command(["git", "apply", "--check", "-"], input_text=patch_text)
        if check["returncode"] != 0:
            return check
        return self.run_command(["git", "apply", "-"], input_text=patch_text, mutate=True)


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


def read_yosys_summary(path: Path) -> dict[str, object]:
    json_path = path / "summary.json"
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text())
