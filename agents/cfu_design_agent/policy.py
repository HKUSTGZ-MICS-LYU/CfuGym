"""Declarative access and command policy for the CFU agent."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_PATH = "agents/specs/agent_access_policy.yaml"


class PolicyError(ValueError):
    """Raised when a path or command violates the agent policy."""


@dataclass(frozen=True)
class StagePolicy:
    read: tuple[str, ...]
    write: tuple[str, ...]
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CommandPolicy:
    executable: str
    argv_prefix: tuple[str, ...] = ()
    required_tokens: tuple[str, ...] = ()
    required_substring: str = ""
    # Substrings that must appear anywhere in the joined argv. Used when a
    # single argv element carries a subcommand plus its flags (sbt runMain).
    required_substrings: tuple[str, ...] = ()
    # Higher priority wins when several declarations match the same argv, so a
    # stricter variant is not shadowed by a looser one declared earlier.
    priority: int = 0


@dataclass(frozen=True)
class AccessPolicy:
    version: str
    deny: tuple[str, ...]
    write_roots: tuple[str, ...]
    stages: dict[str, StagePolicy]
    commands: dict[str, CommandPolicy]

    @classmethod
    def from_file(cls, path: Path) -> "AccessPolicy":
        try:
            data = yaml.safe_load(path.read_text(errors="replace"))
        except OSError as exc:
            raise PolicyError(f"cannot read access policy: {path}") from exc
        if not isinstance(data, dict):
            raise PolicyError(f"access policy must be a mapping: {path}")

        raw_stages = data.get("stages", {})
        raw_commands = data.get("commands", {})
        if not isinstance(raw_stages, dict) or not isinstance(raw_commands, dict):
            raise PolicyError("access policy stages and commands must be mappings")

        stages: dict[str, StagePolicy] = {}
        for name, raw in raw_stages.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise PolicyError(f"invalid stage policy: {name!r}")
            stages[name] = StagePolicy(
                read=_string_tuple(raw.get("read", []), f"{name}.read"),
                write=_string_tuple(raw.get("write", []), f"{name}.write"),
                commands=_string_tuple(raw.get("commands", []), f"{name}.commands"),
            )

        commands: dict[str, CommandPolicy] = {}
        for name, raw in raw_commands.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise PolicyError(f"invalid command policy: {name!r}")
            executable = raw.get("executable")
            if not isinstance(executable, str) or not executable:
                raise PolicyError(f"{name}.executable must be a non-empty string")
            commands[name] = CommandPolicy(
                executable=executable,
                argv_prefix=_string_tuple(raw.get("argv_prefix", []), f"{name}.argv_prefix"),
                required_tokens=_string_tuple(raw.get("required_tokens", []), f"{name}.required_tokens"),
                required_substring=str(raw.get("required_substring", "")),
                required_substrings=_string_tuple(
                    raw.get("required_substrings", []), f"{name}.required_substrings"
                ),
                priority=int(raw.get("priority", 0) or 0),
            )

        return cls(
            version=str(data.get("version", "1")),
            deny=_string_tuple(data.get("deny", []), "deny"),
            write_roots=_string_tuple(data.get("write_roots", []), "write_roots"),
            stages=stages,
            commands=commands,
        )

    def stage(self, name: str) -> StagePolicy:
        try:
            return self.stages[name]
        except KeyError as exc:
            raise PolicyError(f"stage is not declared in access policy: {name}") from exc

    def _matches(self, path: str, patterns: tuple[str, ...]) -> bool:
        return any(_glob_matches(path, pattern) for pattern in patterns)

    def is_denied(self, rel_path: str) -> bool:
        return self._matches(rel_path, self.deny)

    def can_read(self, stage: str, rel_path: str) -> bool:
        stage_policy = self.stage(stage)
        return not self.is_denied(rel_path) and self._matches(rel_path, stage_policy.read)

    def can_write(self, stage: str, rel_path: str) -> bool:
        stage_policy = self.stage(stage)
        return not self.is_denied(rel_path) and self._matches(rel_path, stage_policy.write)

    def check_read(self, stage: str, rel_path: str) -> None:
        if not self.can_read(stage, rel_path):
            reason = "denied by deny rule" if self.is_denied(rel_path) else "not allowlisted for stage"
            raise PolicyError(f"read path is {reason}: {stage}: {rel_path}")

    def check_write(self, stage: str, rel_path: str) -> None:
        if not self.can_write(stage, rel_path):
            reason = "denied by deny rule" if self.is_denied(rel_path) else "not allowlisted for stage"
            raise PolicyError(f"write path is {reason}: {stage}: {rel_path}")
        if not any(_under_root(rel_path, root) for root in self.write_roots):
            raise PolicyError(f"write path is outside write roots: {rel_path}")

    def check_command(self, stage: str, command_id: str, args: list[str], workdir: Path) -> None:
        stage_policy = self.stage(stage)
        if command_id not in stage_policy.commands:
            raise PolicyError(f"command is not allowlisted for stage: {stage}: {command_id}")
        try:
            command = self.commands[command_id]
        except KeyError as exc:
            raise PolicyError(f"command descriptor is missing: {command_id}") from exc
        if not args or args[0] != command.executable:
            raise PolicyError(f"command executable mismatch for {command_id}: {args}")
        if command.argv_prefix and tuple(args[: len(command.argv_prefix)]) != command.argv_prefix:
            raise PolicyError(f"command prefix mismatch for {command_id}: {args}")
        missing = [token for token in command.required_tokens if token not in args]
        if missing:
            raise PolicyError(f"command missing required tokens for {command_id}: {missing}")
        joined = " ".join(args)
        if command.required_substring and command.required_substring not in joined:
            raise PolicyError(f"command missing required text for {command_id}: {command.required_substring}")
        for needle in command.required_substrings:
            if needle not in joined:
                raise PolicyError(f"command missing required text for {command_id}: {needle}")
        _check_command_paths(args, workdir, self)


def load_policy(workdir: Path, path: str = DEFAULT_POLICY_PATH) -> AccessPolicy:
    policy_path = (workdir / path).resolve()
    try:
        policy_path.relative_to(workdir.resolve())
    except ValueError as exc:
        raise PolicyError(f"policy path escapes workdir: {path}") from exc
    return AccessPolicy.from_file(policy_path)


def normalize_repo_path(workdir: Path, raw_path: str | Path) -> str:
    """Resolve a path and return a safe repo-relative POSIX path."""
    raw = Path(raw_path)
    if raw.is_absolute():
        raise PolicyError(f"absolute paths are not allowed: {raw_path}")
    root = workdir.resolve()
    resolved = (root / raw).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PolicyError(f"path escapes workdir: {raw_path}") from exc
    normalized = relative.as_posix()
    if not normalized or normalized == ".":
        raise PolicyError(f"directory root is not a file path: {raw_path}")
    return normalized


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PolicyError(f"{field} must be a list of strings")
    return tuple(value)


def _glob_matches(path: str, pattern: str) -> bool:
    if fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/") and fnmatchcase(path, pattern[3:]):
        return True
    return False


def _under_root(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _check_command_paths(args: list[str], workdir: Path, policy: AccessPolicy) -> None:
    """Check command argv for unsafe paths and shell syntax.

    Full repo confinement is enforced by the read/write/copy layer. Here we
    reject absolute paths, deny-rule hits, and shell metacharacters so a
    declared command cannot escape policy or smuggle a second command.

    Every token is inspected, not just the values of known path options: a
    declared command may take a positional path (for example the Yosys RTL
    argument), and that must be confined too.
    """
    path_options = {"-C", "--load-elf", "--out-dir", "--input", "--output", "--rtl"}
    cwd = _command_cwd(args, workdir)
    next_is_path = False
    for token in args[1:]:
        if _has_shell_metachar(token):
            raise PolicyError(f"shell syntax is not allowed in command argv: {token!r}")
        if next_is_path:
            _check_token_path(token, policy, workdir=workdir, cwd=cwd)
            next_is_path = False
            continue
        if token in path_options:
            next_is_path = True
            continue
        _check_token_path(token, policy, workdir=workdir, cwd=cwd)
    if next_is_path:
        raise PolicyError("command path option has no value")
    _scan_embedded_absolute_paths(args)


def _command_cwd(args: list[str], workdir: Path) -> Path:
    """Effective working directory for a command, honouring `-C <dir>`."""
    if "-C" in args:
        index = args.index("-C")
        if index + 1 < len(args):
            return (workdir / args[index + 1]).resolve()
    return workdir.resolve()


def _scan_embedded_absolute_paths(args: list[str]) -> None:
    import re

    joined = " ".join(args)
    pattern = re.compile(r"(?:--load-elf|--out-dir|--input|--output|--rtl|-C)\s+([^\s]+)")
    for match in pattern.finditer(joined):
        value = match.group(1)
        normalized = _lenient_posix(value)
        if normalized.startswith("/"):
            raise PolicyError(f"absolute paths are not allowed in commands: {value}")
        if any(marker in normalized for marker in COMMAND_DENY_PATTERNS):
            raise PolicyError(f"command path is denied: {value}")


COMMAND_DENY_PATTERNS = (
    ".git/",
    ".env",
    "secret",
    "token",
    "credential",
    "password",
)


def _check_token_path(
    token: str,
    policy: AccessPolicy,
    *,
    workdir: Path | None = None,
    cwd: Path | None = None,
) -> None:
    # Also inspect the value side of KEY=value forms (MAIN=..., OUT_DIR=...).
    candidates = [token]
    if "=" in token:
        candidates.append(token.split("=", 1)[1])
    for candidate in candidates:
        normalized = _lenient_posix(candidate)
        if normalized.startswith("/"):
            raise PolicyError(f"absolute paths are not allowed in commands: {token}")
        if any(pattern in normalized for pattern in COMMAND_DENY_PATTERNS):
            raise PolicyError(f"command path is denied: {normalized}")
        # A relative path may still escape the repository through "..".
        # Resolve against the command's effective directory so the legitimate
        # "make -C sw ... MAIN=../<repo path>" form keeps working.
        if ".." in Path(normalized).parts and workdir is not None and cwd is not None:
            resolved = (cwd / normalized).resolve()
            root = workdir.resolve()
            if resolved != root and root not in resolved.parents:
                raise PolicyError(
                    f"command path escapes the repository: {token} -> {resolved}"
                )


def _lenient_posix(path: str) -> str:
    value = path.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _has_shell_metachar(token: str) -> bool:
    return any(ch in token for ch in (";", "&", "|", "\n", "`", "$("))
