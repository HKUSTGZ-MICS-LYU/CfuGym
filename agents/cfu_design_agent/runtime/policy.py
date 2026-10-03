"""Declarative access and command policy for the CFU agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_PATH = "agents/specs/agent_access_policy.yaml"
DEFAULT_VISIBILITY_LEVELS: tuple[str, ...] = ("basic", "standard", "extended")
DEFAULT_VISIBILITY = "standard"


class PolicyError(ValueError):
    """Raised when a path or command violates the agent policy."""


@dataclass(frozen=True)
class StagePolicy:
    """What one stage may WRITE and RUN.

    Reads are deliberately absent: a run's knowledge is shared, so the whole
    flow sees one tree while only the write/command authority is stage-scoped.
    """

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
    # One shared read tree for the whole flow. Every stage sees the same paths;
    # `is_current_run_path` still scopes run artifacts to the executing run.
    read: tuple[str, ...] = ()
    # Visibility tiers are an ordered ladder (lowest first). A path matched by
    # ``visibility_gated[level]`` may only be read at that level or above.
    visibility: str = DEFAULT_VISIBILITY
    visibility_levels: tuple[str, ...] = DEFAULT_VISIBILITY_LEVELS
    visibility_gated: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path, visibility: str = "") -> "AccessPolicy":
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

        read = _string_tuple(data.get("read", []), "read")
        if not read:
            raise PolicyError("access policy needs a top-level `read` list")

        stages: dict[str, StagePolicy] = {}
        for name, raw in raw_stages.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise PolicyError(f"invalid stage policy: {name!r}")
            if "read" in raw:
                # Reads are shared by the whole flow; a stage-scoped read list
                # would silently narrow what a stage sees, so refuse it.
                raise PolicyError(
                    f"stage {name!r} declares `read`; reads are declared once at "
                    "the top level and shared by every stage"
                )
            stages[name] = StagePolicy(
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

        levels, gated, default_level = _parse_visibility(data.get("visibility", {}))

        return cls(
            version=str(data.get("version", "1")),
            deny=_string_tuple(data.get("deny", []), "deny"),
            write_roots=_string_tuple(data.get("write_roots", []), "write_roots"),
            stages=stages,
            commands=commands,
            read=read,
            visibility=_resolve_visibility(visibility or default_level, levels),
            visibility_levels=levels,
            visibility_gated=gated,
        )

    def level_index(self, level: str) -> int:
        try:
            return self.visibility_levels.index(level)
        except ValueError as exc:
            raise PolicyError(
                f"unknown visibility level {level!r}; expected one of "
                f"{', '.join(self.visibility_levels)}"
            ) from exc

    def required_visibility(self, rel_path: str) -> str:
        """The lowest level allowed to read ``rel_path`` or anything below it.

        A gate is usually declared as a subtree (``ext/SpinalHDL/**``), so the
        directory that roots it must report the same level as its contents:
        otherwise a walk matches nothing on the directory and descends into a
        tree that can never yield a readable file.
        """
        required = ""
        for level, patterns in self.visibility_gated.items():
            hit = self._matches(rel_path, patterns) or any(
                _under_root(rel_path, root)
                for root in map(_subtree_root, patterns)
                if root
            )
            if hit:
                # Several levels may match; the strictest wins.
                if not required or self.level_index(level) > self.level_index(required):
                    required = level
        return required

    def visibility_block(self, rel_path: str) -> str:
        """Name the level that blocks ``rel_path`` now, or "" when readable."""
        required = self.required_visibility(rel_path)
        if required and self.level_index(self.visibility) < self.level_index(required):
            return required
        return ""

    def stage(self, name: str) -> StagePolicy:
        try:
            return self.stages[name]
        except KeyError as exc:
            raise PolicyError(f"stage is not declared in access policy: {name}") from exc

    def _matches(self, path: str, patterns: tuple[str, ...]) -> bool:
        return any(_glob_matches(path, pattern) for pattern in patterns)

    def is_denied(self, rel_path: str) -> bool:
        return self._matches(rel_path, self.deny)

    def is_denied_subtree(self, rel_path: str) -> bool:
        """True when ``rel_path`` or everything below it is denied.

        ``is_denied`` alone misses a directory that only *contains* a denied
        tree (``target`` against ``**/target/**``), so a walk would descend into
        it and spend its enumeration budget on paths it can never return.
        """
        if self.is_denied(rel_path) or self.is_denied(rel_path + "/"):
            return True
        return any(
            _under_root(rel_path, root)
            for root in map(_subtree_root, self.deny)
            if root
        )

    def can_read(self, rel_path: str) -> bool:
        if self.is_denied(rel_path) or not self._matches(rel_path, self.read):
            return False
        return not self.visibility_block(rel_path)

    def _pattern_is_gated_here(self, pattern: str) -> bool:
        """True when the whole ``pattern`` is unreadable at the current level."""
        root = _subtree_root(pattern) or _literal_prefix(pattern)
        if not root:
            return False
        required = self.required_visibility(root)
        return bool(required) and self.level_index(required) > self.level_index(self.visibility)

    def may_contain_readable(self, rel_path: str) -> bool:
        """True when a readable path may exist at or below ``rel_path``.

        Navigation needs this: a directory is never itself in the read
        allowlist, so filtering directories through ``can_read`` hides every
        directory-only tree behind a misleading "nothing here".
        """
        rel = (rel_path or "").strip("/")
        if not rel:
            return True
        if self.visibility_block(rel):
            # The directory roots a gated subtree: nothing below it is readable.
            return False
        if self.can_read(rel):
            return True
        for pattern in self.read:
            if self._pattern_is_gated_here(pattern):
                continue
            prefix = _literal_prefix(pattern)
            if not prefix or _under_root(rel, prefix) or _under_root(prefix, rel):
                return True
        return False

    def _blocked_gate_below(self, rel_path: str) -> str:
        """The strictest gate above the current level that roots a subtree here.

        A directory that only leads to gated content (``skills`` at ``basic``)
        is unreadable *because of the tier*, and saying so is what tells the
        model that rephrasing cannot help.
        """
        blocked = ""
        for level, patterns in self.visibility_gated.items():
            if self.level_index(level) <= self.level_index(self.visibility):
                continue
            for pattern in patterns:
                root = _subtree_root(pattern)
                if root and _under_root(root, rel_path):
                    if not blocked or self.level_index(level) > self.level_index(blocked):
                        blocked = level
        return blocked

    def subtree_exclusion(self, rel_path: str) -> str:
        """Why nothing at or below ``rel_path`` can be read, or "" when it may."""
        if self.may_contain_readable(rel_path):
            return ""
        rel = (rel_path or "").strip("/")
        if self.is_denied_subtree(rel):
            return "denied by deny rule"
        blocked = self.visibility_block(rel) or (
            self._blocked_gate_below(rel) if rel else ""
        )
        if blocked:
            return f"requires visibility level {blocked!r} (current: {self.visibility})"
        return "outside the shared read allowlist"

    def can_write(self, stage: str, rel_path: str) -> bool:
        stage_policy = self.stage(stage)
        return not self.is_denied(rel_path) and self._matches(rel_path, stage_policy.write)

    def check_read(self, rel_path: str) -> None:
        if self.can_read(rel_path):
            return
        if self.is_denied(rel_path):
            reason = "denied by deny rule"
        elif not self._matches(rel_path, self.read):
            reason = "not in the shared read allowlist"
        else:
            blocked = self.visibility_block(rel_path)
            reason = (
                f"requires visibility level {blocked!r} "
                f"(current: {self.visibility})"
            )
        raise PolicyError(f"read path is {reason}: {rel_path}")

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


def load_policy(
    workdir: Path,
    path: str = DEFAULT_POLICY_PATH,
    visibility: str = "",
) -> AccessPolicy:
    policy_path = (workdir / path).resolve()
    try:
        policy_path.relative_to(workdir.resolve())
    except ValueError as exc:
        raise PolicyError(f"policy path escapes workdir: {path}") from exc
    return AccessPolicy.from_file(policy_path, visibility=visibility)


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


def _parse_visibility(raw: Any) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], str]:
    """Parse the optional top-level ``visibility`` section."""
    if raw in (None, {}):
        return DEFAULT_VISIBILITY_LEVELS, {}, DEFAULT_VISIBILITY
    if not isinstance(raw, dict):
        raise PolicyError("visibility must be a mapping")

    levels = _string_tuple(raw.get("levels", list(DEFAULT_VISIBILITY_LEVELS)), "visibility.levels")
    if not levels:
        raise PolicyError("visibility.levels must list at least one level")
    if len(set(levels)) != len(levels):
        raise PolicyError("visibility.levels must not contain duplicates")

    default_level = str(raw.get("default", "") or levels[0])
    if default_level not in levels:
        raise PolicyError(
            f"visibility.default {default_level!r} is not in visibility.levels "
            f"({', '.join(levels)})"
        )

    raw_gated = raw.get("gated", {})
    if not isinstance(raw_gated, dict):
        raise PolicyError("visibility.gated must be a mapping of level -> patterns")
    gated: dict[str, tuple[str, ...]] = {}
    for level, patterns in raw_gated.items():
        if level not in levels:
            raise PolicyError(
                f"visibility.gated level {level!r} is not in visibility.levels "
                f"({', '.join(levels)})"
            )
        gated[level] = _string_tuple(patterns, f"visibility.gated.{level}")
    return levels, gated, default_level


def _resolve_visibility(requested: str, levels: tuple[str, ...]) -> str:
    if requested not in levels:
        raise PolicyError(
            f"unknown visibility level {requested!r}; expected one of {', '.join(levels)}"
        )
    return requested


def _glob_matches(path: str, pattern: str) -> bool:
    if fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/") and fnmatchcase(path, pattern[3:]):
        return True
    return False


def _has_wildcard(text: str) -> bool:
    return any(ch in text for ch in "*?[")


def _literal_prefix(pattern: str) -> str:
    """The longest leading path a pattern cannot match outside of.

    ``src/main/**/*.scala`` -> ``src/main``, ``sw/tests/**`` -> ``sw/tests``.
    An empty result means the pattern may match anywhere.
    """
    parts: list[str] = []
    for part in pattern.split("/"):
        if _has_wildcard(part):
            break
        parts.append(part)
    return "/".join(parts)


def _subtree_root(pattern: str) -> str:
    """The literal directory a subtree pattern covers (``X/**`` -> ``X``).

    Returns "" for patterns that name files or carry a wildcard in the
    directory part, because those cannot prune a whole subtree.
    """
    body = pattern[3:] if pattern.startswith("**/") else pattern
    for suffix in ("/**", "/*", "/"):
        if body.endswith(suffix):
            root = body[: -len(suffix)]
            return root if root and not _has_wildcard(root) else ""
    return ""


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
