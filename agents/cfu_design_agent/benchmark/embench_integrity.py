"""Lightweight, fail-closed structural fairness checks for prepared Embench runs.

This is deliberately not a C compiler or a proof of arbitrary C/RTL honesty.
It freezes the workload/harness structure while allowing computational bodies,
fresh helpers and scratch storage. No subprocesses or benchmark execution occur.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import yaml

INTEGRITY_VERSION = 1
PROTOCOL = "cold_single_invocation_v1"
ENTRY_FUNCTIONS = frozenset({"benchmark", "warm_caches", "initialise_benchmark", "verify_benchmark"})
_RESERVED = ENTRY_FUNCTIONS | {"main", "start_trigger", "stop_trigger", "initialise_board",
    "LOCAL_SCALE_FACTOR", "GLOBAL_SCALE_FACTOR", "WARMUP_HEAT", "USE_AGENT_CFU",
    "EMBENCH_OBS", "EMBENCH_NAME", "HAVE_BOARDSUPPORT_H"}
_KEYWORDS = set("auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while _Bool _Complex _Atomic _Alignas _Alignof".split())
_ATTRIBUTES = {"__attribute__", "__attribute", "__declspec", "_Alignas"}
_FORBIDDEN = {"constructor", "destructor", "alias", "ifunc", "section", "_Pragma", "__pragma"}
_STANDARD_HEADERS = set("assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h math.h setjmp.h signal.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdio.h stdlib.h stdnoreturn.h string.h tgmath.h threads.h time.h uchar.h wchar.h wctype.h".split())
_LEX = re.compile(r'/\*.*?\*/|//[^\n]*|(?:u8|[LuU])?"(?:\\.|[^"\\])*"|(?:[LuU])?\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|(?:\d[\w.]*|\.\d[\w.]*)|\s+|.', re.DOTALL)
_IDENT = re.compile(r"[A-Za-z_]\w*\Z")


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int
    directive: bool = False


def scan(text: str) -> list[Token]:
    """Keep literals/directives, discard comments; retain original byte offsets.

    Line splices are removed before lexing (C translation phase 2), including
    in identifiers, // comments and directives. Digraph directives are supported;
    trigraphs are rejected because interpreting those is compiler-flag dependent.
    """
    if "??" in text:
        raise ValueError("unsupported trigraph-like source")
    logical: list[str] = []
    offsets: list[int] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and (text.startswith("\n", i + 1) or text.startswith("\r\n", i + 1)):
            i += 3 if text.startswith("\r\n", i + 1) else 2
        else:
            logical.append(text[i]); offsets.append(i); i += 1
    source = "".join(logical)
    result: list[Token] = []
    i, line_start = 0, True
    while i < len(source):
        if line_start and (source[i] == "#" or source.startswith("%:", i)):
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            raw = source[i:end]
            if raw.startswith("%:"):
                raw = "#" + raw[2:]
            parts = _plain_tokens(raw)
            # Whitespace between a macro name and '(' changes macro semantics.
            function_macro = bool(re.match(r"#\s*define\s+[A-Za-z_]\w*\(", _uncomment(raw)))
            value = " ".join(parts) + (" @function" if function_macro else "")
            result.append(Token(value, offsets[i], offsets[end - 1] + 1, True))
            i = end; line_start = False
            continue
        match = _LEX.match(source, i)
        assert match is not None
        value = match.group()
        start, end = i, match.end()
        i = end
        if value.isspace():
            if "\n" in value:
                line_start = True
            continue
        if value.startswith(("/*", "//")):
            if value.startswith("/*") and not value.endswith("*/"):
                raise ValueError("unterminated comment")
            if "\n" in value:
                line_start = True
            continue
        if value in {'"', "'"}:
            raise ValueError("unterminated literal")
        line_start = False
        result.append(Token(value, offsets[start], offsets[end - 1] + 1))
    return result


def _uncomment(text: str) -> str:
    return "".join(" " if m.group().startswith(("/*", "//")) else m.group() for m in _LEX.finditer(text))


def _plain_tokens(text: str) -> list[str]:
    return [m.group() for m in _LEX.finditer(text) if not m.group().isspace() and not m.group().startswith(("/*", "//"))]


def normalize_local_scale_factor(sources: dict[str, str]) -> dict[str, str]:
    """Replace the unique source definition, never a comment/literal/CLI flag."""
    definitions = [(name, token) for name, text in sources.items() for token in scan(text)
                   if token.directive and token.value.split()[:3] == ["#", "define", "LOCAL_SCALE_FACTOR"]]
    if len(definitions) != 1:
        raise ValueError(f"Embench requires exactly one source LOCAL_SCALE_FACTOR definition (found {len(definitions)})")
    name, token = definitions[0]
    if any(t.directive and t.value.split()[:3] == ["#", "undef", "LOCAL_SCALE_FACTOR"]
           for text in sources.values() for t in scan(text)):
        raise ValueError("ambiguous LOCAL_SCALE_FACTOR undefinition")
    if token.value.endswith(" @function"):
        raise ValueError("LOCAL_SCALE_FACTOR must be an object-like macro")
    result = dict(sources)
    result[name] = sources[name][:token.start] + "#define LOCAL_SCALE_FACTOR 1" + sources[name][token.end:]
    return result


def omit_warmup(text: str) -> str:
    """Remove warm_caches call statements, preserving all other main statements."""
    tokens = scan(text)
    cuts: list[tuple[int, int]] = []
    for i, token in enumerate(tokens[:-1]):
        if token.directive or token.value != "warm_caches" or tokens[i + 1].value != "(":
            continue
        depth, j = 1, i + 2
        while j < len(tokens) and depth:
            depth += (tokens[j].value == "(") - (tokens[j].value == ")")
            j += 1
        if depth or j >= len(tokens) or tokens[j].value != ";":
            raise ValueError("ambiguous warm_caches call in Embench main")
        cuts.append((token.start, tokens[j].end))
    for start, end in reversed(cuts):
        text = text[:start] + text[end:]
    return text


@dataclass(frozen=True)
class Object:
    name: str
    signature: tuple[str, ...]
    body: tuple[str, ...] | None
    scope: tuple[str, ...]
    start: int
    end: int


@dataclass
class Source:
    tokens: list[Token]
    objects: list[Object]
    directives: list[tuple[Token, tuple[str, ...]]]


def _function_name(header: list[Token]) -> str:
    values = [t.value for t in header]
    if "=" in values or "typedef" in values:
        return ""
    depth, name = 0, ""
    for i, value in enumerate(values):
        if value == "(":
            if depth == 0 and i and _IDENT.fullmatch(values[i - 1]) and values[i - 1] not in _ATTRIBUTES:
                name = values[i - 1]
            depth += 1
        elif value == ")":
            depth -= 1
    # Declaration-generating macro calls can precede a real definition without
    # semicolons (sglib). Keep that prefix fixed, but identify the final actual
    # declarator, not a preceding macro invocation, as the function interface.
    return name


def _parse(text: str) -> Source:
    tokens = scan(text)
    code = [t for t in tokens if not t.directive]
    objects: list[Object] = []
    directives: list[tuple[Token, tuple[str, ...]]] = []
    stack: list[str] = []
    for token in tokens:
        if not token.directive:
            continue
        parts = token.value.split()
        command = parts[1] if len(parts) > 1 else ""
        directives.append((token, tuple(stack)))
        if command in {"if", "ifdef", "ifndef"}:
            stack.append(token.value)
        elif command in {"else", "elif"}:
            if not stack:
                raise ValueError("unmatched preprocessor branch")
            stack[-1] += " / " + token.value
        elif command == "endif":
            if not stack:
                raise ValueError("unmatched #endif")
            stack.pop()
    if stack:
        raise ValueError("unclosed preprocessor condition")

    def scope_at(offset: int) -> tuple[str, ...]:
        active: list[str] = []
        for token, _ in directives:
            if token.start >= offset:
                break
            parts = token.value.split(); command = parts[1] if len(parts) > 1 else ""
            if command in {"if", "ifdef", "ifndef"}:
                active.append(token.value)
            elif command in {"else", "elif"}:
                active[-1] += " / " + token.value
            elif command == "endif":
                active.pop()
        return tuple(active)

    def values(start: int, end: int) -> tuple[str, ...]:
        return tuple(t.value for t in tokens if start <= t.start < end)

    begin, i, parens, brackets, braces = 0, 0, 0, 0, 0
    while i < len(code):
        token = code[i]; value = token.value
        if value == "{" and parens == brackets == braces == 0:
            name = _function_name(code[begin:i])
            if name:
                depth, j = 1, i + 1
                while j < len(code) and depth:
                    depth += (code[j].value == "{") - (code[j].value == "}")
                    j += 1
                if depth:
                    raise ValueError(f"unclosed function {name}")
                start, end = code[begin].start, code[j - 1].end
                objects.append(Object(name, values(start, token.start), values(token.start, end), scope_at(start), start, end))
                begin = i = j
                continue
        parens += (value == "(") - (value == ")")
        brackets += (value == "[") - (value == "]")
        braces += (value == "{") - (value == "}")
        if min(parens, brackets, braces) < 0:
            raise ValueError("unbalanced top-level source")
        if value == ";" and parens == brackets == braces == 0:
            start, end = code[begin].start, token.end
            if i > begin:
                objects.append(Object("", values(start, end), None, scope_at(start), start, end))
            begin = i + 1
        i += 1
    if begin != len(code) or parens or brackets or braces:
        raise ValueError("unknown or unterminated top-level object")
    return Source(tokens, objects, directives)


def _declared_names(obj: Object) -> set[str]:
    if obj.name:
        return {obj.name}
    # Declarator names, tags and enum constants. Initializer identifiers are not
    # declarations; original types may be used freely for fresh scratch storage.
    tokens = [v for v in obj.signature if not v.startswith("#")]
    names: set[str] = set()
    level = 0; segment: list[str] = []; initialized = False
    for i, value in enumerate(tokens):
        if value in {"struct", "union", "enum"} and i + 2 < len(tokens) and _IDENT.fullmatch(tokens[i + 1]) and tokens[i + 2] in {"{", ";"}:
            names.add(tokens[i + 1])
        if level == 0 and value in {",", ";", "="}:
            if not initialized:
                ids = [v for v in segment if _IDENT.fullmatch(v) and v not in _KEYWORDS and v not in _ATTRIBUTES]
                # Array bounds, argument lists and attributes are skipped below.
                if ids:
                    names.add(ids[-1])
            if value == "=":
                initialized = True
            else:
                segment = []; initialized = False
            continue
        if level == 0 and value not in {"(", "[", "{"} and not initialized:
            segment.append(value)
        level += (value in {"(", "[", "{"}) - (value in {")", "]", "}"})
    return names


def _owner(source: Source, token: Token) -> Object | None:
    return next((obj for obj in source.objects if obj.start <= token.start < obj.end), None)


def _directive_key(source: Source, token: Token, scope: tuple[str, ...]):
    owner = _owner(source, token)
    return (token.value, scope, owner.name if owner else "")


def _compare(reference: Source, candidate: Source, protected: set[str],
             external_declarations: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    ref_objects = Counter((o.name, o.signature, o.scope) for o in reference.objects)
    got_objects = Counter((o.name, o.signature, o.scope) for o in candidate.objects)
    for key, count in ref_objects.items():
        if got_objects[key] != count:
            errors.append(f"missing, changed or duplicate protected object {key[0] or 'declaration'}")
    ref_functions = Counter(o.name for o in reference.objects if o.name)
    got_functions = Counter(o.name for o in candidate.objects if o.name)
    for name, count in ref_functions.items():
        if got_functions[name] != count:
            errors.append(f"missing or duplicate original function {name}")
    for obj in reference.objects:
        if obj.name in ENTRY_FUNCTIONS:
            matches = [o for o in candidate.objects if (o.name, o.signature, o.scope) == (obj.name, obj.signature, obj.scope)]
            if len(matches) != 1 or matches[0].body != obj.body:
                errors.append(f"immutable {obj.name}() body was modified")
    declared = set().union(*(_declared_names(o) for o in reference.objects), _RESERVED,
                           external_declarations or set())
    new_names: set[str] = set()
    remaining = ref_objects.copy()
    for obj in candidate.objects:
        key = (obj.name, obj.signature, obj.scope)
        if remaining[key]:
            remaining[key] -= 1
            continue
        names = _declared_names(obj)
        if any(v in {"asm", "__asm", "__asm__"} for v in obj.signature):
            errors.append("unapproved new file-scope assembly or assembly interface alias")
        if not names:
            errors.append("unknown new top-level object")
        if names & (declared | new_names):
            errors.append("new object shadows protected/duplicate symbol: " + ", ".join(sorted(names & (declared | new_names))))
        new_names.update(names)
    # Original macro definitions (even inside a body) and all outside-body
    # directives retain their structure and conditional context.
    def fixed(source: Source):
        return Counter(_directive_key(source, t, scope) for t, scope in source.directives
                       if t.value.split()[1:2] in (["define"], ["undef"], ["pragma"])
                       or not (_owner(source, t) and _owner(source, t).name not in ENTRY_FUNCTIONS and _owner(source, t).body is not None))
    expected = fixed(reference); actual = fixed(candidate)
    for key, count in expected.items():
        if actual[key] != count:
            errors.append(f"missing, changed or duplicate original directive: {key[0]}")
    original = Counter(_directive_key(reference, t, scope) for t, scope in reference.directives)
    for token, scope in candidate.directives:
        key = _directive_key(candidate, token, scope)
        if original[key]:
            original[key] -= 1
            continue
        parts = token.value.split(); command = parts[1] if len(parts) > 1 else ""
        if command in {"define", "undef"}:
            name = parts[2] if len(parts) > 2 else ""
            if not name or name in protected or name in declared:
                errors.append(f"macro shadows protected symbol: {name}")
        elif command == "include":
            target = "".join(parts[2:])
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            elif target.startswith('"') and target.endswith('"'):
                target = target[1:-1]
            else:
                target = ""
            if target not in _STANDARD_HEADERS | {"cfu_intrinsics.h"}:
                errors.append("unapproved new include: " + token.value)
        elif command not in {"if", "ifdef", "ifndef", "elif", "else", "endif"}:
            errors.append("unapproved new preprocessor directive: " + token.value)
    baseline = Counter(v for t in reference.tokens for v in (_plain_tokens(t.value) if t.directive else [t.value]))
    current = Counter(v for t in candidate.tokens for v in (_plain_tokens(t.value) if t.directive else [t.value]))
    for value in _FORBIDDEN:
        if current[value] > baseline[value]:
            errors.append("forbidden new attribute/pragma mechanism: " + value)
    return errors


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_candidate(workdir: Path, run_root: str, name: str,
                       candidates: dict[str, str] | None = None) -> list[str]:
    """Validate actual sources, or pending canonical workdir-relative replacements.

    A full transaction map may include unrelated paths (such as Scala); those
    are ignored. Pending trusted/harness files must still match their hashes.
    Unknown paths inside this run's Embench workspaces always reject.
    Metadata is host-owned: <run_root>/04_benchmark/embench.yaml contains the
    project, measurement protocol and a versioned SHA256 integrity manifest.
    Missing/legacy metadata, missing files and changed trusted bytes all reject.
    """
    from . import embench
    workdir = Path(workdir).resolve()
    problems: list[str] = []
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid benchmark name")
        root = Path(run_root)
        if root.is_absolute() or ".." in root.parts or not root.parts:
            raise ValueError("invalid run root")
        metadata = workdir / root / "04_benchmark/embench.yaml"
        if metadata.is_symlink():
            raise ValueError("symlinked host metadata")
        if not metadata.is_file():
            raise ValueError("missing integrity metadata; start a new run")
        project = yaml.safe_load(metadata.read_text())
        measurement = embench.measurement_parameters()
        if not isinstance(project, dict) or project.get("benchmark") != name or project.get("measurement") != measurement:
            raise ValueError("missing or legacy measurement metadata; start a new run")
        integrity = project.get("integrity")
        if not isinstance(integrity, dict) or integrity.get("version") != INTEGRITY_VERSION:
            raise ValueError("missing or legacy integrity metadata; start a new run")
        refs, harness = integrity.get("reference_sources"), integrity.get("protected_files")
        if not isinstance(refs, dict) or not refs or not isinstance(harness, dict):
            raise ValueError("incomplete integrity hash manifest")
        cfgs = {c: embench.config_workspace(run_root, name, c) for c in embench.CONFIGS}
        all_pending = {} if candidates is None else candidates
        if not isinstance(all_pending, dict) or any(not isinstance(k, str) for k in all_pending):
            raise ValueError("invalid pending candidate map")
        metadata_rel = str(metadata.relative_to(workdir))
        if metadata_rel in all_pending and all_pending[metadata_rel] != metadata.read_text():
            raise ValueError("pending host integrity metadata modification")
        workspace_prefix = f"{root.as_posix()}/workspace/embench_"
        pending = {key: value for key, value in all_pending.items() if key.startswith(workspace_prefix)}
        if any(key.startswith(embench.EMBENCH_ROOT + "/") for key in all_pending):
            raise ValueError("pending read-only Embench source modification")
        allowed = set(refs) | set(harness) | {
            f"{cfgs['cfu']}/src/{Path(rel).name}" for rel in refs
        } | {f"{cfgs['cfu']}/include/cfu_intrinsics.h"}
        if any(key not in allowed or not isinstance(value, str) for key, value in pending.items()):
            raise ValueError("unknown pending candidate source path")
        required = {f"{cfg}/{rel}" for cfg in cfgs.values() for rel in
                    ["embench_main.c", *(f"support/{f}" for f in embench.SUPPORT_FILES), *(f"board/{f}" for f in embench.BOARD_FILES)]}
        if not required <= harness.keys():
            raise ValueError("missing required harness hashes")
        reference_root = workdir / cfgs["reference"] / "src"
        expected_sources = {str(p.relative_to(workdir)) for p in reference_root.glob("*.c")}
        if set(refs) != expected_sources:
            raise ValueError("missing/unknown trusted reference sources")
        parsed: dict[str, Source] = {}
        for rel, digest in {**refs, **harness}.items():
            path = workdir / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or not path.resolve().is_relative_to(workdir / root):
                raise ValueError("invalid integrity manifest path")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"required integrity file missing or symlinked: {rel}")
            text = pending[rel] if rel in pending else path.read_text(encoding="utf-8")
            candidate_bytes = text.encode("utf-8") if rel in pending else path.read_bytes()
            if not isinstance(digest, str) or hashlib.sha256(candidate_bytes).hexdigest() != digest:
                raise ValueError(f"trusted reference/harness hash mismatch: {rel}")
            if rel in refs:
                parsed[Path(rel).name] = _parse(text)
        if not parsed:
            raise ValueError("no trusted reference C source")
        entries = Counter(o.name for source in parsed.values() for o in source.objects if o.name in ENTRY_FUNCTIONS)
        if entries != Counter({name: 1 for name in ENTRY_FUNCTIONS}):
            raise ValueError("missing or duplicate trusted reference entry functions")
        # The required CFU/ref local headers are hashed just like support headers.
        for header in reference_root.glob("*.h"):
            for cfg in cfgs.values():
                if f"{cfg}/src/{header.name}" not in harness:
                    raise ValueError("missing original header hashes")
        protected = set(_RESERVED) | _KEYWORDS
        declared = set().union(*(_declared_names(o) for source in parsed.values() for o in source.objects), _RESERVED)
        for source in parsed.values():
            protected.update(t.value for t in source.tokens if not t.directive and _IDENT.fullmatch(t.value))
            protected.update(v for t, _ in source.directives for v in _plain_tokens(t.value) if _IDENT.fullmatch(v))
        scanned_harness: set[str] = set()
        for rel, digest in harness.items():
            if digest in scanned_harness:
                continue
            scanned_harness.add(digest)
            harness_text = (workdir / rel).read_text()
            # Some vendor headers use declaration-generating macros beyond this
            # scanner's grammar. Their bytes and macro identifiers remain fixed.
            try:
                harness_source = _parse(harness_text)
            except ValueError:
                harness_source = None
            if harness_source is not None:
                declared.update(name for obj in harness_source.objects for name in _declared_names(obj))
            for token in scan(harness_text):
                if not token.directive and _IDENT.fullmatch(token.value):
                    protected.add(token.value)
                elif token.directive:
                    protected.update(v for v in _plain_tokens(token.value) if _IDENT.fullmatch(v))
        cfu = cfgs["cfu"]
        original_headers = {p.name for p in reference_root.glob("*.h")}
        if {p.name for p in (workdir / cfu / "src").glob("*.h")} != original_headers:
            raise ValueError("missing/unknown CFU source headers")
        actual_sources = {p.name for p in (workdir / cfu / "src").glob("*.c")}
        if actual_sources != set(parsed):
            raise ValueError("missing/unknown CFU source files")
        for fname, reference in parsed.items():
            for cfg in cfgs.values():
                required_source = workdir / cfg / "src" / fname
                if not required_source.is_file() or required_source.is_symlink():
                    raise ValueError(f"required config source missing or symlinked: {cfg}/src/{fname}")
            rel = f"{cfu}/src/{fname}"
            path = workdir / rel
            if path.is_symlink():
                raise ValueError("symlinked CFU source")
            text = pending[rel] if rel in pending else path.read_text()
            reference_rel = f"{cfgs['reference']}/src/{fname}"
            if _digest(text) != refs[reference_rel]:
                problems.extend(f"src/{fname}: {error}" for error in _compare(reference, _parse(text), protected, declared))
        if any(p.name != "cfu_intrinsics.h" for p in (workdir / cfu / "include").glob("*.h")):
            raise ValueError("unknown CFU include header")
        header_rel = f"{cfu}/include/cfu_intrinsics.h"
        header = workdir / header_rel
        if header_rel in pending or header.exists():
            if header.is_symlink():
                raise ValueError("symlinked intrinsics header")
            text = pending[header_rel] if header_rel in pending else header.read_text()
            problems.extend("cfu_intrinsics.h: " + error for error in _compare(_parse(""), _parse(text), protected, declared))
    except (OSError, UnicodeError, ValueError, TypeError, IndexError, yaml.YAMLError) as exc:
        problems.append("Embench integrity validation failed: " + str(exc))
    return problems
