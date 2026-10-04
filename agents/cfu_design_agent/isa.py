"""CFU ISA spec parsing, normalization, validation, and Markdown rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .runtime import parsing


ISA_SPEC_VERSION = "1"
TEMPLATE_PATH = "agents/specs/cfu_isa_template.yaml"

# AgentCfuFiber.getCfuBusParameters sets CFU_FUNCTION_ID_W = 3 and the decoder
# reads func3 (bits 14:12), so a contract may only use function ids 0..7.
FUNCTION_ID_BITS = 3
MAX_FUNCTION_ID = (1 << FUNCTION_ID_BITS) - 1
SUBFUNCTION_ID_BITS = 7
MAX_SUBFUNCTION_ID = (1 << SUBFUNCTION_ID_BITS) - 1
COMMAND_KINDS = frozenset({"compute", "control", "load", "store"})
RESERVED_INTRINSIC_SYMBOLS = frozenset({
    "enable", "memory_fence", "agent_counter_read", "agent_counter_elapsed",
})


def command_kind(command: dict[str, Any]) -> str:
    """Legacy function families supply defaults; explicit roles are authoritative."""
    func = command.get("function_id")
    default = {2: "control", 4: "load", 5: "store"}.get(func, "compute") if isinstance(func, int) else "compute"
    return command.get("kind", default)


def _valid_id(value: Any, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum

CUSTOM0_OPCODE = 0x0B
CUSTOM1_OPCODE = 0x2B


def parse_opcode(value: Any) -> int | None:
    """Map a declared opcode label to its 7-bit value, or ``None`` if unknown."""
    text = str(value or "").strip().lower()
    if "custom0" in text or "0x0b" in text:
        return CUSTOM0_OPCODE
    if "custom1" in text or "0x2b" in text:
        return CUSTOM1_OPCODE
    return None


class IsaLintError(ValueError):
    """Raised when a CFU ISA spec violates the schema or consistency rules."""


def load_isa_spec(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(errors="replace")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise IsaLintError(f"CFU ISA spec must be a mapping: {path}")
    return data


def _describe(value: Any) -> str:
    """What the model actually returned, for an actionable shape lint.

    "assumptions must be a mapping" cost run 20261003-130005 ten answer-repair
    rounds: the model kept sending a list of strings and the message never said
    what had been received.
    """
    if value is None:
        return "null"
    if isinstance(value, dict):
        return f"a mapping with {len(value)} key(s)"
    if isinstance(value, list):
        return f"a list of {len(value)} item(s)"
    if isinstance(value, str):
        return "an empty string" if not value else f"a string ({len(value)} chars)"
    if isinstance(value, bool):
        return f"a boolean ({value})"
    return f"a {type(value).__name__}"


def validate_cfu_resources(resources: Any, datapath: Any = None, prefix: str = "resources") -> list[str]:
    """Validate explicit resource declarations; absence is handled by the caller.

    cpu_value indices use the entire CPU value (hardware must bounds-check before
    narrowing). legacy_register_number explicitly retains historical truncation.
    """
    if not isinstance(resources, list):
        return [f"{prefix} must be a list"]
    errors: list[str] = []
    roles: set[str] = set()
    writes = 0
    depth = datapath.get("regDepth") if isinstance(datapath, dict) else None
    known_depth = isinstance(depth, int) and not isinstance(depth, bool) and depth > 0
    for n, resource in enumerate(resources):
        path = f"{prefix}[{n}]"
        if not isinstance(resource, dict):
            errors.append(f"{path} must be a mapping")
            continue
        storage, access, role = (resource.get(k) for k in ("storage", "access", "role"))
        expected = {"storage", "access", "role"}
        if storage == "vector":
            expected.add("index")
        if set(resource) != expected:
            errors.append(f"{path} keys must be exactly {', '.join(sorted(expected))}")
        if not isinstance(storage, str) or storage not in ("cpu", "vector"):
            errors.append(f"{path}.storage must be cpu or vector")
        if not isinstance(access, str) or access not in ("read", "write"):
            errors.append(f"{path}.access must be read or write")
        allowed = ("rs1", "rs2") if storage == "cpu" and access == "read" else ("rd",) if storage == "cpu" and access == "write" else ("vs1", "vs2") if storage == "vector" and access == "read" else ("vd",) if storage == "vector" and access == "write" else ()
        if not isinstance(role, str) or role not in allowed:
            errors.append(f"{path}.role is invalid for storage/access")
        if isinstance(role, str):
            if role in roles:
                errors.append(f"{path}.role duplicates {role}")
            roles.add(role)
        writes += access == "write"
        if storage != "vector":
            continue
        index = resource.get("index")
        if not isinstance(index, dict):
            errors.append(f"{path}.index must be a mapping")
            continue
        source = index.get("source")
        if not isinstance(source, str) or source not in ("cpu_value", "constant", "legacy_register_number"):
            errors.append(f"{path}.index.source must be cpu_value, constant, or legacy_register_number")
            continue
        keys = {"source", "value"} if source == "constant" else {"source", "register"}
        if set(index) != keys:
            errors.append(f"{path}.index keys must be exactly {', '.join(sorted(keys))}")
        if source == "constant":
            value = index.get("value")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{path}.index.value must be a nonnegative integer")
            elif known_depth and value >= depth:
                errors.append(f"{path}.index.value must be less than datapath.regDepth ({depth})")
        elif not isinstance(index.get("register"), str) or index["register"] not in ("rs1", "rs2"):
            errors.append(f"{path}.index.register must be rs1 or rs2 (legacy rd is not supported)")
    if writes > 1:
        errors.append(f"{prefix} permits at most one data write (CPU RD or vector VD)")
    return sorted(errors)


def _resource_description(resource: dict[str, Any]) -> str:
    """Human-readable direction, storage and index provenance for valid resources."""
    role = resource["role"].upper()
    if resource["storage"] == "cpu":
        return f"{resource['access']} CPU {role} value"
    index = resource["index"]
    if index["source"] == "constant":
        origin = f"constant {index['value']}"
    elif index["source"] == "cpu_value":
        origin = f"CPU {index['register'].upper()} value (full-width bounds check)"
    else:
        origin = f"legacy instruction {index['register'].upper()} register number (truncated)"
    return f"{resource['access']} vector {role}, index = {origin}"


def validate_isa_spec(data: dict[str, Any]) -> list[str]:
    """Return a sorted list of non-empty problems; empty list means valid."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["CFU ISA spec must be a mapping"]
    spec_version = data.get("spec_version", ISA_SPEC_VERSION)
    if str(spec_version) != ISA_SPEC_VERSION:
        errors.append(f"spec_version {spec_version!r} is not supported ({ISA_SPEC_VERSION})")

    identity = data.get("identity") or {}
    function_id_width: int | None = None
    if not isinstance(identity, dict):
        errors.append(f"identity must be a mapping (received {_describe(identity)})")
    else:
        for key in ("name", "opcode", "function_id_width"):
            if not identity.get(key):
                errors.append(f"identity.{key} is required")
        width = identity.get("function_id_width")
        if isinstance(width, bool):
            errors.append("identity.function_id_width must be an integer")
        elif isinstance(width, int):
            if not 1 <= width <= FUNCTION_ID_BITS:
                errors.append(
                    f"identity.function_id_width must be in 1..{FUNCTION_ID_BITS} "
                    f"(AgentCfu func3), got {width}"
                )
            else:
                function_id_width = width
        elif identity.get("function_id_width"):
            errors.append("identity.function_id_width must be an integer")
        sub_width = identity.get("subfunction_id_width", SUBFUNCTION_ID_BITS)
        if not isinstance(sub_width, int) or isinstance(sub_width, bool) or sub_width != SUBFUNCTION_ID_BITS:
            errors.append("identity.subfunction_id_width must be 7 (funct7)")
        opcode = parse_opcode(identity.get("opcode", ""))
        if opcode is None:
            errors.append("identity.opcode must name custom0 (0x0B) or custom1 (0x2B)")
        elif opcode != CUSTOM0_OPCODE:
            # render_cfu_intrinsics emits the register-register custom0 form;
            # custom1 decodes its second operand as an immediate (IMM_I), so it
            # cannot be served by the generated intrinsic header.
            errors.append(
                "identity.opcode custom1 (0x2B) is not supported by the generated "
                "register-register intrinsics; declare custom0 (0x0B)"
            )

    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    else:
        seen: dict[tuple[int, int], int] = {}
        names: set[str] = set()
        symbols: set[str] = set()
        macros: set[str] = set()
        for index, command in enumerate(commands):
            prefix = f"commands[{index}]"
            if not isinstance(command, dict):
                errors.append(f"{prefix} must be a mapping (received {_describe(command)})")
                continue
            func = command.get("function_id")
            if func is None:
                errors.append(f"{prefix}.function_id is required")
            elif isinstance(func, bool) or not isinstance(func, int):
                errors.append(
                    f"{prefix}.function_id must be an integer in 0..{MAX_FUNCTION_ID}"
                )
            elif not 0 <= func <= MAX_FUNCTION_ID:
                errors.append(
                    f"{prefix}.function_id {func} is outside the {FUNCTION_ID_BITS}-bit "
                    f"func3 range 0..{MAX_FUNCTION_ID}"
                )
            elif function_id_width is not None and func >= (1 << function_id_width):
                errors.append(
                    f"{prefix}.function_id {func} does not fit the declared "
                    f"function_id_width {function_id_width}"
                )
            sub = command.get("subfunction_id", 0)
            if not _valid_id(sub, MAX_SUBFUNCTION_ID):
                errors.append(f"{prefix}.subfunction_id must be an integer in 0..127")
            elif _valid_id(func, MAX_FUNCTION_ID):
                key = (func, sub)
                if key in seen:
                    errors.append(f"{prefix} encoding duplicates commands[{seen[key]}] (function_id, subfunction_id)")
                else:
                    seen[key] = index
            kind = command_kind(command) if _valid_id(func, MAX_FUNCTION_ID) else command.get("kind", "compute")
            if not isinstance(kind, str) or kind not in COMMAND_KINDS:
                errors.append(f"{prefix}.kind must be compute, control, load, or store")
            name = command.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{prefix}.name is required and must be a non-empty string")
            elif not re.fullmatch(r"[A-Za-z0-9_ .-]+", name) or not re.search(r"[A-Za-z0-9]", name):
                errors.append(f"{prefix}.name must contain ASCII letters/digits with only space, underscore, dot or hyphen separators")
            else:
                symbol = command_symbol(command, func if isinstance(func, int) else 0)
                if name in names:
                    errors.append(f"{prefix}.name duplicates an earlier command")
                if symbol in symbols or symbol.upper() in macros:
                    errors.append(f"{prefix} generated symbol/macro conflicts with an earlier command")
                if symbol.lower() in RESERVED_INTRINSIC_SYMBOLS:
                    errors.append(f"{prefix} generated symbol cfu_{symbol} is a reserved helper")
                names.add(name)
                symbols.add(symbol)
                macros.add(symbol.upper())
            if not command.get("semantics"):
                errors.append(f"{prefix}.semantics is required")
            if "resources" in command:
                errors.extend(validate_cfu_resources(command["resources"], data.get("datapath"), f"{prefix}.resources"))

    datapath = data.get("datapath")
    if datapath is not None and not isinstance(datapath, dict):
        errors.append("datapath must be a mapping when provided"
                      f" (received {_describe(datapath)})")
    if isinstance(datapath, dict) and "implementation" in datapath and not isinstance(datapath["implementation"], dict):
        implementation = datapath["implementation"]
        errors.append("datapath.implementation must be a mapping when provided"
                      f" (received {_describe(implementation)})")
    # Legacy flat algorithm-specific metadata remains readable; new templates
    # deliberately put optional implementation choices in a namespaced mapping.
    # Datapath dimensions are independent, optional design metadata. A direct
    # scalar CFU may not use vector width or a vector RF at all;
    # zero/null/"unused" are therefore valid declarations. Cross-parameter
    # checks belong to the selected hardware implementation, not generic ISA lint.

    if not data.get("integration"):
        errors.append("integration is required")
    software = data.get("software")
    if not isinstance(software, dict):
        errors.append(f"software is required and must be a mapping (received {_describe(software)})")
        software = {}
    if not software.get("intrinsics_header"):
        errors.append("software.intrinsics_header is required")
    if not software.get("scalar_fallback"):
        errors.append("software.scalar_fallback is required")

    validation = data.get("validation") or {}
    if not isinstance(validation, dict):
        errors.append(f"validation must be a mapping (received {_describe(validation)})")
    else:
        requirements = validation.get("requirements", [])
        if not isinstance(requirements, list):
            errors.append("validation.requirements must be a list")
            requirements = []
        issues = [str(item) for item in requirements if item]
        if validation.get("status") in ("implemented", "validated") and not issues:
            errors.append("validation.requirements cannot be empty for implemented/validated status")

    assumptions = data.get("assumptions", {})
    if not isinstance(assumptions, dict):
        errors.append("assumptions must be a mapping of name: text"
                      f" (received {_describe(assumptions)})")
    unknowns = data.get("unknowns", [])
    if not isinstance(unknowns, list):
        errors.append("unknowns must be a list of strings"
                      f" (received {_describe(unknowns)})")

    return sorted(errors)


def validate_isa_extension(previous: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Validate an append-only ISA revision without mutating either input.

    All old command fields (including arbitrary semantic metadata) are frozen.
    Omitted legacy subfunction/kind defaults compare equal to explicit defaults.
    Shared state/memory/datapath/integration/operations_contract and software ABI
    remain identical: structural lint cannot prove a changed width/reset/layout
    preserves old instructions. This deliberately conservative API permits new
    operations over the existing shared ABI, not incompatible architecture edits.
    Analysis/validation provenance may evolve; intrinsic_names may only grow.
    """
    errors: list[str] = []
    for label, spec in (("previous", previous), ("candidate", candidate)):
        if not isinstance(spec, dict):
            errors.append(f"{label} must be a mapping")
        else:
            errors.extend(f"{label}: {error}" for error in validate_isa_spec(spec))
    if errors:
        return sorted(errors)
    old_identity, new_identity = previous["identity"], candidate["identity"]
    for key in (set(old_identity) | set(new_identity)) - {"status", "subfunction_id_width", "opcode"}:
        if old_identity.get(key) != new_identity.get(key):
            errors.append(f"identity.{key} cannot change in an ISA extension")
    if parse_opcode(old_identity["opcode"]) != parse_opcode(new_identity["opcode"]):
        errors.append("identity.opcode cannot change in an ISA extension")
    def normalized(command: dict[str, Any]) -> dict[str, Any]:
        return {**command, "subfunction_id": command.get("subfunction_id", 0), "kind": command_kind(command)}
    new_commands = {(c["function_id"], c.get("subfunction_id", 0)): normalized(c) for c in candidate["commands"]}
    for command in previous["commands"]:
        key = (command["function_id"], command.get("subfunction_id", 0))
        if key not in new_commands:
            errors.append(f"old command {command['name']} at {key} was removed or re-encoded")
        elif normalized(command) != new_commands[key]:
            errors.append(f"old command {command['name']} at {key} changed semantic/encoding fields")
    for key in ("state", "memory", "datapath", "integration", "operations_contract", "assumptions"):
        if previous.get(key, {}) != candidate.get(key, {}):
            errors.append(f"{key} cannot change in a conservative ISA extension")
    old_sw, new_sw = previous["software"], candidate["software"]
    for key in (set(old_sw) | set(new_sw)) - {"intrinsic_names"}:
        if old_sw.get(key) != new_sw.get(key):
            errors.append(f"software.{key} cannot change in an ISA extension")
    old_names, new_names = old_sw.get("intrinsic_names", []), new_sw.get("intrinsic_names", [])
    if not isinstance(old_names, list) or not isinstance(new_names, list) or any(name not in new_names for name in old_names):
        errors.append("software.intrinsic_names must preserve all old entries")
    return sorted(errors)


def _document_candidates(text: str) -> Iterable[str]:
    """Yield the full response and fenced JSON/YAML blocks for compatibility.

    Shared with every other YAML answer (``parsing.document_candidates``) so the
    contract and the workload analysis accept exactly the same shapes.
    """
    return parsing.document_candidates(text)


# A model can wrap the contract in an envelope instead of returning it directly.
# Observed: GLM returning {"analysis": ..., "roofline": {...},
# "isa_contract": "<yaml string>"}. Accept the nested contract when the top level
# is clearly not one, so a formatting slip costs nothing instead of a whole run.
CONTRACT_KEYS = ("identity", "commands")
_WRAPPER_KEYS = ("isa_contract", "cfu_isa", "isa", "cfu_isa_spec", "contract", "spec")


def unwrap_isa_contract(data: Any) -> dict[str, Any]:
    """Return the nested CFU contract when a model wrapped it in an envelope.

    The contract may be nested as a mapping or as a string holding YAML/JSON. The
    top-level object is returned unchanged when it already looks like a contract, or
    when no candidate wrapper holds one.
    """
    if not isinstance(data, dict) or any(key in data for key in CONTRACT_KEYS):
        return data
    for key in _WRAPPER_KEYS:
        nested = data.get(key)
        if isinstance(nested, str):
            try:
                nested = yaml.safe_load(nested)
            except yaml.YAMLError:
                continue
        if isinstance(nested, dict) and any(candidate in nested for candidate in CONTRACT_KEYS):
            return nested
    return data


def validate_isa_text(text: str) -> tuple[dict[str, Any], list[str]]:
    parse_error: Exception | None = None
    for candidate in _document_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            parse_error = exc
            try:
                data = yaml.safe_load(candidate)
            except yaml.YAMLError as yaml_exc:
                parse_error = yaml_exc
                continue
        if isinstance(data, dict):
            data = unwrap_isa_contract(data)
            return data, validate_isa_spec(data)
    if parse_error is not None:
        raise IsaLintError(f"CFU ISA spec does not parse: {parse_error}") from parse_error
    raise IsaLintError("CFU ISA spec must be a mapping")


def _cell(value: Any) -> str:
    """One Markdown table cell: never breaks the row for any JSON-ish value."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _render_mapping(value: Any) -> str:
    """Bullet lines for a mapping, tolerating a rejected candidate's malformed value.

    A candidate that failed validation can hold a list or a scalar where the schema
    wants a mapping; rendering must degrade to bullets instead of raising.
    """
    if isinstance(value, dict):
        return "\n".join(f"- {key}: {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)) and value:
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, str) and value:
        return f"- {value}"
    return ""


def _render_sequence(value: Any) -> str:
    """Bullet lines for a sequence, tolerating malformed values the same way."""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return _render_mapping(value)


def render_isa_markdown(data: dict[str, Any]) -> str:
    """Render a CFU ISA contract as Markdown.

    This formatter also runs on candidates that FAILED validation, because the isa
    rung records every attempt and feeds the repair prompt. It must therefore never
    raise on malformed input: a crash here aborts the run instead of reporting the
    lint problems that are supposed to drive the repair. Unknown or wrong-typed
    fields render as empty cells or "n/a".
    """
    if not isinstance(data, dict):
        data = {}
    identity = data.get("identity") if isinstance(data.get("identity"), dict) else {}
    commands = data.get("commands") if isinstance(data.get("commands"), list) else []
    lines: list[str] = [
        "# CFU ISA Spec",
        f"- Name: {identity.get('name', '')}",
        f"- Opcode: {identity.get('opcode', '')} (function_id width {identity.get('function_id_width', '')} bits)",
        f"- Status: {identity.get('status', 'draft')}",
        f"- Reference workload: {identity.get('reference_workload', '')}",
        f"- Spec version: {data.get('spec_version', ISA_SPEC_VERSION)}",
        "",
        "## Commands",
        "| function_id | subfunction_id | Kind | Command | Operand | Result | Semantics | Response |",
        "| ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for command in commands:
        if not isinstance(command, dict):
            lines.append(f"| | | | {_cell(command)} | | | | |")
            continue
        lines.append(
            "| {flag} | {sub} | {kind} | {name} | {operand} | {result} | {semantics} | {response} |".format(
                flag=_cell(command.get("function_id", "")),
                sub=_cell(command.get("subfunction_id", 0)),
                kind=_cell(command_kind(command)),
                name=_cell(command.get("name", "")),
                operand=_cell(command.get("operand", "")),
                result=_cell(command.get("result", "")),
                semantics=_cell(command.get("semantics", "")),
                response=_cell(command.get("response", "")),
            )
        )

    explicit = [c for c in commands if isinstance(c, dict) and "resources" in c]
    if explicit:
        lines += ["", "## CfuResource bindings", "| Command | Resources | CPU completion |", "| --- | --- | --- |"]
        for command in explicit:
            resources = command["resources"]
            problems = validate_cfu_resources(resources, data.get("datapath"))
            bindings = "; ".join(_resource_description(r) for r in resources) if not problems else "INVALID: " + "; ".join(problems)
            cpu_rd = not problems and any(r["storage"] == "cpu" and r["access"] == "write" for r in resources)
            lines.append(f"| {_cell(command.get('name', ''))} | {bindings or 'none (ack only)'} | {'result to CPU RD' if cpu_rd else 'ack; CPU rd=x0'} |")

    works = data.get("operations_contract", "")
    if isinstance(works, dict):
        works = ", ".join(f"{key}: {value}" for key, value in works.items())
    lines += [
        "",
        "## Operations Contract",
        str(works or "n/a"),
        "",
        "## Roofline",
        _render_mapping(data.get("roofline")) or "n/a",
        "",
        "## State & Reset",
        _render_mapping(data.get("state")) or "n/a",
        "",
        "## Memory & Ordering",
        _render_mapping(data.get("memory")) or "n/a",
        "",
        "## Datapath",
        _render_mapping(data.get("datapath")) or "n/a",
        "",
        "## Software",
        _render_mapping(data.get("software")) or "n/a",
        "",
        "## Integration",
        _render_mapping(data.get("integration")) or "n/a",
        "",
        "## Validation",
        _render_mapping(data.get("validation")) or "n/a",
        "",
        "## Assumptions",
        _render_mapping(data.get("assumptions")) or "n/a",
        "",
        "## Unknowns",
        _render_sequence(data.get("unknowns")) or "n/a",
    ]
    return "\n".join(lines) + "\n"


CFU_OPCODE_CUSTOM0 = 0x0B


def command_symbol(command: dict[str, Any], func_id: int) -> str:
    """C identifier render_cfu_intrinsics uses for one command."""
    name = str(command.get("name", f"op{func_id}") or f"op{func_id}")
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or f"op{func_id}"


def intrinsic_symbols(data: dict[str, Any]) -> list[str]:
    """CFU intrinsic function names a spec renders (for example ``cfu_compute``).

    Used to verify that the accelerated software actually calls the CFU
    datapath instead of reverting to a scalar loop inside the CFU build. Only
    commands the renderer can encode (integer function id in the func3 range)
    are reported.
    """
    commands = data.get("commands", []) if isinstance(data, dict) else []
    symbols: list[str] = []
    if not isinstance(commands, list):
        return symbols
    for command in commands:
        if not isinstance(command, dict):
            continue
        func_id = command.get("function_id")
        if isinstance(func_id, bool) or not isinstance(func_id, int):
            continue
        if not 0 <= func_id <= MAX_FUNCTION_ID or not _valid_id(command.get("subfunction_id", 0), MAX_SUBFUNCTION_ID):
            continue
        symbols.append(f"cfu_{command_symbol(command, func_id)}")
    return symbols


def _resource_intrinsic(command: dict[str, Any], datapath: Any) -> list[str]:
    resources = command["resources"]
    errors = validate_cfu_resources(resources, datapath)
    if errors:
        raise IsaLintError("; ".join(errors))
    values: set[str] = set()
    physical: set[str] = set()
    cpu_rd = False
    for resource in resources:
        if resource["storage"] == "cpu":
            if resource["access"] == "read":
                values.add(resource["role"])
            else:
                cpu_rd = True
        else:
            index = resource["index"]
            if index["source"] == "cpu_value":
                values.add(index["register"])
            elif index["source"] == "legacy_register_number":
                physical.add(index["register"])
    physical |= values
    func, sub = command["function_id"], command.get("subfunction_id", 0)
    symbol = command_symbol(command, func)
    descriptions = "; ".join(_resource_description(r) for r in resources) or "no data resources; ack only"
    parameters = ", ".join(f"uint32_t {r}" for r in ("rs1", "rs2") if r in values) or "void"
    lines = [f"/* function_id {func}: {symbol} - {descriptions}; completion {'CPU RD' if cpu_rd else 'ack, rd=x0'} */",
             f"#define CFU_{symbol.upper()}_FUNC3 {func}",
             f"#define CFU_{symbol.upper()}_SUBFUNCTION_ID {sub}",
             f"static inline {'uint32_t' if cpu_rd else 'void'} cfu_{symbol}({parameters})", "{"]
    if cpu_rd:
        lines.append('  register uint32_t rd asm("a0") = 0;')
    inputs = []
    for n, role in ((1, "rs1"), (2, "rs2")):
        if role in physical:
            lines.append(f'  register uint32_t r{n} asm("a{n}") = {role if role in values else "0"};')
            inputs.append(f'"r"(r{n})')
    # Every nonzero raw register field has an explicit compiler operand, including
    # legacy number-only fields whose actual value is irrelevant. Unused fields
    # are x0; vector-only/ack instructions cannot clobber an undeclared CPU RD.
    rd, rs1, rs2 = (10 if cpu_rd else 0), (11 if "rs1" in physical else 0), (12 if "rs2" in physical else 0)
    sub_encoding = f" | ({sub} << 25)" if sub else ""
    outputs = '\"+r\"(rd)' if cpu_rd else ""
    lines += ["  __asm__ volatile(",
              f'      ".word ((0x0B) | ({rd} << 7) | ({rs1} << 15) | ({rs2} << 20) | ({func} << 12){sub_encoding})"',
              f'      : {outputs} : {", ".join(inputs)} : "memory");']
    if cpu_rd:
        lines.append("  return rd;")
    return lines + ["}", ""]


def render_cfu_intrinsics(data: dict[str, Any]) -> str:
    """Render custom0 intrinsics for an AgentCfu ISA spec.

    Encoding follows the repo convention (CfuPlugin + custom_asm.h):
    opcode | rd<<7 | func3<<12 | rs1<<15 | rs2<<20 | subfunction_id<<25.
    subfunction_id defaults to zero for legacy contracts. Used rd/rs1/rs2 fields
    are pinned to a0/a1/a2 with explicit compiler operands. Absent resources keep
    the legacy two-argument uint32_t ABI byte-for-byte; explicit resources emit
    only CPU-value arguments, x0 for unused fields, and void when CPU RD is not
    a declared destination. Legacy register-number indices retain pinned fields.

    Only custom0 (0x0B) is emitted: it wires RS2 into the CFU input, which is
    what a two-register intrinsic needs. custom1 (0x2B) decodes an immediate
    instead, so ``validate_isa_spec`` rejects it for generated intrinsics.
    """
    if not isinstance(data, dict):
        data = {}
    identity = data.get("identity", {}) if isinstance(data.get("identity"), dict) else {}
    name = str(identity.get("name", "agent_cfu") or "agent_cfu")
    commands = [
        c for c in (data.get("commands") if isinstance(data.get("commands"), list) else [])
        if isinstance(c, dict)
    ]
    lines = [
        "/* Generated from the CFU ISA spec. Do not edit by hand. */",
        "#ifndef CFU_INTRINSICS_H",
        "#define CFU_INTRINSICS_H",
        "",
        "#include <stdint.h>",
        "",
        "/* CSR 0xBC0 bit 31 enables the CFU. */",
        "static inline void cfu_enable(void)",
        "{",
        '  __asm__ volatile("li t1, 0x80000000\\n\\t csrs 0xBC0, t1" ::: "t1");',
        "}",
        "",
        "static inline void cfu_memory_fence(void)",
        "{",
        '  __asm__ volatile("fence rw, rw" ::: "memory");',
        "}",
        "",
    ]
    for command in commands:
        func_id = command.get("function_id")
        if isinstance(func_id, bool) or not isinstance(func_id, int):
            continue
        if not 0 <= func_id <= MAX_FUNCTION_ID:
            # func3 is 3 bits; a wider id would overwrite the rs1 field.
            continue
        sub_id = command.get("subfunction_id", 0)
        if not _valid_id(sub_id, MAX_SUBFUNCTION_ID):
            continue
        if "resources" in command:
            lines += _resource_intrinsic(command, data.get("datapath"))
            continue
        # Keep legacy assembly text byte-identical when the new field is zero.
        sub_encoding = f" | ({sub_id} << 25)" if sub_id else ""
        cmd_name = str(command.get("name", f"op{func_id}") or f"op{func_id}")
        symbol = command_symbol(command, func_id)
        semantics = str(command.get("semantics", "")).replace("*/", "* /")
        lines += [
            f"/* function_id {func_id}: {cmd_name} - {semantics} */",
            f"#define CFU_{symbol.upper()}_FUNC3 {func_id}",
            f"#define CFU_{symbol.upper()}_SUBFUNCTION_ID {sub_id}",
            f"static inline uint32_t cfu_{symbol}(uint32_t rs1, uint32_t rs2)",
            "{",
            '  register uint32_t rd asm("a0") = 0;',
            '  register uint32_t r1 asm("a1") = rs1;',
            '  register uint32_t r2 asm("a2") = rs2;',
            "  __asm__ volatile(",
            f'      ".word ((0x0B) | (10 << 7) | (11 << 15) | (12 << 20) | ({func_id} << 12){sub_encoding})"',
            '      : "+r"(rd) : "r"(r1), "r"(r2) : "memory");',
            "  return rd;",
            "}",
            "",
        ]
    lines += [f"#endif /* CFU_INTRINSICS_H - {name} */", ""]
    return "\n".join(lines)
