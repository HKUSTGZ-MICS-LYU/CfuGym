"""CFU ISA spec parsing, normalization, validation, and Markdown rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ISA_SPEC_VERSION = "1"
TEMPLATE_PATH = "agents/specs/cfu_isa_template.yaml"

# AgentCfuFiber.getCfuBusParameters sets CFU_FUNCTION_ID_W = 3 and the decoder
# reads func3 (bits 14:12), so a contract may only use function ids 0..7.
FUNCTION_ID_BITS = 3
MAX_FUNCTION_ID = (1 << FUNCTION_ID_BITS) - 1
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


def validate_isa_spec(data: dict[str, Any]) -> list[str]:
    """Return a sorted list of non-empty problems; empty list means valid."""
    errors: list[str] = []
    spec_version = data.get("spec_version", ISA_SPEC_VERSION)
    if str(spec_version) != ISA_SPEC_VERSION:
        errors.append(f"spec_version {spec_version!r} is not supported ({ISA_SPEC_VERSION})")

    identity = data.get("identity") or {}
    function_id_width: int | None = None
    if not isinstance(identity, dict):
        errors.append("identity must be a mapping")
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
        seen: dict[int, int] = {}
        for index, command in enumerate(commands):
            prefix = f"commands[{index}]"
            if not isinstance(command, dict):
                errors.append(f"{prefix} must be a mapping")
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
            elif func in seen:
                errors.append(f"{prefix}.function_id duplicates commands[{seen[func]}].function_id")
            else:
                seen[func] = index
            if not command.get("name"):
                errors.append(f"{prefix}.name is required")
            if not command.get("semantics"):
                errors.append(f"{prefix}.semantics is required")

    datapath = data.get("datapath")
    if datapath is not None and not isinstance(datapath, dict):
        errors.append("datapath must be a mapping when provided")
    # Datapath dimensions are independent, optional design metadata. A direct
    # scalar CFU may not use vector width, MAC slices, or a vector RF at all;
    # zero/null/"unused" are therefore valid declarations. Cross-parameter
    # checks belong to the selected hardware implementation, not generic ISA lint.

    if not data.get("integration"):
        errors.append("integration is required")
    software = data.get("software")
    if not software:
        errors.append("software is required")
    if not software.get("intrinsics_header"):
        errors.append("software.intrinsics_header is required")
    if not software.get("scalar_fallback"):
        errors.append("software.scalar_fallback is required")

    validation = data.get("validation") or {}
    if not isinstance(validation, dict):
        errors.append("validation must be a mapping")
    else:
        issues = [str(item) for item in validation.get("requirements", []) if item]
        if validation.get("status") in {"implemented", "validated"} and not issues:
            errors.append("validation.requirements cannot be empty for implemented/validated status")

    if not isinstance(data.get("assumptions", {}), dict):
        errors.append("assumptions must be a mapping")
    if not isinstance(data.get("unknowns", []), list):
        errors.append("unknowns must be a list")

    return sorted(errors)


def _document_candidates(text: str) -> Iterable[str]:
    """Yield the full response and fenced JSON/YAML blocks for compatibility."""
    yield text
    for match in re.finditer(r"```(?:json|yaml|yml)\s*\n(?P<body>.*?)```", text, re.IGNORECASE | re.DOTALL):
        yield match.group("body")


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
            return data, validate_isa_spec(data)
    if parse_error is not None:
        raise IsaLintError(f"CFU ISA spec does not parse: {parse_error}") from parse_error
    raise IsaLintError("CFU ISA spec must be a mapping")


def render_isa_markdown(data: dict[str, Any]) -> str:
    identity = data.get("identity", {})
    lines: list[str] = [
        "# CFU ISA Spec",
        f"- Name: {identity.get('name', '')}",
        f"- Opcode: {identity.get('opcode', '')} (function_id width {identity.get('function_id_width', '')} bits)",
        f"- Status: {identity.get('status', 'draft')}",
        f"- Reference workload: {identity.get('reference_workload', '')}",
        f"- Spec version: {data.get('spec_version', ISA_SPEC_VERSION)}",
        "",
        "## Commands",
        "| function_id | Command | Operand | Result | Semantics | Response |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for command in data.get("commands", []):
        lines.append(
            "| {flag} | {name} | {operand} | {result} | {semantics} | {response} |".format(
                flag=command.get("function_id", ""),
                name=command.get("name", ""),
                operand=command.get("operand", ""),
                result=command.get("result", ""),
                semantics=command.get("semantics", ""),
                response=command.get("response", ""),
            )
        )

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
        "\n".join(f"- {key}: {value}" for key, value in (data.get("assumptions") or {}).items()) or "n/a",
        "",
        "## Unknowns",
        "\n".join(f"- {item}" for item in data.get("unknowns") or []) or "n/a",
    ]
    return "\n".join(lines) + "\n"


def _render_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return "\n".join(f"- {key}: {item}" for key, item in value.items())


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
        if not 0 <= func_id <= MAX_FUNCTION_ID:
            continue
        symbols.append(f"cfu_{command_symbol(command, func_id)}")
    return symbols


def render_cfu_intrinsics(data: dict[str, Any]) -> str:
    """Render custom0 intrinsics for an AgentCfu ISA spec.

    Encoding follows the repo convention (CfuPlugin + custom_asm.h):
    opcode | rd<<7 | func3<<12 | rs1<<15 | rs2<<20. rd/rs1/rs2 are pinned to
    a0/a1/a2 so the emitted functions are self-contained.

    Only custom0 (0x0B) is emitted: it wires RS2 into the CFU input, which is
    what a two-register intrinsic needs. custom1 (0x2B) decodes an immediate
    instead, so ``validate_isa_spec`` rejects it for generated intrinsics.
    """
    identity = data.get("identity", {}) if isinstance(data.get("identity"), dict) else {}
    name = str(identity.get("name", "agent_cfu") or "agent_cfu")
    commands = [c for c in data.get("commands", []) if isinstance(c, dict)]
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
        cmd_name = str(command.get("name", f"op{func_id}") or f"op{func_id}")
        symbol = command_symbol(command, func_id)
        semantics = str(command.get("semantics", "")).replace("*/", "* /")
        lines += [
            f"/* function_id {func_id}: {cmd_name} - {semantics} */",
            f"#define CFU_{symbol.upper()}_FUNC3 {func_id}",
            f"static inline uint32_t cfu_{symbol}(uint32_t rs1, uint32_t rs2)",
            "{",
            '  register uint32_t rd asm("a0") = 0;',
            '  register uint32_t r1 asm("a1") = rs1;',
            '  register uint32_t r2 asm("a2") = rs2;',
            "  __asm__ volatile(",
            f'      ".word ((0x0B) | (10 << 7) | (11 << 15) | (12 << 20) | ({func_id} << 12))"',
            '      : "+r"(rd) : "r"(r1), "r"(r2) : "memory");',
            "  return rd;",
            "}",
            "",
        ]
    lines += [f"#endif /* CFU_INTRINSICS_H - {name} */", ""]
    return "\n".join(lines)

def render_list(value: Iterable[Any]) -> str:
    return "\n".join(f"- {item}" for item in value or [])
