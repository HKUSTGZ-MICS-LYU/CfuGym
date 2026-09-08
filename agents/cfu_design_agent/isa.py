"""CFU ISA spec parsing, normalization, validation, and Markdown rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


ISA_SPEC_VERSION = "1"
TEMPLATE_PATH = "agents/specs/cfu_isa_template.yaml"


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
    if not isinstance(identity, dict):
        errors.append("identity must be a mapping")
    else:
        for key in ("name", "opcode", "function_id_width"):
            if not identity.get(key):
                errors.append(f"identity.{key} is required")

    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    else:
        seen: dict[str, int] = {}
        for index, command in enumerate(commands):
            prefix = f"commands[{index}]"
            if not isinstance(command, dict):
                errors.append(f"{prefix} must be a mapping")
                continue
            func = command.get("function_id")
            if func is None:
                errors.append(f"{prefix}.function_id is required")
            else:
                entry = str(func)
                if entry in seen:
                    errors.append(f"{prefix}.function_id duplicates commands[{seen[entry]}].function_id")
                else:
                    seen[entry] = index
            if not command.get("name"):
                errors.append(f"{prefix}.name is required")
            if not command.get("semantics"):
                errors.append(f"{prefix}.semantics is required")

    datapath = data.get("datapath") or {}
    if not isinstance(datapath, dict):
        errors.append("datapath must be a mapping")
    else:
        for key in ("vlen", "xlen", "maclen", "accWidth"):
            value = datapath.get(key)
            if value is None:
                errors.append(f"datapath.{key} is required")
        if isinstance(datapath.get("vlen"), int) and isinstance(datapath.get("xlen"), int):
            if datapath["xlen"] and datapath["vlen"] % datapath["xlen"] != 0:
                errors.append("datapath.vlen must be a multiple of datapath.xlen")
        if isinstance(datapath.get("vlen"), int) and isinstance(datapath.get("maclen"), int):
            if datapath["maclen"] and datapath["vlen"] % datapath["maclen"] != 0:
                errors.append("datapath.vlen must be a multiple of datapath.maclen")
        reg_depth = datapath.get("regDepth")
        if isinstance(reg_depth, int) and reg_depth < 2:
            errors.append("datapath.regDepth must be at least 2")

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


def validate_isa_text(text: str) -> tuple[dict[str, Any], list[str]]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise IsaLintError(f"CFU ISA spec does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise IsaLintError("CFU ISA spec must be a mapping")
    return data, validate_isa_spec(data)


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


def render_list(value: Iterable[Any]) -> str:
    return "\n".join(f"- {item}" for item in value or [])
