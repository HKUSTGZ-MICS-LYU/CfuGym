"""WorkloadSpec ingestion and normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .embench import list_benchmarks
from .llm import OpenAiTextClient
from .state import RepoContext, WorkloadSpec


SPEC_VERSION = "1"
SPEC_REQUIRED = ("name", "source_kind", "operation", "pattern", "validation", "unknowns")
DEFAULT_SCALE_FACTOR = 1
DEFAULT_WARMUP_HEAT = 1

SPEC_SYSTEM_PROMPT = """You convert CFU workload requests into a normalized YAML WorkloadSpec.
Use only facts present in the input and repo context. For missing high-impact fields, add them to `unknowns` instead of guessing.
Return YAML only, with keys: spec_version, name, source_kind, description, project_root, kernel_file, kernel_function, build_cmd, run_cmd, test_cmd, operation, pattern, element_type, signedness, output_type, vector_bytes, memory_access, alignment_bytes, cfu_style, validation, unknowns."""


def normalize_workload_spec(
    *,
    task: str,
    input_mode: str,
    context: RepoContext,
    workdir: Path,
    workload_spec_path: str = "",
    project_root: str = "",
    kernel_file: str = "",
    kernel_function: str = "",
    build_cmd: str = "",
    run_cmd: str = "",
    test_cmd: str = "",
    benchmark_name: str = "",
    llm: OpenAiTextClient | None = None,
) -> tuple[WorkloadSpec, str]:
    if input_mode == "embench" or benchmark_name:
        spec = infer_embench_spec(task, benchmark_name or infer_benchmark_name(task, workdir))
        return sanitize_spec(spec), yaml.safe_dump(dict(spec), sort_keys=False)

    if input_mode == "formatted_spec":
        spec = load_formatted_spec(workdir / workload_spec_path)
        spec.setdefault("source_kind", "formatted_spec")
        spec.setdefault("description", task)
        spec.setdefault("unknowns", [])
        return sanitize_spec(spec), yaml.safe_dump(dict(spec), sort_keys=False)

    if input_mode == "c_project":
        code = read_optional(workdir / kernel_file)
        fallback = infer_spec_from_text(
            "\n".join([task, code]),
            source_kind="c_project",
            project_root=project_root,
            kernel_file=kernel_file,
            kernel_function=kernel_function,
            build_cmd=build_cmd,
            run_cmd=run_cmd,
            test_cmd=test_cmd,
        )
        spec = llm_extract_or_fallback(task, context, fallback, llm)
        spec.update(
            {
                "source_kind": "c_project",
                "description": task,
                "project_root": project_root,
                "kernel_file": kernel_file,
                "kernel_function": kernel_function,
                "build_cmd": build_cmd,
                "run_cmd": run_cmd,
                "test_cmd": test_cmd,
            }
        )
        return sanitize_spec(spec), yaml.safe_dump(dict(spec), sort_keys=False)

    fallback = infer_spec_from_text(task, source_kind="natural_language")
    spec = llm_extract_or_fallback(task, context, fallback, llm)
    spec.setdefault("source_kind", "natural_language")
    return sanitize_spec(spec), yaml.safe_dump(dict(spec), sort_keys=False)


def infer_benchmark_name(task: str, workdir: Path) -> str:
    """Pick a benchmark name mentioned in the task, else the first available."""
    lowered = task.lower()
    for name in list_benchmarks(workdir):
        if name.lower() in lowered:
            return name
    return ""


def infer_embench_spec(task: str, name: str) -> WorkloadSpec:
    if not name:
        raise ValueError("embench workload requires a benchmark name")
    return WorkloadSpec(
        spec_version=SPEC_VERSION,
        name=name.replace("-", "_"),
        source_kind="embench",
        description=clean_description(task),
        benchmark_suite="embench-iot",
        benchmark_name=name,
        benchmark_scale=DEFAULT_SCALE_FACTOR,
        warmup_heat=DEFAULT_WARMUP_HEAT,
        operation="embench_workload",
        pattern="unknown",
        cfu_style="tilelink_vector_rf",
        validation={
            "scalar_reference": True,
            "bench_verify": True,
            "soc_sim": True,
            "cycle_profile": True,
            "yosys_cost": True,
        },
        unknowns=[],
    )


def load_formatted_spec(path: Path) -> WorkloadSpec:
    text = path.read_text(errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"workload spec must be a mapping: {path}")
    data = strip_unknown_config_keys(data)
    return WorkloadSpec(**data)


def strip_unknown_config_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Drop config keys that are not part of the known WorkloadSpec schema."""
    allowed = set(WorkloadSpec.__annotations__)
    allowed.update({"spec_version", "hot_spot"})
    return {key: value for key, value in data.items() if key in allowed}


def llm_extract_or_fallback(
    task: str,
    context: RepoContext,
    fallback: WorkloadSpec,
    llm: OpenAiTextClient | None,
) -> WorkloadSpec:
    if llm is None or not llm.available:
        return fallback
    context_paths = "\n".join(sorted(context.get("files", {}).keys()))
    prompt = f"""Input request:
{task}

Known repo context files:
{context_paths}

Fallback heuristic spec:
{yaml.safe_dump(dict(fallback), sort_keys=False)}

Return the normalized YAML WorkloadSpec."""
    result = llm.complete(SPEC_SYSTEM_PROMPT, prompt, fallback=yaml.safe_dump(dict(fallback), sort_keys=False))
    try:
        parsed = yaml.safe_load(result.text)
    except yaml.YAMLError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    merged = dict(fallback)
    merged.update({k: v for k, v in parsed.items() if v is not None})
    return WorkloadSpec(**merged)


def infer_spec_from_text(text: str, *, source_kind: str, **overrides: str) -> WorkloadSpec:
    lowered = text.lower()
    operation = infer_operation(lowered)
    pattern = infer_pattern(lowered, operation)
    element_type = infer_element_type(lowered)
    vector_bytes = infer_int_after(lowered, ["vector_bytes", "vector bytes", "aligned to", "alignment"], default=32)
    spec: WorkloadSpec = {
        "spec_version": SPEC_VERSION,
        "name": infer_name(operation),
        "source_kind": source_kind,  # type: ignore[typeddict-item]
        "description": clean_description(text),
        "operation": operation,
        "pattern": pattern,
        "element_type": element_type,
        "signedness": "signed" if "int8" in lowered and "uint8" not in lowered else "unsigned",
        "output_type": "uint32_t" if "uint32" in lowered or operation in {"weighted_average", "dot_product", "sum_reduce"} else "",
        "vector_bytes": vector_bytes,
        "memory_access": "sequential" if any(word in lowered for word in ["array", "stride", "sequential", "loop"]) else "",
        "alignment_bytes": vector_bytes if "align" in lowered else 0,
        "cfu_style": infer_cfu_style(lowered, pattern),
        "validation": {
            "scalar_reference": True,
            "soc_sim": True,
            "cycle_profile": True,
            "yosys_cost": True,
        },
        "unknowns": infer_unknowns(lowered, operation, element_type),
    }
    for key, value in overrides.items():
        if value:
            spec[key] = value  # type: ignore[literal-required]
    return spec


def sanitize_spec(spec: WorkloadSpec) -> WorkloadSpec:
    spec.setdefault("spec_version", SPEC_VERSION)
    spec.setdefault("name", infer_name(spec.get("operation", "cfu_workload")))
    spec.setdefault("source_kind", "natural_language")
    spec.setdefault("description", "")
    spec.setdefault("operation", "unknown")
    spec.setdefault("pattern", "unknown")
    spec.setdefault("validation", {})
    spec.setdefault("unknowns", [])
    if not isinstance(spec.get("unknowns"), list):
        spec["unknowns"] = [str(spec["unknowns"])]
    return spec


def validate_workload_spec(spec: WorkloadSpec, *, strict: bool = False) -> list[str]:
    """Return a sorted list of validation problems; empty means the spec is valid."""
    errors: list[str] = []
    for key in SPEC_REQUIRED:
        value = spec.get(key)
        # An empty unknowns list is a valid answer ("nothing unknown"); only a
        # missing value is an error.
        if value is None or value == "" or (value == [] and key != "unknowns"):
            errors.append(f"{key} is required")
    source_kind = spec.get("source_kind")
    if source_kind not in {"natural_language", "formatted_spec", "c_project", "embench"}:
        errors.append(
            "source_kind must be natural_language|formatted_spec|c_project|embench, "
            f"got {source_kind!r}"
        )
    operation = spec.get("operation")
    if operation == "unknown":
        errors.append("operation is unknown; resolve before proceeding")
    validation = spec.get("validation")
    if not isinstance(validation, dict):
        errors.append("validation must be a mapping")
    if spec.get("vector_bytes") is not None and not isinstance(spec.get("vector_bytes"), int):
        errors.append("vector_bytes must be an integer")
    unknowns = spec.get("unknowns")
    if unknowns is not None and not isinstance(unknowns, list):
        errors.append("unknowns must be a list")
    if strict and unknown_high_impact_fields(operation, spec):
        errors.append("high-impact fields are unresolved and listed in unknowns")
    return sorted(errors)


def unknown_high_impact_fields(operation: str | None, spec: WorkloadSpec) -> bool:
    op = operation or spec.get("operation", "")
    return bool(
        any(unknown in {"operation", "element_type", "signedness", "overflow_policy"} for unknown in spec.get("unknowns", []))
    )


def infer_operation(lowered: str) -> str:
    if ("weighted" in lowered and "average" in lowered) or "wavg" in lowered:
        return "weighted_average"
    if "dot" in lowered:
        return "dot_product"
    if "sum" in lowered or "reduce" in lowered:
        return "sum_reduce"
    if "add" in lowered:
        return "elementwise_add"
    if "mul" in lowered or "multiply" in lowered:
        return "elementwise_mul"
    return "unknown"


def infer_pattern(lowered: str, operation: str) -> str:
    if operation in {"weighted_average", "dot_product"}:
        return "vector_mul_reduce"
    if operation == "sum_reduce":
        return "vector_to_scalar"
    if operation.startswith("elementwise"):
        return "vector_to_vector"
    if "scale" in lowered:
        return "scalar_on_vector"
    return "unknown"


def infer_element_type(lowered: str) -> str:
    for candidate in ["uint8_t", "int8_t", "uint16_t", "int16_t", "uint32_t", "int32_t"]:
        if candidate in lowered:
            return candidate
    for candidate in ["uint8", "int8", "u8", "i8"]:
        if candidate in lowered:
            return candidate
    return ""


def infer_cfu_style(lowered: str, pattern: str) -> str:
    if any(word in lowered for word in ["tilelink", "memory", "load", "store", "vector rf", "register file", "resident"]):
        return "tilelink_vector_rf"
    if pattern in {"vector_mul_reduce", "vector_to_vector"}:
        return "tilelink_vector_rf"
    if pattern == "vector_to_scalar":
        return "direct_or_tilelink"
    return ""


def infer_unknowns(lowered: str, operation: str, element_type: str) -> list[str]:
    unknowns: list[str] = []
    if operation == "unknown":
        unknowns.append("operation")
    if not element_type:
        unknowns.append("element_type")
    if "signed" not in lowered and "unsigned" not in lowered and "uint" not in lowered:
        unknowns.append("signedness")
    if "overflow" not in lowered and operation in {"weighted_average", "dot_product", "sum_reduce"}:
        unknowns.append("overflow_policy")
    if "align" not in lowered and "memory" in lowered:
        unknowns.append("alignment_bytes")
    return unknowns


def infer_int_after(lowered: str, labels: list[str], *, default: int) -> int:
    for label in labels:
        match = re.search(re.escape(label) + r"\D+(\d+)", lowered)
        if match:
            return int(match.group(1))
    return default


def infer_name(operation: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", operation.lower()).strip("_")
    return safe or "cfu_workload"


def clean_description(text: str) -> str:
    one_line = " ".join(text.strip().split())
    return one_line[:1000]


def read_optional(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(errors="replace")[:20000]
