"""Versioned ladder: ISA Contract -> visibility -> dev conformance -> IR.

The v3 ladder uses explicit off/dev modes and never promotes dev tests to a gate.
Visibility-none is the existing basic tier, not zero context. Historical v1/v2
layers remain readable for archive collection; ``full`` stays a separate preset.
"""
from __future__ import annotations

ABLATION_SCHEME = "isa-visibility-devconf-ir-v3"
V2_ABLATION_SCHEME = "isa-visibility-ir-v2"
_DEV_DEFAULTS = {"conformance_promote": False, "max_conformance_runs": 2}
ABLATION_LAYERS: dict[str, dict] = {
    "l0-none": {
        **_DEV_DEFAULTS,
        "flags": ("--no-isa-contract", "--no-conformance", "--no-llvm-ir"),
        "visibility": "basic", "conformance_mode": "off",
        "adds": "baseline: no ISA Contract, no extra source visibility, no conformance or IR",
    },
    "l1-isa": {
        **_DEV_DEFAULTS,
        "flags": ("--no-conformance", "--no-llvm-ir"),
        "visibility": "basic", "conformance_mode": "off",
        "adds": "ISA Contract: the contract stage authors `isa_spec`",
    },
    "l2-vis": {
        **_DEV_DEFAULTS,
        "flags": ("--no-conformance", "--no-llvm-ir"),
        "visibility": "extended", "conformance_mode": "off",
        "adds": "Visibility: basic -> extended (repository/skills/SpinalHDL reads)",
    },
    "l3-devconf": {
        **_DEV_DEFAULTS,
        "flags": ("--no-llvm-ir",),
        "visibility": "extended", "conformance_mode": "dev",
        "adds": "Dev conformance: agent-authored tests as evidence, not a gate",
    },
    "l4-ir": {
        **_DEV_DEFAULTS,
        "flags": (),
        "visibility": "extended", "conformance_mode": "dev",
        "adds": "LLVM IR: optional `.ll` / feature-XML evidence and `ir_features`",
    },
    "full": {
        "flags": (),
        "adds": "legacy full flow: ISA Contract + conformance (agent default mode) + LLVM IR",
    },
}
ABLATION_LADDER = ("l0-none", "l1-isa", "l2-vis", "l3-devconf", "l4-ir")
DEFAULT_LAYERS = ("full",)
LAYER_ALIASES = {
    "none": "l0-none", "l0": "l0-none",
    "isa": "l1-isa", "l1": "l1-isa",
    "vis": "l2-vis", "visibility": "l2-vis", "l2": "l2-vis",
    "conf": "l3-devconf", "conformance": "l3-devconf", "devconf": "l3-devconf", "l3": "l3-devconf",
    "llvm-ir": "l4-ir", "ir": "l4-ir", "l4": "l4-ir",
}
# Historical vocabularies never define a new schedule.
LEGACY_ABLATION_LAYERS = {
    "l0-none": {"flags": ("--no-isa-contract", "--no-conformance", "--no-llvm-ir"),
                "adds": "baseline: no ISA Contract, conformance or LLVM IR"},
    "l1-isa": {"flags": ("--no-conformance", "--no-llvm-ir"), "adds": "ISA Contract"},
    "l2-conf": {"flags": ("--no-llvm-ir",), "adds": "Conformance"},
    "full": {"flags": (), "adds": "LLVM IR"},
}
V2_ABLATION_LAYERS = {
    "l0-none": {**LEGACY_ABLATION_LAYERS["l0-none"], "visibility": "basic"},
    "l1-isa": {**LEGACY_ABLATION_LAYERS["l1-isa"], "visibility": "basic"},
    "l2-vis": {"flags": ("--no-conformance", "--no-llvm-ir"),
               "visibility": "extended", "adds": "Visibility: basic -> extended"},
    "l3-ir": {"flags": ("--no-conformance",), "visibility": "extended", "adds": "LLVM IR"},
}


def resolve_layers(spec) -> list[str]:
    """Resolve a v3 schedule; old l2-conf/l3-ir names are archive-only."""
    items = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not items:
        raise SystemExit("--layers must name at least one layer")
    if any(t in ("ladder", "all") for t in items):
        if len(items) != 1:
            raise SystemExit("--layers ladder/all cannot be mixed with other entries")
        return list(ABLATION_LADDER)
    picked: set[str] = set()
    for item in items:
        name = LAYER_ALIASES.get(item, item)
        if name not in ABLATION_LAYERS:
            raise SystemExit(
                f"unknown --layers entry {item!r} (new ladder: {', '.join(ABLATION_LADDER)}; "
                "legacy full is separate; l2-conf/l3-ir are archive-only)")
        picked.add(name)
    return [name for name in (*ABLATION_LADDER, "full") if name in picked]


def layer_flags(layer: str) -> tuple[str, ...]:
    entry = (ABLATION_LAYERS.get(str(layer or "full")) or V2_ABLATION_LAYERS.get(str(layer))
             or LEGACY_ABLATION_LAYERS.get(str(layer)))
    return tuple(entry["flags"]) if entry else ()


def layer_visibility(layer: str, fallback: str = "standard") -> str:
    return str(ABLATION_LAYERS.get(str(layer), {}).get("visibility") or fallback)


def job_visibility(job: dict, fallback: str = "standard") -> str:
    return str(job.get("visibility") or layer_visibility(job.get("layer", ""), fallback))


ABLATION_FLAG_NAMES = ("--no-isa-contract", "--no-conformance", "--no-llvm-ir")


def effective_ablation_flags(flags) -> tuple[str, ...]:
    """Canonical disabled flags for reports, distinct from rendered CLI argv."""
    picked = {str(flag) for flag in flags or ()}
    if "--no-isa-contract" in picked:
        picked.add("--no-conformance")
    return tuple(flag for flag in ABLATION_FLAG_NAMES if flag in picked)


def job_layer_flags(job: dict) -> tuple[str, ...]:
    if "ablation_flags" in job:
        return tuple(job["ablation_flags"] or ())
    return layer_flags(job.get("layer", ""))


def job_run_flags(job: dict) -> tuple[str, ...]:
    """Render explicit v3 modes; preserve legacy jobs' existing argv semantics."""
    flags = job_layer_flags(job)
    mode = job.get("conformance_mode")
    if mode is None:
        return flags
    if mode not in ("off", "dev") or job.get("conformance_promote"):
        raise ValueError("v3 ladder requires off/dev conformance without promotion")
    budget = int(job.get("max_conformance_runs", 2))
    if budget < 1:
        raise ValueError("v3 conformance stage-local budget must be >= 1")
    return (*[flag for flag in flags if flag != "--no-conformance"],
            "--conformance-mode", str(mode), "--max-conformance-runs", str(budget))


def _layer_order(layer) -> tuple[int, str]:
    name = str(layer or "full")
    order = (*ABLATION_LADDER, "l3-ir", "l2-conf", "full")
    return (order.index(name), name) if name in order else (len(order), name)
