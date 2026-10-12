"""Ablation-layer vocabulary: the capability ladder shared by every experiment preset.

Each layer adds exactly ONE capability to the previous layer, in the order
ISA Contract -> Conformance -> LLVM IR.  The layer name doubles as the
run-name token and the ``layer`` column value; ``full`` is the un-ablated
reference flow and keeps the canonical (unsuffixed) run name.  The flags are
the agent CLI's ``--no-*`` switches (see "Ablation flags" in agents/README.md);
``--no-isa-contract`` implies ``--no-conformance``, so l0-none spells both out
explicitly to keep the recorded argv honest.
"""

from __future__ import annotations

ABLATION_LAYERS: dict[str, dict] = {
    "l0-none": {
        "flags": ("--no-isa-contract", "--no-conformance", "--no-llvm-ir"),
        "adds": "nothing: the agent authors its own instructions and "
                "`cfu_intrinsics.h`; asm/summary evidence only",
    },
    "l1-isa": {
        "flags": ("--no-conformance", "--no-llvm-ir"),
        "adds": "ISA Contract: the contract stage authors `isa_spec`",
    },
    "l2-conf": {
        "flags": ("--no-llvm-ir",),
        "adds": "Conformance: the `hw_conformance` rung gets its oracle",
    },
    "full": {
        "flags": (),
        "adds": "LLVM IR: `.ll` / feature-XML evidence and `ir_features`",
    },
}
# Ladder order: capability additions ISA Contract -> Conformance -> LLVM IR.
ABLATION_LADDER = ("l0-none", "l1-isa", "l2-conf", "full")
DEFAULT_LAYERS = ("full",)
LAYER_ALIASES = {
    "none": "l0-none", "l0": "l0-none",
    "isa": "l1-isa", "l1": "l1-isa",
    "conf": "l2-conf", "conformance": "l2-conf", "l2": "l2-conf",
    "llvm-ir": "full", "ir": "full", "l3": "full",
}


def resolve_layers(spec) -> list[str]:
    """Resolve a ``--layers`` spec to ladder-ordered layer names.

    Accepts a comma list of layer names/aliases, or ``ladder``/``all`` for the
    whole capability ladder.  Unknown names are fatal: a typo must never
    silently drop a layer from the experiment.
    """
    items = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not items:
        raise SystemExit("--layers must name at least one layer")
    if any(t in ("ladder", "all") for t in items):
        return list(ABLATION_LADDER)
    picked: set[str] = set()
    for item in items:
        name = LAYER_ALIASES.get(item, item)
        if name not in ABLATION_LAYERS:
            raise SystemExit(
                f"unknown --layers entry {item!r} "
                f"(layers: {', '.join(ABLATION_LADDER)}; aliases: {', '.join(sorted(LAYER_ALIASES))})"
            )
        picked.add(name)
    return [name for name in ABLATION_LADDER if name in picked]


def layer_flags(layer: str) -> tuple[str, ...]:
    """The agent CLI flags a layer adds on top of the full flow."""
    entry = ABLATION_LAYERS.get(str(layer or "full"))
    return tuple(entry["flags"]) if entry else ()


#: The agent CLI's three ablation switches.  They may also arrive through a
#: preset's ``--`` passthrough (e.g. a plain ``--no-conformance`` run), in
#: which case they are folded into the run's recorded ablation flags.
ABLATION_FLAG_NAMES = ("--no-isa-contract", "--no-conformance", "--no-llvm-ir")


def effective_ablation_flags(flags) -> tuple[str, ...]:
    """Deduped ablation switches in canonical order, with implications closed.

    ``--no-isa-contract`` implies ``--no-conformance`` (the agent drops the
    conformance requirement and its report's "## Ablation" section records
    BOTH switches), so the recorded/expected set must contain the implied one
    too or the collection cross-check would warn on every honest run.
    """
    picked = {str(flag) for flag in flags or ()}
    if "--no-isa-contract" in picked:
        picked.add("--no-conformance")
    return tuple(flag for flag in ABLATION_FLAG_NAMES if flag in picked)


def job_layer_flags(job: dict) -> tuple[str, ...]:
    """The ablation flags of a scheduled job.

    The job's own ``ablation_flags`` win (callers and tests may set them
    directly); otherwise the job's layer name maps to its canonical flags.
    """
    return tuple(job.get("ablation_flags") or ()) or layer_flags(job.get("layer", ""))


def _layer_order(layer) -> tuple[int, str]:
    """Ladder position for sorting (unknown layers land after the ladder)."""
    name = str(layer or "full")
    try:
        return (ABLATION_LADDER.index(name), name)
    except ValueError:
        return (len(ABLATION_LADDER), name)
