"""Framework-owned test data for the conformance rung.

The model owns the *semantics* of a command — the independent scalar `reference`
and, where a command needs it, the `setup` that seeds a vector slot. It does not
own the *data*: this module generates the boundary and random argument vectors
from a fixed seed, owns the byte block the DUT reads, and binds every argument.
Two failure classes disappear with it.

* A run can no longer pass on a hand-picked degenerate input set. Run
  20261005-073745 declared a single zero-length message, which exercises padding
  and the round function but never a source-byte transfer.
* No case argument is model-authored ground truth. The model chooses the *shape*
  of the input space (`kind`, and a bound for `length`/`index`), never the values,
  so a table that disagrees with the workload cannot be hidden behind easy inputs.

The generator is pure and seed-fixed: the same oracle always renders the same
test, so re-running a rung re-runs the same experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

#: Fixed seed. The framework owns it; a model-supplied ``seed`` is rejected so
#: the test data cannot be steered.
SEED = 0x5A5A2026
#: Cases rendered for one command: the boundary sweep plus the random draws.
MAX_CASES = 12
DEFAULT_RANDOM_CASES = 4
MAX_RANDOM_CASES = 16
MAX_BUFFER_BYTES = 1024
MIN_BUFFER_BYTES = 4
#: Ceiling on the bytes one command transfers across all its cases. The rung is a
#: lightweight instruction check, not a benchmark: with `buffer: 1024` the sweep
#: plus the draws would otherwise hand the simulated CPU ~4.6 KB of scalar
#: reference work per command. Cases are kept in priority order (the full-block
#: case first, then the sweep, then the draws) until this budget is reached.
CASE_BYTE_BUDGET = 2048
ROLE_KINDS = ("address", "length", "index", "value")
_WIDTH = 0xFFFFFFFF


class _Rng:
    """A tiny deterministic LCG; enough for reproducible test vectors."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _WIDTH

    def next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & _WIDTH
        return self.state

    def below(self, bound: int) -> int:
        return self.next() % bound if bound > 0 else 0

    def between(self, low: int, high: int) -> int:
        return low + self.below(high - low + 1)


def _seed_for(name: str, salt: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{salt}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _macro(name: str, suffix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).upper() + suffix


def _array(name: str) -> str:
    return "cfu_conf_data_" + re.sub(r"[^A-Za-z0-9_]", "_", name).lower()


@dataclass(frozen=True)
class DataPlan:
    """One command's generated test data, ready to render as C."""

    #: One tuple of C argument expressions per case, in rs1/rs2 order.
    cases: tuple[tuple[str, ...], ...] = ()
    #: Harness-owned bytes the DUT reads; empty when the command declares none.
    buffer: bytes = b""
    array_name: str = ""
    base_macro: str = ""
    bytes_macro: str = ""

    #: Bytes this command transfers across all rendered cases.
    bytes_total: int = 0

    @property
    def has_buffer(self) -> bool:
        return bool(self.array_name)


@dataclass(frozen=True)
class _Role:
    kind: str
    bound: int = _WIDTH
    #: C expressions; the first one is the "corner" value for the sweep.
    values: tuple[str, ...] = ()


def _int_or_none(value: Any) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


def _boundary_values(kind: str, bound: int) -> list[int]:
    if kind == "length":
        values = [0, 1, min(2, bound), bound // 2, max(0, bound - 1), bound]
    elif kind == "index":
        values = [1, max(1, bound // 2), max(1, bound)]
    else:  # value
        values = [0, 1, 0x7FFFFFFF, _WIDTH]
    values = [min(max(v, 0), bound) for v in values]
    if kind == "index":
        values = [max(v, 1) for v in values]
    return sorted(set(values))


def plan(name: str, data: Any, role_count: int) -> tuple[DataPlan | None, list[str]]:
    """Generate one command's data, or report why the ``data`` spec is unusable."""
    if not isinstance(data, Mapping):
        return None, [
            "data must be a mapping (data.buffer/data.roles): the harness generates the "
            "test vectors; model-authored `cases` are no longer accepted"
        ]
    problems: list[str] = []
    if "seed" in data:
        problems.append(
            "data.seed is not allowed: the harness fixes the seed so the test data cannot "
            "be steered"
        )
    raw_roles = data.get("roles", [])
    if raw_roles is None:
        raw_roles = []
    if not isinstance(raw_roles, list):
        problems.append("data.roles must be a list with one entry per CPU value argument")
        return None, problems
    if len(raw_roles) != role_count:
        problems.append(
            f"data.roles has {len(raw_roles)} entry(s) but the command takes {role_count} "
            "CPU value argument(s) (rs1/rs2 order): declare exactly one input-space "
            "descriptor per argument, or omit data.roles when the command takes none"
        )
        return None, problems

    buffer_size = data.get("buffer", 0) or 0
    if _int_or_none(buffer_size) is None or buffer_size < 0:
        problems.append("data.buffer must be a non-negative integer")
        return None, problems
    if buffer_size and not MIN_BUFFER_BYTES <= buffer_size <= MAX_BUFFER_BYTES:
        problems.append(
            f"data.buffer must be between {MIN_BUFFER_BYTES} and {MAX_BUFFER_BYTES} bytes"
        )
        return None, problems
    if buffer_size % 4:
        problems.append("data.buffer must be a multiple of 4 so every argument is aligned")
        return None, problems

    count = data.get("count", DEFAULT_RANDOM_CASES)
    if _int_or_none(count) is None or not 0 <= count <= MAX_RANDOM_CASES:
        problems.append(f"data.count must be an integer between 0 and {MAX_RANDOM_CASES}")
        return None, problems

    array_name = base_macro = bytes_macro = ""
    if buffer_size:
        array_name = _array(name)
        base_macro = _macro(name, "_CONF_DATA_BASE")
        bytes_macro = _macro(name, "_CONF_DATA_BYTES")

    roles: list[_Role] = []
    for index, raw in enumerate(raw_roles):
        if not isinstance(raw, Mapping):
            problems.append(f"data.roles[{index}] must be a mapping with a kind")
            continue
        kind = str(raw.get("kind") or "")
        if kind == "address":
            if not buffer_size:
                problems.append(
                    f"data.roles[{index}] is an address but data.buffer is missing: the "
                    "harness owns the block the DUT reads, so declare its size"
                )
                continue
            roles.append(_Role(kind="address", bound=0, values=(base_macro,)))
            continue
        if kind == "length":
            bound = _int_or_none(raw.get("max"))
            if bound is None or not 1 <= bound <= buffer_size:
                problems.append(
                    f"data.roles[{index}].max must be between 1 and data.buffer "
                    f"({buffer_size or 'unset'}): it bounds the transferred length"
                )
                continue
        elif kind in ("index", "value"):
            bound = _int_or_none(raw.get("max"))
            if kind == "index" and bound is None:
                problems.append(f"data.roles[{index}].max is required: it bounds the index")
                continue
            bound = _WIDTH if bound is None else bound
            if not 1 <= bound <= _WIDTH:
                problems.append(f"data.roles[{index}].max must be between 1 and 0xFFFFFFFF")
                continue
        else:
            problems.append(f"data.roles[{index}].kind must be one of {', '.join(ROLE_KINDS)}")
            continue
        roles.append(
            _Role(kind=kind, bound=bound,
                  values=tuple(str(v) for v in _boundary_values(kind, bound)))
        )
    if problems:
        return None, problems

    buffer = b""
    if buffer_size:
        rng = _Rng(_seed_for(name, "buffer"))
        buffer = bytes(rng.below(256) for _ in range(buffer_size))

    cases, bytes_total = _assemble(name, roles, count)
    if not cases:
        problems.append("data generated no case: declare data.count >= 1 or a bounded role")
        return None, problems
    return (
        DataPlan(cases=cases, buffer=buffer, array_name=array_name,
                 base_macro=base_macro, bytes_macro=bytes_macro,
                 bytes_total=bytes_total),
        [],
    )


def _assemble(name: str, roles: Sequence[_Role], count: int,
              budget: int = CASE_BYTE_BUDGET) -> tuple[tuple[tuple[str, ...], ...], int]:
    """Boundary sweep, then deterministic random draws; ordered, budgeted, capped.

    Priority order is the full-block case, then the rest of the sweep, then the
    random draws, so hitting the byte budget drops a random case rather than the
    one transfer that matters. The full-block case is kept whatever the budget:
    a budget below one block would otherwise hide the transfer entirely.
    """
    if not roles:
        return ((),) if count else (), 0
    corners = [role.values[0] for role in roles]
    sweep: list[tuple[str, ...]] = []
    for index, role in enumerate(roles):
        for value in role.values:
            case = list(corners)
            case[index] = value
            sweep.append(tuple(case))
    full_block = None
    for index, role in enumerate(roles):
        if role.kind == "length":
            case = list(corners)
            case[index] = str(role.bound)
            full_block = tuple(case)
            break
    ordered: list[tuple[str, ...]] = []
    for case in ([full_block] if full_block is not None else []) + sweep:
        if case not in ordered:
            ordered.append(case)
    rng = _Rng(_seed_for(name, "cases"))
    for _ in range(count):
        case = tuple(_random_value(role, rng) for role in roles)
        if case not in ordered:
            ordered.append(case)
    kept: list[tuple[str, ...]] = []
    total = 0
    for case in ordered:
        weight = _case_bytes(case, roles)
        if kept and total + weight > budget:
            continue
        kept.append(case)
        total += weight
    return tuple(kept[:MAX_CASES]), total


def _case_bytes(case: Sequence[str], roles: Sequence[_Role]) -> int:
    """Bytes one case transfers: the largest declared `length` operand."""
    widths = [_as_int(value) for role, value in zip(roles, case) if role.kind == "length"]
    return max((width for width in widths if width is not None), default=0)


def _as_int(value: Any) -> int | None:
    """The integer a rendered argument expression denotes, or ``None``."""
    parsed = _int_or_none(value)
    if parsed is not None:
        return parsed
    try:
        return int(str(value).strip(), 0)
    except (TypeError, ValueError):
        return None


def _random_value(role: _Role, rng: _Rng) -> str:
    if role.kind == "address":
        return role.values[0]
    if role.kind == "length":
        return str(rng.between(0, role.bound))
    if role.kind == "index":
        return str(rng.between(1, role.bound))
    return str(rng.between(0, role.bound))


def render_data(plan: DataPlan) -> str:
    """The C a plan needs: its owned byte block, plus the base/size macros."""
    if not plan.has_buffer:
        return ""
    body = ",".join(f"0x{byte:02x}" for byte in plan.buffer)
    return (
        f"/* harness-owned data: seed {SEED:#x}, {len(plan.buffer)} bytes */\n"
        f"static uint8_t {plan.array_name}[{len(plan.buffer)}] "
        f"__attribute__((aligned(4))) = {{{body}}};\n"
        f"#define {plan.base_macro} ((uint32_t)(uintptr_t){plan.array_name})\n"
        f"#define {plan.bytes_macro} ((uint32_t){len(plan.buffer)}u)"
    )
