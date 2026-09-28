"""Templated differential micro-test for a freshly designed CFU.

The ``hw_conformance`` rung sits between the SoC build and the software rung. It
runs the smallest possible program that exercises each declared instruction and
compares it against an independent scalar reference, so a contract/encoding/
writeback mistake is found before any workload software is written.

Division of ownership:

* The **harness** owns the C test (``render_conformance``). It is never a model
  write target, so a hardware repair cannot weaken it.
* The **model** owns only the oracle YAML: the scalar reference expression, the
  optional setup statements and the boundary cases.

Anti-self-grading rules:

* A command that writes CPU ``rd`` is compared case-by-case against ``reference``,
  an independent scalar C expression.
* A command with no CPU result cannot be read back generically, so the model must
  supply a ``verify`` C expression instead. That is **weaker** evidence (it is the
  model attesting its own hardware) and is documented as such.
* A command with any vector resource must supply ``setup`` to seed the register
  file; the harness never guesses.
* Missing reference/verify/setup/cases is a coverage FAIL, never a silent skip.
* Lazy oracles are rejected: ``reference``/``verify`` must actually observe the
  command's effect, so constant/tautological expressions and ``cfu_*``
  self-references fail validation (a reference that calls the DUT proves
  nothing), as does an ``a``/``b`` reference for a role the command never
  declared (the class behind a ``'b' undeclared`` build failure).
* Oracle ``helpers`` blocks must define each C symbol once; a helper needed by
  several commands must live in one entry (the class behind a generated-test
  ``redefinition of 'ref_rol32'`` build failure).
* For rd-writing commands the harness appends a bounded set of deterministic
  extra differential cases (edge values plus fixed-seed pseudo-random inputs
  derived from the command name) after the model's own cases, so the model's
  ``reference`` is exercised on inputs the model did not pick itself. An entry
  whose reference cannot model arbitrary inputs (address-taking or stateful
  commands) opts out with ``edge_cases: false``; the resulting weaker coverage
  is an accepted trade-off over wrong comparisons.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from .isa import MAX_FUNCTION_ID, MAX_SUBFUNCTION_ID, command_symbol

STATUS_SENTINEL = "CFU_CONFORMANCE"
# sbt prefixes every line of the forked simulator's output ("[info] CFU_CONFORMANCE
# PASS ..."), so the parser must tolerate the level marker. A mock that feeds bare
# lines hides this, which is why the pattern allows the prefix explicitly.
_SBT_PREFIX_RE = re.compile(r"^\s*\[(?:info|error|warn|success)\]\s?")
STATUS_RE = re.compile(
    r"^CFU_CONFORMANCE\s+(?P<status>PASS|FAIL)\s+"
    r"commands=(?P<commands>\d+)/(?P<total>\d+)\s+"
    r"cases=(?P<cases>\d+)\s+failed=(?P<failed>\S+)\s*$",
    re.MULTILINE,
)
DETAIL_RE = re.compile(r"^CFU_CONFORMANCE_(?:MISSING|CASE)\b.*$", re.MULTILINE)

# Oracle `cases[].args` are bound to these C names in rs1/rs2 order.
ROLE_NAMES = {"rs1": "a", "rs2": "b"}

# Harness-side extra differential cases for rd-writing commands: edge values
# swept across the roles plus fixed-seed pseudo-random inputs, bounded.
EDGE_CASE_VALUES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
MAX_EXTRA_CASES = 5


def _commands(isa: Any) -> list[dict[str, Any]]:
    if not isinstance(isa, dict):
        return []
    commands = isa.get("commands")
    return [c for c in commands if isinstance(c, dict)] if isinstance(commands, list) else []


def _int_ok(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def encodable(command: dict[str, Any]) -> bool:
    """True when the command has an encoding the intrinsics header can emit."""
    func = command.get("function_id")
    if not _int_ok(func) or not 0 <= func <= MAX_FUNCTION_ID:
        return False
    sub = command.get("subfunction_id", 0)
    return _int_ok(sub) and 0 <= sub <= MAX_SUBFUNCTION_ID


def enabled_commands(isa: Any) -> list[dict[str, Any]]:
    """Commands the conformance test must cover, in ISA order."""
    return [c for c in _commands(isa) if encodable(c)]


def intrinsic_name(command: dict[str, Any]) -> str:
    return f"cfu_{command_symbol(command, int(command.get('function_id', 0)))}"


# C helpers the generated header/test provide besides the command intrinsics.
RESERVED_INTRINSIC_HELPERS = (
    "cfu_enable",
    "cfu_memory_fence",
    "cfu_agent_counter_read",
    "cfu_agent_counter_elapsed",
)


def allowed_intrinsics(isa: Any) -> set[str]:
    """Every ``cfu_*`` callable from the oracle's C (generated + reserved)."""
    names = set(RESERVED_INTRINSIC_HELPERS)
    for command in _commands(isa):
        if encodable(command):
            names.add(intrinsic_name(command))
    return names


def unknown_intrinsic_calls(entry: dict[str, Any], isa: Any) -> list[str]:
    """``cfu_*`` calls in the oracle C that the generated header cannot resolve.

    Without this lint an invented name only surfaces as a linker error
    (``undefined reference to `cfu_load'``), and when the repair feedback omits
    the linker output the model repairs blind and repeats the same invented
    name until the budget is gone (observed 2026-09-26).
    """
    import difflib

    allowed = sorted(allowed_intrinsics(isa))
    problems: list[str] = []
    for field in ("helpers", "setup", "reference", "verify"):
        text = str(entry.get(field) or "")
        for call in sorted(set(re.findall(r"\b(cfu_[A-Za-z0-9_]+)\s*\(", text))):
            if call in allowed:
                continue
            # Cutoff 0.5: an invented short name like `cfu_load` is only ~0.59
            # similar to the real `cfu_vector_load_i32`, below difflib's 0.6.
            suggestion = difflib.get_close_matches(call, allowed, n=1, cutoff=0.5)
            hint = f" (did you mean {suggestion[0]}?)" if suggestion else ""
            problems.append(
                f"{field} calls unknown intrinsic {call}{hint}: the generated "
                f"cfu_intrinsics.h defines only {', '.join(allowed)}"
            )
    return problems


def cpu_value_roles(command: dict[str, Any]) -> list[str]:
    """CPU-value arguments of the generated intrinsic, in rs1/rs2 order.

    Mirrors ``isa._resource_intrinsic``: CPU reads and ``cpu_value`` vector
    indices each consume one argument; an absent ``resources`` key means the
    legacy two-argument ABI.
    """
    resources = command.get("resources")
    if not isinstance(resources, list):
        return ["rs1", "rs2"]
    needed: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if resource.get("storage") == "cpu" and resource.get("access") == "read":
            needed.add(str(resource.get("role")))
        index = resource.get("index")
        if (
            resource.get("storage") == "vector"
            and isinstance(index, dict)
            and index.get("source") == "cpu_value"
        ):
            needed.add(str(index.get("register")))
    return [role for role in ("rs1", "rs2") if role in needed]


def writes_cpu_rd(command: dict[str, Any]) -> bool:
    resources = command.get("resources")
    if not isinstance(resources, list):
        # Legacy ABI always returns into rd.
        return True
    return any(
        isinstance(r, dict)
        and r.get("storage") == "cpu"
        and r.get("access") == "write"
        and r.get("role") == "rd"
        for r in resources
    )


def has_vector_resource(command: dict[str, Any]) -> bool:
    resources = command.get("resources")
    if not isinstance(resources, list):
        return False
    return any(isinstance(r, dict) and r.get("storage") == "vector" for r in resources)


def _oracle_entries(oracle: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(oracle, dict):
        return {}
    commands = oracle.get("commands")
    if not isinstance(commands, list):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for entry in commands:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            entries.setdefault(entry["name"], entry)
    return entries


def _c_literal(value: Any) -> str:
    """Render an oracle argument as a uint32 C literal, or "" when unusable."""
    if _int_ok(value):
        return f"(uint32_t){int(value) & 0xFFFFFFFF}u"
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = int(text, 0)
        except ValueError:
            return ""
        return f"(uint32_t){parsed & 0xFFFFFFFF}u"
    return ""


_C_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_C_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_C_NUMBER_RE = re.compile(r"\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+(?:\.\d+)?)(?:[uUlLfF]*)\b")
_C_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_C_TYPE_TOKENS = frozenset(
    """
    void char short int long float double signed unsigned const volatile restrict
    _Bool bool size_t ssize_t ptrdiff_t uintptr_t intptr_t
    uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t
    """.split()
)
_C_TYPE_CAST_RE = re.compile(r"\((?:\s*[A-Za-z_]\w*)+\s*\)")


def _strip_c_comments_and_literals(text: str) -> str:
    """Blank out C comments and string/char literals, keeping offsets irrelevant."""
    return _C_LITERAL_RE.sub('""', _C_COMMENT_RE.sub(" ", str(text)))


def _drop_type_cast(match: "re.Match[str]") -> str:
    """Drop ``(uint32_t)``-style cast groups; keep real expression parens."""
    inner = match.group(0)[1:-1]
    tokens = inner.split()
    return " " if tokens and all(t in _C_TYPE_TOKENS for t in tokens) else match.group(0)


def _expression_is_constant(expression: str) -> bool:
    """True when the expression has no identifier left after stripping noise.

    Conservative by design: comments, string/char literals, numeric literals
    with ``u``/``l`` suffixes and known-type casts are removed, then any
    remaining identifier (function name or variable) makes the expression
    observational. No full C parsing is attempted.
    """
    text = _strip_c_comments_and_literals(expression)
    previous = None
    while previous != text:  # nested casts like (uint32_t)(unsigned long)0
        previous = text
        text = _C_TYPE_CAST_RE.sub(_drop_type_cast, text)
    text = _C_NUMBER_RE.sub(" ", text)
    return _C_IDENTIFIER_RE.search(text) is None


_C_KEYWORDS = frozenset("if else for while switch return do sizeof".split())
_HELPER_DEF_RE = re.compile(
    r"\b(?:static\s+|inline\s+|extern\s+|const\s+|unsigned\s+|signed\s+|volatile\s+)*"
    r"(?:[A-Za-z_]\w*(?:\s*\*\s*|\s+))+([A-Za-z_]\w*)\s*\([^;()]*\)\s*\{"
)


def _helper_definitions(helpers_text: str) -> list[str]:
    """Top-level C function definition names in an oracle ``helpers`` block.

    Conservative: only a parenthesised signature immediately followed by a body
    brace counts, so call statements and prototypes are never mistaken for
    definitions. Source order, duplicates preserved (duplicates are themselves
    redefinitions).
    """
    text = _strip_c_comments_and_literals(helpers_text)
    return [m.group(1) for m in _HELPER_DEF_RE.finditer(text) if m.group(1) not in _C_KEYWORDS]


def _extra_cases(command: dict[str, Any], entry: dict[str, Any]) -> list[list[int]]:
    """Harness-side deterministic extra differential cases (bounded).

    Appended after the model's own cases for rd-writing commands only, so the
    model's ``reference`` is exercised against edge and fixed-seed pseudo-random
    inputs the model did not pick. The seed derives from the command name via
    sha256, so the vectors are reproducible across runs and machines.

    Trade-off (accepted, see tests): ``reference`` must model these arbitrary
    inputs. A command whose reference cannot (address-taking or stateful
    commands) opts out with ``edge_cases: false`` in its oracle entry, keeping
    only the model's own cases — weaker coverage instead of wrong comparisons.
    """
    if entry.get("edge_cases") is False:
        return []
    if not writes_cpu_rd(command):
        return []  # verify expressions may not model arbitrary inputs
    roles = cpu_value_roles(command)
    if not roles:
        return []
    seed = int.from_bytes(hashlib.sha256(str(command.get("name", "")).encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    extras: list[list[int]] = []
    for offset in (0, 3):  # two edge sweeps over EDGE_CASE_VALUES
        extras.append([EDGE_CASE_VALUES[(offset + i) % len(EDGE_CASE_VALUES)] for i in range(len(roles))])
    for _ in range(3):  # three seeded pseudo-random pairs
        extras.append([rng.getrandbits(32) for _ in roles])
    return extras[:MAX_EXTRA_CASES]


def _expression_problems(command: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Lazy-oracle lints on the oracle C expressions (reference/verify).

    Regression class (2026-09-28): ``verify: "1"``, ``reference: "0"`` and a
    self-referential ``reference: "cfu_compute(a, b)"`` all validated clean, so
    a design could pass hw_conformance on a tautology and then fail the
    benchmark (run 20260928-103622: conformance 2/2, Embench verify failed).
    """
    problems: list[str] = []
    for field in ("reference", "verify"):
        text = str(entry.get(field) or "").strip()
        if not text:
            continue  # emptiness is reported by the required-field checks
        calls = sorted(set(re.findall(r"\b(cfu_[A-Za-z0-9_]*)\s*\(", text)))
        if calls:
            problems.append(
                f"{field} calls the cfu_* intrinsic {calls[0]}: reference/verify must not call "
                "cfu_* intrinsics: the reference must be an independent scalar C expression"
            )
        if _expression_is_constant(text):
            problems.append(
                f"{field} must actually observe the command's effect: reference/verify must use "
                "the case arguments or state set by setup, not a constant"
            )
        roles = cpu_value_roles(command)
        bound = {ROLE_NAMES[role] for role in roles}
        used = set(re.findall(r"(?<![.>\w])([ab])(?!\w)", _strip_c_comments_and_literals(text)))
        for identifier in sorted(used - bound):
            bound_names = ", ".join(ROLE_NAMES[r] for r in roles) or "no case arguments"
            problems.append(
                f"{field} references undeclared identifier '{identifier}': this command's "
                f"declared CPU-value roles bind only {bound_names} (rs1->a, rs2->b); declare "
                "the CPU read in `resources` or stop using that identifier"
            )
    return problems


def _coverage_problem_list(command: dict[str, Any], entry: dict[str, Any] | None) -> list[str]:
    """Every reason this command cannot be verified, or [] when it can.

    The caller prefixes the command name, so messages here stay name-free.
    Rendered as one MISSING line when non-empty, so an unverifiable command can
    never look like a pass even if ``validate_oracle`` was bypassed.
    """
    if entry is None:
        return ["no oracle entry"]
    problems: list[str] = []
    cases = entry.get("cases")
    if not isinstance(cases, list) or not cases:
        problems.append("cases must be a non-empty list")
    if writes_cpu_rd(command):
        if not str(entry.get("reference") or "").strip():
            problems.append(
                "reference is required because the command writes CPU rd; provide an "
                "independent scalar C expression over the case arguments"
            )
    elif not str(entry.get("verify") or "").strip():
        problems.append(
            "this command has no CPU result, so it cannot be compared; provide a "
            "verify C expression that is non-zero after the call (weaker evidence "
            "than reference, but it must not be omitted)"
        )
    if has_vector_resource(command) and not str(entry.get("setup") or "").strip():
        problems.append(
            "setup is required because the command uses the vector register file; "
            "the harness does not guess how to seed it"
        )
    if "edge_cases" in entry and not isinstance(entry.get("edge_cases"), bool):
        problems.append("edge_cases must be a boolean: false opts out of the harness extra cases")
    problems.extend(_expression_problems(command, entry))
    if isinstance(cases, list):
        roles = cpu_value_roles(command)
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                problems.append(f"cases[{index}] must be a mapping with an args list")
                continue
            args = case.get("args")
            if not isinstance(args, list):
                problems.append(f"cases[{index}].args must be a list")
                continue
            if len(args) != len(roles):
                problems.append(
                    f"cases[{index}].args has {len(args)} value(s) but the command takes "
                    f"{len(roles)} CPU value argument(s) ({', '.join(roles) or 'none'}): "
                    f"give exactly one literal per role, in rs1/rs2 order, and move any "
                    f"data/address preparation into `setup`"
                )
                continue
            for position, value in enumerate(args):
                if not _c_literal(value):
                    problems.append(
                        f"cases[{index}].args[{position}] = {value!r} must be an integer or "
                        "a C-style integer string such as 0xFFFFFFFF; casts, addresses, "
                        "identifiers and expressions are rejected because the harness embeds "
                        "each arg as a compile-time constant into the generated intrinsic call "
                        "(move the address/data preparation into `setup` and pass a literal "
                        "value instead)"
                    )
                    break
    return problems


def _coverage_problem(command: dict[str, Any], entry: dict[str, Any] | None) -> str:
    """First coverage problem, or "" when the command can be verified."""
    problems = _coverage_problem_list(command, entry)
    return problems[0] if problems else ""


def validate_oracle(isa: Any, oracle: Any) -> list[str]:
    """Structural problems that must be repaired before the test can be built."""
    errors: list[str] = []
    if not isinstance(oracle, dict):
        return ["oracle must be a mapping with a 'commands' list"]
    raw_commands = oracle.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        return ["oracle.commands must be a non-empty list"]
    commands = enabled_commands(isa)
    if not commands:
        return ["the ISA contract declares no encodable command to test"]
    seen: set[str] = set()
    for index, entry in enumerate(raw_commands):
        if not isinstance(entry, dict):
            errors.append(f"oracle.commands[{index}] must be a mapping")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"oracle.commands[{index}].name is required")
            continue
        if name in seen:
            errors.append(f"oracle.commands[{index}].name duplicates {name!r}")
            continue
        seen.add(name)
        if name not in {str(c.get("name", "")) for c in commands}:
            errors.append(f"oracle.commands[{index}] names unknown command {name!r}")
    for command in commands:
        name = str(command.get("name", ""))
        entry = _oracle_entries(oracle).get(name)
        for problem in _coverage_problem_list(command, entry):
            errors.append(f"{name}: {problem}")
        for issue in unknown_intrinsic_calls(entry or {}, isa):
            errors.append(f"{name}: {issue}")
    # Cross-entry helper symbol collisions (regression class, run 20260928-103622:
    # `redefinition of 'ref_rol32'` — two entries each defined the helper). A
    # helper needed by several commands must live in exactly one entry.
    symbol_counts: dict[str, int] = {}
    for entry in _oracle_entries(oracle).values():
        for symbol in _helper_definitions(str(entry.get("helpers") or "")):
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    for symbol in sorted(symbol_counts):
        if symbol_counts[symbol] > 1:
            errors.append(
                f"helpers symbol '{symbol}' is defined by more than one command: "
                "rename it or share it once"
            )
    return errors


def _render_case(command: dict[str, Any], entry: dict[str, Any]) -> str:
    """One case block: bind arguments, call the intrinsic and compare.

    The model's own cases come first; harness extras (see ``_extra_cases``)
    follow and count into ``g_cases`` like any other case. Both case-argument
    names are declared unconditionally at case scope — the undeclared-identifier
    class must never fail the C build even for a stale oracle (the validator
    lint rejects such references first, with a message that names them).
    """
    name = str(command.get("name", ""))
    roles = cpu_value_roles(command)
    intrinsic = intrinsic_name(command)
    check = "reference" if writes_cpu_rd(command) else "verify"
    expression = str(entry.get(check) or "").strip()
    model_cases = list(entry.get("cases") or [])
    cases = model_cases + [{"args": args} for args in _extra_cases(command, entry)]
    lines: list[str] = []
    for index, case in enumerate(cases):
        args = case.get("args") or []
        lines.append("    {")
        lines.append("      uint32_t a = 0;")
        lines.append("      uint32_t b = 0;")
        for role, value in zip(roles, args):
            lines.append(f"      {ROLE_NAMES[role]} = {_c_literal(value)};")
        lines.append("      (void)a; (void)b;")
        call = f"{intrinsic}({', '.join(ROLE_NAMES[r] for r in roles)})"
        if writes_cpu_rd(command):
            lines.append(f"      uint32_t got = (uint32_t){call};")
            lines.append(f"      uint32_t ref = (uint32_t)({expression});")
            lines.append("      g_cases++;")
            lines.append("      if (got != ref) {")
            lines.append("        cmd_ok = 0;")
            lines.append(
                f'        printf("CFU_CONFORMANCE_CASE FAIL {name}[{index}] '
                'got=0x%08lx ref=0x%08lx\\n", (unsigned long)got, (unsigned long)ref);'
            )
            lines.append("      }")
        else:
            lines.append(f"      (void){call};")
            lines.append("      g_cases++;")
            lines.append(f"      if (!({expression})) {{")
            lines.append("        cmd_ok = 0;")
            lines.append(
                f'        printf("CFU_CONFORMANCE_CASE FAIL {name}[{index}] verify=0\\n");'
            )
            lines.append("      }")
        lines.append("    }")
    return "\n".join(lines)


def render_conformance(isa: Any, oracle: Any) -> str:
    """Render the harness-owned C micro-test.

    Coverage problems are rendered as explicit failures (and counted against the
    ``commands=ok/total`` total), so an unverifiable command can never look like
    a pass. ``validate_oracle`` normally rejects them before this runs.
    """
    commands = enabled_commands(isa)
    entries = _oracle_entries(oracle)
    # Helper emission is collision-proof: identical blocks are emitted once, and
    # when two distinct blocks define the same symbol the first wins (the render
    # stays compilable; validate_oracle rejects the oracle with a rename hint).
    helper_blocks: list[str] = []
    seen_blocks: set[str] = set()
    emitted_symbols: set[str] = set()
    for command in commands:
        block = str(entries.get(str(command.get("name", "")), {}).get("helpers") or "").strip()
        if not block or block in seen_blocks:
            continue
        seen_blocks.add(block)
        symbols = set(_helper_definitions(block))
        if symbols & emitted_symbols:
            continue
        emitted_symbols |= symbols
        helper_blocks.append(block)
    body: list[str] = []
    for command in commands:
        name = str(command.get("name", ""))
        entry = entries.get(name, {})
        problem = "; ".join(_coverage_problem_list(command, entries.get(name)))
        body.append(f"  /* ---- {name} ---- */")
        body.append("  {")
        if problem:
            body.append("    int cmd_ok = 0;")
            body.append(
                f'    printf("CFU_CONFORMANCE_MISSING {name}: {problem}\\n");'
            )
        else:
            body.append("    int cmd_ok = 1;")
            setup = str(entry.get("setup") or "").strip()
            if setup:
                body.append("    /* oracle setup */")
                body.extend(f"    {line}" if line.strip() else "" for line in setup.splitlines())
            body.append(_render_case(command, entry))
        body.append(f'    if (cmd_ok) g_commands_ok++; else conf_fail("{name}");')
        body.append("  }")
        body.append("")
    rendered_helpers = "\n\n".join(helper_blocks)
    return f"""/* Generated by CfuGym for the hw_conformance rung. Do not edit by hand.
 *
 * The model owns only the oracle YAML; this test is harness-owned, so a hardware
 * repair cannot weaken it. `main()` returns non-zero on the first mismatch and the
 * linker-provided `fail` symbol then stops the simulation.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfu_intrinsics.h"

#define CONF_COMMANDS {len(commands)}

/* ---- oracle helpers ---- */
{rendered_helpers}

static uint32_t g_cases = 0;
static uint32_t g_commands_ok = 0;
static char g_failed[512];

static void conf_fail(const char *name) {{
  if (g_failed[0]) {{
    strncat(g_failed, ",", sizeof(g_failed) - strlen(g_failed) - 1);
  }}
  strncat(g_failed, name, sizeof(g_failed) - strlen(g_failed) - 1);
}}

int main(void) {{
  cfu_enable();

{chr(10).join(body)}
  printf("CFU_CONFORMANCE %s commands=%lu/%lu cases=%lu failed=%s\\n",
         (g_commands_ok == CONF_COMMANDS) ? "PASS" : "FAIL",
         (unsigned long)g_commands_ok, (unsigned long)CONF_COMMANDS,
         (unsigned long)g_cases, g_failed[0] ? g_failed : "none");
  return (g_commands_ok == CONF_COMMANDS) ? 0 : 1;
}}
"""


def parse_conformance(text: str) -> dict[str, Any]:
    """Parse the ``CFU_CONFORMANCE`` verdict line.

    A missing line is reported as ``status="missing"`` and must be treated as a
    failure: silence is not evidence.
    """
    body = "\n".join(
        _SBT_PREFIX_RE.sub("", line) for line in str(text or "").splitlines()
    )
    matches = list(STATUS_RE.finditer(body))
    details = [line.strip() for line in DETAIL_RE.findall(body)]
    if not matches:
        return {
            "status": "missing",
            "ok": False,
            "commands": 0,
            "total": 0,
            "cases": 0,
            "failed": [],
            "details": details,
        }
    match = matches[-1]
    raw_failed = match.group("failed")
    failed = [] if raw_failed in ("none", "-", "") else [n for n in raw_failed.split(",") if n]
    status = match.group("status").lower()
    commands = int(match.group("commands"))
    total = int(match.group("total"))
    # A PASS line that does not actually cover every command is still a failure.
    ok = status == "pass" and commands == total and total > 0 and not failed
    return {
        "status": status,
        "ok": ok,
        "commands": commands,
        "total": total,
        "cases": int(match.group("cases")),
        "failed": failed,
        "details": details,
    }
