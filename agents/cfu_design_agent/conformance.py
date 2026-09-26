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
"""

from __future__ import annotations

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


def _coverage_problem(command: dict[str, Any], entry: dict[str, Any] | None) -> str:
    """Why this command cannot be verified, or "" when it can.

    The caller prefixes the command name, so messages here stay name-free.
    """
    if entry is None:
        return "no oracle entry"
    cases = entry.get("cases")
    if not isinstance(cases, list) or not cases:
        return "cases must be a non-empty list"
    if writes_cpu_rd(command):
        if not str(entry.get("reference") or "").strip():
            return (
                "reference is required because the command writes CPU rd; provide an "
                "independent scalar C expression over the case arguments"
            )
    elif not str(entry.get("verify") or "").strip():
        return (
            "this command has no CPU result, so it cannot be compared; provide a "
            "verify C expression that is non-zero after the call (weaker evidence "
            "than reference, but it must not be omitted)"
        )
    if has_vector_resource(command) and not str(entry.get("setup") or "").strip():
        return (
            "setup is required because the command uses the vector register file; "
            "the harness does not guess how to seed it"
        )
    roles = cpu_value_roles(command)
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return f"cases[{index}] must be a mapping with an args list"
        args = case.get("args")
        if not isinstance(args, list):
            return f"cases[{index}].args must be a list"
        if len(args) != len(roles):
            return (
                f"cases[{index}].args has {len(args)} value(s) but the command takes "
                f"{len(roles)} CPU value argument(s) ({', '.join(roles) or 'none'})"
            )
        for position, value in enumerate(args):
            if not _c_literal(value):
                return (
                    f"cases[{index}].args[{position}] must be an integer or a C-style "
                    "integer string such as 0xFFFFFFFF"
                )
    return ""


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
        problem = _coverage_problem(command, _oracle_entries(oracle).get(name))
        if problem:
            errors.append(f"{name}: {problem}")
    return errors


def _render_case(command: dict[str, Any], entry: dict[str, Any]) -> str:
    """One case block: bind arguments, call the intrinsic and compare."""
    name = str(command.get("name", ""))
    roles = cpu_value_roles(command)
    intrinsic = intrinsic_name(command)
    check = "reference" if writes_cpu_rd(command) else "verify"
    expression = str(entry.get(check) or "").strip()
    lines: list[str] = []
    for index, case in enumerate(entry.get("cases") or []):
        args = case.get("args") or []
        lines.append("    {")
        for role, value in zip(roles, args):
            lines.append(f"      uint32_t {ROLE_NAMES[role]} = {_c_literal(value)};")
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
    ``commands=ok/total`` total), so an unverifiable command can never look like a
    pass. ``validate_oracle`` normally rejects them before this runs.
    """
    commands = enabled_commands(isa)
    entries = _oracle_entries(oracle)
    helpers = [
        str(entry.get("helpers") or "").strip()
        for entry in (
            entries.get(str(c.get("name", "")), {}) for c in commands
        )
        if str(entry.get("helpers") or "").strip()
    ]
    body: list[str] = []
    for command in commands:
        name = str(command.get("name", ""))
        entry = entries.get(name, {})
        problem = _coverage_problem(command, entries.get(name))
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
    rendered_helpers = "\n\n".join(helpers)
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
