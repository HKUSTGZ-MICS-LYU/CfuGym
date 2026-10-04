"""Harness-owned differential micro-test: rendering, oracle lint and verdict parsing."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.conformance import (
    _c_literal,
    cpu_value_roles,
    parse_conformance,
    render_conformance,
    validate_oracle,
    writes_cpu_rd,
)
from agents.cfu_design_agent.runtime import commands as commands_mod

CPU_COMMAND = {
    "function_id": 1,
    "subfunction_id": 0,
    "name": "compute",
    "kind": "compute",
    "semantics": "rd = rs1 + rs2",
    "resources": [
        {"storage": "cpu", "access": "read", "role": "rs1"},
        {"storage": "cpu", "access": "read", "role": "rs2"},
        {"storage": "cpu", "access": "write", "role": "rd"},
    ],
}
VECTOR_COMMAND = {
    "function_id": 1,
    "subfunction_id": 1,
    "name": "vset",
    "kind": "control",
    "semantics": "write a vector slot",
    "resources": [
        {"storage": "cpu", "access": "read", "role": "rs1"},
        {
            "storage": "vector",
            "access": "write",
            "role": "vd",
            "index": {"source": "cpu_value", "register": "rs2"},
        },
    ],
}
LEGACY_COMMAND = {"function_id": 2, "subfunction_id": 0, "name": "config", "kind": "control"}
ONE_ARG_COMMAND = {
    "function_id": 4,
    "subfunction_id": 0,
    "name": "onesel",
    "kind": "compute",
    "semantics": "rd = f(rs1)",
    "resources": [
        {"storage": "cpu", "access": "read", "role": "rs1"},
        {"storage": "cpu", "access": "write", "role": "rd"},
    ],
}
CONST_INDEX_COMMAND = {
    "function_id": 5,
    "subfunction_id": 0,
    "name": "vslot0",
    "kind": "control",
    "semantics": "vd[0] = rs1",
    "resources": [
        {"storage": "cpu", "access": "read", "role": "rs1"},
        {
            "storage": "vector",
            "access": "write",
            "role": "vd",
            "index": {"source": "constant", "value": 0},
        },
    ],
}


def isa(*commands: dict) -> dict:
    return {"spec_version": "1", "commands": [dict(c) for c in commands]}


class ResourceShapeTest(unittest.TestCase):
    def test_cpu_value_roles_follow_the_generated_intrinsic(self) -> None:
        self.assertEqual(cpu_value_roles(CPU_COMMAND), ["rs1", "rs2"])
        # A cpu_value vector index consumes rs2, and there is no CPU result.
        self.assertEqual(cpu_value_roles(VECTOR_COMMAND), ["rs1", "rs2"])
        # Absent resources means the legacy two-argument ABI.
        self.assertEqual(cpu_value_roles(LEGACY_COMMAND), ["rs1", "rs2"])
        self.assertTrue(writes_cpu_rd(CPU_COMMAND))
        self.assertFalse(writes_cpu_rd(VECTOR_COMMAND))
        self.assertTrue(writes_cpu_rd(LEGACY_COMMAND))

    def test_only_needed_cpu_value_arguments_are_passed(self) -> None:
        command = {
            "function_id": 1,
            "name": "oneArg",
            "resources": [
                {"storage": "cpu", "access": "read", "role": "rs2"},
                {"storage": "cpu", "access": "write", "role": "rd"},
            ],
        }
        self.assertEqual(cpu_value_roles(command), ["rs2"])


class RenderTest(unittest.TestCase):
    def oracle(self, command: dict, **fields) -> dict:
        entry = {"name": command["name"], "cases": [{"args": [1, 2]}], **fields}
        return {"commands": [entry]}

    def test_cpu_command_compares_intrinsic_against_reference(self) -> None:
        text = render_conformance(
            isa(CPU_COMMAND),
            self.oracle(
                CPU_COMMAND,
                helpers="static uint32_t ref_compute(uint32_t a, uint32_t b) { return a + b; }",
                reference="ref_compute(a, b)",
            ),
        )
        self.assertIn('#include "cfu_intrinsics.h"', text)
        self.assertIn("cfu_enable();", text)
        self.assertIn("uint32_t got = (uint32_t)cfu_compute(a, b);", text)
        self.assertIn("uint32_t ref = (uint32_t)(ref_compute(a, b));", text)
        self.assertIn("CFU_CONFORMANCE ", text)
        self.assertIn("return (g_commands_ok == CONF_COMMANDS) ? 0 : 1;", text)

    def test_generated_test_has_no_performance_instrumentation(self) -> None:
        # The rung is a functional check only: no counters, cycles or profile hooks.
        text = render_conformance(
            isa(CPU_COMMAND),
            self.oracle(CPU_COMMAND, reference="a + b"),
        )
        for banned in ("counter", "cycle", "CFU_AGENT_PROFILE", "clock", "timer"):
            self.assertNotIn(banned, text.lower())

    def test_vector_command_without_setup_is_a_coverage_failure(self) -> None:
        oracle = self.oracle(
            VECTOR_COMMAND, helpers="static uint32_t observe(uint32_t a, uint32_t b) { return 1; }",
            verify="observe(a, b)",
        )
        text = render_conformance(isa(VECTOR_COMMAND), oracle)
        self.assertIn("CFU_CONFORMANCE_MISSING vset:", text)
        self.assertIn("setup is required", text)
        # The command must not be counted as covered.
        self.assertIn("int cmd_ok = 0;", text)
        self.assertNotIn("int cmd_ok = 1;", text)
        self.assertTrue(validate_oracle(isa(VECTOR_COMMAND), oracle))

    def test_vector_command_with_setup_uses_verify(self) -> None:
        text = render_conformance(
            isa(VECTOR_COMMAND),
            self.oracle(
                VECTOR_COMMAND,
                setup="seed_slot(1);",
                verify="slot_is(1)",
            ),
        )
        self.assertIn("seed_slot(1);", text)
        self.assertIn("(void)cfu_vset(a, b);", text)
        self.assertIn("if (!(slot_is(1))) {", text)
        self.assertIn("g_commands_ok++;", text)

    def test_missing_coverage_never_passes_silently(self) -> None:
        # No oracle entry at all: the command is reported missing and cannot pass.
        text = render_conformance(isa(CPU_COMMAND), {"commands": []})
        self.assertIn("CFU_CONFORMANCE_MISSING compute:", text)
        self.assertIn("int cmd_ok = 0;", text)
        self.assertNotIn("int cmd_ok = 1;", text)


class ValidateOracleTest(unittest.TestCase):
    def entry(self, **fields) -> dict:
        return {"name": "compute", "cases": [{"args": [1, 2]}], **fields}

    def test_valid_oracle(self) -> None:
        self.assertEqual(
            validate_oracle(isa(CPU_COMMAND), {"commands": [self.entry(reference="a + b")]}), []
        )

    def test_missing_reference_for_cpu_rd_command(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [self.entry()]})
        self.assertTrue(any("reference is required" in p for p in problems), problems)

    def test_cpu_rd_command_may_not_use_verify_instead(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [self.entry(verify="1")]})
        self.assertTrue(any("reference is required" in p for p in problems), problems)

    def test_missing_verify_for_command_without_cpu_result(self) -> None:
        entry = {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();"}
        problems = validate_oracle(isa(VECTOR_COMMAND), {"commands": [entry]})
        self.assertTrue(any("verify" in p for p in problems), problems)

    def test_missing_cases_and_argument_mismatch(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{"name": "compute", "reference": "a"}]})
        self.assertTrue(any("cases must be a non-empty list" in p for p in problems), problems)
        problems = validate_oracle(
            isa(CPU_COMMAND),
            {"commands": [self.entry(reference="a", cases=[{"args": [1]}])]},
        )
        self.assertTrue(any("1 value(s)" in p for p in problems), problems)

    def test_unknown_and_duplicate_command_names(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [self.entry(reference="a", name="nope")]})
        self.assertTrue(any("unknown command" in p for p in problems), problems)
        problems = validate_oracle(
            isa(CPU_COMMAND),
            {"commands": [self.entry(reference="a"), self.entry(reference="b")]},
        )
        self.assertTrue(any("duplicates" in p for p in problems), problems)

    def test_every_declared_command_must_be_covered(self) -> None:
        problems = validate_oracle(
            isa(CPU_COMMAND, VECTOR_COMMAND),
            {"commands": [self.entry(reference="a")]},
        )
        self.assertTrue(any("vset: no oracle entry" in p for p in problems), problems)

    def test_empty_oracle_is_rejected(self) -> None:
        self.assertTrue(validate_oracle(isa(CPU_COMMAND), {"commands": []}))
        self.assertTrue(validate_oracle(isa(CPU_COMMAND), None))


class ParseVerdictTest(unittest.TestCase):
    def test_pass_line(self) -> None:
        verdict = parse_conformance(
            "boot\nCFU_CONFORMANCE PASS commands=2/2 cases=3 failed=none\n"
        )
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["status"], "pass")
        self.assertEqual((verdict["commands"], verdict["total"], verdict["cases"]), (2, 2, 3))

    def test_fail_line_lists_commands_and_details(self) -> None:
        verdict = parse_conformance(
            "CFU_CONFORMANCE_CASE FAIL compute[0] got=0x00000001 ref=0x00000002\n"
            "CFU_CONFORMANCE FAIL commands=1/2 cases=2 failed=vset\n"
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed"], ["vset"])
        self.assertTrue(any("compute[0]" in d for d in verdict["details"]))

    def test_missing_line_is_a_failure(self) -> None:
        # Silence is not evidence: a test that did not report must not pass.
        verdict = parse_conformance("some unrelated log output\n")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["status"], "missing")

    def test_incomplete_pass_line_is_a_failure(self) -> None:
        verdict = parse_conformance("CFU_CONFORMANCE PASS commands=1/2 cases=2 failed=none\n")
        self.assertEqual(verdict["status"], "pass")
        self.assertFalse(verdict["ok"])

    def test_last_verdict_wins(self) -> None:
        verdict = parse_conformance(
            "CFU_CONFORMANCE FAIL commands=0/1 cases=0 failed=compute\n"
            "CFU_CONFORMANCE PASS commands=1/1 cases=1 failed=none\n"
        )
        self.assertTrue(verdict["ok"])

    def test_build_command_links_the_test_as_main_only(self) -> None:
        # The sw Makefile links `$(OBJS) $(MAIN).c`, so passing the test in
        # EXTRA_SOURCES as well fails a real build with "multiple definition of
        # `main`". The generated source is the MAIN translation unit.
        from agents.cfu_design_agent import graph

        state = {"run_root": "agents/generated/runs/x-00000000", "run_id": "x-00000000"}
        header = ("agents/generated/runs/x-00000000/workspace/embench_crc32/cfu/"
                  "include/cfu_intrinsics.h")
        argv, elf = commands_mod._conformance_build_command(state, header)
        joined = " ".join(argv)
        self.assertIn(commands_mod.CONFORMANCE_BIN, joined)
        self.assertIn("MAIN=", joined)
        self.assertIn("EXTRA_INCLUDES=", joined)
        self.assertNotIn("EXTRA_SOURCES=", joined)
        self.assertTrue(elf.endswith(f"{commands_mod.CONFORMANCE_BIN}.elf"), elf)

    def test_sbt_prefixed_verdict_is_parsed(self) -> None:
        # Real MiCoSocSim output is sbt-framed ("[info] CFU_CONFORMANCE PASS ...").
        # A mock that feeds bare lines passes while the real path reports "missing".
        text = (
            "[info] [Progress] Start MiCoSocSim test simulation with seed 42\n"
            "[info] CFU_CONFORMANCE PASS commands=4/4 cases=7 failed=none\n"
            "[info] [Done] Simulation done in 3718.700 ms\n"
            "[success] Total time: 8 s\n"
        )
        verdict = parse_conformance(text)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["commands"], 4)
        self.assertEqual(verdict["cases"], 7)

    def test_sbt_prefixed_case_failure_is_parsed(self) -> None:
        text = (
            '[info] CFU_CONFORMANCE_CASE FAIL add[0] got=0x00000000 ref=0x00000003\n'
            "[info] CFU_CONFORMANCE FAIL commands=0/1 cases=2 failed=add\n"
            '[error] Exception in thread "main" spinal.sim.SimFailure: boom\n'
        )
        verdict = parse_conformance(text)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed"], ["add"])
        self.assertTrue(any("add[0]" in d for d in verdict["details"]), verdict)


class ArgGuidanceTest(unittest.TestCase):
    """Oracle-arg rejections must explain the fix, not just the format.

    Regression (2026-09-26, matmult-int): the model oscillated three oracle
    attempts between the arg-count and arg-literal errors because neither the
    validator message nor the prompt said that args are compile-time constants
    and addresses belong in `setup`.
    """

    def entry(self, **fields) -> dict:
        return {"name": "compute", "cases": [{"args": [1, 2]}], **fields}

    def test_literal_error_names_value_and_setup_fix(self) -> None:
        problems = validate_oracle(
            isa(CPU_COMMAND),
            {"commands": [self.entry(reference="a", cases=[{"args": ["(uintptr_t)probe", 2]}])]},
        )
        message = next(p for p in problems if "args[0]" in p)
        self.assertIn("(uintptr_t)probe", message)  # the offending value
        self.assertIn("setup", message)  # where the address belongs
        self.assertIn("compile-time constant", message)

    def test_count_error_says_one_literal_per_role(self) -> None:
        problems = validate_oracle(
            isa(CPU_COMMAND),
            {"commands": [self.entry(reference="a", cases=[{"args": ["0x80000000", 0, 3]}])]},
        )
        message = next(p for p in problems if "value(s)" in p)
        self.assertIn("one literal per role", message)
        self.assertIn("setup", message)

    def test_prompt_documents_literal_args_and_repair_discipline(self) -> None:
        from agents.cfu_design_agent.prompts import conformance_prompt

        prompt = conformance_prompt("task", isa(CPU_COMMAND), {}, feedback="compute: problems")
        self.assertIn("INTEGER LITERAL", prompt)
        self.assertIn("ORACLE REPAIR MODE", prompt)
        self.assertIn("at once", prompt)
        self.assertIn("setup", prompt)
        # The prompt names the generated intrinsics so setup/verify cannot
        # invent names like cfu_load (2026-09-26 link failure).
        self.assertIn("cfu_compute(", prompt)
        self.assertIn("undefined reference", prompt)


class UnknownIntrinsicLintTest(unittest.TestCase):
    """An invented cfu_* name must fail validation, not the linker."""

    def test_invented_intrinsic_is_rejected_before_the_linker(self) -> None:
        entry = {
            "name": "compute", "cases": [{"args": [1, 2]}], "reference": "a + b",
            "setup": "cfu_load((uintptr_t)buf, 0);",
        }
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [entry]})
        message = next(p for p in problems if "unknown intrinsic" in p)
        self.assertIn("cfu_load", message)
        self.assertIn("defines only", message)
        self.assertIn("cfu_compute", message)

    def test_suggests_the_generated_name(self) -> None:
        command = dict(CPU_COMMAND, name="vector_load_i32")
        entry = {
            "name": "vector_load_i32", "cases": [{"args": [1, 2]}], "reference": "a + b",
            "setup": "cfu_load(p, 0);",
        }
        problems = validate_oracle(isa(command), {"commands": [entry]})
        message = next(p for p in problems if "unknown intrinsic" in p)
        self.assertIn("did you mean cfu_vector_load_i32", message)

    def test_generated_names_and_helpers_are_allowed(self) -> None:
        entry = {
            "name": "compute", "cases": [{"args": [1, 2]}], "reference": "a + b",
            "setup": "cfu_enable(); cfu_memory_fence();",
        }
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [entry]})
        self.assertFalse(any("unknown intrinsic" in p for p in problems), problems)


class LazyOracleRejectionTest(unittest.TestCase):
    """A lazy oracle must fail validation, never pass silently.

    Regression class (2026-09-28, run 20260928-103622-049268-95608f2c): the
    design passed hw_conformance 2/2 and then failed the Embench verify at
    0.75x, because a one-case oracle with a weak expression cannot distinguish
    a wrong design. ``verify: "1"``, ``reference: "0"`` and a self-referential
    ``reference: "cfu_compute(a, b)"`` all validated clean before this lint.
    """

    def entry(self, **fields) -> dict:
        return {"name": "compute", "cases": [{"args": [1, 2]}], **fields}

    def test_constant_reference_rejected(self) -> None:
        for constant in ("1", "0", "0x1234u", "(uint32_t)0", "(unsigned long)0"):
            with self.subTest(constant=constant):
                problems = validate_oracle(
                    isa(CPU_COMMAND), {"commands": [self.entry(reference=constant)]}
                )
                message = next((p for p in problems if "not a constant" in p), None)
                self.assertIsNotNone(message, problems)
                self.assertIn("must actually observe the command's effect", message)
                self.assertIn("state set by setup", message)

    def test_tautology_verify_rejected(self) -> None:
        # Non-rd command: verify is the operative expression.
        entry = {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();", "verify": "1"}
        problems = validate_oracle(isa(VECTOR_COMMAND), {"commands": [entry]})
        message = next((p for p in problems if "verify" in p and "not a constant" in p), None)
        self.assertIsNotNone(message, problems)

    def test_self_referential_reference_rejected(self) -> None:
        problems = validate_oracle(
            isa(CPU_COMMAND), {"commands": [self.entry(reference="cfu_compute(a, b)")]}
        )
        message = next((p for p in problems if "must not call cfu_*" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("cfu_compute", message)
        self.assertIn("independent scalar C expression", message)

    def test_self_referential_verify_rejected(self) -> None:
        entry = {
            "name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
            "verify": "cfu_vset(a, b)",
        }
        problems = validate_oracle(isa(VECTOR_COMMAND), {"commands": [entry]})
        self.assertTrue(any("must not call cfu_*" in p for p in problems), problems)

    def test_identifier_bearing_expression_passes_the_constant_check(self) -> None:
        # Conservative on purpose: any identifier passes; no full C parsing.
        for expression in ("a + b", "ref_rol32(a, b)", "a * 0 + b", "a ? 1 : 2"):
            with self.subTest(expression=expression):
                self.assertEqual(
                    validate_oracle(
                        isa(CPU_COMMAND), {"commands": [self.entry(reference=expression)]}
                    ),
                    [],
                )


class InformativeOracleTest(unittest.TestCase):
    """An oracle that decides its own outcome is rejected, not just a constant.

    Regression class (2026-10-03, run 20261003-090854-136683-d98e5be9): nine
    conformance cases passed while the benchmark computed a wrong digest, because
    every check was self-fulfilling — ``verify: "a == a"``, ``verify:
    "(a | b) == 0u"`` with the single case ``[0, 0]``, ``verify:
    "probe_marker == 1u"`` where setup had written ``probe_marker = 1u``, and a
    readback ``reference`` that returned the reset IV table.
    """

    def commands(self, *entries: dict) -> dict:
        return {"commands": list(entries)}

    def problems(self, contract: dict, entry: dict) -> list[str]:
        return validate_oracle(contract, self.commands(entry))

    def assert_message(self, problems: list[str], needle: str) -> None:
        message = next((p for p in problems if needle in p), None)
        self.assertIsNotNone(message, problems)

    def test_operand_compared_with_itself_is_rejected(self) -> None:
        for expression in ("a == a", "(a) == a", "b != b"):
            with self.subTest(expression=expression):
                entry = {"name": "vset", "cases": [{"args": [1, 2]}],
                         "setup": "seed();", "verify": expression}
                self.assert_message(self.problems(isa(VECTOR_COMMAND), entry),
                                    "compares an operand with itself")

    def test_argument_only_verify_is_rejected(self) -> None:
        entry = {"name": "vset", "cases": [{"args": [1, 0]}],
                 "setup": "seed();", "verify": "(a | b) == 0u"}
        self.assert_message(self.problems(isa(VECTOR_COMMAND), entry),
                            "verify only references the case arguments")

    def test_expression_comparing_its_own_value_is_rejected(self) -> None:
        """A value the oracle wrote next to the check cannot verify anything."""
        marker = {"name": "vset", "cases": [{"args": [1, 2]}],
                  "setup": "static volatile uint32_t marker = 1u; seed(); marker = 1u;",
                  "verify": "marker == 1u"}
        helper = {"name": "compute", "cases": [{"args": [1, 2]}],
                  "helpers": "static uint32_t expected;\n"
                             "static void set_expected(void) { expected = 4u; }",
                  "setup": "set_expected();", "reference": "expected"}
        for entry, contract in ((marker, isa(VECTOR_COMMAND)), (helper, isa(CPU_COMMAND))):
            with self.subTest(expression=entry.get("verify") or entry["reference"]):
                self.assert_message(self.problems(contract, entry), "assigned a literal to")

    def test_all_zero_case_arguments_are_rejected(self) -> None:
        entry = {"name": "compute", "cases": [{"args": [0, 0]}], "reference": "a + b"}
        self.assert_message(self.problems(isa(CPU_COMMAND), entry),
                            "only zero arguments")
        # One representative non-zero case is enough.
        entry = {"name": "compute", "cases": [{"args": [0, 0]}, {"args": [3, 4]}],
                 "reference": "a + b"}
        self.assertEqual(self.problems(isa(CPU_COMMAND), entry), [])

    def test_genuine_observations_still_validate(self) -> None:
        """No false positives: seed it in setup, read it back, and it is evidence."""
        control = {"function_id": 3, "subfunction_id": 0, "name": "ctl",
                   "kind": "control", "resources": []}
        cases = [
            # A no-role command whose verify reads memory a helper seeded.
            (isa(control), {"name": "ctl", "cases": [{"args": []}],
                       "helpers": "static void seed_memory(void) { volatile uint32_t *p = "
                                  "(volatile uint32_t *)0x80000000u; p[0] = 4u; }\n"
                                  "static uint32_t ctl_seen(void) { volatile uint32_t *p = "
                                  "(volatile uint32_t *)0x80000000u; return p[0] == 4u; }",
                       "setup": "seed_memory();", "verify": "ctl_seen()"}),
            (isa(CPU_COMMAND), {"name": "compute", "cases": [{"args": [4, 2]}],
                           "helpers": "static uint32_t w[8];",
                           "setup": "w[0] = 0x80000000u;", "reference": "w[0] >> 4"}),
            (isa(VECTOR_COMMAND), {"name": "vset", "cases": [{"args": [4, 2]}],
                              "helpers": "static uint32_t check_loaded(void) { "
                                         "volatile uint32_t *p = (volatile uint32_t *)0x80000000u; "
                                         "return p[0] == 4u; }",
                              "setup": "seed();", "verify": "check_loaded()"}),
            # `p = (T *)addr` publishes an address; reading through p is the observation.
            (isa(CPU_COMMAND), {"name": "compute", "cases": [{"args": [4, 2]}],
                           "helpers": "static volatile uint32_t *p;",
                           "setup": "p = (volatile uint32_t *)0x80000000u;",
                           "reference": "p[a]"}),
        ]
        for contract, entry in cases:
            with self.subTest(expression=entry.get("verify") or entry["reference"]):
                self.assertEqual(validate_oracle(contract, self.commands(entry)), [])


class HelperOracleIndependenceTest(unittest.TestCase):
    """Oracle expressions cannot hide DUT evaluation in shared C helpers."""

    def entry(self, **fields) -> dict:
        return {"name": "compute", "cases": [{"args": [1, 2]}], **fields}

    def assert_rejected(self, contract: dict, oracle: dict, name: str, *symbols: str) -> None:
        problems = validate_oracle(contract, oracle)
        message = next((p for p in problems if p.startswith(name + ":") and
                        "must not call cfu_*" in p), None)
        self.assertIsNotNone(message, problems)
        for symbol in symbols:
            self.assertIn(symbol, message)
        # Defense in depth: a caller bypassing validate must still render FAIL,
        # not evaluate the self-grading expression or count the command covered.
        text = render_conformance(contract, oracle)
        self.assertIn(f"CFU_CONFORMANCE_MISSING {name}:", text)
        section = text.split(f"/* ---- {name} ---- */", 1)[1].split("/* ----", 1)[0]
        self.assertIn("int cmd_ok = 0;", section)
        self.assertNotIn("g_cases++;", section)

    def test_reference_helper_directly_calls_dut(self) -> None:
        entry = self.entry(
            helpers="static uint32_t ref_sum(uint32_t x, uint32_t y) { return cfu_compute(x, y); }",
            reference="ref_sum(a, b)",
        )
        self.assert_rejected(isa(CPU_COMMAND), {"commands": [entry]}, "compute", "ref_sum", "cfu_compute")

    def test_reference_transitively_calls_dut(self) -> None:
        entry = self.entry(helpers="""
            static uint32_t leaf(uint32_t x, uint32_t y) { return cfu_compute(x, y); }
            static uint32_t middle(uint32_t x, uint32_t y) { if (x) { return leaf(x, y); } return y; }
            static uint32_t ref_sum(uint32_t x, uint32_t y) { return middle(x, y); }
            """, reference="ref_sum(a, b)")
        self.assert_rejected(isa(CPU_COMMAND), {"commands": [entry]}, "compute",
                             "ref_sum", "middle", "leaf", "cfu_compute")

    def test_verify_helper_calls_dut(self) -> None:
        entry = {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
                 "helpers": "static int observe(uint32_t x, uint32_t y) { cfu_vset(x, y); return x; }",
                 "verify": "observe(a, b)"}
        self.assert_rejected(isa(VECTOR_COMMAND), {"commands": [entry]}, "vset", "observe", "cfu_vset")

    def test_cross_entry_helper_is_shared_for_reference_and_verify(self) -> None:
        for helpers in (
            "static uint32_t shared(uint32_t x, uint32_t y) { return cfu_compute(x, y); }",
            "#define shared(x, y) DUT((x), (y))\n#define DUT cfu_compute",
        ):
            with self.subTest(helpers=helpers):
                oracle = {"commands": [
                    self.entry(reference="shared(a, b)"),
                    {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
                     "helpers": helpers, "verify": "shared(a, b)"},
                ]}
                for name in ("compute", "vset"):
                    self.assert_rejected(isa(CPU_COMMAND, VECTOR_COMMAND), oracle, name,
                                         "shared", "cfu_compute")

    def test_macro_wrappers_and_object_aliases(self) -> None:
        wrappers = (
            "#define REF(x, y) cfu_compute((x), (y))",
            "#define DUT cfu_compute\n#define REF(x, y) DUT((x), (y))",
            "#define REF(x, y) \\\n  cfu_compute((x), (y))",
            "#define DUT(x, y) cfu_compute((x), (y))\n"
            "static uint32_t REF(uint32_t x, uint32_t y) { return DUT(x, y); }",
        )
        for helpers in wrappers:
            with self.subTest(helpers=helpers):
                oracle = {"commands": [self.entry(helpers=helpers, reference="REF(a, b)")]}
                self.assert_rejected(isa(CPU_COMMAND), oracle, "compute", "REF", "cfu_compute")

    def test_pointer_aliases_and_callback_references(self) -> None:
        forms = (
            {"helpers": "static uint32_t (*ref_ptr)(uint32_t, uint32_t) = &cfu_compute;",
             "reference": "ref_ptr(a, b)"},
            {"helpers": "static uint32_t (*ref_ptr)(uint32_t, uint32_t);",
             "setup": "ref_ptr = cfu_compute;", "reference": "(*ref_ptr)(a, b)"},
            {"helpers": "static uint32_t ref_sum(uint32_t x, uint32_t y) {"
                        " uint32_t (*f)(uint32_t, uint32_t) = cfu_compute; return f(x, y); }",
             "reference": "ref_sum(a, b)"},
            {"helpers": "static uint32_t bad(uint32_t x, uint32_t y) { return cfu_compute(x, y); }"
                        " static uint32_t (*ref_ptr)(uint32_t, uint32_t) = bad;",
             "reference": "ref_ptr(a, b)"},
            {"reference": "apply(cfu_compute, a, b)"},
        )
        for fields in forms:
            with self.subTest(fields=fields):
                self.assert_rejected(isa(CPU_COMMAND), {"commands": [self.entry(**fields)]},
                                     "compute", "cfu_compute")

    def test_pure_shared_helpers_and_macros_are_accepted(self) -> None:
        oracle = {"commands": [
            self.entry(reference="REF(a, b)", helpers="""
                #define REF(x, y) ref_sum((x), (y))
                static uint32_t add(uint32_t x, uint32_t y) { return x + y; }
                static uint32_t ref_sum(uint32_t x, uint32_t y) {
                  /* cfu_compute(x, y) */
                  const char *note = "cfu_compute(x, y) { }"; (void)note;
                  if (x) { return add(x, y); } return y;
                }
                """),
            {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();", "verify": "REF(a, b)"},
        ]}
        self.assertEqual(validate_oracle(isa(CPU_COMMAND, VECTOR_COMMAND), oracle), [])
        self.assertNotIn("CFU_CONFORMANCE_MISSING", render_conformance(isa(CPU_COMMAND, VECTOR_COMMAND), oracle))

    def test_setup_only_helper_and_macro_may_call_approved_dut(self) -> None:
        entry = self.entry(helpers="""
            #define SEED(x, y) cfu_compute((x), (y))
            static void seed_inner(void) { cfu_enable(); SEED(1, 2); cfu_memory_fence(); }
            static void seed(void) { seed_inner(); }
            static uint32_t ref_sum(uint32_t x, uint32_t y) { return x + y; }
            """, setup="seed();", reference="ref_sum(a, b)")
        oracle = {"commands": [entry]}
        self.assertEqual(validate_oracle(isa(CPU_COMMAND), oracle), [])
        text = render_conformance(isa(CPU_COMMAND), oracle)
        self.assertNotIn("CFU_CONFORMANCE_MISSING", text)
        self.assertIn("seed();", text)
        self.assertIn("uint32_t ref = (uint32_t)(ref_sum(a, b));", text)

    def test_cycles_terminate_and_still_find_dut(self) -> None:
        for leaf in ("first(x, y)", "first(x, y) + cfu_compute(x, y)"):
            entry = self.entry(helpers=
                "static uint32_t first(uint32_t x, uint32_t y) { return second(x, y); }"
                f" static uint32_t second(uint32_t x, uint32_t y) {{ return {leaf}; }}",
                reference="first(a, b)")
            oracle = {"commands": [entry]}
            if "cfu_" in leaf:
                self.assert_rejected(isa(CPU_COMMAND), oracle, "compute", "cfu_compute")
            else:
                self.assertEqual(validate_oracle(isa(CPU_COMMAND), oracle), [])


class UndeclaredArgumentLintTest(unittest.TestCase):
    """The generated case scope only binds declared roles' arguments.

    Regression (run 20260928-105332-728135-7f9a4132, report.md:920): a verify
    expression referencing ``b`` for a command that never declared rs2 died at
    build time with ``'b' undeclared (first use in this function)``. The lint
    names the identifier and the fix at validation time instead.
    """

    def test_verify_referencing_undeclared_role_rejected(self) -> None:
        entry = {
            "name": "vslot0", "cases": [{"args": [7]}], "setup": "seed();",
            "verify": "slot_at(b) == 0",
        }
        problems = validate_oracle(isa(CONST_INDEX_COMMAND), {"commands": [entry]})
        message = next((p for p in problems if "undeclared identifier" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("'b'", message)
        self.assertIn("resources", message)
        self.assertIn("rs1->a, rs2->b", message)

    def test_reference_referencing_undeclared_role_rejected(self) -> None:
        entry = {"name": "onesel", "cases": [{"args": [3]}], "reference": "a + b"}
        problems = validate_oracle(isa(ONE_ARG_COMMAND), {"commands": [entry]})
        self.assertTrue(
            any("'b'" in p and "undeclared identifier" in p for p in problems), problems
        )

    def test_declared_roles_may_be_referenced(self) -> None:
        entry = {"name": "onesel", "cases": [{"args": [3]}], "reference": "a * 2u + 1u"}
        self.assertEqual(validate_oracle(isa(ONE_ARG_COMMAND), {"commands": [entry]}), [])


class HelperCollisionTest(unittest.TestCase):
    """Helper symbols must be defined exactly once across oracle entries.

    Regression (run 20260928-103622-049268-95608f2c, report.md:998): two
    entries each defined ``ref_rol32`` and the generated test failed with
    ``redefinition of 'ref_rol32'`` at build time.
    """

    HELPER = "static uint32_t ref_rol32(uint32_t a, uint32_t b) { return a + (b << 1); }"

    def oracle(self, second_helper: str) -> dict:
        return {
            "commands": [
                {
                    "name": "compute", "cases": [{"args": [1, 2]}],
                    "reference": "ref_rol32(a, b)", "helpers": self.HELPER,
                },
                {
                    "name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
                    "verify": "ref_rol32(a, b)", "helpers": second_helper,
                },
            ]
        }

    def test_duplicate_helpers_rejected_redefinition_class(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND, VECTOR_COMMAND), self.oracle(self.HELPER))
        message = next((p for p in problems if "helpers symbol" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("ref_rol32", message)
        self.assertIn("defined by more than one command", message)
        self.assertIn("rename it or share it once", message)

    def test_identical_helpers_emitted_once(self) -> None:
        text = render_conformance(isa(CPU_COMMAND, VECTOR_COMMAND), self.oracle(self.HELPER))
        self.assertEqual(text.count("static uint32_t ref_rol32"), 1)

    def test_colliding_distinct_helpers_first_definition_wins(self) -> None:
        # The render stays compilable even without validation: the colliding
        # block is dropped whole, so two definitions can never be emitted.
        distinct = self.HELPER + "\nstatic uint32_t other_h(uint32_t a) { return a; }"
        text = render_conformance(isa(CPU_COMMAND, VECTOR_COMMAND), self.oracle(distinct))
        self.assertEqual(text.count("static uint32_t ref_rol32"), 1)
        self.assertNotIn("other_h", text)

    def test_duplicate_file_scope_object_rejected(self) -> None:
        """Run 20261003-130005: a variable counted as a redefinition too.

        Both entries defined ``md5_oracle_mem``; the function-only symbol scan
        accepted the oracle, and the delivered C then failed to compile with
        ``redefinition of 'md5_oracle_mem'`` in the hw_conformance rung.
        """
        pointer = ("static volatile uint32_t *const md5_oracle_mem = "
                   "(volatile uint32_t *)0x80000000u;")
        oracle = {
            "commands": [
                {"name": "compute", "cases": [{"args": [1, 2]}], "reference": "a + b",
                 "helpers": pointer},
                {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
                 "verify": "md5_oracle_mem[0] != 0u",
                 "helpers": pointer + "\nstatic uint32_t extra(uint32_t a) { return a; }"},
            ]
        }
        problems = validate_oracle(isa(CPU_COMMAND, VECTOR_COMMAND), oracle)
        message = next((p for p in problems if "md5_oracle_mem" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("defined by more than one command", message)

    def test_function_local_variables_are_not_symbols(self) -> None:
        """Only file-scope objects count; locals in a body must never collide."""
        first = ("static uint32_t ref_a(uint32_t a) { uint32_t scratch = a; return scratch; }")
        second = ("static uint32_t ref_b(uint32_t a) { uint32_t scratch = a + 1u; "
                  "return scratch; }")
        oracle = {
            "commands": [
                {"name": "compute", "cases": [{"args": [1, 2]}], "reference": "ref_a(a)",
                 "helpers": first},
                {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();",
                 "verify": "ref_b(a) != 0u", "helpers": second},
            ]
        }
        self.assertEqual(validate_oracle(isa(CPU_COMMAND, VECTOR_COMMAND), oracle), [])

    def test_distinct_helper_symbols_are_all_emitted(self) -> None:
        distinct = "static uint32_t other_h(uint32_t a) { return a; }"
        text = render_conformance(isa(CPU_COMMAND, VECTOR_COMMAND), self.oracle(distinct))
        self.assertIn("ref_rol32", text)
        self.assertIn("other_h", text)


class ModelChosenCasesTest(unittest.TestCase):
    """The model owns every input and size: no hidden random/edge expansion."""

    def entry(self, **fields) -> dict:
        return {"name": "compute", "cases": [{"args": [1, 2]}], **fields}

    def test_only_declared_cases_are_rendered_in_order(self) -> None:
        entry = self.entry(reference="a + b", cases=[{"args": [11, 2]}, {"args": [22, 8]}])
        text = render_conformance(isa(CPU_COMMAND), {"commands": [entry]})
        self.assertEqual(text.count("g_cases++;"), 2)
        self.assertEqual(text.count("uint32_t got ="), 2)
        self.assertLess(text.index("a = (uint32_t)11u"), text.index("a = (uint32_t)22u"))
        self.assertNotIn("2147483648u", text)
        self.assertNotIn("4294967295u", text)

    def test_large_seed_does_not_invent_large_iteration_counts(self) -> None:
        entry = self.entry(reference="a ^ b", cases=[
            {"args": [0, 0]}, {"args": [0, 1]}, {"args": [4294967295, 8]},
            {"args": [12345, 1024]},
        ])
        text = render_conformance(isa(CPU_COMMAND), {"commands": [entry]})
        self.assertEqual(text.count("g_cases++;"), 4)
        self.assertIn("a = (uint32_t)4294967295u", text)
        self.assertNotIn("b = (uint32_t)4294967295u", text)
        self.assertNotIn("b = (uint32_t)880296978u", text)

    def test_legacy_edge_flag_never_adds_cases(self) -> None:
        for flag in (True, False):
            with self.subTest(flag=flag):
                entry = self.entry(reference="a + b", edge_cases=flag)
                self.assertEqual(validate_oracle(isa(CPU_COMMAND), {"commands": [entry]}), [])
                text = render_conformance(isa(CPU_COMMAND), {"commands": [entry]})
                self.assertEqual(text.count("g_cases++;"), 1)

    def test_legacy_edge_flag_must_still_be_boolean(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [
            self.entry(reference="a + b", edge_cases="no")]})
        self.assertTrue(any("edge_cases must be a boolean" in p for p in problems), problems)

    def test_non_rd_cases_are_not_expanded(self) -> None:
        entry = {"name": "vset", "cases": [{"args": [1, 2]}], "setup": "seed();", "verify": "slot_is(1)"}
        text = render_conformance(isa(VECTOR_COMMAND), {"commands": [entry]})
        self.assertEqual(text.count("g_cases++;"), 1)

    def test_prompt_requires_model_chosen_small_quick_tests(self) -> None:
        from agents.cfu_design_agent.prompts import conformance_prompt
        for tools in (True, False):
            with self.subTest(tools=tools):
                prompt = conformance_prompt("test", isa(CPU_COMMAND), {}, agent_tools=tools)
                self.assertIn("You choose ALL test inputs and data sizes", prompt)
                self.assertIn("SMALL DATA", prompt)
                self.assertIn("32000 cycles", prompt)
                self.assertIn("never appends hidden edge or random inputs", prompt)
                self.assertIn("do NOT use 0xFFFFFFFF", prompt)

    def test_case_scope_binds_a_and_b_unconditionally(self) -> None:
        # Render backstop for stale oracles: undeclared identifiers must never
        # fail the C build; the validator lint rejects them first with a fix.
        entry = {"name": "onesel", "cases": [{"args": [3]}], "reference": "a"}
        text = render_conformance(isa(ONE_ARG_COMMAND), {"commands": [entry]})
        self.assertIn("uint32_t a = 0;", text)
        self.assertIn("uint32_t b = 0;", text)
        self.assertIn("(void)a; (void)b;", text)
        self.assertIn("a = (uint32_t)3u;", text)
        self.assertNotIn("b = (uint32_t)", text)  # undeclared role stays 0


if __name__ == "__main__":
    unittest.main()
