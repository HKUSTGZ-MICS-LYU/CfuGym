"""Harness-owned differential micro-test: rendering, oracle lint and verdict parsing."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.conformance import (
    cpu_value_roles,
    parse_conformance,
    render_conformance,
    validate_oracle,
    writes_cpu_rd,
)

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
        argv, elf = graph._conformance_build_command(state, header)
        joined = " ".join(argv)
        self.assertIn(graph.CONFORMANCE_BIN, joined)
        self.assertIn("MAIN=", joined)
        self.assertIn("EXTRA_INCLUDES=", joined)
        self.assertNotIn("EXTRA_SOURCES=", joined)
        self.assertTrue(elf.endswith(f"{graph.CONFORMANCE_BIN}.elf"), elf)

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


if __name__ == "__main__":
    unittest.main()
