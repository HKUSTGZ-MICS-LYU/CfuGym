"""Harness-owned differential micro-test: data generator, rendering, oracle lint.

The model owns the oracle's *semantics*. The harness owns the *data*: the
``data.buffer``/``data.roles`` spec selects the shape of the input space and
``conformance_data`` generates the boundary sweep and the random draws from a
fixed seed. These tests pin both halves, and every rejection message they assert
has to name the fix, not just the format.
"""

from __future__ import annotations

import unittest

from agents.cfu_design_agent import conformance_data
from agents.cfu_design_agent.conformance import (
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


def data_for(command: dict, *, buffer: int = 0, max_value: int = 255) -> dict:
    """The input-space spec the harness generator needs for one command."""
    spec: dict = {
        "roles": [{"kind": "value", "max": max_value} for _ in cpu_value_roles(command)]
    }
    if buffer:
        spec["buffer"] = buffer
    return spec


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
        entry = {"name": command["name"], "data": data_for(command), **fields}
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


class HarnessDataTest(unittest.TestCase):
    """The harness generates the data; the oracle only declares its shape.

    Regression class (run 20261005-073745): the oracle declared one zero-length
    case, so the rung never transferred a source byte and a wrong scalar table
    could not be told apart from wrong hardware.
    """

    def plan(self, data: dict, roles: int = 2, name: str = "hash"):
        plan, problems = conformance_data.plan(name, data, roles)
        self.assertEqual(problems, [], problems)
        self.assertIsNotNone(plan)
        return plan

    def test_boundary_sweep_covers_zero_half_and_full_block(self) -> None:
        plan = self.plan({"buffer": 64,
                          "roles": [{"kind": "address"}, {"kind": "length", "max": 64}]})
        lengths = {case[1] for case in plan.cases}
        self.assertIn("0", lengths)
        self.assertIn("64", lengths)
        self.assertIn("32", lengths)
        self.assertGreater(len(plan.cases), 2)
        self.assertTrue(all(case[0] == plan.base_macro for case in plan.cases))

    def test_data_is_deterministic_and_not_model_chosen(self) -> None:
        spec = {"buffer": 64, "roles": [{"kind": "address"}, {"kind": "length", "max": 64}]}
        first = self.plan(spec)
        second = self.plan(spec)
        self.assertEqual(first.cases, second.cases)
        self.assertEqual(first.buffer, second.buffer)
        self.assertEqual(len(first.buffer), 64)
        self.assertNotEqual(first.buffer, bytes(64), "a zero block would hide transfer bugs")

    def test_random_draws_add_to_the_boundary_sweep(self) -> None:
        spec = {"count": 4, "roles": [{"kind": "value", "max": 0xFFFFFFFF}]}
        plan = self.plan(spec, roles=1)
        self.assertEqual(len(plan.cases), 4 + 4)  # four boundaries plus four draws
        self.assertIn(("4294967295",), plan.cases)
        self.assertIn(("0",), plan.cases)

    def test_buffer_renders_as_owned_block_with_a_fence(self) -> None:
        oracle = {"commands": [{
            "name": "hash",
            "reference": "ref_hash(a, b)",
            "helpers": "static uint32_t ref_hash(uint32_t p, uint32_t n) { return p + n; }",
            "data": {"buffer": 16, "roles": [{"kind": "address"}, {"kind": "length", "max": 16}]},
        }]}
        command = dict(CPU_COMMAND, name="hash")
        text = render_conformance(isa(command), oracle)
        self.assertIn("CFU_CONFORMANCE", text)
        self.assertIn("static uint8_t cfu_conf_data_hash[16]", text)
        self.assertIn("#define HASH_CONF_DATA_BASE", text)
        self.assertIn("#define HASH_CONF_DATA_BYTES ((uint32_t)16u)", text)
        # Harness-owned data means the harness also owns the ordering fence.
        self.assertIn("cfu_memory_fence();", text)
        # The rendered argument is the owned block, never a model literal address.
        self.assertIn("a = (uint32_t)(HASH_CONF_DATA_BASE);", text)

    def test_legacy_cases_field_is_rejected_with_the_migration(self) -> None:
        oracle = {"commands": [{"name": "compute", "cases": [{"args": [1, 2]}],
                                "reference": "a + b"}]}
        problems = validate_oracle(isa(CPU_COMMAND), oracle)
        message = next((p for p in problems if "cases is no longer accepted" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("data.buffer", message)
        self.assertIn("data.roles", message)

    def test_model_supplied_seed_is_rejected(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{
            "name": "compute", "reference": "a + b",
            "data": {"seed": 1, **data_for(CPU_COMMAND)}}]})
        self.assertTrue(any("seed is not allowed" in p for p in problems), problems)

    def test_legacy_edge_flag_is_rejected(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [
            {"name": "compute", "reference": "a + b", "data": data_for(CPU_COMMAND),
             "edge_cases": True}]})
        self.assertTrue(any("edge_cases is no longer accepted" in p for p in problems), problems)

    def test_role_count_must_match_the_command(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{
            "name": "compute", "reference": "a + b",
            "data": {"roles": [{"kind": "value"}]}}]})
        message = next((p for p in problems if "data.roles has 1 entry" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("rs1/rs2 order", message)

    def test_missing_data_block_names_the_replacement(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND),
                                   {"commands": [{"name": "compute", "reference": "a + b"}]})
        message = next((p for p in problems if "data must be a mapping" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("cases", message)

    def test_address_role_requires_a_buffer(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{
            "name": "compute", "reference": "a + b",
            "data": {"roles": [{"kind": "address"}, {"kind": "value"}]}}]})
        self.assertTrue(any("data.buffer is missing" in p for p in problems), problems)

    def test_length_bound_may_not_exceed_the_owned_block(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{
            "name": "compute", "reference": "a + b",
            "data": {"buffer": 64,
                     "roles": [{"kind": "address"}, {"kind": "length", "max": 128}]}}]})
        self.assertTrue(any("bounds the transferred length" in p for p in problems), problems)

    def test_buffer_size_and_count_are_bounded(self) -> None:
        cases = [
            ({"buffer": 6, "roles": [{"kind": "value"}, {"kind": "value"}]}, "multiple of 4"),
            ({"buffer": 4096, "roles": [{"kind": "value"}, {"kind": "value"}]}, "between 4 and 1024"),
            ({"count": 99, **data_for(CPU_COMMAND)}, "data.count must be an integer between"),
        ]
        for data, needle in cases:
            with self.subTest(data=data):
                problems = validate_oracle(isa(CPU_COMMAND), {"commands": [
                    {"name": "compute", "reference": "a + b", "data": data}]})
                self.assertTrue(any(needle in p for p in problems), problems)

    def test_unknown_role_kind_lists_the_vocabulary(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [{
            "name": "compute", "reference": "a + b",
            "data": {"roles": [{"kind": "banana"}, {"kind": "value"}]}}]})
        message = next((p for p in problems if "kind must be one of" in p), None)
        self.assertIsNotNone(message, problems)
        for kind in conformance_data.ROLE_KINDS:
            self.assertIn(kind, message)

    def test_role_less_command_gets_one_invocation(self) -> None:
        control = {"function_id": 3, "subfunction_id": 0, "name": "ctl",
                   "kind": "control", "resources": []}
        oracle = {"commands": [{
            "name": "ctl", "verify": "ctl_seen()",
            "helpers": "static uint32_t ctl_seen(void) { return 1u; }"}]}
        self.assertEqual(validate_oracle(isa(control), oracle), [])
        text = render_conformance(isa(control), oracle)
        self.assertEqual(text.count("g_cases++;"), 1)

    def test_byte_budget_keeps_the_full_block_and_drops_the_rest(self) -> None:
        """A big buffer must not turn the rung into a mini-benchmark."""
        plan = self.plan({"buffer": 1024, "count": 4,
                          "roles": [{"kind": "address"}, {"kind": "length", "max": 1024}]})
        self.assertLessEqual(plan.bytes_total, conformance_data.CASE_BYTE_BUDGET)
        lengths = [case[1] for case in plan.cases]
        self.assertIn("1024", lengths, "the full-block transfer is the case that matters")
        self.assertIn("0", lengths)
        self.assertNotIn("1023", lengths, "budget is spent before the near-full boundary")

    def test_small_buffer_keeps_the_whole_sweep(self) -> None:
        plan = self.plan({"buffer": 64,
                          "roles": [{"kind": "address"}, {"kind": "length", "max": 64}]})
        self.assertEqual(plan.bytes_total, 339)
        self.assertEqual(len(plan.cases), 10)

    def test_prompt_states_the_harness_owns_the_data(self) -> None:
        from agents.cfu_design_agent.prompts import conformance_prompt

        for tools in (True, False):
            with self.subTest(tools=tools):
                prompt = conformance_prompt("test", isa(CPU_COMMAND), {}, agent_tools=tools)
                self.assertIn("YOU DO NOT CHOOSE THE TEST DATA", prompt)
                self.assertIn("data.roles", prompt)
                self.assertIn("data.buffer", prompt)
                self.assertNotIn("cases:\n      - args:", prompt)


class ValidateOracleTest(unittest.TestCase):
    def entry(self, **fields) -> dict:
        return {"name": "compute", "data": data_for(CPU_COMMAND), **fields}

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
        entry = {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();"}
        problems = validate_oracle(isa(VECTOR_COMMAND), {"commands": [entry]})
        self.assertTrue(any("verify" in p for p in problems), problems)

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


class DataSpecGuidanceTest(unittest.TestCase):
    """A rejected data spec must explain the fix, not just the format."""

    def entry(self, **fields) -> dict:
        return {"name": "compute", **fields}

    def test_missing_roles_error_says_one_descriptor_per_role(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [
            self.entry(reference="a + b", data={"buffer": 16})]})
        message = next(p for p in problems if "data.roles has 0 entry" in p)
        self.assertIn("rs1/rs2 order", message)
        self.assertIn("omit data.roles when the command takes none", message)

    def test_index_role_requires_its_bound(self) -> None:
        problems = validate_oracle(isa(CPU_COMMAND), {"commands": [
            self.entry(reference="a + b", data={
                "roles": [{"kind": "index"}, {"kind": "value"}]})]})
        message = next(p for p in problems if "max is required" in p)
        self.assertIn("bounds the index", message)

    def test_prompt_documents_the_data_block_and_repair_discipline(self) -> None:
        from agents.cfu_design_agent.prompts import conformance_prompt

        prompt = conformance_prompt("task", isa(CPU_COMMAND), {}, feedback="compute: problems")
        self.assertIn("data:", prompt)
        self.assertIn("kind: address", prompt)
        self.assertIn("kind: length", prompt)
        self.assertIn("ORACLE REPAIR MODE", prompt)
        self.assertIn("at once", prompt)
        # The prompt names the generated intrinsics so setup/verify cannot
        # invent names like cfu_load (2026-09-26 link failure).
        self.assertIn("cfu_compute(", prompt)
        self.assertIn("undefined reference", prompt)


class UnknownIntrinsicLintTest(unittest.TestCase):
    """An invented cfu_* name must fail validation, not the linker."""

    def test_invented_intrinsic_is_rejected_before_the_linker(self) -> None:
        entry = {
            "name": "compute", "data": data_for(CPU_COMMAND), "reference": "a + b",
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
            "name": "vector_load_i32", "data": data_for(CPU_COMMAND), "reference": "a + b",
            "setup": "cfu_load(p, 0);",
        }
        problems = validate_oracle(isa(command), {"commands": [entry]})
        message = next(p for p in problems if "unknown intrinsic" in p)
        self.assertIn("did you mean cfu_vector_load_i32", message)

    def test_generated_names_and_helpers_are_allowed(self) -> None:
        entry = {
            "name": "compute", "data": data_for(CPU_COMMAND), "reference": "a + b",
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
        return {"name": "compute", "data": data_for(CPU_COMMAND), **fields}

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
        entry = {"name": "vset", "data": data_for(VECTOR_COMMAND),
                 "setup": "seed();", "verify": "1"}
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
            "name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
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
                entry = {"name": "vset", "data": data_for(VECTOR_COMMAND),
                         "setup": "seed();", "verify": expression}
                self.assert_message(self.problems(isa(VECTOR_COMMAND), entry),
                                    "compares an operand with itself")

    def test_argument_only_verify_is_rejected(self) -> None:
        entry = {"name": "vset", "data": data_for(VECTOR_COMMAND),
                 "setup": "seed();", "verify": "(a | b) == 0u"}
        self.assert_message(self.problems(isa(VECTOR_COMMAND), entry),
                            "verify only references the case arguments")

    def test_expression_comparing_its_own_value_is_rejected(self) -> None:
        """A value the oracle wrote next to the check cannot verify anything."""
        marker = {"name": "vset", "data": data_for(VECTOR_COMMAND),
                  "setup": "static volatile uint32_t marker = 1u; seed(); marker = 1u;",
                  "verify": "marker == 1u"}
        helper = {"name": "compute", "data": data_for(CPU_COMMAND),
                  "helpers": "static uint32_t expected;\n"
                             "static void set_expected(void) { expected = 4u; }",
                  "setup": "set_expected();", "reference": "expected"}
        for entry, contract in ((marker, isa(VECTOR_COMMAND)), (helper, isa(CPU_COMMAND))):
            with self.subTest(expression=entry.get("verify") or entry["reference"]):
                self.assert_message(self.problems(contract, entry), "assigned a literal to")

    def test_genuine_observations_still_validate(self) -> None:
        """No false positives: seed it in setup, read it back, and it is evidence."""
        control = {"function_id": 3, "subfunction_id": 0, "name": "ctl",
                   "kind": "control", "resources": []}
        cases = [
            # A no-role command whose verify reads memory a helper seeded.
            (isa(control), {"name": "ctl", "data": {"count": 1},
                            "helpers": "static void seed_memory(void) { volatile uint32_t *p = "
                                       "(volatile uint32_t *)0x80000000u; p[0] = 4u; }\n"
                                       "static uint32_t ctl_seen(void) { volatile uint32_t *p = "
                                       "(volatile uint32_t *)0x80000000u; return p[0] == 4u; }",
                            "setup": "seed_memory();", "verify": "ctl_seen()"}),
            (isa(CPU_COMMAND), {"name": "compute", "data": data_for(CPU_COMMAND),
                                "helpers": "static uint32_t w[8];",
                                "setup": "w[0] = 0x80000000u;", "reference": "w[0] >> 4"}),
            (isa(VECTOR_COMMAND), {"name": "vset", "data": data_for(VECTOR_COMMAND),
                                   "helpers": "static uint32_t check_loaded(void) { "
                                              "volatile uint32_t *p = (volatile uint32_t *)0x80000000u; "
                                              "return p[0] == 4u; }",
                                   "setup": "seed();", "verify": "check_loaded()"}),
            # `p = (T *)addr` publishes an address; reading through p is the observation.
            (isa(CPU_COMMAND), {"name": "compute", "data": data_for(CPU_COMMAND),
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
        return {"name": "compute", "data": data_for(CPU_COMMAND), **fields}

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
        entry = {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
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
                    {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
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
            {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
             "verify": "REF(a, b)"},
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
            "name": "vslot0", "data": data_for(CONST_INDEX_COMMAND, max_value=7),
            "setup": "seed();", "verify": "slot_at(b) == 0",
        }
        problems = validate_oracle(isa(CONST_INDEX_COMMAND), {"commands": [entry]})
        message = next((p for p in problems if "undeclared identifier" in p), None)
        self.assertIsNotNone(message, problems)
        self.assertIn("'b'", message)
        self.assertIn("resources", message)
        self.assertIn("rs1->a, rs2->b", message)

    def test_reference_referencing_undeclared_role_rejected(self) -> None:
        entry = {"name": "onesel", "data": data_for(ONE_ARG_COMMAND), "reference": "a + b"}
        problems = validate_oracle(isa(ONE_ARG_COMMAND), {"commands": [entry]})
        self.assertTrue(
            any("'b'" in p and "undeclared identifier" in p for p in problems), problems
        )

    def test_declared_roles_may_be_referenced(self) -> None:
        entry = {"name": "onesel", "data": data_for(ONE_ARG_COMMAND), "reference": "a * 2u + 1u"}
        self.assertEqual(validate_oracle(isa(ONE_ARG_COMMAND), {"commands": [entry]}), [])

    def test_case_scope_binds_a_and_b_unconditionally(self) -> None:
        # Render backstop for stale oracles: undeclared identifiers must never
        # fail the C build; the validator lint rejects them first with a fix.
        entry = {"name": "onesel", "data": data_for(ONE_ARG_COMMAND), "reference": "a"}
        text = render_conformance(isa(ONE_ARG_COMMAND), {"commands": [entry]})
        self.assertIn("uint32_t a = 0;", text)
        self.assertIn("uint32_t b = 0;", text)
        self.assertIn("(void)a; (void)b;", text)
        self.assertNotIn("b = (uint32_t)(", text)  # undeclared role stays 0


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
                    "name": "compute", "data": data_for(CPU_COMMAND),
                    "reference": "ref_rol32(a, b)", "helpers": self.HELPER,
                },
                {
                    "name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
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
                {"name": "compute", "data": data_for(CPU_COMMAND), "reference": "a + b",
                 "helpers": pointer},
                {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
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
                {"name": "compute", "data": data_for(CPU_COMMAND), "reference": "ref_a(a)",
                 "helpers": first},
                {"name": "vset", "data": data_for(VECTOR_COMMAND), "setup": "seed();",
                 "verify": "ref_b(a) != 0u", "helpers": second},
            ]
        }
        self.assertEqual(validate_oracle(isa(CPU_COMMAND, VECTOR_COMMAND), oracle), [])

    def test_distinct_helper_symbols_are_all_emitted(self) -> None:
        distinct = "static uint32_t other_h(uint32_t a) { return a; }"
        text = render_conformance(isa(CPU_COMMAND, VECTOR_COMMAND), self.oracle(distinct))
        self.assertIn("ref_rol32", text)
        self.assertIn("other_h", text)


if __name__ == "__main__":
    unittest.main()
