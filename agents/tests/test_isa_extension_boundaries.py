"""Append-only ABI boundaries; fixtures are pure memory, never run artifacts."""
from copy import deepcopy
import unittest

from agents.cfu_design_agent.isa import (
    render_cfu_intrinsics, validate_isa_extension, validate_isa_spec,
)
from agents.cfu_design_agent.runtime import commands


def minimal_spec():
    return {
        "spec_version": "1",
        "identity": {"name": "example", "opcode": "custom0", "function_id_width": 3},
        "commands": [{
            "name": "mulmod", "function_id": 1, "operand": "rs1, rs2",
            "result": "rd", "semantics": "exact modular product", "response": "value",
            "resources": [
                {"storage": "cpu", "access": "read", "role": "rs1"},
                {"storage": "cpu", "access": "read", "role": "rs2"},
                {"storage": "cpu", "access": "write", "role": "rd"},
            ],
        }],
        "state": {"reset_behavior": "clear", "register_slots": 2},
        "memory": {"load": False, "store": False, "ordering": "in order"},
        "datapath": {"xlen": 32, "vlen": 256, "regDepth": 2, "scalarRegDepth": 4},
        "software": {
            "intrinsics_header": "cfu_intrinsics.h", "intrinsic_names": ["cfu_mulmod"],
            "scalar_fallback": True, "compiler_march": "rv32imc", "soc_flags": [],
        },
        "integration": {
            "hardware_component": "AgentCfu", "hardware_fiber": "AgentCfuFiber",
            "soc_params": [], "soc_cli_flags": [], "bus_style": "TilelinkCfuSpec",
            "single_cpu_bus_owner": True,
        },
        "operations_contract": {"input_words": "low word first"},
        "assumptions": {"input_domain": "every nonzero unsigned modulus"},
    }


def append_command(previous):
    candidate = deepcopy(previous)
    candidate["commands"].append({
        "name": "mulmod4", "function_id": 3, "subfunction_id": 0,
        "semantics": "fourth power of modular product", "response": "value",
    })
    candidate["software"]["intrinsic_names"].append("cfu_mulmod4")
    return candidate


class IsaExtensionBoundariesTest(unittest.TestCase):
    def assertFrozen(self, previous, candidate, path, category="cannot change"):
        self.assertEqual(validate_isa_spec(previous), [])
        self.assertEqual(validate_isa_spec(candidate), [])
        before, after = deepcopy(previous), deepcopy(candidate)
        errors = validate_isa_extension(previous, candidate)
        matching = [error for error in errors if path in error and category in error]
        self.assertTrue(matching, errors)
        self.assertTrue(all("before=" in error and "after=" in error for error in matching))
        self.assertEqual(previous, before)
        self.assertEqual(candidate, after)
        return errors

    def test_attempt_007_extension_kernel_plan_only_rejection_regression(self):
        # Minimal shape of the latest attempt_007: valid appended commands/names,
        # unchanged shared ABI, and a changed multiline software plan. Historically
        # its sole error was "software.extension_kernel_plan cannot change".
        previous = minimal_spec()
        previous["software"]["kernel_plan"] = "Keep the scalar branch and use the existing intrinsics."
        previous["software"]["extension_kernel_plan"] = "Use the earlier fused chain."
        candidate = append_command(previous)
        candidate["software"]["extension_kernel_plan"] = (
            "Append-only extension (function_id 3); fuse the algorithm's own chains.\n"
            "  cfu_enable() once before the loop nest;\n"
            "  lo = cfu_mulmod4(a); hi = cfu_gcd_uhi();\n"
            "Keep scalar checks and the original #else branch unchanged.\n"
        )
        before, after = deepcopy(previous), deepcopy(candidate)
        self.assertEqual(validate_isa_spec(candidate), [])
        self.assertEqual(validate_isa_extension(previous, candidate), [])
        self.assertEqual(previous, before)
        self.assertEqual(candidate, after)

    def test_known_advisory_fields_can_be_added_changed_or_removed(self):
        fields = (
            ("software", "kernel_plan", "call existing instructions"),
            ("software", "extension_kernel_plan", "call new instructions"),
            ("software", "notes", "explanation"),
            ("software", "rationale", "why this schedule"),
            ("datapath", "implementation", {"pipeline": 2, "rationale": "area"}),
            ("integration", "notes", "the overlay stays isolated"),
            ("integration", "rationale", "reuse fixed wiring"),
        )
        for section, field, value in fields:
            with self.subTest(section=section, field=field):
                previous = minimal_spec()
                added = append_command(previous)
                added[section][field] = value
                self.assertEqual(validate_isa_extension(previous, added), [])
                changed = deepcopy(added)
                changed[section][field] = {"pipeline": 4} if field == "implementation" else "revised explanation"
                self.assertEqual(validate_isa_extension(added, changed), [])
                removed = deepcopy(changed)
                del removed[section][field]
                self.assertEqual(validate_isa_extension(changed, removed), [])

    def test_advisory_changes_do_not_change_command_flags_or_intrinsics(self):
        previous = minimal_spec()
        candidate = deepcopy(previous)
        candidate["datapath"]["implementation"] = {
            "xlen": 128, "regDepth": 999, "pipeline": 4, "multiplier_count": 2,
        }
        candidate["software"].update(kernel_plan="unroll", notes="advisory")
        candidate["integration"]["notes"] = "no wiring change"
        self.assertEqual(validate_isa_extension(previous, candidate), [])
        self.assertEqual(commands.agent_cfu_datapath_flags(previous), commands.agent_cfu_datapath_flags(candidate))
        self.assertEqual(render_cfu_intrinsics(previous), render_cfu_intrinsics(candidate))

    def test_old_command_fields_including_unknown_metadata_remain_exact(self):
        for field, value in (
            ("name", "renamed"), ("operand", "rs2"), ("result", "no CPU result"),
            ("response", "ack"), ("kind", "control"), ("semantics", "approximate"),
            ("new_semantic_metadata", {"carry": False}), ("notes", "new meaning"),
            ("rationale", "new meaning"), ("implementation", {"pipeline": 2}),
        ):
            with self.subTest(field=field):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate["commands"][0][field] = value
                self.assertFrozen(previous, candidate, f"commands[0].{field}", "old command")

    def test_old_command_removal_or_reencoding_has_path_and_values(self):
        previous = minimal_spec()
        for field, value in (("function_id", 7), ("subfunction_id", 1)):
            candidate = append_command(previous)
            candidate["commands"][0][field] = value
            self.assertFrozen(previous, candidate, f"commands[0].{field}", "old command")
        candidate = append_command(previous)
        del candidate["commands"][0]
        self.assertFrozen(previous, candidate, "commands[0]", "was removed or re-encoded")

    def test_resource_leaf_feedback_uses_candidate_index_after_reordering(self):
        previous = minimal_spec()
        candidate = append_command(previous)
        candidate["commands"].reverse()
        candidate["commands"][1]["resources"].pop(1)
        errors = self.assertFrozen(previous, candidate, "commands[1].resources[1].access", "old command")
        self.assertTrue(any("before='read'; after='write'" in error for error in errors), errors)

    def test_resource_index_and_resource_presence_are_frozen(self):
        previous = minimal_spec()
        previous["commands"][0]["resources"] = [{
            "storage": "vector", "access": "read", "role": "vs1",
            "index": {"source": "cpu_value", "register": "rs1"},
        }]
        candidate = append_command(previous)
        candidate["commands"][0]["resources"][0]["index"]["register"] = "rs2"
        self.assertFrozen(previous, candidate, "commands[0].resources[0].index.register", "old command")
        candidate = append_command(previous)
        del candidate["commands"][0]["resources"]
        self.assertFrozen(previous, candidate, "commands[0].resources", "old command")

    def test_shared_state_reset_memory_domain_and_operations_stay_frozen(self):
        for section, field, value in (
            ("state", "reset_behavior", "retain"), ("state", "register_slots", 4),
            ("state", "notes", "a new state guarantee"),
            ("memory", "load", True), ("memory", "store", True),
            ("memory", "ordering", "unordered"), ("memory", "reset_behavior", "retain"),
            ("assumptions", "input_domain", "only small moduli"),
            ("operations_contract", "input_words", "high word first"),
        ):
            with self.subTest(section=section, field=field):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate[section][field] = value
                self.assertFrozen(previous, candidate, f"{section}.{field}")

    def test_datapath_top_level_parameters_and_unknowns_stay_frozen(self):
        for field, value in (("xlen", 64), ("vlen", 128), ("regDepth", 4),
                             ("scalarRegDepth", 8), ("accWidth", 64), ("notes", "unclassified")):
            with self.subTest(field=field):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate["datapath"]["implementation"] = {field: value}
                candidate["datapath"][field] = value
                self.assertFrozen(previous, candidate, f"datapath.{field}")

    def test_integration_wiring_and_unknown_fields_stay_frozen(self):
        for field, value in (("hardware_component", "OtherCfu"), ("hardware_fiber", "OtherFiber"),
                             ("soc_params", ["other"]), ("soc_cli_flags", ["--other"]),
                             ("bus_style", "OtherBus"), ("single_cpu_bus_owner", False),
                             ("future_wiring", {"notes": "different wire"})):
            with self.subTest(field=field):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate["integration"].update(notes="harmless", rationale="harmless")
                candidate["integration"][field] = value
                self.assertFrozen(previous, candidate, f"integration.{field}")

    def test_software_abi_and_unknown_fields_stay_frozen(self):
        for field, value in (("intrinsics_header", "other.h"), ("scalar_fallback", "required scalar implementation"),
                             ("compiler_march", "rv64im"), ("soc_flags", ["--other"]),
                             ("future_abi", {"notes": "unknown ABI"}), ("metadata", {"rationale": "unknown"})):
            with self.subTest(field=field):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate["software"]["extension_kernel_plan"] = "allowed independently"
                candidate["software"][field] = value
                self.assertFrozen(previous, candidate, f"software.{field}")

    def test_unknown_fields_cannot_be_added_changed_removed_or_null(self):
        for section in (None, "identity", "software", "datapath", "integration"):
            with self.subTest(section=section):
                previous = minimal_spec()
                candidate = append_command(previous)
                target = candidate if section is None else candidate[section]
                target["future_abi"] = None
                path = "future_abi" if section is None else f"{section}.future_abi"
                self.assertFrozen(previous, candidate, path)
                with_old = deepcopy(previous)
                old_target = with_old if section is None else with_old[section]
                old_target["future_abi"] = {"notes": "old", "rationale": "old"}
                candidate = append_command(with_old)
                target = candidate if section is None else candidate[section]
                target["future_abi"]["notes"] = "new"
                self.assertFrozen(with_old, candidate, path + ".notes")
                del target["future_abi"]
                self.assertFrozen(with_old, candidate, path)

    def test_freeze_does_not_confuse_boolean_and_integer(self):
        previous = minimal_spec()
        candidate = append_command(previous)
        candidate["integration"]["single_cpu_bus_owner"] = 1
        self.assertFrozen(previous, candidate, "integration.single_cpu_bus_owner")

    def test_legacy_defaults_and_intrinsic_growth_still_work(self):
        previous = minimal_spec()
        candidate = append_command(previous)
        candidate["identity"].update(status="approved", opcode="custom0 (0x0B)", subfunction_id_width=7)
        candidate["commands"][0].update(kind="compute", subfunction_id=0)
        candidate["validation"] = {"status": "draft"}
        candidate["roofline"] = {"bottleneck": "updated estimate"}
        candidate["unknowns"] = ["unmeasured"]
        self.assertEqual(validate_isa_extension(previous, candidate), [])
        candidate["software"]["intrinsic_names"] = ["cfu_mulmod4"]
        self.assertFrozen(previous, candidate, "software.intrinsic_names", "must preserve all old entries")

    def test_original_empty_shared_section_defaults_are_preserved(self):
        previous = minimal_spec()
        candidate = append_command(previous)
        for key in ("state", "memory", "datapath", "operations_contract", "assumptions"):
            previous.pop(key)
            candidate[key] = {}
        candidate["datapath"]["implementation"] = {"schedule": "sequential"}
        self.assertEqual(validate_isa_extension(previous, candidate), [])

    def test_feedback_is_leaf_precise_bounded_and_sorted(self):
        previous = minimal_spec()
        previous["software"]["unknown_abi"] = {"word_order": "low " + "a" * 5000}
        candidate = append_command(previous)
        candidate["software"]["unknown_abi"]["word_order"] = "high " + "b" * 5000
        candidate["state"]["reset_behavior"] = "retain"
        errors = self.assertFrozen(previous, candidate, "software.unknown_abi.word_order")
        self.assertEqual(errors, sorted(errors))
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(len(error) < 650 for error in errors), errors)
        self.assertTrue(any("before='low " in error and "after='high " in error for error in errors), errors)

    def test_advisory_exceptions_do_not_weaken_initial_contract_lint(self):
        for invalid, expected in (("implementation", "datapath.implementation must be a mapping"),
                                  ("duplicate", "encoding duplicates"),
                                  ("domain", "assumptions must be a mapping"),
                                  ("fallback", "software.scalar_fallback is required"),
                                  ("opcode", "custom1")):
            with self.subTest(invalid=invalid):
                previous = minimal_spec()
                candidate = append_command(previous)
                candidate["software"]["extension_kernel_plan"] = "allowed prose"
                if invalid == "implementation":
                    candidate["datapath"]["implementation"] = "not a mapping"
                elif invalid == "duplicate":
                    candidate["commands"].append(deepcopy(candidate["commands"][0]))
                elif invalid == "domain":
                    candidate["assumptions"] = ["narrowed"]
                elif invalid == "fallback":
                    candidate["software"]["scalar_fallback"] = False
                else:
                    candidate["identity"]["opcode"] = "custom1"
                errors = validate_isa_extension(previous, candidate)
                self.assertTrue(any(error.startswith("candidate: ") and expected in error for error in errors), errors)
        previous = minimal_spec()
        previous["datapath"]["implementation"] = []
        self.assertTrue(any(error.startswith("previous: ") for error in validate_isa_extension(previous, minimal_spec())))


if __name__ == "__main__":
    unittest.main()
