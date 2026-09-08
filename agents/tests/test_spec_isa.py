"""WorkloadSpec and CFU ISA spec validation."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.isa import load_isa_spec, validate_isa_spec
from agents.cfu_design_agent.spec import load_formatted_spec, validate_workload_spec
from agents.tests.common import REPO_ROOT, repo_files


class IsaValidationTest(unittest.TestCase):
    def test_reference_u8_isa_is_valid(self) -> None:
        spec = load_isa_spec(repo_files("agents/specs/u8_weighted_average_isa.yaml"))
        self.assertEqual(validate_isa_spec(spec), [])

    def test_template_is_incomplete(self) -> None:
        spec = load_isa_spec(repo_files("agents/specs/cfu_isa_template.yaml"))
        problems = validate_isa_spec(spec)
        self.assertTrue(problems)
        self.assertTrue(any("identity.name is required" in p for p in problems))
        self.assertTrue(any("software.intrinsics_header is required" in p for p in problems))

    def test_duplicate_function_id_rejected(self) -> None:
        spec = load_isa_spec(repo_files("agents/specs/u8_weighted_average_isa.yaml"))
        spec["commands"] = [dict(spec["commands"][0]), dict(spec["commands"][0])]
        problems = validate_isa_spec(spec)
        self.assertTrue(any("duplicates" in p for p in problems))

    def test_vlen_multiple_of_xlen_enforced(self) -> None:
        spec = load_isa_spec(repo_files("agents/specs/u8_weighted_average_isa.yaml"))
        spec["datapath"]["vlen"] = 255  # not a multiple of xlen=32
        problems = validate_isa_spec(spec)
        self.assertTrue(any("datapath.vlen must be a multiple of datapath.xlen" in p for p in problems))


class WorkloadSpecValidationTest(unittest.TestCase):
    def test_formatted_spec_strips_unknown_keys(self) -> None:
        spec = load_formatted_spec(repo_files("agents/specs/u8_weighted_average.spec"))
        self.assertNotIn("not_a_real_field", spec)
        self.assertEqual(spec["operation"], "weighted_average")

    def test_valid_natural(self) -> None:
        from agents.cfu_design_agent.spec import normalize_workload_spec

        spec, _ = normalize_workload_spec(
            task="Accelerate uint8 weighted average x*w / n",
            input_mode="natural_language",
            context={},
            workdir=REPO_ROOT,
            llm=None,
        )
        self.assertEqual(validate_workload_spec(spec), [])


if __name__ == "__main__":
    unittest.main()
