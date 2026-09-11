"""WorkloadSpec and CFU ISA spec validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.isa import load_isa_spec, validate_isa_text
from agents.cfu_design_agent.spec import load_formatted_spec, validate_workload_spec
from agents.tests.common import REPO_ROOT, repo_files

# Self-contained fixtures: the reference specs under agents/specs/ are examples
# that may be pruned, so the validation tests must not depend on them.
VALID_ISA_YAML = """spec_version: "1"
identity:
  name: u8_wavg_agent
  status: draft
  opcode: custom0 (0x0B)
  function_id_width: 3
  reference_workload: uint8 weighted average
commands:
  - function_id: 1
    name: compute
    operand: rs1, rs2
    result: sum
    semantics: multiply and accumulate uint8 lanes
    response: status
datapath:
  vlen: 256
  xlen: 32
  maclen: 32
  accWidth: 32
  regDepth: 2
software:
  intrinsics_header: cfu_intrinsics.h
  scalar_fallback: true
integration:
  hardware_component: AgentCfu
  hardware_fiber: AgentCfuFiber
  bus_style: CfuBus
  single_cpu_bus_owner: true
validation:
  status: draft
  requirements: [scalar reference equals CFU output]
assumptions: {}
unknowns: []"""

VALID_SPEC_YAML = """spec_version: "1"
name: u8_weighted_average
source_kind: formatted_spec
operation: weighted_average
pattern: reduction
element_type: uint8
vector_bytes: 32
not_a_real_field: ignored
"""


def _valid_isa() -> dict:
    spec, problems = validate_isa_text(VALID_ISA_YAML)
    assert not problems, problems
    return spec


class IsaValidationTest(unittest.TestCase):
    def test_reference_isa_is_valid(self) -> None:
        spec = _valid_isa()
        _again, problems = validate_isa_text(VALID_ISA_YAML)
        self.assertEqual(problems, [])
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")

    def test_template_is_incomplete(self) -> None:
        spec = load_isa_spec(repo_files("agents/specs/cfu_isa_template.yaml"))
        from agents.cfu_design_agent.isa import validate_isa_spec

        problems = validate_isa_spec(spec)
        self.assertTrue(problems)
        self.assertTrue(any("identity.name is required" in p for p in problems))
        self.assertTrue(any("software.intrinsics_header is required" in p for p in problems))

    def test_duplicate_function_id_rejected(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["commands"] = [dict(spec["commands"][0]), dict(spec["commands"][0])]
        problems = validate_isa_spec(spec)
        self.assertTrue(any("duplicates" in p for p in problems))

    def test_function_id_outside_func3_range_rejected(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["commands"][0]["function_id"] = 8
        problems = validate_isa_spec(spec)
        self.assertTrue(any("func3 range" in p for p in problems), problems)

    def test_function_id_must_be_an_integer(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["commands"][0]["function_id"] = "1"
        problems = validate_isa_spec(spec)
        self.assertTrue(any("must be an integer" in p for p in problems), problems)

    def test_function_id_must_fit_declared_width(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["identity"]["function_id_width"] = 2
        spec["commands"][0]["function_id"] = 5
        problems = validate_isa_spec(spec)
        self.assertTrue(any("declared" in p for p in problems), problems)

    def test_custom1_opcode_rejected_for_generated_intrinsics(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["identity"]["opcode"] = "custom1 (0x2B)"
        problems = validate_isa_spec(spec)
        self.assertTrue(any("custom1" in p for p in problems), problems)

    def test_unknown_opcode_rejected(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["identity"]["opcode"] = "custom9 (0x9B)"
        problems = validate_isa_spec(spec)
        self.assertTrue(any("identity.opcode" in p for p in problems), problems)

    def test_fenced_yaml_contract_is_accepted(self) -> None:
        spec, problems = validate_isa_text(f"```yaml\n{VALID_ISA_YAML}\n```")
        self.assertEqual(problems, [])
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")

    def test_json_contract_is_accepted(self) -> None:
        import json
        import yaml

        spec, problems = validate_isa_text(json.dumps(yaml.safe_load(VALID_ISA_YAML)))
        self.assertEqual(problems, [])
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")

    def test_datapath_dimensions_are_independent_and_optional(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["datapath"] = {
            "vlen": 255,
            "xlen": 32,
            "maclen": 0,
            "accWidth": None,
            "regDepth": 0,
        }
        self.assertEqual(validate_isa_spec(spec), [])

    def test_datapath_mapping_can_be_empty_for_scalar_cfu(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["datapath"] = {}
        self.assertEqual(validate_isa_spec(spec), [])


class WorkloadSpecValidationTest(unittest.TestCase):
    def test_formatted_spec_strips_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.spec"
            path.write_text(VALID_SPEC_YAML)
            spec = load_formatted_spec(path)
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
