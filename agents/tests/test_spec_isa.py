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
  implementation:
    result_width_bits: 32
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
            "implementation": {"result_width_bits": None},
            "regDepth": 0,
        }
        self.assertEqual(validate_isa_spec(spec), [])

    def test_legacy_algorithm_metadata_remains_readable(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec
        spec = _valid_isa()
        spec["datapath"] = {"maclen": 32, "accWidth": 64}
        self.assertEqual(validate_isa_spec(spec), [])

    def test_algorithm_specific_metadata_is_optional_and_namespaced(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec
        for parameters in ({}, {"crc_polynomial": "0x04c11db7"}, {"permutation_width_bits": 32}):
            spec = _valid_isa()
            spec["datapath"] = {"implementation": parameters}
            self.assertEqual(validate_isa_spec(spec), [])
        spec["datapath"]["implementation"] = 32
        message = next(m for m in validate_isa_spec(spec)
                       if m.startswith("datapath.implementation must be a mapping"))
        # The message names what was received, so a repair can act on it.
        self.assertIn("received a int", message)

    def test_shape_lint_names_what_was_received(self) -> None:
        """Run 20261003-130005: ten answer-repair rounds on one terse message.

        The model kept returning ``assumptions`` as a sequence of strings, and
        "assumptions must be a mapping" never said so.
        """
        from agents.cfu_design_agent.isa import validate_isa_spec
        spec = _valid_isa()
        spec["assumptions"] = ["one", "two"]
        message = next(m for m in validate_isa_spec(spec) if m.startswith("assumptions"))
        self.assertIn("a mapping of name: text", message)
        self.assertIn("received a list of 2 item(s)", message)
        spec["unknowns"] = {"not": "a list"}
        self.assertTrue(any(m.startswith("unknowns must be a list of strings")
                            and "received a mapping with 1 key(s)" in m
                            for m in validate_isa_spec(spec)))

    def test_datapath_mapping_can_be_empty_for_scalar_cfu(self) -> None:
        from agents.cfu_design_agent.isa import validate_isa_spec

        spec = _valid_isa()
        spec["datapath"] = {}
        self.assertEqual(validate_isa_spec(spec), [])


class IsaRendererRobustnessTest(unittest.TestCase):
    """Renderers also run on candidates that FAILED validation.

    The isa rung records every attempt and feeds the repair prompt, so a formatter
    crash would abort the run instead of reporting the lint problems.
    """

    MALFORMED = {
        # Regression: Assumptions was the one section that called .items() directly,
        # so a model returning a list crashed the whole run.
        "assumptions-list": {"assumptions": ["free-form"], "unknowns": []},
        "identity-list": {"identity": ["x"]},
        "commands-with-non-dict": {"commands": [None, 3, "s", {"function_id": 1, "name": "c"}]},
        "commands-not-list": {"commands": {"a": 1}},
        "unknowns-int": {"unknowns": 7},
        "datapath-scalar": {"datapath": 3, "roofline": "text"},
        "everything-wrong-type": {
            "identity": 5, "commands": 5, "roofline": "x", "state": [1], "memory": None,
            "datapath": 3, "software": [1], "integration": 2, "validation": "v",
            "assumptions": 9, "unknowns": 9, "operations_contract": [1, 2],
        },
        "empty": {},
    }

    def test_markdown_renders_rejected_candidate_without_raising(self) -> None:
        from agents.cfu_design_agent.isa import render_isa_markdown

        for name, data in self.MALFORMED.items():
            with self.subTest(case=name):
                text = render_isa_markdown(data)
                self.assertTrue(text.startswith("# CFU ISA Spec"), text[:80])
                for section in ("## Commands", "## Assumptions", "## Unknowns"):
                    self.assertIn(section, text)

    def test_markdown_renders_list_valued_assumptions(self) -> None:
        from agents.cfu_design_agent.isa import render_isa_markdown

        text = render_isa_markdown({"assumptions": ["free-form note"], "unknowns": ["u"]})
        self.assertIn("- free-form note", text)
        self.assertIn("- u", text)

    def test_markdown_escapes_table_separators(self) -> None:
        from agents.cfu_design_agent.isa import render_isa_markdown

        text = render_isa_markdown({"commands": [{"function_id": 1, "name": "a|b"}]})
        self.assertIn("a\\|b", text)

    def test_intrinsics_render_rejected_candidate_without_raising(self) -> None:
        from agents.cfu_design_agent.isa import render_cfu_intrinsics

        for name, data in self.MALFORMED.items():
            with self.subTest(case=name):
                self.assertIn("#ifndef CFU_INTRINSICS_H", render_cfu_intrinsics(data))


class IsaEnvelopeToleranceTest(unittest.TestCase):
    """A model may wrap the contract; accept the nested one instead of failing the run.

    Observed with GLM: {"analysis": ..., "roofline": {...}, "isa_contract": "<yaml>"}.
    Every top-level key then looked missing, so the rung burned its whole repair
    budget re-emitting the same envelope.
    """

    def test_nested_contract_string_is_unwrapped(self) -> None:
        import json

        import yaml

        response = json.dumps({
            "analysis": "hot loop analysis",
            "roofline": {"bottleneck": "memory-bound"},
            "isa_contract": yaml.safe_dump(yaml.safe_load(VALID_ISA_YAML)),
        })
        spec, problems = validate_isa_text(response)
        self.assertEqual(problems, [])
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")

    def test_nested_contract_mapping_is_unwrapped(self) -> None:
        import json

        import yaml

        response = json.dumps({
            "analysis": "x",
            "cfu_isa": yaml.safe_load(VALID_ISA_YAML),
        })
        spec, problems = validate_isa_text(response)
        self.assertEqual(problems, [])
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")

    def test_envelope_without_a_contract_is_still_rejected(self) -> None:
        import json

        response = json.dumps({"analysis": "x", "roofline": {"bottleneck": "unknown"}})
        spec, problems = validate_isa_text(response)
        self.assertTrue(problems)
        # The parsed envelope survives to the artifact so the report shows what came back.
        self.assertIn("analysis", spec)

    def test_flat_contract_is_not_rewritten(self) -> None:
        import json

        import yaml

        contract = yaml.safe_load(VALID_ISA_YAML)
        contract["isa_contract"] = "decoy"
        spec, problems = validate_isa_text(json.dumps(contract))
        self.assertEqual(problems, [])
        self.assertEqual(spec["isa_contract"], "decoy")
        self.assertEqual(spec["identity"]["name"], "u8_wavg_agent")


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
