"""Roofline requirements are present in every CFU design prompt."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.graph import _isa_from_template, deterministic_roofline_analysis
from agents.cfu_design_agent.isa import render_isa_markdown
from agents.cfu_design_agent.prompts import (
    contract_prompt,
    embench_instructions,
    hardware_prompt,
    repair_routing_prompt,
    software_need_prompt,
    software_prompt,
    workload_prompt,
)


class RooflinePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {
            "name": "test_workload",
            "operation": "dot_product",
            "pattern": "vector_to_scalar",
            "unknowns": [],
        }
        self.context = {
            "files": {
                "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala": "// scaffold",
                "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala": "// fiber",
            }
        }
        self.analysis = """roofline:
  memory_boundary: CPU-to-CFU
  arithmetic_intensity: {value: 1.0, unit: ops/byte, kind: algorithmic}
  bottleneck: memory-bound
  design_action: keep vectors resident
"""

    def test_workload_prompt_requires_structured_roofline(self) -> None:
        prompt = workload_prompt("accelerate it", self.spec, self.context)
        for field in (
            "hot_function",
            "hot_loop",
            "trip_count",
            "arithmetic_intensity",
            "compute_peak",
            "bandwidth_peak",
            "ridge_point",
            "predicted_ceiling",
            "bottleneck",
            "verification_metric",
        ):
            self.assertIn(field, prompt)
        self.assertIn("Return exactly one YAML mapping", prompt)
        self.assertIn("algorithmic", prompt)
        contract = contract_prompt("accelerate it", self.spec, self.analysis, self.context)
        self.assertIn("independent optional knobs", contract)
        self.assertIn("regDepth: 0", contract)
        self.assertIn("achieved", prompt)

    def test_fallback_keeps_roofline_values_conservative(self) -> None:
        roofline = deterministic_roofline_analysis({"operation": "dot_product"})
        self.assertEqual(roofline["arithmetic_intensity"]["value"], "unknown")
        self.assertEqual(roofline["compute_peak"]["value"], "unknown")
        self.assertEqual(roofline["bandwidth_peak"]["value"], "unknown")
        self.assertEqual(roofline["bottleneck"], "unknown")
        self.assertTrue(roofline["unknowns"])

        isa = _isa_from_template("", self.spec, "")
        self.assertEqual(isa["datapath"]["regDepth"], 0)
        self.assertEqual(isa["datapath"]["maclen"], 0)
        self.assertIn("roofline", isa)
        self.assertIn("unknown", render_isa_markdown(isa).lower())

    def test_downstream_prompts_preserve_roofline_decision(self) -> None:
        prompts = (
            contract_prompt("accelerate it", self.spec, self.analysis, self.context),
            hardware_prompt("accelerate it", self.spec, self.analysis, "roofline: {}", self.context),
            software_prompt(
                "accelerate it",
                self.spec,
                self.analysis,
                "roofline: {}",
                self.context,
                targets=[("agents/generated/runs/test/main.c", "int main(void) { return 0; }")],
            ),
        )
        for prompt in prompts:
            self.assertIn("Roofline", prompt)
            self.assertIn("memory-bound", prompt)
        self.assertIn("roofline`", prompts[0])
        self.assertIn("external bytes/cycle", prompts[2])
        self.assertIn("vector width", prompts[1])
        self.assertIn("packing", prompts[2])


class RepairPromptTest(unittest.TestCase):
    """Repair prompts must ground the model in the exact tool error log."""

    def setUp(self) -> None:
        self.spec = {
            "name": "test_workload",
            "operation": "dot_product",
            "pattern": "vector_to_scalar",
            "unknowns": [],
        }
        self.context = {
            "files": {
                "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala": "// scaffold",
                "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala": "// fiber",
            }
        }
        self.error_log = "AgentCfu.scala:203:30: type mismatch; found: Boolean, required: spinal.core.Bool"

    def test_hardware_repair_prompt_requires_diagnosis(self) -> None:
        prompt = hardware_prompt(
            "accelerate it",
            self.spec,
            "analysis",
            "contract",
            self.context,
            feedback=f"### hw_build (SoC generation) errors\n{self.error_log}",
        )
        self.assertIn("HARDWARE REPAIR MODE", prompt)
        self.assertIn("REPAIR_DIAGNOSIS", prompt)
        self.assertIn("Observed errors", prompt)
        self.assertIn("Repair mapping", prompt)
        self.assertIn(self.error_log, prompt)

    def test_hardware_first_attempt_has_no_repair_mode(self) -> None:
        prompt = hardware_prompt("accelerate it", self.spec, "analysis", "contract", self.context)
        self.assertNotIn("HARDWARE REPAIR MODE", prompt)
        self.assertIn("first hardware attempt", prompt)

    def test_software_repair_prompt_requires_diagnosis(self) -> None:
        prompt = software_prompt(
            "accelerate it",
            self.spec,
            "analysis",
            "contract",
            self.context,
            targets=[("agents/generated/runs/test/main.c", "int main(void) { return 0; }")],
            feedback=f"### sw_build errors\n{self.error_log}",
        )
        self.assertIn("SOFTWARE REPAIR MODE", prompt)
        self.assertIn("REPAIR_DIAGNOSIS", prompt)
        self.assertIn(self.error_log, prompt)

    def test_isa_repair_prompt_carries_linter_problems(self) -> None:
        prompt = contract_prompt(
            "accelerate it",
            self.spec,
            "analysis",
            self.context,
            feedback="- datapath must be a mapping when provided",
        )
        self.assertIn("ISA REPAIR MODE", prompt)
        self.assertIn("datapath must be a mapping when provided", prompt)

    def test_isa_first_attempt_has_no_repair_mode(self) -> None:
        prompt = contract_prompt("accelerate it", self.spec, "analysis", self.context)
        self.assertNotIn("ISA REPAIR MODE", prompt)


class EditProtocolPromptTest(unittest.TestCase):
    """Targeted edits are the preferred delivery protocol, not full rewrites."""

    def _prompts(self) -> tuple[str, str]:
        spec = {"name": "w", "operation": "dot_product", "unknowns": []}
        context = {
            "files": {
                "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala": "// scaffold",
                "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala": "// fiber",
            }
        }
        return (
            hardware_prompt("t", spec, "a", "c", context),
            software_prompt(
                "t",
                spec,
                "a",
                "c",
                context,
                targets=[("agents/generated/runs/test/main.c", "int main(void) { return 0; }")],
            ),
        )

    def test_edit_protocol_is_documented_and_preferred(self) -> None:
        for prompt in self._prompts():
            self.assertIn("EDIT_BEGIN", prompt)
            self.assertIn("<<<<<<< SEARCH", prompt)
            self.assertIn(">>>>>>> REPLACE", prompt)
            self.assertIn("Protocol A (preferred): targeted edits", prompt)
            # Edit blocks must be offered before the whole-file fallback.
            self.assertLess(
                prompt.index("targeted edits"), prompt.index("whole-file replacement")
            )
            # The old behavior forced full rewrites; the fallback must be scoped.
            self.assertIn("only for a new file or a large restructure", prompt)


class SearchBlockGuidanceTest(unittest.TestCase):
    """A large, comment-heavy SEARCH block must be discouraged explicitly."""

    def _context(self, target: str) -> dict:
        return {"files": {target: "LIVE_BYTES", "other.c": "KEEP"}, "visibility": "standard"}

    def test_context_digest_omits_the_rungs_own_targets(self) -> None:
        from agents.cfu_design_agent.prompts import context_digest

        target = "agents/generated/runs/x/workspace/design/AgentCfu.scala"
        digest = context_digest(self._context(target), skip=[target])
        self.assertNotIn("LIVE_BYTES", digest)
        self.assertIn("KEEP", digest)
        self.assertIn("shown in full under", digest)

    def test_target_file_is_rendered_exactly_once(self) -> None:
        target = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        prompt = hardware_prompt("t", {}, "", "", self._context(target))
        self.assertEqual(prompt.count("LIVE_BYTES"), 1)

        benchmark = "agents/generated/runs/x/workspace/embench_b/cfu/src/b.c"
        prompt = software_prompt(
            "t", {}, "", "", self._context(benchmark), targets=[(benchmark, "LIVE_BYTES")]
        )
        self.assertEqual(prompt.count("LIVE_BYTES"), 1)

    def test_delivery_protocols_warn_about_large_comment_search_blocks(self) -> None:
        from agents.cfu_design_agent.prompts import DELIVERY_PROTOCOLS

        self.assertIn("Do NOT wrap large comment blocks", DELIVERY_PROTOCOLS)
        self.assertIn("copy it\nbyte-for-byte", DELIVERY_PROTOCOLS)
        self.assertIn("do not resend the same block", DELIVERY_PROTOCOLS)


class EmbenchSimLogGuidanceTest(unittest.TestCase):
    def test_embench_instructions_explain_the_sim_log(self) -> None:
        text = embench_instructions({"source_kind": "embench", "benchmark_name": "depthconv"})
        self.assertIn("fail symbol", text)
        self.assertIn("EMBENCH_VERIFY_FAIL", text)
        self.assertIn("correctness bug", text)
        self.assertIn("hang", text)

    def test_non_embench_workloads_get_no_embench_guidance(self) -> None:
        self.assertEqual(embench_instructions({"source_kind": "natural_language"}), "")


class RouterPromptTest(unittest.TestCase):
    """Both routers must state what each answer actually changes."""

    ISA = {
        "identity": {"name": "depthconv_cfu"},
        "commands": [{"function_id": 1, "name": "compute"}],
    }

    def test_repair_routing_explains_the_consequence_of_each_target(self) -> None:
        text = repair_routing_prompt(
            "accelerate it", "verify failed", "log", {"name": "depthconv"}, isa_spec=self.ISA
        )
        # Software only: the hardware is not rebuilt.
        self.assertIn("hardware is\n  left untouched and only the software changes", text)
        # Hardware: the software is re-checked, so both sides may change.
        self.assertIn("software is re-checked against the new hardware", text)
        self.assertIn("updated as well when the new hardware needs it", text)

    def test_software_need_prompt_explains_both_answers(self) -> None:
        text = software_need_prompt(
            "accelerate it",
            {"name": "depthconv"},
            self.ISA,
            hardware_summary="hw",
            software_summary="sw",
            calls=[{"file": "a.c", "line": 3, "name": "cfu_compute", "text": "cfu_compute(1,2);"}],
            evidence="the current kernel already calls cfu_compute",
        )
        self.assertIn('"software_change": "required" or "not_required"', text)
        self.assertIn("a.c:3", text)
        self.assertIn("only risk breaking working software", text)


if __name__ == "__main__":
    unittest.main()
