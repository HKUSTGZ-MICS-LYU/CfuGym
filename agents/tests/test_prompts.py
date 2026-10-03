"""Roofline requirements are present in every CFU design prompt."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.stages.contract import _isa_from_template
from agents.cfu_design_agent.stages.fallbacks import deterministic_roofline_analysis
from agents.cfu_design_agent.isa import render_isa_markdown
from agents.cfu_design_agent.prompts import (
    HARDWARE_TARGETS,
    conformance_prompt,
    contract_prompt,
    embench_instructions,
    hardware_prompt,
    optimization_prompt,
    paths_digest,
    software_prompt,
    workload_digest_block,
    workload_prompt,
)
from agents.cfu_design_agent.stages import contract as contract_mod
from agents.cfu_design_agent.stages import fallbacks


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

    def test_analysis_prompt_states_the_raw_yaml_wire_format(self) -> None:
        # Observed 2026-09-29 (twice): a prose preamble plus a ```yaml fence made
        # the whole analysis unparseable. The prompt must say the reply is a raw
        # YAML mapping and nothing else.
        prompt = workload_prompt("accelerate it", self.spec, self.context)
        self.assertIn("and nothing else", prompt)
        self.assertIn("Markdown fences", prompt)
        self.assertIn("YAML parser", prompt)
        self.assertIn("not valid YAML", prompt)

    def test_analysis_repair_prompt_quotes_the_previous_rejection(self) -> None:
        prompt = workload_prompt(
            "accelerate it", self.spec, self.context,
            feedback="\n## Previous attempt rejected\n- not a mapping\n",
        )
        self.assertIn("## Previous attempt rejected", prompt)
        self.assertIn("not a mapping", prompt)
        self.assertNotIn(
            "Previous attempt rejected",
            workload_prompt("accelerate it", self.spec, self.context),
        )

    def test_opt_in_approximate_ir_prompt_is_bounded_and_not_measured(self) -> None:
        evidence = {
            "mode": "approx", "status": "approximate", "reason": "reduced ISA",
            "toolchain": {"target_match": False},
            "ir_files": ["agents/generated/runs/r1/07_analysis/ir/kernel.ll"],
            "units": [{"large_ir": "@" * 100_000}],
        }
        prompt = workload_prompt("accelerate it", self.spec, self.context, ir_evidence=evidence)
        self.assertIn("target_match: False", prompt)
        self.assertIn("do not infer loop counts, cycles", prompt)
        self.assertIn("kernel.ll", prompt)
        self.assertNotIn("large_ir", prompt)
        self.assertNotIn("@" * 100, prompt)
        default = workload_prompt("accelerate it", self.spec, self.context)
        self.assertNotIn("LLVM IR export", default)

    def test_observation_tool_prompt_names_only_measured_hotspots(self) -> None:
        index = {"observations": {
            "obs_loop_cold": {"cycles": 12, "calls": 1, "kind": "loop"},
            "obs_loop_hot": {"cycles": 1200, "calls": 6, "kind": "loop"},
            "obs_fn_context": {"cycles": 1500, "calls": 1, "kind": "function"},
            "obs_loop_unmeasured": {"cycles": None, "calls": None, "kind": "loop"},
        }}
        prompt = workload_prompt("accelerate", self.spec, self.context,
                                 agent_tools=True, observation_index=index)
        self.assertIn("Call inspect_observation", prompt)
        self.assertIn("top measured loop", prompt)
        self.assertLess(prompt.index("obs_loop_hot"), prompt.index("obs_loop_cold"))
        self.assertNotIn("obs_loop_unmeasured", prompt)
        self.assertNotIn("obs_fn_context", prompt)
        self.assertNotIn("inspect_observation", workload_prompt(
            "accelerate", self.spec, self.context,
            agent_tools=False, observation_index=index,
        ))

    def test_fallback_keeps_roofline_values_conservative(self) -> None:
        roofline = deterministic_roofline_analysis({"operation": "dot_product"})
        self.assertEqual(roofline["arithmetic_intensity"]["value"], "unknown")
        self.assertEqual(roofline["compute_peak"]["value"], "unknown")
        self.assertEqual(roofline["bandwidth_peak"]["value"], "unknown")
        self.assertEqual(roofline["bottleneck"], "unknown")
        self.assertTrue(roofline["unknowns"])

        isa = _isa_from_template("", self.spec, "")
        self.assertEqual(isa["datapath"]["regDepth"], 0)
        self.assertNotIn("maclen", isa["datapath"])
        self.assertNotIn("accWidth", isa["datapath"])
        self.assertEqual(isa["datapath"]["implementation"], {})
        self.assertIn("roofline", isa)
        import yaml
        from agents.tests.common import REPO_ROOT
        template = yaml.safe_load((REPO_ROOT / "agents/specs/cfu_isa_template.yaml").read_text())
        for field in ("maclen", "accWidth", "computePipe", "rfRam", "rfSync"):
            self.assertNotIn(field, template["datapath"])
        self.assertEqual(template["datapath"]["implementation"], {})
        instruction_prompt = contract_prompt("crc32", self.spec, self.analysis, self.context)
        self.assertNotIn("maclen", instruction_prompt)
        self.assertNotIn("accWidth", instruction_prompt)
        self.assertIn("datapath.implementation", instruction_prompt)
        self.assertIn("unknown", render_isa_markdown(isa).lower())

    def test_resource_contract_is_explicit_for_new_commands(self) -> None:
        prompt = contract_prompt("accelerate it", self.spec, self.analysis, self.context)
        for term in ("CfuResource", "cpu_value", "legacy_register_number", "resources",
                     "CPU rd OR vector vd", "completion response", "void intrinsic"):
            self.assertIn(term, prompt)
        hw = hardware_prompt("accelerate it", self.spec, self.analysis, "contract", self.context)
        for term in ("CfuResource", "VS1/VS2", "ctx.read", "ctx.complete",
                     "vector source snapshots", "LegacyRegisterNumber",
                     "AgentCfuLib.scala", "AgentCfuBase", "read-only platform",
                     "import AgentCfuBase.CfuResource._"):
            self.assertIn(term, hw)
        # The platform is read-only and must not be presented as an edit target.
        self.assertIn("Never edit,", hw)
        self.assertEqual(
            [path for path in HARDWARE_TARGETS if "AgentCfuLib" in path], [],
        )
        sw = software_prompt("accelerate it", self.spec, self.analysis, "contract", self.context,
                             targets=[("agents/generated/runs/test/main.c", "")])
        for term in ("return void", "rd=x0", "vector/memory effects"):
            self.assertIn(term, sw)
        isa = _isa_from_template("", self.spec, "")
        for command in isa["commands"]:
            self.assertEqual(command["resources"], [
                {"storage": "cpu", "access": "read", "role": "rs1"},
                {"storage": "cpu", "access": "read", "role": "rs2"},
                {"storage": "cpu", "access": "write", "role": "rd"},
            ])

    def test_contract_prompt_forbids_wrapping_the_contract(self) -> None:
        # GLM wrapped the contract in {"analysis", "roofline", "isa_contract": "<yaml>"}
        # and the rung could not recover, so the prompt must forbid it explicitly.
        prompt = contract_prompt("accelerate it", self.spec, self.analysis, self.context)
        for term in ("must BE the top level", "isa_contract", "as a string",
                     "thoughts/analysis/reasoning"):
            self.assertIn(term, prompt)

    def test_conformance_prompt_mentions_json_for_json_mode(self) -> None:
        # The conformance request sets json_object mode; a prompt without the word
        # "json" makes OpenAI-compatible gateways answer HTTP 400.
        from agents.cfu_design_agent.prompts import conformance_prompt

        isa = {
            "spec_version": "1",
            "identity": {"name": "x", "opcode": "custom0", "function_id_width": 3},
            "commands": [{
                "function_id": 1, "subfunction_id": 1, "name": "add", "semantics": "rd = a + b",
                "resources": [
                    {"storage": "cpu", "access": "read", "role": "rs1"},
                    {"storage": "cpu", "access": "write", "role": "rd"},
                ],
            }],
        }
        prompt = conformance_prompt("accelerate it", isa, {"name": "w"})
        self.assertIn("json", prompt.lower())
        self.assertIn("commands", prompt)

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


PROMPT_BODY = "BODY_SENTINEL_" + ("x" * 6000)


def _prompt_context() -> dict:
    return {
        "files": {
            "skills/cfu-designer/SKILL.md": PROMPT_BODY,
            "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala": PROMPT_BODY,
            "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala": PROMPT_BODY,
            "benchmarks/embench-iot/src/crc32/crc_32.c": PROMPT_BODY,
        },
        "visibility": "standard",
        "workdir": "/repo/root",
        "run_root": "agents/generated/runs/r1",
    }


def _stage_prompts(agent_tools: bool) -> dict[str, str]:
    context = _prompt_context()
    spec = {"name": "w", "operation": "crc32", "unknowns": []}
    run = "agents/generated/runs/r1"
    return {
        "analysis": workload_prompt(
                "t", spec, context, agent_tools=agent_tools, tool_rounds=6,
                artifact_paths=[(f"{run}/05_profile/result.yaml", "profile")],
            ),
        "contract": contract_prompt(
            "t", spec, "ANALYSIS_BODY", context, agent_tools=agent_tools, tool_rounds=6,
            artifact_paths=[(f"{run}/07_analysis/workload_analysis.yaml", "analysis")],
        ),
        "conformance": conformance_prompt(
            "t", {"commands": []}, spec, agent_tools=agent_tools, tool_rounds=6,
            workdir="/repo/root", run_root=run,
            artifact_paths=[(f"{run}/08_isa/cfu_isa.yaml", "contract")],
        ),
        "hardware": hardware_prompt(
            "t", spec, "ANALYSIS_BODY", "CONTRACT_BODY", context,
            agent_tools=agent_tools, tool_rounds=6,
            artifact_paths=[(f"{run}/08_isa/cfu_isa.yaml", "contract")],
        ),
        "software": software_prompt(
            "t", spec, "ANALYSIS_BODY", "CONTRACT_BODY", context,
            agent_tools=agent_tools, tool_rounds=6,
            targets=[(f"{run}/workspace/embench_crc32/cfu/crc_32.c", PROMPT_BODY)],
            intrinsics_path=f"{run}/workspace/embench_crc32/cfu/cfu_intrinsics.h",
            artifact_paths=[(f"{run}/08_isa/cfu_isa.yaml", "contract")],
        ),
        "optimize": optimization_prompt(
            "t", spec, "score", "rounds", {}, "", "", 2, "performance",
            agent_tools=agent_tools, tool_rounds=6,
            workdir="/repo/root", run_root=run,
            artifact_paths=[(f"{run}/10_validation/embench.yaml", "validation")],
        ),
    }


class PromptBudgetTest(unittest.TestCase):
    """Tools-enabled prompts are path-based briefs; bodies are never inlined."""

    BODY = PROMPT_BODY

    def _context(self) -> dict:
        return _prompt_context()

    def _prompts(self, agent_tools: bool) -> dict[str, str]:
        return _stage_prompts(agent_tools)

    def test_tools_enabled_prompts_are_path_briefs(self) -> None:
        budgets = {
            "analysis": 9000,
            "contract": 12000,
            "conformance": 9000,
            "hardware": 14000,
            "software": 14000,
            "optimize": 9000,
        }
        for name, prompt in self._prompts(True).items():
            with self.subTest(stage=name):
                self.assertNotIn("BODY_SENTINEL", prompt)
                self.assertNotIn("ANALYSIS_BODY", prompt)
                self.assertNotIn("CONTRACT_BODY", prompt)
                self.assertIn("working directory: /repo/root", prompt)
                self.assertIn("run directory: agents/generated/runs/r1/", prompt)
                self.assertIn("read_file", prompt)
                self.assertIn("## Deliverable", prompt)
                self.assertIn("## Stage artifacts", prompt)
                self.assertLess(len(prompt), budgets[name], f"{name} prompt grew: {len(prompt)} chars")
        prompts = self._prompts(True)
        # The shared repo index is present wherever the prompt has a repo context.
        for name in ("analysis", "contract", "hardware", "software"):
            self.assertIn("## Key paths", prompts[name])
            self.assertIn("skills/cfu-designer/SKILL.md", prompts[name])
        self.assertIn("08_isa/cfu_isa.yaml", prompts["contract"])
        self.assertIn("08_isa/cfu_isa.yaml", prompts["conformance"])
        self.assertIn("09_implementation/hw_plan.md", prompts["optimize"])
        self.assertIn("cfu_intrinsics.h", prompts["software"])

    def test_no_tools_prompts_keep_the_legacy_bodies(self) -> None:
        prompts = self._prompts(False)
        self.assertIn("BODY_SENTINEL", prompts["analysis"])
        self.assertIn("ANALYSIS_BODY", prompts["contract"])
        self.assertIn("ANALYSIS_BODY", prompts["hardware"])
        self.assertIn("BODY_SENTINEL", prompts["software"])
        self.assertIn("ISA spec:", prompts["optimize"])
        self.assertIn("No tool loop is available", prompts["analysis"])
        # The stable index is shared by every stage, including body-inlining calls.
        self.assertIn("## Key paths", prompts["analysis"])


class PromptCacheLayoutTest(unittest.TestCase):
    """Stable content first: the provider caches only a request's leading bytes.

    Regression class (run 20261003-090854): two stages of the same run shared six
    characters of user prompt, because ``## Stage`` and the stage-specific tool
    block came first. Everything before ``## Stage`` must now be byte-identical.
    """

    def test_shared_head_precedes_every_stage_specific_section(self) -> None:
        # The canonical order: every shared section first, in one fixed sequence,
        # then the stage block. A section present in some stages only (the shared
        # workload digests) must not jump ahead of the ones all stages share.
        order = [
            "## Where you are", "## Key paths", "Normalized WorkloadSpec:",
            "Measured profile:", "Embench workload:", "## Stage:", "## Tools",
            "## Deliverable", "## Stage artifacts",
        ]
        prompts = _stage_prompts(True)
        spec = {"name": "w", "operation": "crc32", "unknowns": []}
        shared = workload_digest_block(spec, None)
        shared_head = prompts["analysis"][:prompts["analysis"].index("## Stage:")]
        for name, prompt in prompts.items():
            with self.subTest(stage=name):
                expected = [marker for marker in order if marker in prompt]
                found = sorted((prompt.index(marker), marker) for marker in expected)
                self.assertEqual([marker for _, marker in found], expected, name)
        for name in ("analysis", "contract", "hardware", "software"):
            self.assertIn(shared, prompts[name])
        # Everything before the first stage block is one shared prefix, and the
        # first divergence is the stage name itself.
        for name in ("contract", "hardware", "software"):
            self.assertTrue(prompts[name].startswith(shared_head), f"{name} diverged early")
        self.assertTrue(shared_head.startswith("Task:"))
        self.assertTrue(prompts["hardware"][len(shared_head):].startswith("## Stage: hardware"))

    def test_a_run_id_never_reaches_the_request(self) -> None:
        from agents.cfu_design_agent.runtime import llm_step

        sent = []
        for run_root in ("agents/generated/runs/aaa", "agents/generated/runs/bbb"):
            state = {"run_root": run_root}
            required = [(f"{run_root}/07_analysis/workload_analysis.yaml", "analysis")]
            for name, prompt in _stage_prompts(True).items():
                with self.subTest(stage=name, run=run_root):
                    request = llm_step._virtual_paths(state, "s", prompt)[1]
                    self.assertNotIn(run_root, request)
                    self.assertIn("{{run-dir}}", request)
            sent.append(llm_step._virtual_paths(state, "s", workload_prompt(
                "t", {"name": "w", "operation": "crc32", "unknowns": []},
                {**_prompt_context(), "run_root": run_root}, None,
                agent_tools=True, tool_rounds=6, artifact_paths=required))[1])
        # Same workload, different run id: the request is byte-identical, which is
        # what lets a provider serve the second run from its prefix cache.
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(llm_step._virtual_paths({}, "s", "no run here"), ("s", "no run here"))


class PathsDigestTest(unittest.TestCase):
    """The path index lists where to look and never a file body."""

    def test_paths_digest_lists_paths_without_bodies(self) -> None:
        context = {
            "files": {
                "skills/cfu-designer/SKILL.md": "BODY_SENTINEL",
                ".git/config": "secret",
                "agents/generated/runs/other/report.md": "other run body",
            },
            "visibility": "standard",
            "run_root": "agents/generated/runs/mine",
        }
        digest = paths_digest(context, artifact_paths=[
            ("agents/generated/runs/mine/08_isa/cfu_isa.yaml", "contract"),
        ])
        self.assertIn("skills/cfu-designer/SKILL.md", digest)
        self.assertIn("agents/generated/runs/mine/08_isa/cfu_isa.yaml", digest)
        self.assertIn("contract", digest)
        self.assertNotIn("BODY_SENTINEL", digest)
        self.assertNotIn(".git/config", digest)
        self.assertNotIn("agents/generated/runs/other/report.md", digest)
        self.assertIn("## Visibility: standard", digest)

    def test_paths_digest_is_empty_without_anything_to_show(self) -> None:
        self.assertEqual(paths_digest({"files": {}}), "")


