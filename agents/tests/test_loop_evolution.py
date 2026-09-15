"""Regression coverage for bounded exploration and append-only ISA rounds."""
import copy
import hashlib
import json
import shutil
import unittest
from unittest import mock

import yaml

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.artifacts import RunArtifacts
from agents.cfu_design_agent.cli import build_parser
from agents.cfu_design_agent.llm import LlmResult
from agents.cfu_design_agent.policy import load_policy
from agents.tests.common import REPO_ROOT
from agents.tests.test_graph_ladder import VALID_ISA


class EvolutionLoopTest(unittest.TestCase):
    def setUp(self):
        self.run_id = "test-loop-evolution-00000000"
        self.root = REPO_ROOT / "agents/generated/runs" / self.run_id
        self.addCleanup(shutil.rmtree, self.root, True)
        self.isa = yaml.safe_load(VALID_ISA)
        self.state = {"task": "accelerate", "workdir": str(REPO_ROOT),
                      "run_id": self.run_id, "run_root": self.root.relative_to(REPO_ROOT).as_posix(),
                      "loop_optimize": True, "max_iters": 5, "isa_spec": self.isa,
                      "optimization_plan": {"target": "extend_isa", "reason": "batch updates"}}

    def candidate(self):
        result = copy.deepcopy(self.isa)
        result["commands"].append({"function_id": 1, "subfunction_id": 1,
                                   "name": "batch", "semantics": "batched updates"})
        return result

    def complete(self, text):
        return (LlmResult(text, used_llm=True), "raw.txt")

    def test_cli_defaults_to_bounded_exploration(self):
        args = build_parser().parse_args(["--loop"])
        self.assertEqual(args.loop_policy, "explore")
        self.assertEqual(args.loop_patience, 3)

    def test_stop_is_reconsidered_and_can_extend_isa(self):
        replies = [self.complete('{"continue":false,"reason":"reasonable speedup"}'),
                   self.complete('{"continue":true,"target":"extend_isa","reason":"batch updates"}')]
        with mock.patch.object(graph, "_complete_llm", side_effect=replies) as call:
            target, decision, _ = graph._choose_optimization(self.state, [])
        self.assertEqual(target, "extend_isa")
        self.assertEqual(call.call_count, 2)
        self.assertIn("RECONSIDER EARLY STOP", call.call_args.args[4])

    def test_reconsideration_is_bounded(self):
        with mock.patch.object(graph, "_complete_llm", return_value=self.complete('{"continue":false}')) as call:
            target, decision, _ = graph._choose_optimization(self.state, [])
        self.assertEqual(target, "")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(decision["source"], "planner_exhausted")

    def test_patience_is_framework_owned(self):
        rounds = [{"target": "baseline", "improved": True}] + [
            {"target": "implement_sw", "improved": False} for _ in range(3)]
        with mock.patch.object(graph, "_complete_llm") as call:
            target, decision, _ = graph._choose_optimization(self.state, rounds)
        call.assert_not_called()
        self.assertEqual(decision["source"], "patience")

    def test_extension_accepts_new_encoding_and_keeps_base(self):
        with mock.patch.object(graph, "_complete_llm", return_value=self.complete(json.dumps(self.candidate()))):
            update = graph.extend_isa(self.state)
        self.assertTrue(update["gates"]["isa"]["ok"])
        self.assertEqual(update["gates"]["isa"]["next"], "implement_hw")
        self.assertEqual(update["isa_revision"], 1)
        self.assertEqual(update["isa_added_symbols"], ["cfu_batch"])
        self.assertEqual(update["isa_extension_base"], self.isa)
        self.assertTrue((self.root / "08_isa/revisions/001.yaml").is_file())

    def test_rejected_extension_does_not_replace_incumbent(self):
        candidate = self.candidate()
        candidate["commands"][0]["semantics"] = "incompatible"
        with mock.patch.object(graph, "_complete_llm", return_value=self.complete(json.dumps(candidate))):
            update = graph.extend_isa(self.state)
        self.assertFalse(update["gates"]["isa"]["ok"])
        self.assertEqual(update["gates"]["isa"]["next"], "extend_isa")
        self.assertNotIn("isa_spec", update)
        self.assertEqual(self.state["isa_spec"], self.isa)

    def test_extension_prevents_software_reuse(self):
        with mock.patch.object(graph, "_software_targets", return_value=[("kernel.c", "cfu_compute(a,b);")]):
            ok, reason, _, _ = graph._software_reuse_evidence(self.state, mock.Mock())
        self.assertFalse(ok)
        self.assertIn("adaptation", reason)

    def test_precise_cycles_rank_same_display_speedup(self):
        old = {"reference_cycles": 26650231, "cfu_cycles": 14352815,
               "speedup_x100": 185, "cost": {"cells": 98238}}
        new = {**old, "cfu_cycles": 14334785}
        self.assertTrue(graph._round_improved(new, old))
        self.assertFalse(graph._round_improved(old, new))
        self.assertFalse(graph._round_improved({**new, "validated": False}, old))

    def test_artifact_versions_survive_latest_overwrite(self):
        run = RunArtifacts.create(REPO_ROOT, load_policy(REPO_ROOT), run_id=self.run_id)
        first = run.write_text("review", "12_review/feedback.txt", "old")
        second = run.write_text("review", "12_review/feedback.txt", "new")
        for ref, text in [(first, "old"), (second, "new")]:
            self.assertEqual((REPO_ROOT / ref["immutable_path"]).read_text(), text)
            self.assertEqual(ref["sha256"], hashlib.sha256(text.encode()).hexdigest())
        run.write_text("review", "report.md", "report")

    def test_resolved_build_error_is_not_reinjected(self):
        results = [{"rung": "hw_build", "returncode": 1, "stdout_tail": "old"},
                   {"rung": "hw_build", "returncode": 0},
                   {"rung": "hw_build", "returncode": 1, "stdout_tail": "current"}]
        text = graph.check_feedback({"validation_results": results}, "hw_build")
        self.assertNotIn("old", text)
        self.assertIn("current", text)

    def test_long_build_output_keeps_the_diagnostic_line(self):
        # sbt/SpinalHDL print the diagnostic before a long stack trace. A tail-only
        # excerpt used to keep only the "at ..." frames, so hardware repairs never
        # saw the cause of the failure.
        stack = "\n".join(
            f"[error]     at some.pkg.Cls.m(File.scala:{i})" for i in range(400)
        )
        stdout = "[info] welcome to sbt\n[error] Vec address width mismatch.\n" + stack
        results = [{"rung": "hw_build", "returncode": 1, "stdout_tail": stdout}]
        text = graph.check_feedback({"validation_results": results}, "hw_build")
        self.assertIn("Vec address width mismatch.", text)
        self.assertLess(len(text), 6000)

    def test_output_excerpt_flags_read_only_platform_failures(self):
        # A failure located in the read-only platform is only actionable as "change
        # the design" or "report a platform defect", so the excerpt must say so.
        platform = (
            '[error] Vec address width mismatch.\n'
            '[error]     at vexiiriscv.soc.mico.AgentCfuBase.vectorRead(AgentCfuLib.scala:97)\n'
        )
        out = graph.output_excerpt(platform, 2500)
        self.assertIn('read-only platform note', out)
        self.assertIn('AgentCfuLib.scala', out)
        self.assertIn('AgentCfu.scala', out)
        design = (
            '[error] WIDTH MISMATCH (32 bits <- 31 bits) on (Bits ^ Bits)[32 bits]\n'
            '[error]     at vexiiriscv.soc.mico.AgentCfu.x(AgentCfu.scala:155)\n'
        )
        self.assertNotIn('read-only platform note', graph.output_excerpt(design, 2500))

    def test_output_excerpt_keeps_spinal_diagnostics_over_stack_frames(self):
        # Elaboration failure: banner, message, "- key : value" details, then frames.
        text = (
            '[error] Exception in thread "main" spinal.core.SpinalExit:\n'
            '[error]  Error detected in phase PhaseCreateComponent\n'
            f"[error] {'*' * 80}\n"
            '[error] Vec address width mismatch.\n'
            '[error] - Vec : vectorRegs : Vec of 1 elements\n'
            '[error] - Address width : 1\n'
            '[error]     at vexiiriscv.soc.mico.AgentCfuBase.vectorReadBeat(AgentCfuLib.scala:101)\n'
            + "".join(
                f"[error]     at spinal.core.Phase$anonfun$impl${i}(Phase.scala:{3000 + i})\n"
                for i in range(200)
            )
            + "[error] Design's errors are listed above.\n"
            "[error] Nonzero exit code returned from runner: 1\n"
        )
        out = graph.output_excerpt(text, 2500)
        for kept in (
            'Vec address width mismatch.',
            '- Vec : vectorRegs : Vec of 1 elements',
            '- Address width : 1',
            'Design\'s errors are listed above.',
            'Nonzero exit code returned from runner: 1',
            'AgentCfuLib.scala:101',
        ):
            self.assertIn(kept, out)
        # The highlighted section must not contain stack frames.
        self.assertNotIn('at spinal.core', out.split('first source location')[0])
        self.assertNotIn('Phase$anonfun$impl$199', out.split('[truncated]')[0])

    def test_output_excerpt_reports_width_mismatch_and_design_location(self):
        text = (
            '[error] Exception in thread "main" spinal.core.SpinalExit:\n'
            '[error]  Error detected in phase PhaseNormalizeNodeInputs\n'
            '[error] WIDTH MISMATCH (32 bits <- 31 bits) on (Bits ^ Bits)[32 bits]\n'
            '[error] - Left  operand : (UInt -> Bits of 31 bits)\n'
            '[error] - Right operand : (top/??? :  Bits[32 bits])\n'
            '[error]   at \n'
            '[error]     at vexiiriscv.soc.mico.AgentCfu.$anonfun$instructionTable$9(AgentCfu.scala:155)\n'
            + "".join(
                f"[error]     at spinal.core.Phase$anonfun$impl${i}(Phase.scala:{3000 + i})\n"
                for i in range(200)
            )
        )
        out = graph.output_excerpt(text, 2500)
        self.assertIn('WIDTH MISMATCH (32 bits <- 31 bits)', out)
        self.assertIn('- Left  operand : (UInt -> Bits of 31 bits)', out)
        self.assertIn('- Right operand :', out)
        self.assertIn('AgentCfu.scala:155', out)

    def test_output_excerpt_prefers_design_frames_over_stdlib_and_framework(self):
        # sbt tags the SpinalHDL simulation error as [info], and the first frames are
        # stdlib/framework; the useful frame is the one in project code.
        text = (
            '[info] [Error] Simulation failed at time=421\n'
            '[error] Exception in thread "main" java.lang.AssertionError: assertion failed: boom\n'
            '[error] \tat scala.Predef$.assert(Predef.scala:223)\n'
            '[error] \tat spinal.core.package$.assert(core.scala:548)\n'
            '[error] \tat vexiiriscv.soc.mico.AgentCfuSuboperationSim$.boundedWait$1'
            '(AgentCfuSuboperationSim.scala:93)\n'
            + "".join(
                f"[error] \tat spinal.core.Phase$anonfun$impl${i}(Phase.scala:{3000 + i})\n"
                for i in range(200)
            )
        )
        out = graph.output_excerpt(text, 2500)
        self.assertIn('Simulation failed at time=421', out)
        self.assertIn('assertion failed: boom', out)
        self.assertIn('first source location', out)
        self.assertIn('AgentCfuSuboperationSim.scala:93', out)
        self.assertNotIn('first source location: at scala.Predef', out)

    def test_output_excerpt_keeps_scala_compile_errors(self):
        text = (
            '[error] /w/design/AgentCfu.scala:14:46: not found: type CfuResource\n'
            '[error]       def ins(name: String)(fn: InstructionContext => Unit) =\n'
            '[error]                                              ^\n'
            '[error] one error found\n'
            '[error] (Compile / compileIncremental) Compilation failed\n'
        )
        out = graph.output_excerpt(text, 2500)
        self.assertIn('not found: type CfuResource', out)
        self.assertIn('one error found', out)
        self.assertIn('Compilation failed', out)

    def test_output_excerpt_falls_back_to_window_without_diagnostics(self):
        out = graph.output_excerpt('[info] nothing useful\n' * 500, 2500)
        self.assertLessEqual(len(out), 2600)
        self.assertIn('[truncated]', out)

    def test_call_evidence_ignores_strings_and_inactive_cfu_branches(self):
        source = '''const char *s = "cfu_batch(a,b)";
#if 0
cfu_batch(a,b);
#endif
#ifndef USE_AGENT_CFU
cfu_batch(a,b);
#else
cfu_compute(a,b);
#endif
'''
        toolbox = mock.Mock()
        toolbox.read_text.return_value = source
        sites = graph._cfu_call_sites(self.state, toolbox, [("kernel.c", source)], max_sites=None)
        self.assertEqual([s["name"] for s in sites], ["cfu_compute"])
        toolbox.read_text.return_value = "cfu_compute(a,b);\n" * 30 + "cfu_batch(a,b);"
        sites = graph._cfu_call_sites(self.state, toolbox, [("kernel.c", "")], max_sites=None)
        self.assertIn("cfu_batch", {s["name"] for s in sites})

    def test_second_extension_preserves_first_added_command(self):
        first = self.candidate()
        second = copy.deepcopy(first)
        second["commands"].append({"function_id": 1, "subfunction_id": 2,
                                    "name": "finish", "semantics": "finish batch"})
        state = {**self.state, "isa_spec": first, "isa_extension_base": first, "isa_revision": 1}
        with mock.patch.object(graph, "_complete_llm", return_value=self.complete(json.dumps(second))):
            update = graph.extend_isa(state)
        self.assertEqual(update["isa_revision"], 2)
        self.assertEqual(update["isa_added_symbols"], ["cfu_finish"])
        self.assertEqual(update["isa_spec"]["commands"][:2], first["commands"])

    def test_failed_experiment_retains_best_snapshot(self):
        candidate = {"round": 0, "manifest": "candidate.yaml", "files": []}
        state = {**self.state, "best_candidate": candidate,
                 "optimization_round": 1,
                 "optimization_rounds": [{"round": 0, "target": "baseline", "improved": True,
                                           "measured": True, "speedup_x100": 200}],
                 "gates": {"workload_spec": {"ok": True}, "isa": {"ok": True},
                           "implementation_hw": {"ok": False}, "hw_build": {"ok": False}}}
        with mock.patch.object(graph, "_choose_optimization") as choose:
            result = graph.review_and_route(state)
        choose.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["best_candidate"], candidate)
        self.assertEqual(result["delivered_round"], 0)

    def test_candidate_snapshot_is_complete_and_missing_source_fails(self):
        path = self.root / "workspace/design/AgentCfu.scala"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("verified hardware")
        state = {**self.state, "design_files": [path.relative_to(REPO_ROOT).as_posix()]}
        entry = {"round": 0}
        result = graph._snapshot_candidate(state, entry)
        snapshot = REPO_ROOT / result["files"][0]["snapshot"]
        self.assertEqual(snapshot.read_text(), "verified hardware")
        path.write_text("later broken hardware")
        self.assertEqual(snapshot.read_text(), "verified hardware")
        state["design_files"] = [str(self.root.relative_to(REPO_ROOT) / "workspace/design/AgentCfuFiber.scala")]
        with self.assertRaisesRegex(ValueError, "missing"):
            graph._snapshot_candidate(state, {"round": 1})

    def test_both_runners_have_extension_node(self):
        self.assertIs(graph.python_nodes()["extend_isa"], graph.extend_isa)
        self.assertIs(graph.PY_ROUTERS["extend_isa"], graph._route_after_isa)


if __name__ == "__main__":
    unittest.main()
