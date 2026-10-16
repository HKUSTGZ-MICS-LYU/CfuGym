"""Offline simulation outcome contracts: no LLM, compiler, or simulator calls."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent.runtime import tools
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.verify import software


class CandidateBaselineTest(unittest.TestCase):
    def test_final_hw_and_sw_sources_compare_independently_to_initial_baseline(self):
        state = {"design_baseline": {"version": 1, "hardware": "hw0", "software": "sw0"}}
        cases = (
            ({"hardware": "hw0", "software": "sw0"}, {"hardware": False, "software": False}),
            ({"hardware": "hw1", "software": "sw0"}, {"hardware": True, "software": False}),
            ({"hardware": "hw0", "software": "sw1"}, {"hardware": False, "software": True}),
            ({"hardware": "hw1", "software": "sw1"}, {"hardware": True, "software": True}),
        )
        for current, expected in cases:
            with self.subTest(current=current), mock.patch.object(
                    software.intrinsics, "_design_source_hashes", return_value=current):
                self.assertEqual(
                    software.intrinsics._candidate_change_domains(state, mock.Mock()), expected)

    def test_legacy_state_falls_back_to_both_touched_path_ledgers(self):
        state = {"changed_design_files": [], "changed_sw_files": ["kernel.c"]}
        self.assertEqual(
            software.intrinsics._candidate_change_domains(state, mock.Mock()),
            {"hardware": False, "software": True},
        )

    def test_host_seeded_intrinsics_header_matches_initial_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            run_root = "agents/generated/runs/header-baseline"
            hardware_path = next(iter(tools.design_workspace_paths(run_root).values()))
            software_path = f"{run_root}/workspace/demo/kernel.c"
            for path in (hardware_path, software_path):
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("initial\n")
            toolbox = SimpleNamespace(
                workdir=workdir,
                read_text=lambda path, limit=2_000_000: (workdir / path).read_text()[:limit],
            )
            state = {
                "workdir": str(workdir), "run_root": run_root, "isa_contract": False,
                "apply_patch": True,
                "benchmark_project": {"kind": "generated_benchmark",
                                      "workspace": f"{run_root}/workspace/demo",
                                      "source_path": software_path},
                "workload_spec": {}, "changed_design_files": [], "changed_sw_files": [],
            }
            state["design_baseline"] = intrinsics._initial_design_baseline(state, toolbox)
            header = intrinsics.intrinsics_header_path(state)
            header_path = workdir / header
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(intrinsics.intrinsics_template_text(str(workdir)))
            self.assertEqual(intrinsics._candidate_change_domains(state, toolbox),
                             {"hardware": False, "software": False})
            header_path.write_text("agent-authored header\n")
            self.assertEqual(intrinsics._candidate_change_domains(state, toolbox),
                             {"hardware": False, "software": True})

    def test_source_hashes_detect_either_final_side_against_initial_files(self):
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            run_root = "agents/generated/runs/baseline"
            hardware_paths = list(tools.design_workspace_paths(run_root).values())
            software_path = f"{run_root}/workspace/demo/kernel.c"
            originals = {path: "baseline\n" for path in (*hardware_paths, software_path)}
            for path, text in originals.items():
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            toolbox = SimpleNamespace(
                workdir=workdir,
                read_text=lambda path, limit=2_000_000: (workdir / path).read_text()[:limit],
            )
            state = {
                "workdir": str(workdir), "run_root": run_root, "isa_contract": True,
                "benchmark_project": {"kind": "generated_benchmark", "source_path": software_path},
                "workload_spec": {}, "changed_design_files": [], "changed_sw_files": [],
            }
            state["design_baseline"] = intrinsics._initial_design_baseline(state, toolbox)
            self.assertEqual(intrinsics._candidate_change_domains(state, toolbox),
                             {"hardware": False, "software": False})

            (workdir / software_path).write_text("software-only repair\n")
            self.assertEqual(intrinsics._candidate_change_domains(state, toolbox),
                             {"hardware": False, "software": True})
            (workdir / software_path).write_text(originals[software_path])
            (workdir / hardware_paths[0]).write_text("hardware-only repair\n")
            self.assertEqual(intrinsics._candidate_change_domains(state, toolbox),
                             {"hardware": True, "software": False})


class SimulationOutcomesTest(unittest.TestCase):
    def test_software_only_change_is_preserved_in_simulation_evidence(self):
        result = self.evaluate(changed_domains={"hardware": False, "software": True})
        self.assertTrue(result["design_changed"])
        self.assertEqual(result["changed_domains"], {"hardware": False, "software": True})
        self.assertTrue(result["performance_valid"])

    def evaluate(self, *, name="xgboost", ref=None, got=None, rc=0, verify="pass", category="", **extra):
        reference = {"cycles": 1000, "result": 42, "verify": 1} if ref is None else ref
        profile = {"cycles": 500, "result": 42, "verify": 1} if got is None else got
        result = {"returncode": rc, "parsed": {"verify": {"status": verify, "benchmark": name},
                  "embench_profile": {name: profile}}, "failure_observation": {"category": category}}
        return software._simulation_outcome(results=[result], name=name, reference=reference,
            profile=profile, verification="bench_verify", changed=True, integrity_errors=[], **extra)

    def test_xgboost_reference_mismatch_overrides_verify_pass_and_speed(self):
        result = self.evaluate(got={"cycles": 100, "result": 43, "verify": 1})
        self.assertEqual(result["terminal_reason"], "result_mismatch")
        self.assertFalse(result["correctness_ok"])
        self.assertFalse(result["performance_valid"])
        self.assertIsNone(result["speedup_x100"])
        self.assertIn("computed result=43", " ".join(result["errors"]))

    def test_picojpeg_wrong_and_slow_is_correctness_not_floor(self):
        result = self.evaluate(name="picojpeg", got={"cycles": 2000, "result": 43, "verify": 0},
                               rc=1, verify="fail", category="fail_symbol")
        self.assertEqual(result["terminal_reason"], "verify_failed")
        self.assertIsNone(result["speedup_x100"])
        self.assertNotIn("meaningfully faster", " ".join(result["errors"]))

    def test_nsichneu_assertion_without_cycles_is_not_zero_speed(self):
        result = self.evaluate(name="nsichneu", got={}, rc=1, verify="", category="assertion")
        self.assertEqual(result["terminal_reason"], "assertion")
        self.assertIsNone(result["cfu_cycles"])
        self.assertIsNone(result["speedup_x100"])
        self.assertFalse(result["timing_ok"])

    def test_explicit_wrong_result_without_timing_is_still_correctness(self):
        result = self.evaluate(got={"result": 43, "verify": 0}, verify="fail")
        self.assertEqual(result["terminal_reason"], "verify_failed")
        self.assertFalse(result["correctness_ok"])
        self.assertIsNone(result["speedup_x100"])

    def test_nonzero_exit_and_timeout_cannot_be_blessed_by_pass(self):
        for rc, category, reason in ((1, "simulation", "execution_error"),
                                     (-9, "timeout", "timeout"), (1, "assertion", "assertion")):
            with self.subTest(reason=reason):
                result = self.evaluate(rc=rc, category=category)
                self.assertEqual(result["terminal_reason"], reason)
                self.assertFalse(result["performance_valid"])
                self.assertIsNone(result["speedup_x100"])

    def test_result_missing_on_either_side_fails_closed(self):
        for ref, got in (({"cycles": 1000}, {"cycles": 500, "result": 42, "verify": 1}),
                         ({"cycles": 1000, "result": 42}, {"cycles": 500, "verify": 1})):
            with self.subTest(ref=ref, got=got):
                self.assertEqual(self.evaluate(ref=ref, got=got)["terminal_reason"], "result_missing")

    def test_legacy_verify_only_is_supported_but_requires_verdict(self):
        result = self.evaluate(ref={"cycles": 1000}, got={"cycles": 500, "verify": 1})
        self.assertEqual(result["terminal_reason"], "passed")
        self.assertEqual(result["speedup_x100"], 200)
        result = self.evaluate(ref={"cycles": 1000}, got={"cycles": 500}, verify="")
        self.assertEqual(result["terminal_reason"], "no_verification")
        self.assertIsNone(result["speedup_x100"])

    def test_valid_slow_measurement_retains_speed_for_floor(self):
        result = self.evaluate(got={"cycles": 2000, "result": 42, "verify": 1})
        self.assertEqual(result["terminal_reason"], "speedup_floor")
        self.assertTrue(result["performance_valid"])
        self.assertEqual(result["speedup_x100"], 50)

    def test_passing_verify_without_cycles_is_no_timing(self):
        result = self.evaluate(got={"result": 42, "verify": 1})
        self.assertEqual(result["terminal_reason"], "no_timing")
        self.assertTrue(result["correctness_ok"])
        self.assertFalse(result["timing_ok"])
        self.assertIsNone(result["cfu_cycles"])
        self.assertIsNone(result["speedup_x100"])

    def test_report_shows_current_reason_and_invalid_performance(self):
        from agents.cfu_design_agent.stages.review import report
        lines = report.measurement_lines({"benchmark_project": {"kind": "embench_benchmark"},
            "gates": {"sim": {"terminal_reason": "result_mismatch", "correctness_ok": False,
                              "timing_ok": True, "performance_valid": False}}})
        self.assertIn("current simulation outcome: result_mismatch", "\n".join(lines))
        self.assertIn("speedup: unavailable", "\n".join(lines))

    def test_invalid_cycle_counts_are_null(self):
        for value in (None, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(software._cycle_count(value))

    def test_semantic_failure_has_current_id_and_repair_feedback(self):
        from agents.cfu_design_agent.runtime import observations
        from agents.cfu_design_agent.stages import feedback
        state = {"workdir": "/tmp/offline-fixture"}
        real = {"rung": "sim", "command": "mock-sim", "returncode": 0, "log_path": "sim.log"}
        evidence = self.evaluate(got={"cycles": 500, "result": 43, "verify": 1})
        check = software._semantic_failure_result(state, evidence, [real])
        self.assertEqual(real["returncode"], 0)
        self.assertEqual(check["kind"], "host_gate_check")
        self.assertFalse(check["executed"])
        combined = {"validation_results": [real, check], "gates": {"sim": evidence}}
        failures = observations.failure_index(combined)["failures"]
        observation = check["failure_observation"]
        self.assertEqual(failures[observation["id"]]["status"], "current")
        self.assertEqual(observation["category"], "result_mismatch")
        self.assertTrue(observation["log_path"].endswith("10_validation/sim.yaml"))
        self.assertEqual(check["source_log_paths"], ["sim.log"])
        rendered = feedback.sim_feedback(combined)
        self.assertIn(observation["id"], rendered)
        self.assertIn("result_mismatch", rendered)
        self.assertIn("computed result=43", rendered)

    def test_simulation_accepts_software_only_change_against_initial_baseline(self):
        name = "xgboost"
        state = {
            "run_commands": True,
            "changed_design_files": [],  # final repair touched software only
            "changed_sw_files": ["kernel.c"],
            "design_baseline": {"version": 1, "hardware": "hw0", "software": "sw0"},
            "benchmark_project": {"kind": "embench_benchmark", "benchmark": name,
                                  "verification": "bench_verify"},
            "workload_spec": {"benchmark_name": name},
            "embench_profiles": {"reference": {name: {"cycles": 1000, "result": 42}}},
            "gates": {}, "max_iters": 1,
        }
        real = {"returncode": 0, "parsed": {
            "verify": {"status": "pass", "benchmark": name},
            "embench_profile": {name: {"cycles": 500, "result": 42, "verify": 1}},
        }}
        artifacts = mock.Mock()
        with mock.patch.object(software.stage_services, "_toolbox_for_stage", return_value=mock.Mock()), \
             mock.patch.object(software.intrinsics, "_design_source_hashes",
                               return_value={"hardware": "hw0", "software": "sw1"}), \
             mock.patch.object(software.stage_services, "benchmark_integrity_problems", return_value=[]), \
             mock.patch.object(software.embench, "workspace_tampering", return_value=[]), \
             mock.patch.object(software.commands_mod, "_software_sim_commands", return_value=[["mock-sim"]]), \
             mock.patch.object(software.effects, "run_command", return_value=real), \
             mock.patch.object(software.stage_services, "_artifacts_for_stage", return_value=artifacts), \
             mock.patch.object(software.stage_services, "_write_ladder_artifact"), \
             mock.patch.object(software.provenance, "candidate_provenance", return_value={}), \
             mock.patch.object(software.ladder, "_ladder_transition", return_value=({}, "estimate_cost")):
            update = software.simulate(state)
        gate = update["gates"]["sim"]
        self.assertTrue(gate["ok"])
        self.assertTrue(gate["design_changed"])
        self.assertEqual(gate["changed_domains"], {"hardware": False, "software": True})
        self.assertTrue(gate["performance_valid"])
        self.assertEqual(gate["speedup_x100"], 200)

    def test_simulate_mismatch_appends_semantic_verdict_without_changing_exit(self):
        from agents.cfu_design_agent.runtime import observations
        name = "xgboost"
        state = {"run_commands": True, "changed_design_files": ["design"],
                 "benchmark_project": {"kind": "embench_benchmark", "benchmark": name,
                     "verification": "bench_verify"},
                 "embench_profiles": {"reference": {name: {"cycles": 1000, "result": 42}}},
                 "gates": {}, "max_iters": 1}
        real = {"rung": "sim", "returncode": 0, "parsed": {"verify": {"status": "pass", "benchmark": name},
                "embench_profile": {name: {"cycles": 500, "result": 43, "verify": 1}}}}
        with mock.patch.object(software.stage_services, "_toolbox_for_stage", return_value=mock.Mock()), \
             mock.patch.object(software.stage_services, "benchmark_integrity_problems", return_value=[]), \
             mock.patch.object(software.embench, "workspace_tampering", return_value=[]), \
             mock.patch.object(software.commands_mod, "_software_sim_commands", return_value=[["mock-sim"]]), \
             mock.patch.object(software.effects, "run_command", return_value=real), \
             mock.patch.object(software.stage_services, "_artifacts_for_stage", return_value=mock.Mock()), \
             mock.patch.object(software.stage_services, "_write_ladder_artifact"), \
             mock.patch.object(software.ladder, "_ladder_transition", return_value=({}, "estimate_cost")):
            update = software.simulate(state)
        self.assertEqual(real["returncode"], 0)
        self.assertEqual(len(update["validation_results"]), 2)
        ident = update["gates"]["sim"]["failure_id"]
        current = observations.failure_index(update)["failures"][ident]
        self.assertEqual(current["status"], "current")
        self.assertEqual(current["category"], "result_mismatch")
        self.assertIsNone(update["gates"]["sim"]["speedup_x100"])

    def test_simulation_persists_same_current_reason_in_gate_and_yaml(self):
        name = "nsichneu"
        state = {"run_commands": True, "changed_design_files": ["design"],
                 "benchmark_project": {"kind": "embench_benchmark", "benchmark": name,
                     "verification": "bench_verify"},
                 "embench_profiles": {"reference": {name: {"cycles": 1000, "result": 42}}},
                 "errors": ["OLD verify failed and slow"], "gates": {}, "max_iters": 1}
        result = {"returncode": 1, "failure_observation": {"category": "assertion"},
                  "stdout_tail": "HDL assertion failure", "stderr_tail": "", "parsed": {}}
        artifacts = mock.Mock()
        with mock.patch.object(software.stage_services, "_toolbox_for_stage", return_value=mock.Mock()), \
             mock.patch.object(software.stage_services, "benchmark_integrity_problems", return_value=[]), \
             mock.patch.object(software.embench, "workspace_tampering", return_value=[]), \
             mock.patch.object(software.commands_mod, "_software_sim_commands", return_value=[["mock-sim"]]), \
             mock.patch.object(software.effects, "run_command", return_value=result) as run, \
             mock.patch.object(software.stage_services, "_artifacts_for_stage", return_value=artifacts), \
             mock.patch.object(software.stage_services, "_write_ladder_artifact") as ladder_artifact, \
             mock.patch.object(software.ladder, "_ladder_transition", return_value=({}, "estimate_cost")):
            update = software.simulate(state)
        run.assert_called_once()
        gate = update["gates"]["sim"]
        embench_yaml = artifacts.write_yaml.call_args.args[2]
        sim_yaml = ladder_artifact.call_args.kwargs["evidence"]
        for evidence in (gate, embench_yaml, sim_yaml):
            self.assertEqual(evidence["terminal_reason"], "assertion")
            self.assertIsNone(evidence["cfu_cycles"])
            self.assertIsNone(evidence["speedup_x100"])
            self.assertFalse(evidence["performance_valid"])
            self.assertNotIn("OLD", " ".join(evidence["errors"]))


if __name__ == "__main__":
    unittest.main()
