"""Fast FPGA integration contracts: immutable hardware inputs and configuration."""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from agents.cfu_design_agent import cli
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import commands, dependencies, effects, ppa, processes, rtl_snapshot, stage_services, validation_parallel, vivado
from agents.cfu_design_agent.stages.verify import cost
from agents.cfu_design_agent.flow import ladder
from contextlib import ExitStack
from agents.cfu_design_agent.flow import optimize


class _Artifacts:
    def __init__(self, root, run):
        self.root, self.run = root, run

    def write_text(self, stage, relative, text, **kwargs):
        path = self.root / self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return {"path": str(path)}


class FpgaInputsTest(unittest.TestCase):
    def test_frequency_matches_soc_and_sim(self):
        state = {"run_root": "agents/generated/runs/clock", "target_frequency_mhz": 100.0}
        soc = commands.agent_cfu_overlay_commands(state, {})[0][-1]
        sim = commands.agent_cfu_sim_commands(state, {}, "demo.elf")[0][-1]
        for command in (soc, sim):
            self.assertIn("--system-frequency 100000000", command)

    def test_part_changes_only_cost_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {"workdir": directory, "cost_backend": "vivado", "fpga_part": "xc7a200tsbg484-1"}
            before = dependencies.input_fingerprints(state)
            after = dependencies.input_fingerprints({**state, "fpga_part": "xc7a100tcsg324-1"})
            self.assertNotEqual(before["cost"], after["cost"])
            self.assertEqual(before["sim"], after["sim"])
            self.assertEqual(before["hw_build"], after["hw_build"])

    def test_frequency_invalidates_hardware_and_sim(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {"workdir": directory, "target_frequency_mhz": 100.0}
            before = dependencies.input_fingerprints(state)
            after = dependencies.input_fingerprints({**state, "target_frequency_mhz": 125.0})
            for key in ("hw_build", "sim", "cost"):
                self.assertNotEqual(before[key], after[key])

    def test_snapshot_freezes_complete_fileset_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = "agents/generated/runs/fpga-input-test"
            source = root / run / "workspace/soc"
            source.mkdir(parents=True)
            (source / "MiCoSoc.v").write_text('module MiCoSoc; initial $readmemb("ram.bin", mem); endmodule')
            (source / "ram.bin").write_text("01\n10\n")
            state = {"workdir": directory, "run_root": run, "target_frequency_mhz": 100.0}
            with mock.patch.object(stage_services, "_policy", return_value=mock.Mock()), mock.patch.object(
                stage_services, "_artifacts_for_stage", return_value=_Artifacts(root, run)):
                snapshot = rtl_snapshot.freeze(state)
            state["rtl_snapshot"] = snapshot
            self.assertEqual(len(snapshot["files"]), 2)
            rtl_snapshot.validate(state)
            (source / "MiCoSoc.v").write_text("later candidate")
            self.assertIn("module MiCoSoc", (root / snapshot["root"] / "MiCoSoc.v").read_text())
            (root / snapshot["root"] / "ram.bin").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                rtl_snapshot.validate(state)

    def test_missing_memory_reference_refuses_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = "agents/generated/runs/missing-image"
            source = root / run / "workspace/soc"
            source.mkdir(parents=True)
            (source / "MiCoSoc.v").write_text('module MiCoSoc; initial $readmemh("missing.hex", mem); endmodule')
            with self.assertRaisesRegex(ValueError, "missing RTL"):
                rtl_snapshot.freeze({"workdir": directory, "run_root": run})


class FpgaReportingTest(unittest.TestCase):
    def state(self, root):
        state = {"workdir": root, "cost_backend": "vivado", "fpga_part": "xc7a200tsbg484-1",
                 "target_frequency_mhz": 100.0, "evaluation_id": "current-eval", "flow_version": 2}
        data = {"backend": "vivado", "part": state["fpga_part"], "target_frequency_mhz": 100.0,
                "status": "passed", "vivado": {"version": "2025.2"},
                "soc": {"area": {"lut": 100}, "power": {"device_static": {"value": 0.15}}},
                "cfu": {"instance": "agentCfu", "area": {"lut": 0, "ff": 0},
                        "power": {"dynamic": {"upper_bound": 0.001}},
                        "timing": {"internal": {"setup": {"exists": False}, "hold": {"exists": False}}}},
                "evidence": {"input_sha256": dependencies.input_fingerprints(state)["cost"],
                             "evaluation_id": state["evaluation_id"]}}
        state["gates"] = {"cost": {"ok": True, "ppa": data}}
        return state

    def test_current_historical_and_cancelled_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            state = self.state(root)
            self.assertTrue(ppa.view(state)["valid"])
            changed = {**state, "fpga_part": "xc7a100tcsg324-1"}
            self.assertFalse(ppa.view(changed)["current"])
            self.assertFalse(ppa.view({**state, "evaluation_id": "new-eval"})["current"])
            state["gates"]["cost"]["cancelled"] = True
            self.assertFalse(ppa.view(state)["valid"])
            self.assertEqual(optimize._cost_metric(state), {})

    def test_unknown_bound_and_zero_survive_reporting(self):
        with tempfile.TemporaryDirectory() as root:
            state = self.state(root)
            text = "\n".join(ppa.lines(state))
            self.assertIn("<0.001", text)
            self.assertIn("not applicable (no path)", text)
            self.assertIn("not apportioned", text)
            self.assertIn("| CFU hierarchy | 0 | 0 | n/a | n/a |", text)
            self.assertEqual(optimize._cost_metric(state)["lut"], 0)

    def test_vivado_pareto_never_treats_zero_as_unknown_or_crosses_basis(self):
        base_cost = {"backend": "vivado", "metric": "cfu_post_route_lut", "part": "p",
                     "target_frequency_mhz": 100, "lut": 0}
        best = {"cost": base_cost, "speedup_x100": 100}
        self.assertFalse(optimize._round_improved({"cost": {**base_cost, "lut": 1},
                                                  "speedup_x100": 110}, best))
        self.assertTrue(optimize._round_improved({"cost": base_cost,
                                                 "speedup_x100": 110}, best))
        for delta in ({"part": "other"}, {"target_frequency_mhz": 125}, {"backend": "yosys"}):
            self.assertFalse(optimize._round_improved({"cost": {**base_cost, **delta},
                                                      "speedup_x100": 200}, best))


class VivadoCostWorkerTest(unittest.TestCase):
    def call(self, *, status="passed", returncode=0, skip=False, fault=None):
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            state = {"workdir": root, "run_root": "agents/generated/runs/worker-test",
                     "cost_backend": "vivado", "run_commands": True,
                     "evaluation_id": "eval-worker", "eda_timeout": 2, "skip_cost": skip}
            stack.enter_context(mock.patch.object(stage_services, "_toolbox_for_stage", return_value=mock.Mock()))
            write = stack.enter_context(mock.patch.object(stage_services, "_write_cost_artifact"))
            stack.enter_context(mock.patch.object(rtl_snapshot, "validate", return_value={"root": "snapshot", "sha256": "rtl"}))
            stack.enter_context(mock.patch.object(vivado, "build_command", return_value=["python3", "fixed-runner"]))
            run = stack.enter_context(mock.patch.object(effects, "run_command", return_value={"returncode": returncode}, side_effect=fault))
            load = stack.enter_context(mock.patch.object(vivado, "load_summary", return_value={
                "backend": "vivado", "status": status,
                "issues": [{"code": "power_assumptions", "severity": "warning", "detail": "vectorless estimate"}]
                          + ([] if status == "passed" else [{"code": "timing_not_closed", "detail": "setup -1 ns"}])}))
            notify = stack.enter_context(mock.patch.object(validation_parallel, "command_failed"))
            stack.enter_context(mock.patch.object(ladder, "_ladder_transition", side_effect=AssertionError("worker spent repair")))
            update = cost.estimate_cost(state, transition=False)
            return update, run, load, notify, write

    def test_success_retains_evaluation_bound_ppa_without_ladder_or_generation(self):
        update, run, load, notify, write = self.call()
        gate = update["gates"]["cost"]
        self.assertTrue(gate["ok"])
        self.assertEqual(gate["ppa"]["evidence"]["evaluation_id"], "eval-worker")
        self.assertEqual(run.call_args.kwargs["timeout"], 32)
        self.assertEqual(run.call_args.kwargs["command_id"], "vivado_cost")
        self.assertNotIn("next", gate)
        notify.assert_not_called()
        load.assert_called_once()

    def test_zero_exit_semantic_failure_notifies_and_retains_partial_ppa(self):
        update, run, load, notify, write = self.call(status="failed")
        self.assertFalse(update["gates"]["cost"]["ok"])
        self.assertEqual(update["cost_results"][0]["parsed"]["ppa"]["status"], "failed")
        notify.assert_called_once()
        self.assertIn("timing_not_closed", ";".join(update["errors"]))
        self.assertNotIn("power_assumptions", ";".join(update["errors"]))
        self.assertEqual(update["cost_results"][-1]["kind"], "host_gate_check")

    def test_tool_error_still_loads_available_partial_reports(self):
        update, run, load, notify, write = self.call(status="failed", returncode=1)
        self.assertFalse(update["gates"]["cost"]["ok"])
        load.assert_called_once()
        notify.assert_called_once()

    def test_skip_does_not_launch_or_certify(self):
        update, run, load, notify, write = self.call(skip=True)
        self.assertTrue(update["gates"]["cost"]["skipped"])
        self.assertFalse(update["gates"]["cost"]["ok"])
        run.assert_not_called()
        load.assert_not_called()
        notify.assert_not_called()

    def test_unexpected_python_error_and_cancellation_propagate_without_budget(self):
        with self.assertRaisesRegex(TypeError, "bug"):
            self.call(fault=TypeError("bug"))
        with self.assertRaises(processes.ValidationCancelled):
            self.call(fault=processes.ValidationCancelled("sibling failed"))


class FpgaCliTest(unittest.TestCase):
    def test_defaults_and_explicit_vivado_state_are_threaded(self):
        args = cli.build_parser().parse_args([])
        self.assertEqual((args.cost_backend, args.validation_mode), ("yosys", "parallel"))
        self.assertEqual((args.fpga_part, args.target_frequency_mhz), ("xcau25p-ffvb676-2-e", 100.0))
        self.assertEqual(args.fpga_part, vivado.DEFAULT_PART)
        self.assertEqual(cli.build_parser().parse_args([
            "--fpga-part", "xc7a200tsbg484-1"]).fpga_part, "xc7a200tsbg484-1")
        with mock.patch.object(cli, "export_graph_visualization", return_value={}), mock.patch.object(
            cli, "run_agent", return_value={"status": "planned"}) as run, redirect_stdout(io.StringIO()):
            cli.main(["--task", "offline", "--dry-run", "--cost-backend", "vivado",
                      "--validation-mode", "serial", "--skip-cost"])
        state = run.call_args.args[0]
        self.assertEqual((state["flow_version"], state["cost_backend"], state["validation_mode"]), (2, "vivado", "serial"))
        self.assertTrue(state["skip_cost"])
        self.assertEqual(state["fpga_part"], vivado.DEFAULT_PART)

    def test_default_fingerprint_matches_explicit_au25p_not_old_artix7(self):
        with tempfile.TemporaryDirectory() as root:
            state = {"workdir": root, "cost_backend": "vivado"}
            implicit = dependencies.input_fingerprints(state)
            explicit = dependencies.input_fingerprints({**state, "fpga_part": vivado.DEFAULT_PART})
            old = dependencies.input_fingerprints({**state, "fpga_part": "xc7a200tsbg484-1"})
            self.assertEqual(implicit, explicit)
            self.assertNotEqual(implicit["cost"], old["cost"])
            self.assertEqual(implicit["sim"], old["sim"])

    def test_ultrascale_resource_labels_parse_without_7series_assumptions(self):
        area = vivado.parse_utilization("""
| CLB LUTs* | 42 |
| CLB Registers | 64 |
| Block RAM Tile | 1.5 |
| RAMB36E2 only | 1 |
| RAMB18E2 only | 1 |
| DSP48E2 only | 2 |
""")
        self.assertEqual((area["lut"], area["ff"], area["bram_36k"], area["dsp"]), (42, 64, 1.5, 2))
        self.assertEqual((area["ramb18"], area["ramb36"]), (1, 1))

    def test_invalid_eda_options_fail_before_graph(self):
        for extra in (["--cost-backend", "vivado", "--skip-yosys"],
                      ["--target-frequency-mhz", "nan"], ["--target-frequency-mhz", "0"],
                      ["--vivado-threads", "0"], ["--eda-timeout", "-1"],
                      ["--fpga-part", "part;exec_bad"]):
            with self.subTest(extra=extra), mock.patch.object(cli, "run_agent") as run:
                with self.assertRaises(SystemExit):
                    cli.main(["--task", "offline", *extra])
                run.assert_not_called()

    def test_resume_only_records_explicit_eda_overrides(self):
        with mock.patch.object(cli, "export_graph_visualization", return_value={}), mock.patch.object(
            cli, "run_agent", return_value={"status": "planned"}) as run, redirect_stdout(io.StringIO()):
            cli.main(["--resume", "prior-run", "--cost-backend=vivado"])
        self.assertEqual(run.call_args.args[0]["resume_eda_options"], {"cost_backend": "vivado"})


if __name__ == "__main__":
    unittest.main()
