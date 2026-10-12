"""Exercise the production Stage 3 node with no network or compiler."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent import conformance, graph
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox, design_workspace_paths
from agents.cfu_design_agent.flow import ladder
from agents.cfu_design_agent.runtime import checks
from agents.cfu_design_agent.runtime import effects
from agents.cfu_design_agent.runtime import conformance_delivery
from agents.tests.test_conformance import CPU_COMMAND, isa
from agents.cfu_design_agent.runtime import commands
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.runtime import observations
from agents.cfu_design_agent.runtime import stage_services as services
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.implement import unified


class _RoundClient:
    available = True
    model = "scripted-stage3"
    _stage_round_client = object()  # Never create a provider SDK client.

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.messages: list[list[dict]] = []

    def _chat_round(self, sdk, messages, *, allow_tools, tools, response_format):
        self.seen.append([item["function"]["name"] for item in tools])
        self.messages.append(list(messages))
        if not self.turns:
            raise AssertionError("unexpected model call")
        name, arguments = self.turns.pop(0)
        return SimpleNamespace(
            text="", reasoning="", reasoning_source="", finish_reason="tool_calls",
            extra_keys=(), input_tokens=1, output_tokens=1, total_tokens=2,
            message=SimpleNamespace(tool_calls=[SimpleNamespace(
                id=f"call_{len(self.seen)}", function=SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)),
            )]),
        )



class _BatchClient(_RoundClient):
    """A round client that can emit several tool calls in one turn."""

    def _chat_round(self, sdk, messages, *, allow_tools, tools, response_format):
        self.seen.append([item["function"]["name"] for item in tools])
        self.messages.append(list(messages))
        turn = self.turns.pop(0)
        calls = turn if isinstance(turn, list) else [turn]
        return SimpleNamespace(
            text="", reasoning="", reasoning_source="", finish_reason="tool_calls",
            extra_keys=(), input_tokens=1, output_tokens=1, total_tokens=2,
            message=SimpleNamespace(tool_calls=[
                SimpleNamespace(id=f"call_{len(self.seen)}_{index}",
                                function=SimpleNamespace(
                                    name=name, arguments=json.dumps(arguments)))
                for index, (name, arguments) in enumerate(calls)
            ]),
        )

class Stage3IntegrationTest(unittest.TestCase):
    def setUp(self):
        # The delivered oracle is preflighted with a host compiler command; these
        # tests script the model, so stub the command instead of a toolchain.
        preflight = mock.patch.object(effects, "run_command", return_value={
            "returncode": 0, "stdout_tail": "", "stderr_tail": ""})
        preflight.start()
        self.addCleanup(preflight.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        real = Path(__file__).resolve().parents[1] / "specs/agent_access_policy.yaml"
        policy_path = self.root / "agents/specs/agent_access_policy.yaml"
        policy_path.parent.mkdir(parents=True)
        policy_path.write_bytes(real.read_bytes())
        self.run_root = "agents/generated/runs/disposable-stage3"
        self.hw = next(iter(design_workspace_paths(self.run_root).values()))
        self.sw = f"{self.run_root}/workspace/demo_benchmark/kernel.c"
        for name, content in ((self.hw, "class AgentCfu {}\n"),
                              (self.sw, "int kernel(void) { return 1; }\n")):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        self.oracle = json.dumps({"commands": [
            {"name": "compute", "reference": "a + b",
             "data": {"roles": [{"kind": "value", "max": 255},
                                {"kind": "value", "max": 255}]}}]})
        self.state = {
            "workdir": str(self.root), "run_id": "disposable-stage3", "run_root": self.run_root,
            "task": "accelerate", "isa_spec": isa(CPU_COMMAND), "isa_spec_text": "accepted ISA",
            "workload_spec": {}, "workload_analysis": "candidate",
            "benchmark_project": {"kind": "generated_benchmark", "workspace":
                                  f"{self.run_root}/workspace/demo_benchmark", "source_path": self.sw},
            "agent_tools": True, "apply_patch": True, "run_commands": True,
            "stage3_autonomy": True, "max_tool_rounds": 6, "max_tool_calls": 10,
            "gates": {"isa": {"ok": True}},
        }

    def test_hw_and_sw_then_independent_gates(self):
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }", "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("check_hardware_elaboration", {}),  # permitted BEFORE hardware edit
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { /* accelerated */ }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "HW, SW and oracle ready for independent validation"}),
        ])
        from agents.cfu_design_agent.runtime.implementation_tools import ImplementationToolset
        real_run = ImplementationToolset.run
        observed = []
        def audit_run(toolset, name, arguments=None):
            response = real_run(toolset, name, arguments)
            observed.append((name, response))
            return response
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(ImplementationToolset, "run", audit_run), \
             mock.patch.object(commands, "agent_cfu_check_commands", return_value=[["sbt", "mock"]]), \
             mock.patch.object(checks, "_run_check", return_value={"returncode": 1,
                                "stderr_tail": "development check failed"}), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            try:
                update = unified.implement_stage3(self.state)
            except Exception as exc:
                self.fail(f"Stage 3 failed: {exc}; calls={observed!r}")
        self.assertEqual(client.turns, [])
        self.assertEqual(update["gates"]["implementation_hw"]["ok"], True)
        self.assertEqual(update["gates"]["implementation_sw"]["ok"], True)
        self.assertNotIn("hw_build", update["gates"])
        self.assertNotIn("hw_conformance", update["gates"])
        self.assertNotIn("sw_build", update["gates"])
        self.assertIn("accelerated", (self.root / self.hw).read_text())
        self.assertIn(self.sw, update["changed_sw_files"], update["changed_sw_files"])
        self.assertIn("cfu_compute", (self.root / self.sw).read_text(), update["tool_calls"])
        self.assertEqual(graph._route_after_isa({**self.state,
            "gates": {"isa": {"ok": True, "next": "implement_stage3"}}}), "implement_stage3")
        self.assertEqual(ladder._route_after_hw_conformance(
            {**self.state, "gates": update["gates"]}), "build_sw")
        self.assertEqual(graph._route_after_hw_conformance({**self.state,
            "gates": {**update["gates"], "hw_conformance": {"ok": True}}}), "build_sw")

    def test_first_internal_failure_does_not_relax_either_delivery_domain(self):
        cases = (
            ("check_hardware_elaboration", "edit_hardware", {
                "edits": [{"path": self.hw, "search": "class AgentCfu {}",
                           "replace": "class AgentCfu { int changed; }"}]}, "software"),
            ("check_software_compile", "edit_software", {
                "path": self.sw, "search": "int kernel(void) { return 1; }",
                "replace": "int kernel(void) { return cfu_compute(1, 2); }"}, "hardware"),
        )
        for check_name, edit_name, edit, missing_domain in cases:
            with self.subTest(check=check_name):
                client = _RoundClient([
                    (check_name, {}), (edit_name, edit),
                    ("submit_stage", {"delivery": "only repaired the internal failure"}),
                    ("submit_stage", {"delivery": "still only one domain"}),
                ])
                with mock.patch.object(llm_step, "_llm", return_value=client), \
                     mock.patch.object(commands, "agent_cfu_check_commands", return_value=[["sbt", "mock"]]), \
                     mock.patch.object(commands, "_isolated_software_check_commands", return_value=[["make", "compile"]]), \
                     mock.patch.object(checks, "_run_check", return_value={
                         "returncode": 1, "stderr_tail": "error: first internal failure"}), \
                     mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
                    update = unified.implement_stage3(self.state)
                    self.assertFalse(update["gates"]["implementation_hw"]["ok"])
                    self.assertFalse(update["gates"]["implementation_sw"]["ok"])
                    self.assertIn(f"no {missing_domain} target was edited", " ".join(update["errors"]))
                self.assertEqual(client.turns, [])
                self.assertIn("inspect_failure", client.seen[0])
                self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
                self.assertEqual((self.root / self.sw).read_text(), "int kernel(void) { return 1; }\n")

    def _approve_conformance(self, state, box):
        session = conformance_delivery.ConformanceDelivery(state, box, self.sw + ".h")
        record, problems = session.write(self.oracle)
        self.assertEqual(problems, [])
        state["conformance_delivery"] = record
        session.commit()
        return record

    def test_tool_mode_failure_restores_host_intrinsics_header(self):
        path = self.root / intrinsics.intrinsics_header_path(self.state)
        for existed in (False, True):
            with self.subTest(existed=existed):
                if existed:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("old accepted header bytes\n")
                elif path.exists():
                    path.unlink()
                with mock.patch.object(llm_step, "_llm", return_value=_RoundClient([])), \
                     mock.patch.object(llm_step, "_run_agent_step", side_effect=RuntimeError("model down")):
                    with self.assertRaisesRegex(RuntimeError, "model down"):
                        unified.implement_with_tools(self.state)
                if existed:
                    self.assertEqual(path.read_text(), "old accepted header bytes\n")
                else:
                    self.assertFalse(path.exists())

    def _conformance_bytes(self):
        return {path: (self.root / path).read_bytes() if (self.root / path).is_file() else None
                for path in conformance_delivery.delivery_paths(self.state).values()}

    def test_first_delivery_requires_write_conformance_after_hw_and_sw_edits(self):
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw, "search": "class AgentCfu {}",
                                        "replace": "class AgentCfu { int changed; }"}]}),
            ("submit_stage", {"delivery": "HW and SW applied but no oracle"}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "HW, SW and independent oracle applied"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        self.assertEqual(client.turns, [])
        recovery = " ".join(str(item.get("content", "")) for item in client.messages[3])
        self.assertIn("call write_conformance", recovery)
        box = services._toolbox_for_stage(self.state, "implementation")
        record, problems = conformance_delivery.inspect_delivery({**self.state, **update}, box)
        self.assertEqual(problems, [])
        self.assertEqual(record["oracle"], json.loads(self.oracle))
        self.assertEqual(record["oracle_path"],
                         f"{self.run_root}/09_implementation/conformance/oracle.yaml")
        self.assertEqual(record["source_path"],
                         f"{self.run_root}/09_implementation/conformance/cfu_conformance.c")
        for available in client.seen:
            self.assertIn("write_conformance", available)
            self.assertNotIn("run_conformance", available)
            self.assertNotIn("run_simulation", available)

    def _hardware_repair_state(self, software="int kernel(void) { return cfu_compute(1, 2); }\n"):
        (self.root / self.sw).write_text(software)
        state = dict(self.state)
        box = services._toolbox_for_stage(state, "implementation")
        state["gates"] = {
            **self.state["gates"],
            "implementation_hw": {"ok": True, "applied": True,
                "candidate_hash": intrinsics._hardware_candidate_hash(state, box)},
            "implementation_sw": {"ok": True, "applied": True,
                "candidate_hash": intrinsics._software_candidate_hash(
                    box, services._software_targets(state, box)),
                "isa_hash": intrinsics._isa_digest(state)},
        }
        self._approve_conformance(state, box)
        failure = observations.build_failure_observation("hw_build", {"returncode": 1},
            stderr="AgentCfu.scala:7: error: WIDTH MISMATCH")
        state["validation_results"] = [{"failure_observation": failure}]
        diagnosis = {"failure_ids": [failure["id"]],
            "observed_errors": ["AgentCfu.scala:7 WIDTH MISMATCH"],
            "root_cause": "incorrect hardware width", "repair_mapping": ["fix hardware width"],
            "regression_risks": ["preserve the existing software ABI"]}
        return state, diagnosis

    def _hardware_repair_turns(self, diagnosis):
        return [
            ("submit_repair_diagnosis", diagnosis),
            ("edit_hardware", {"edits": [{"path": self.hw, "search": "class AgentCfu {}",
                "replace": "class AgentCfu { int changed; }"}]}),
            ("submit_stage", {"delivery": "fixed HW; unchanged validated CFU kernel"}),
        ]

    def test_hardware_only_repair_reuses_hash_verified_prior_actual_cfu_software(self):
        state, diagnosis = self._hardware_repair_state()
        original_sw = (self.root / self.sw).read_bytes()
        client = _RoundClient(self._hardware_repair_turns(diagnosis))
        # Keep the production CFU-usage check: unchanged bytes must really call it.
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(state)
        self.assertEqual(client.turns, [])
        self.assertEqual((self.root / self.sw).read_bytes(), original_sw)
        self.assertEqual(update["changed_sw_files"], [])
        self.assertEqual(update["changed_design_files"], [self.hw])
        hw_gate = update["gates"]["implementation_hw"]
        sw_gate = update["gates"]["implementation_sw"]
        self.assertTrue(hw_gate["ok"])
        self.assertFalse(hw_gate["reused"])
        self.assertTrue(sw_gate["ok"])
        self.assertTrue(sw_gate["reused"])
        self.assertEqual(sw_gate["candidate_hash"], state["gates"]["implementation_sw"]["candidate_hash"])
        self.assertEqual(sw_gate["isa_hash"], state["gates"]["implementation_sw"]["isa_hash"])
        self.assertNotEqual(hw_gate["candidate_hash"], state["gates"]["implementation_hw"]["candidate_hash"])

    def test_hardware_only_repair_may_revise_the_oracle(self):
        """The oracle is editable in a design repair, not frozen.

        Regression class (run 20261003-090854-136683-d98e5be9): the model noticed
        its oracle was blind, tried to strengthen it during a design repair, and
        the host refused with "conformance delivery is frozen".
        """
        state, diagnosis = self._hardware_repair_state()
        before = self._conformance_bytes()
        revised = json.loads(self.oracle)
        revised["commands"][0]["data"]["count"] = 8
        turns = self._hardware_repair_turns(diagnosis)
        turns.insert(1, ("write_conformance", {"oracle": json.dumps(revised)}))
        client = _RoundClient(turns)
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(state)
        self.assertEqual(client.turns, [])
        self.assertNotEqual(self._conformance_bytes(), before)
        self.assertEqual(update["conformance_delivery"]["oracle"], revised)
        # The delivered C is the host's rendering of the revised oracle.
        self.assertEqual(
            (self.root / update["conformance_delivery"]["source_path"]).read_text(),
            conformance.render_conformance(state["isa_spec"], revised),
        )

    def test_repair_round_cannot_submit_without_a_diagnosis(self):
        """The oracle may be staged first; the repair still has to justify itself.

        Regression class (run 20261005-073745): the already-correct oracle was
        refused because the diagnosis had not been filed yet, so the round was
        spent on tool ordering. Staging is allowed now, but the stage gate keeps
        the hard requirement.
        """
        state, _diagnosis = self._hardware_repair_state()
        state["validation_results"] = [{
            "rung": "hw_build", "returncode": 1, "command": "sbt GenAgentCfu",
            "stderr_tail": "WIDTH MISMATCH",
            "failure_observation": observations.build_failure_observation(
                "hw_build", {"returncode": 1}, stderr="WIDTH MISMATCH"),
        }]
        before = self._conformance_bytes()
        revised = json.loads(self.oracle)
        revised["commands"][0]["data"]["count"] = 8
        client = _RoundClient([
            ("write_conformance", {"oracle": json.dumps(revised)}),
            ("submit_stage", {"delivery": "revised oracle; no diagnosis filed"}),
            ("submit_stage", {"delivery": "still no diagnosis"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(state)
            self.assertFalse(update["gates"]["implementation_hw"]["ok"])
            self.assertIn("submit_repair_diagnosis", " ".join(update["errors"]))
        self.assertEqual(client.turns, [])
        self.assertEqual(self._conformance_bytes(), before)

    def test_oracle_only_repair_reuses_healthy_hw_and_sw_without_dummy_edits(self):
        state, diagnosis = self._hardware_repair_state()
        state["repair_scope"] = "conformance"
        failure = observations.build_failure_observation("hw_conformance", {"returncode": 1},
            stderr="error: oracle test harness needs another bounded case")
        state["validation_results"] = [{"failure_observation": failure}]
        diagnosis = {**diagnosis, "failure_ids": [failure["id"]],
                     "observed_errors": ["oracle test harness needs another bounded case"],
                     "root_cause": "oracle case coverage needs repair",
                     "repair_mapping": ["add a bounded independent oracle case"]}
        before = {path: (self.root / path).read_bytes() for path in (self.hw, self.sw)}
        revised = json.loads(self.oracle)
        revised["commands"][0]["data"]["count"] = 8
        client = _RoundClient([
            ("submit_repair_diagnosis", diagnosis),
            ("write_conformance", {"oracle": json.dumps(revised)}),
            ("submit_stage", {"delivery": "repaired oracle; healthy HW and SW untouched"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(state)
        self.assertEqual(client.turns, [])
        self.assertEqual({path: (self.root / path).read_bytes() for path in before}, before)
        self.assertEqual(update["changed_design_files"], [])
        self.assertEqual(update["changed_sw_files"], [])
        self.assertTrue(update["gates"]["implementation_hw"]["reused"])
        self.assertTrue(update["gates"]["implementation_sw"]["reused"])
        self.assertEqual(update["conformance_delivery"]["oracle"], revised)
        self.assertNotEqual(update["conformance_delivery"]["oracle_sha"],
                            state["conformance_delivery"]["oracle_sha"])
        box = services._toolbox_for_stage(state, "implementation")
        self.assertEqual(conformance_delivery.inspect_delivery({**state, **update}, box)[1], [])

    def test_hardware_only_repair_cannot_reuse_invalid_prior_software_gate(self):
        for invalid in ({"candidate_hash": "wrong-source-hash"}, {"isa_hash": "wrong-isa-hash"},
                        {"stale": True}, {"applied": False}, None):
            with self.subTest(prior_gate=invalid):
                state, diagnosis = self._hardware_repair_state()
                if invalid is None:
                    state["gates"].pop("implementation_sw")
                else:
                    state["gates"]["implementation_sw"].update(invalid)
                original_sw = (self.root / self.sw).read_bytes()
                client = _RoundClient(self._hardware_repair_turns(diagnosis) + [
                    ("submit_stage", {"delivery": "unchanged software still has invalid reuse evidence"}),
                ])
                with mock.patch.object(llm_step, "_llm", return_value=client), \
                     mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
                    update = unified.implement_stage3(state)
                    self.assertFalse(update["gates"]["implementation_sw"]["ok"])
                    self.assertIn("no software target was edited", " ".join(update["errors"]))
                self.assertEqual(client.turns, [])
                self.assertEqual((self.root / self.sw).read_bytes(), original_sw)
                self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")

    def test_hardware_only_repair_cannot_reuse_hash_matched_non_cfu_software(self):
        for software in ("int kernel(void) { return 1; }\n",
                         "int kernel(void) { cfu_enable(); return 1; } // cfu_compute(1, 2)\n"):
            with self.subTest(software=software):
                state, diagnosis = self._hardware_repair_state(software)
                client = _RoundClient(self._hardware_repair_turns(diagnosis) + [
                    ("submit_stage", {"delivery": "still using scalar software"}),
                ])
                with mock.patch.object(llm_step, "_llm", return_value=client), \
                     mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
                    update = unified.implement_stage3(state)
                    self.assertFalse(update["gates"]["implementation_sw"]["ok"])
                    self.assertIn("does not exercise the CFU", " ".join(update["errors"]))
                self.assertEqual((self.root / self.sw).read_text(), software)
                self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")

    def test_broken_stage_audit_restores_all_edits(self):
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "HW and SW ready"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"), \
             mock.patch.object(services, "_write_hardware_artifact", side_effect=OSError("audit failed")):
            with self.assertRaisesRegex(OSError, "audit failed"):
                unified.implement_stage3(self.state)
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertEqual((self.root / self.sw).read_text(), "int kernel(void) { return 1; }\n")
        self.assertTrue(all(data is None for data in self._conformance_bytes().values()))

    def test_rejected_submission_restores_all_cross_domain_writes(self):
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }", "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "premature"}),
            # The first rejection is correctable, so the run only fails after the
            # retry is rejected too.
            ("submit_stage", {"delivery": "still premature"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(False, "no intrinsic")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
            self.assertFalse(update["gates"]["implementation_hw"]["ok"])
            self.assertFalse(update["gates"]["implementation_sw"]["ok"])
            self.assertIn("gate rejected", " ".join(update["errors"]))
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertEqual((self.root / self.sw).read_text(), "int kernel(void) { return 1; }\n")
        self.assertTrue(all(data is None for data in self._conformance_bytes().values()))

    def test_the_stage3_prompt_requires_applied_edits_and_a_summary_delivery(self):
        # Observed twice (crc32-run-0001/0002): the prompt's only emphasis was
        # that prose cannot pass, so the model pasted the whole design into
        # submit_stage and never called an editing tool. Both runs died there.
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "applied the design and the kernel"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        prompt = "\n".join(str(item.get("content", "")) for item in client.messages[0])
        for expected in (
            "edit_hardware", "edit_software", "write_conformance(oracle)", "submit_stage(delivery)",
            "Apply the change to the workspace files with the editing tools",
            "is a summary of what you changed",
            "reports no applied change",
            "not applied",
        ):
            self.assertIn(expected, prompt)
        # The delivered prompt must not resurrect the legacy block protocol.
        self.assertNotIn("FILE_BEGIN", prompt)
        self.assertNotIn("the harness applies it", prompt)
        self.assertTrue(update["gates"]["implementation_hw"]["ok"])

    def test_the_crc32_failure_mode_recovers_after_the_gate_explains_itself(self):
        # The exact crc32-run-0002 sequence: read, then paste the whole design
        # into submit_stage without editing anything. It must not end the run —
        # the reason goes back, the model applies the edits, and the retry passes.
        client = _RoundClient([
            ("read_file", {"path": self.hw}),
            ("submit_stage", {"delivery": "## 1. Hardware (FULL FILE, write verbatim)\n"
                                          "```scala\nclass AgentCfu {}\n```"}),
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "applied edit_hardware + edit_software"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        self.assertIn("changed", (self.root / self.hw).read_text())
        self.assertIn("cfu_compute", (self.root / self.sw).read_text())
        self.assertTrue(update["gates"]["implementation_hw"]["ok"])
        # The model saw the gate's reason before the corrective round.
        recovery_prompt = "\n".join(
            str(item.get("content", "")) for item in client.messages[2]
        )
        self.assertIn("no hardware target was edited", recovery_prompt)

    def test_a_submission_batched_with_edits_is_applied_before_the_gate(self):
        # Observed 2026-09-30: the implementation stage ended with
        # "submit_stage must be the sole tool call" because the model put its
        # submission in the same turn as another call. The batch must be run,
        # not fatal.
        client = _BatchClient([[
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "applied all three, submitting in the same turn"}),
        ]])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        self.assertIn("changed", (self.root / self.hw).read_text())
        self.assertIn("cfu_compute", (self.root / self.sw).read_text())
        self.assertTrue(update["gates"]["implementation_hw"]["ok"])

    def test_the_batched_submission_failure_recovers_end_to_end(self):
        # The literal crc32-run-0002 sequence: read, then submit a full-file
        # delivery batched with another read. It must reach the gate, report the
        # missing edits, and let the model apply them in the corrective round.
        client = _BatchClient([
            [("read_file", {"path": self.hw}),
             ("submit_stage", {"delivery": "FULL FILE, write verbatim:\nclass AgentCfu {}"})],
            [("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                                "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
             ("edit_hardware", {"edits": [{"path": self.hw,
               "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]})],
            [("write_conformance", {"oracle": self.oracle}),
             ("submit_stage", {"delivery": "applied all three after the gate explained itself"})],
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        self.assertIn("changed", (self.root / self.hw).read_text())
        self.assertIn("cfu_compute", (self.root / self.sw).read_text())
        self.assertTrue(update["gates"]["implementation_hw"]["ok"])
        # The model saw the gate's reason before the corrective round.
        recovery = " ".join(str(item.get("content", "")) for item in client.messages[1])
        self.assertIn("no hardware target was edited", recovery)

    def test_a_rejected_gate_reopens_the_edit_budgets(self):
        # Run-0003: after the CFU-usage check rejected the kernel, every later
        # `edit_software` answered "budget exhausted", so the demanded change was
        # impossible. The gate now re-opens the local budgets on rejection.
        client = _RoundClient([
            ("read_file", {"path": self.hw}),
            ("submit_stage", {"delivery": "nothing applied"}),
            ("submit_stage", {"delivery": "still nothing"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(unified.ImplementationToolset, "grant_correction",
                               autospec=True) as grant, \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
            self.assertFalse(update["gates"]["implementation_hw"]["ok"])
            self.assertIn("no hardware target was edited", " ".join(update["errors"]))
        self.assertEqual(grant.call_count, 2)

    def test_a_cfu_usage_rejection_is_answerable_in_the_corrective_round(self):
        # Run-0003: the submission that added the CFU intrinsic calls was refused
        # at the edit budget, so the gate's own rejection ("the accelerated path
        # does not exercise the CFU") could not be answered. With the refill the
        # same sequence completes.
        client = _RoundClient([
            ("edit_software", {"path": self.sw, "search": "int kernel(void) { return 1; }",
                               "replace": "int kernel(void) { return 2; }  /* step 1 */"}),
            ("edit_hardware", {"edits": [{"path": self.hw,
              "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }"}]}),
            ("write_conformance", {"oracle": self.oracle}),
            ("submit_stage", {"delivery": "first attempt without the intrinsic"}),
            ("edit_software", {"path": self.sw,
                               "search": "int kernel(void) { return 2; }  /* step 1 */",
                               "replace": "int kernel(void) { return cfu_compute(1, 2); }"}),
            ("submit_stage", {"delivery": "kernel now calls the intrinsic"}),
        ])
        # First submission, corrective submission, and final accepted audit.
        checks = [(False, "the CFU build does not call any CFU datapath intrinsic"),
                  (True, ""), (True, "")]
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(unified.ImplementationToolset, "max_software_edits", 1), \
             mock.patch.object(intrinsics, "_software_exercises_cfu", side_effect=checks) as usage_check, \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
        self.assertEqual(usage_check.call_count, 3)
        self.assertIn("cfu_compute", (self.root / self.sw).read_text())
        self.assertTrue(update["gates"]["implementation_sw"]["ok"])
        # The refused-intrinsic reason reached the model before the corrective edit.
        recovery = " ".join(str(item.get("content", "")) for item in client.messages[4])
        self.assertIn("does not call any CFU datapath intrinsic", recovery)

    def test_a_delivery_without_applied_edits_is_rejected_with_a_reason(self):
        # Observed twice (crc32-run-0001/0002): the model pasted the whole design
        # into submit_stage and never called an edit tool. The gate must say so
        # instead of failing with a bare "rejected".
        client = _RoundClient([
            ("submit_stage", {"delivery": "here is the full file:\nclass AgentCfu {}"}),
            ("submit_stage", {"delivery": "and here it is again"}),
        ])
        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.sw + ".h"):
            update = unified.implement_stage3(self.state)
            self.assertFalse(update["gates"]["implementation_hw"]["ok"])
            self.assertIn("no hardware target was edited", " ".join(update["errors"]))
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")


if __name__ == "__main__":
    unittest.main()
