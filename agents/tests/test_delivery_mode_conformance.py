"""Offline delivery-mode ordering, domain boundaries and rollback contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from agents.cfu_design_agent.runtime import conformance_delivery, llm_step, stage_services, tools
from agents.cfu_design_agent.runtime.policy import AccessPolicy, PolicyError, StagePolicy
from agents.cfu_design_agent.stages import implement
from agents.cfu_design_agent.stages.implement import conformance, delivery, intrinsics
from agents.tests.common import REPO_ROOT  # noqa: F401 (network guard)


class DeliveryModeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = "agents/generated/runs/no-tool-unit"
        self.workspace = self.run + "/workspace/software"
        self.source = self.workspace + "/kernel.c"
        self.header = self.workspace + "/cfu_intrinsics.h"
        self.state = {"workdir": str(self.root), "run_root": self.run,
                      "task": "add", "agent_tools": False, "apply_patch": True,
                      "run_commands": False,
                      "benchmark_project": {"kind": "generated_benchmark",
                                            "workspace": self.workspace, "source_path": self.source},
                      "isa_spec": {"commands": [{"name": "add", "function_id": 1}]}}
        policy = AccessPolicy(version="1", deny=(), write_roots=(self.run,),
                              stages={"implementation": StagePolicy(write=(self.run + "/**",), commands=())},
                              commands={}, read=(self.run + "/**",))
        self.box = tools.RepoToolbox(self.root, policy=policy, stage="implementation",
                                     run_root=self.run, apply_patch=True)
        self.hardware = list(tools.design_workspace_paths(self.run).values())
        self.paths = [*self.hardware, self.source, self.header,
                      *conformance_delivery.delivery_paths(self.state).values()]
        for index, path in enumerate(self.paths):
            self.box.write_text(path, f"entry-{index}\n")
        self.oracle = {"commands": [{"name": "add", "reference": "a + b",
                                    "cases": [{"args": [1, 2]}]}]}
        self.before = self.bytes()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box).start()
        mock.patch.object(llm_step, "_llm", return_value=SimpleNamespace(available=False)).start()

    def bytes(self):
        return {path: (self.root / path).read_bytes() if (self.root / path).is_file() else None
                for path in self.paths}

    def hardware_update(self, state):
        self.box.write_text(self.hardware[0], "new hardware\n")
        return {"hardware_plan": "accepted hardware", "changed_design_files": [self.hardware[0]],
                "errors": [], "gates": {"implementation_hw": {"ok": True, "applied": True}}}

    def software_update(self, state):
        self.assertEqual(state["hardware_plan"], "accepted hardware")
        self.assertTrue(state["gates"]["implementation_hw"]["ok"])
        self.box.write_text(self.source, "new software\n")
        self.box.write_text(self.header, "new header\n")
        return {"software_plan": "accepted software", "changed_sw_files": [self.source],
                "errors": [], "gates": {**state["gates"],
                    "implementation_sw": {"ok": True, "applied": True}}}

    def test_hardware_then_software_then_oracle_see_updated_state(self):
        order = []
        def hw(state):
            order.append("hw")
            return self.hardware_update(state)
        def sw(state):
            order.append("sw")
            return self.software_update(state)
        def oracle(state):
            order.append("oracle")
            self.assertEqual(state["hardware_plan"], "accepted hardware")
            self.assertEqual(state["software_plan"], "accepted software")
            return {"conformance_delivery": {"accepted": True}}
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=hw), \
                mock.patch.object(implement.software, "implement_sw", side_effect=sw), \
                mock.patch.object(conformance, "write_conformance", side_effect=oracle):
            result = implement.implement(self.state)
        self.assertEqual(order, ["hw", "sw", "oracle"])
        self.assertTrue(result["conformance_delivery"]["accepted"])
        self.assertNotEqual(self.bytes(), self.before)

    def test_software_rejection_restores_all_entry_bytes(self):
        def rejected(state):
            update = self.software_update(state)
            update["gates"]["implementation_sw"].update(ok=False, errors=["bad kernel"])
            update["errors"] = ["bad kernel"]
            return update
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=rejected), \
                mock.patch.object(conformance, "write_conformance") as oracle:
            result = implement.implement(self.state)
        oracle.assert_not_called()
        self.assertEqual(self.bytes(), self.before)
        for gate in result["gates"].values():
            self.assertFalse(gate["ok"])
            self.assertFalse(gate["applied"])
            self.assertTrue(gate["rolled_back"])
        self.assertEqual(result["changed_design_files"], [])
        self.assertEqual(result["changed_sw_files"], [])

    def test_hardware_rejection_skips_software_and_oracle(self):
        def rejected(state):
            update = self.hardware_update(state)
            update["gates"]["implementation_hw"]["ok"] = False
            return update
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=rejected), \
                mock.patch.object(implement.software, "implement_sw") as sw, \
                mock.patch.object(conformance, "write_conformance") as oracle:
            implement.implement(self.state)
        sw.assert_not_called()
        oracle.assert_not_called()
        self.assertEqual(self.bytes(), self.before)

    def test_model_failure_after_source_delivery_restores_everything(self):
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm", side_effect=RuntimeError("provider failed")):
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                implement.implement(self.state)
        self.assertEqual(self.bytes(), self.before)

    def test_source_write_failure_restores_hw_sw_header_and_pair(self):
        real_write = self.box.write_text
        def fail_source(path, text):
            result = real_write(path, text)
            if path == conformance_delivery.delivery_paths(self.state)["source_path"]:
                raise OSError("source write failed after mutation")
            return result
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm", return_value=(
                    SimpleNamespace(text=yaml.safe_dump(self.oracle)), "raw")), \
                mock.patch.object(self.box, "write_text", side_effect=fail_source):
            with self.assertRaisesRegex(OSError, "source write failed"):
                implement.implement(self.state)
        self.assertEqual(self.bytes(), self.before)

    def test_invalid_oracle_rejection_restores_absent_files_and_headers(self):
        for path in self.paths:
            (self.root / path).unlink()
        before = self.bytes()
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm", return_value=(SimpleNamespace(text="[]"), "raw")):
            result = implement.implement(self.state)
        self.assertTrue(result["errors"])
        self.assertEqual(self.bytes(), before)
        self.assertFalse(result["gates"]["implementation_hw"]["ok"])

    def healthy_gates(self):
        self.box.write_text(self.source, "int kernel(void) { return cfu_add(1, 2); }\n")
        self.before = self.bytes()
        self.state["gates"] = {
            "implementation_hw": {"ok": True, "applied": True,
                "candidate_hash": intrinsics._hardware_candidate_hash(self.state, self.box)},
            "implementation_sw": {"ok": True, "applied": True,
                "candidate_hash": intrinsics._software_candidate_hash(
                    self.box, stage_services._software_targets(self.state, self.box)),
                "isa_hash": intrinsics._isa_digest(self.state)}}

    def test_conformance_only_reuses_only_hash_verified_healthy_gates(self):
        self.healthy_gates()
        self.state["repair_scope"] = "conformance"
        with mock.patch.object(implement.hardware, "implement_hw") as hw, \
                mock.patch.object(implement.software, "implement_sw") as sw, \
                mock.patch.object(llm_step, "_complete_llm", return_value=(
                    SimpleNamespace(text=yaml.safe_dump(self.oracle)), "raw")):
            result = implement.implement(self.state)
        hw.assert_not_called()
        sw.assert_not_called()
        self.assertEqual(result["repair_scope"], "")
        from agents.cfu_design_agent import isa
        self.assertEqual((self.root / self.header).read_text(),
                         isa.render_cfu_intrinsics(self.state["isa_spec"]))
        self.assertEqual(conformance_delivery.inspect_delivery(
            {**self.state, **result}, self.box)[1], [])
        self.assertEqual((self.root / self.source).read_bytes(), self.before[self.source])

    def test_approved_oracle_is_reused_when_no_repair_is_pending(self):
        session = conformance_delivery.ConformanceDelivery(self.state, self.box, self.header)
        record, errors = session.write(yaml.safe_dump(self.oracle))
        self.assertEqual(errors, [])
        session.commit()
        self.state["conformance_delivery"] = record
        pair = {path: (self.root / path).read_bytes()
                for path in conformance_delivery.delivery_paths(self.state).values()}
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm") as oracle_model:
            result = implement.implement(self.state)
        oracle_model.assert_not_called()
        self.assertEqual(result["conformance_delivery"], record)
        self.assertEqual(pair, {path: (self.root / path).read_bytes() for path in pair})

    def approved(self) -> dict:
        """Commit one valid oracle and return the host-approved record."""
        session = conformance_delivery.ConformanceDelivery(self.state, self.box, self.header)
        record, errors = session.write(yaml.safe_dump(self.oracle))
        self.assertEqual(errors, [])
        session.commit()
        self.state["conformance_delivery"] = record
        return record

    def test_repair_can_strengthen_the_approved_oracle(self):
        """A repair round re-offers the oracle so a blind contract can be fixed.

        Regression class (run 20261003-090854-136683-d98e5be9): the oracle was
        frozen, so every design repair kept the same nine checks that never
        observed a compression result.
        """
        record = self.approved()
        self.state["repair_scope"] = "design"
        revised = {"commands": [{"name": "add", "reference": "a + b",
                                 "cases": [{"args": [3, 4]}]}]}
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm", return_value=(
                    SimpleNamespace(text=yaml.safe_dump(revised)), "raw")) as oracle_model:
            result = implement.implement(self.state)
        oracle_model.assert_called_once()
        self.assertFalse(result.get("errors"), result.get("errors"))
        self.assertEqual(result["conformance_delivery"]["oracle"], revised)
        paths = conformance_delivery.delivery_paths(self.state)
        self.assertNotEqual((self.root / paths["oracle_path"]).read_bytes(),
                            yaml.safe_dump(record).encode("utf-8"))

    def test_rejected_revision_keeps_the_approved_oracle_and_the_repair(self):
        """A bad revision must not fail the design repair it was offered for."""
        record = self.approved()
        self.state["repair_scope"] = "design"
        pair = {path: (self.root / path).read_bytes()
                for path in conformance_delivery.delivery_paths(self.state).values()}
        blind = {"commands": [{"name": "add", "reference": "a == a",
                               "cases": [{"args": [1, 2]}]}]}
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update), \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update), \
                mock.patch.object(llm_step, "_complete_llm", return_value=(
                    SimpleNamespace(text=yaml.safe_dump(blind)), "raw")) as oracle_model:
            result = implement.implement(self.state)
        oracle_model.assert_called_once()
        self.assertFalse(result.get("errors"), result.get("errors"))
        self.assertEqual(result["conformance_delivery"]["oracle"], record["oracle"])
        self.assertEqual(pair, {path: (self.root / path).read_bytes() for path in pair})
        self.assertTrue(any("kept the approved oracle" in line
                            for line in result.get("iteration_history", [])))

    def test_stale_hardware_gate_requires_hw_but_reuses_healthy_software(self):
        self.healthy_gates()
        self.state["repair_scope"] = "conformance"
        self.state["gates"]["implementation_hw"]["candidate_hash"] = "stale"
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update) as hw, \
                mock.patch.object(implement.software, "implement_sw") as sw, \
                mock.patch.object(conformance, "write_conformance", return_value={}):
            implement.implement(self.state)
        hw.assert_called_once()
        sw.assert_not_called()

    def test_hash_matching_scalar_software_is_not_reusable(self):
        self.healthy_gates()
        self.box.write_text(self.source, "int kernel(void) { return 3; }\n")
        self.state["gates"]["implementation_sw"]["candidate_hash"] = intrinsics._software_candidate_hash(
            self.box, stage_services._software_targets(self.state, self.box))
        self.state["repair_scope"] = "conformance"
        with mock.patch.object(implement.hardware, "implement_hw") as hw, \
                mock.patch.object(implement.software, "implement_sw", return_value={}) as sw, \
                mock.patch.object(conformance, "write_conformance", return_value={}):
            implement.implement(self.state)
        hw.assert_not_called()
        sw.assert_called_once()

    def test_standalone_author_header_failure_restores_header_and_pair(self):
        real_write = self.box.write_text
        def fail_header(path, text):
            result = real_write(path, text)
            if path == self.header:
                raise OSError("header failure after mutation")
            return result
        with mock.patch.object(llm_step, "_complete_llm", return_value=(
                SimpleNamespace(text=yaml.safe_dump(self.oracle)), "raw")), \
                mock.patch.object(self.box, "write_text", side_effect=fail_header):
            with self.assertRaisesRegex(OSError, "header failure"):
                conformance.write_conformance(self.state)
        self.assertEqual(self.bytes(), self.before)

    def test_conformance_scope_without_healthy_gates_requires_both_sources(self):
        self.state["repair_scope"] = "conformance"
        with mock.patch.object(implement.hardware, "implement_hw", side_effect=self.hardware_update) as hw, \
                mock.patch.object(implement.software, "implement_sw", side_effect=self.software_update) as sw, \
                mock.patch.object(conformance, "write_conformance", return_value={}):
            implement.implement(self.state)
        hw.assert_called_once()
        sw.assert_called_once()

    def test_dry_run_and_no_apply_tool_models_fallback_without_file_writes(self):
        for flags in ({"dry_run": True}, {"apply_patch": False}):
            with self.subTest(flags=flags), \
                    mock.patch.object(llm_step, "_llm", return_value=SimpleNamespace(available=True)), \
                    mock.patch.object(implement.unified, "implement_with_tools") as native, \
                    mock.patch.object(implement.hardware, "implement_hw", return_value={}), \
                    mock.patch.object(implement.software, "implement_sw", return_value={}), \
                    mock.patch.object(llm_step, "_complete_llm") as oracle_model:
                implement.implement({**self.state, "agent_tools": True, "run_commands": True, **flags})
                native.assert_not_called()
                oracle_model.assert_not_called()
                self.assertEqual(self.bytes(), self.before)

    def test_no_apply_rejected_proposal_still_plans_software(self):
        state = {**self.state, "apply_patch": False}
        with mock.patch.object(implement.hardware, "implement_hw", return_value={
                "gates": {"implementation_hw": {"ok": False, "applied": False}},
                "errors": ["application disabled"]}), \
                mock.patch.object(implement.software, "implement_sw", return_value={}) as sw, \
                mock.patch.object(llm_step, "_complete_llm") as oracle_model:
            implement.implement(state)
        sw.assert_called_once()
        oracle_model.assert_not_called()
        self.assertEqual(self.bytes(), self.before)

    def test_snapshot_rejects_noncurrent_denied_or_symlink_paths_before_model(self):
        for bad in ("agents/generated/runs/other/workspace/kernel.c", "private/kernel.c"):
            state = deepcopy(self.state)
            state["benchmark_project"]["source_path"] = bad
            with self.subTest(path=bad), mock.patch.object(implement.hardware, "implement_hw") as hw:
                with self.assertRaises(PolicyError):
                    implement.implement(state)
                hw.assert_not_called()
        full = self.root / self.header
        full.unlink()
        full.symlink_to(self.root / self.source)
        with self.assertRaises(PolicyError):
            implement.implement(self.state)


class DeliveryDomainTest(DeliveryModeTest):
    @staticmethod
    def response(protocol, path):
        if protocol == "file":
            return f"FILE_BEGIN {path}\nreplacement\nFILE_END"
        if protocol == "edit":
            return f"EDIT_BEGIN {path}\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\nEDIT_END"
        return (f"PATCH_BEGIN\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\nPATCH_END")

    def test_all_protocols_reject_cross_domain_and_host_owned_targets(self):
        pair = conformance_delivery.delivery_paths(self.state)
        for domain, path in (("hardware", self.source), ("software", self.hardware[0]),
                             ("software", self.header), ("software", pair["source_path"]),
                             ("hardware", pair["oracle_path"]),
                             ("software", self.workspace + "/unexpected.c")):
            for protocol in ("edit", "file", "patch"):
                with self.subTest(domain=domain, path=path, protocol=protocol):
                    result = delivery._apply_agent_output(self.state, self.response(protocol, path),
                                                          toolbox=self.box, domain=domain)
                    self.assertFalse(result["applied"])
                    self.assertTrue(result["errors"])
                    self.assertEqual(self.bytes(), self.before)

    def test_legal_tracked_hardware_alias_and_software_source_validate(self):
        tracked = next(iter(tools.design_workspace_paths(self.run)))
        for domain, path in (("hardware", tracked), ("software", self.source)):
            with self.subTest(domain=domain), mock.patch.object(
                    self.box, "apply_file_blocks", return_value={"returncode": 0}) as apply:
                result = delivery._apply_agent_output(self.state, self.response("file", path),
                                                      toolbox=self.box, domain=domain)
                self.assertTrue(result["gate_ok"])
                apply.assert_called_once()

    def test_legal_patch_targets_validate_and_reach_application(self):
        tracked = next(iter(tools.design_workspace_paths(self.run)))
        for domain, path in (("hardware", tracked), ("software", self.source)):
            with self.subTest(domain=domain), mock.patch.object(
                    self.box, "apply_patch_text", return_value={"returncode": 0}) as apply:
                answer = self.response("patch", path)
                result = delivery._apply_agent_output(self.state, answer, toolbox=self.box, domain=domain)
                self.assertTrue(result["gate_ok"])
                apply.assert_called_once_with(tools.extract_patch(answer))

    def test_union_compatibility_still_denies_host_owned_files(self):
        result = delivery._apply_agent_output(self.state, self.response("file", self.header), toolbox=self.box)
        self.assertFalse(result["gate_ok"])
        self.assertEqual(self.bytes(), self.before)

    def test_mixed_legal_and_illegal_blocks_validate_before_any_apply(self):
        answer = self.response("file", self.source) + "\n" + self.response("file", self.header)
        with mock.patch.object(self.box, "apply_file_blocks") as apply:
            result = delivery._apply_agent_output(self.state, answer, toolbox=self.box, domain="software")
        apply.assert_not_called()
        self.assertFalse(result["gate_ok"])


if __name__ == "__main__":
    unittest.main()
