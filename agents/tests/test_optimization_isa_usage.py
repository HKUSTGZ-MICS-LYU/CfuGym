"""Both implementation modes share the ISA-extension usage contract, offline."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import stage_services
from agents.cfu_design_agent.stages.implement import intrinsics, repair_scope, submission


class OptimizationIsaUsageTest(unittest.TestCase):
    def setUp(self):
        self.state = {"optimization_plan": {"target": "extend_isa"},
                      "isa_added_symbols": ["cfu_batch"],
                      "isa_spec": {"commands": [
                          {"name": "compute", "function_id": 1, "kind": "compute"},
                          {"name": "batch", "function_id": 2, "kind": "control"}]}}
        self.box = mock.MagicMock()
        self.box.workdir.__truediv__.return_value.is_file.return_value = True
        self.targets = [("kernel.c", "")]

    def usage(self, text, *, state=None):
        self.box.read_text.return_value = text
        return intrinsics._software_usage_check(state or self.state, self.box, self.targets)

    def test_only_old_datapath_does_not_exercise_added_instruction(self):
        ok, reason = self.usage("int k() { return cfu_compute(1, 2); }")
        self.assertFalse(ok)
        self.assertIn("newly added", reason)

    def test_new_control_plus_old_datapath_is_valid(self):
        self.assertTrue(self.usage("int k() { cfu_batch(); return cfu_compute(1, 2); }")[0])
        self.assertFalse(self.usage("int k() { cfu_batch(); return 0; }")[0])

    def test_strings_comments_and_inactive_branches_are_not_usage(self):
        for hidden in ('const char *s = "cfu_batch()";', '/* cfu_batch(); */',
                       '#if 0\ncfu_batch();\n#endif',
                       '#ifndef USE_AGENT_CFU\ncfu_batch();\n#endif'):
            with self.subTest(hidden=hidden):
                self.assertFalse(self.usage(hidden + '\nint k() { return cfu_compute(1,2); }')[0])

    def test_new_instruction_after_many_old_calls_is_found(self):
        self.assertTrue(self.usage('cfu_compute(1,2);\n' * 40 + 'cfu_batch();')[0])

    def test_regular_round_does_not_require_added_instruction(self):
        state = {**self.state, "optimization_plan": {"target": "implement"}}
        self.assertTrue(self.usage('int k() { return cfu_compute(1,2); }', state=state)[0])

    def test_native_submission_runs_the_same_extension_check(self):
        tools = SimpleNamespace(repair_route={}, diagnosis_required=False,
                                hardware_session=object(), software_session=object())
        for text, accepted in (("int k() { return cfu_compute(1,2); }", False),
                               ("int k() { cfu_batch(); return cfu_compute(1,2); }", True)):
            with self.subTest(text=text), \
                 mock.patch.object(stage_services, "benchmark_integrity_problems", return_value=[]), \
                 mock.patch.object(stage_services, "hardware_integrity_problems", return_value=[]), \
                 mock.patch.object(stage_services, "_software_targets", return_value=self.targets), \
                 mock.patch.object(repair_scope, "changed_paths", side_effect=[set(), {"kernel.c"}]):
                self.box.read_text.return_value = text
                ok, reason = submission.audit_delivery(
                    self.state, self.box, "summary", toolsets=[tools], oracle_session=None,
                    hardware_targets=frozenset({"hw.scala"}), software_targets=frozenset({"kernel.c"}),
                    need_hw=False, need_sw=True)
                self.assertEqual(ok, accepted, reason)


if __name__ == "__main__":
    unittest.main()
