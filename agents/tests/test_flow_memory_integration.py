"""Cross-node run memory survives stages/rewinds, never transaction failures."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import agent_tools, flow_memory, llm_step
from agents.cfu_design_agent.runtime.policy import AccessPolicy, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox


class FlowMemoryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_root = "agents/generated/runs/memory-integration"
        (self.root / self.run_root).mkdir(parents=True)
        self.path = "src/platform.scala"
        (self.root / "src").mkdir()
        (self.root / self.path).write_text("stable platform semantics\n")
        self.policy = AccessPolicy(version="test", deny=(), write_roots=(),
            stages={"analysis": StagePolicy(write=(), commands=())}, commands={},
            read=("src/**", "agents/generated/runs/**"), visibility="standard")
        self.toolbox = RepoToolbox(self.root, policy=self.policy, run_root=self.run_root)
        self.state = {"workdir": str(self.root), "run_id": "memory-integration",
                      "run_root": self.run_root, "visibility": "standard", "isa_spec": {}}

    def test_analysis_to_isa_to_implementation_and_rewind(self):
        def analyze_workload(state):
            agent_tools.ReadOnlyToolset(self.toolbox).run("read_file", {"path": self.path})
            return {"workload_analysis": "chosen operation: byte CRC update",
                    "gates": {"analysis": {"ok": True}}}

        with mock.patch.object(instrument.dependencies, "input_fingerprints", return_value={}), \
             mock.patch("agents.cfu_design_agent.runtime.stage_services._toolbox_for_stage", return_value=self.toolbox):
            update = instrument._wrap_node(analyze_workload)(self.state)
        state = {**self.state, **update}
        isa_prompt = llm_step._with_run_memory(state, self.toolbox, "design_contract", "ISA task")
        self.assertIn("chosen operation", isa_prompt)
        self.assertIn("stable platform semantics", isa_prompt)
        state.update(isa_spec={"commands": [{"name": "crc32_byte"}]},
                     gates={"analysis": {"ok": True}, "isa": {"ok": True}})
        implementation = llm_step._with_run_memory(state, self.toolbox, "implement", "HW/SW task")
        self.assertIn("crc32_byte", implementation)
        self.assertIn("stable platform semantics", implementation)
        state["gates"]["hw_build"] = {"ok": False}
        revisit = llm_step._with_run_memory(state, self.toolbox, "implement", "repair HW")
        self.assertIn("stable platform semantics", revisit)
        (self.root / self.path).write_text("mutated platform\n")
        stale = llm_step._with_run_memory(state, self.toolbox, "implement", "repair HW")
        self.assertNotIn("stable platform semantics", stale)
        self.assertNotIn("mutated platform", stale)
        self.assertFalse(state["gates"]["hw_build"]["ok"])

    def test_exception_after_read_does_not_publish_node_memory(self):
        def broken(state):
            agent_tools.ReadOnlyToolset(self.toolbox).run("read_file", {"path": self.path})
            raise RuntimeError("transaction rolled back")
        with self.assertRaisesRegex(RuntimeError, "rolled back"):
            instrument._wrap_node(broken)(self.state)
        self.assertNotIn("flow_memory", self.state)
        with flow_memory.collect_tool_transcripts() as collector:
            self.assertEqual(collector, [])


if __name__ == "__main__":
    unittest.main()
