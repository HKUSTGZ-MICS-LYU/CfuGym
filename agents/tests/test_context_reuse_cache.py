"""Offline contracts for cache-first, bounded cross-stage context reuse."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent import prompts
from agents.cfu_design_agent.runtime import flow_memory, llm_step, request_audit, stage_history
from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.policy import AccessPolicy, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox


class _ProbeClient:
    """Capture application messages; never construct a provider SDK."""

    available = True
    model = "offline-context-probe"
    _stage_round_client = object()

    def __init__(self):
        self.requests = []

    def complete(self, system, user, **kwargs):
        self.requests.append([{"role": "system", "content": system},
                              {"role": "user", "content": user}])
        return LlmResult("offline delivery", used_llm=True)

    def complete_with_tools(self, system, user, **kwargs):
        return self.complete(system, user, **kwargs)

    def _chat_round(self, sdk, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        return SimpleNamespace(
            text="", reasoning="", reasoning_source="", finish_reason="tool_calls",
            extra_keys=(), input_tokens=0, output_tokens=0, total_tokens=0,
            cached_input_tokens=None,
            message=SimpleNamespace(content="", tool_calls=[SimpleNamespace(
                id="offline-submit", function=SimpleNamespace(
                    name="submit_stage", arguments='{"delivery":"offline delivery"}'))]),
        )


class ContextReuseCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run_root = "agents/generated/runs/context-reuse-offline"
        self.policy = AccessPolicy(
            version="context-reuse-test", deny=(), write_roots=(),
            stages={"analysis": StagePolicy(write=(), commands=())}, commands={},
            read=("docs/**", "src/**", "agents/generated/runs/**"),
            visibility="standard",
        )
        self.toolbox = RepoToolbox(self.root, policy=self.policy, stage="analysis",
                                   run_root=self.run_root)
        self.tools = ReadOnlyToolset(self.toolbox)
        self.state = {
            "workdir": str(self.root), "run_id": "context-reuse-offline",
            "run_root": self.run_root, "visibility": "standard", "isa_spec": {},
        }
        self.reference = "docs/context-reference.txt"
        self.put(self.reference, "UNOBSERVED_HEAD\nREUSED_RANGE_TWO\nUNOBSERVED_GAP\n"
                                 "REUSED_RANGE_FOUR\nUNOBSERVED_TAIL\n")

    def put(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def memory_for_reads(self, *reads):
        with flow_memory.collect_tool_transcripts() as captures:
            for arguments in reads:
                result = self.tools.run("read_file", arguments)
                self.assertNotIn("denied by access policy:", result)
                self.assertNotIn("usage error:", result)
        return flow_memory.derive_memory(self.state, {}, captures, self.toolbox)

    def test_virtual_artifact_and_overlay_reads_capture_canonical_paths(self):
        artifact = f"{self.run_root}/07_analysis/workload_analysis.yaml"
        overlay = f"{self.run_root}/workspace/design/AgentCfu.scala"
        tracked = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        self.put(artifact, "UNOBSERVED_HEAD\nARTIFACT_RANGE\nUNOBSERVED_TAIL\n")
        self.put(overlay, "UNOBSERVED_HEAD\nLIVE_OVERLAY_RANGE\nUNOBSERVED_TAIL\n")
        self.put(tracked, "TRACKED_SCAFFOLD_MUST_NOT_BE_CAPTURED\n")
        cases = (
            ("{{run-dir}}/07_analysis/workload_analysis.yaml", artifact, "ARTIFACT_RANGE"),
            ("{{run-dir}}/workspace/design/AgentCfu.scala", overlay, "LIVE_OVERLAY_RANGE"),
            (tracked, overlay, "LIVE_OVERLAY_RANGE"),
        )
        for selector, canonical, marker in cases:
            with self.subTest(selector=selector):
                memory = self.memory_for_reads({"path": selector, "start_line": 2, "end_line": 2})
                self.assertEqual([entry["path"] for entry in memory["entries"]], [canonical])
                rendered = flow_memory.render_memory(memory, self.state, self.toolbox,
                                                     "implementation", max_chars=4000)
                self.assertIn(marker, rendered)
                self.assertNotIn("UNOBSERVED", rendered)
                self.assertNotIn("TRACKED_SCAFFOLD_MUST_NOT_BE_CAPTURED", rendered)

    def _prompt(self, stage):
        context = {
            "workdir": str(self.root), "run_root": self.run_root,
            "visibility": "standard", "files": {self.reference: "not inlined"},
        }
        return "Task:\naccelerate the same workload\n\n" + prompts.stage_brief(
            stage, deliverable="Produce this stage's artifact.", context=context,
            tool_names=("read_file",), tool_rounds=2,
            shared=prompts.workload_digest_block({"name": "same-workload"}, None),
        )

    def _run_request(self, mode, state, *, stage="analyze_workload", label="analysis"):
        client = _ProbeClient()
        user = self._prompt(label)
        with mock.patch.object(llm_step, "_record_llm_output", return_value="offline-raw.txt"), \
             mock.patch.object(llm_step, "_write_tool_calls_artifact"), \
             mock.patch.object(llm_step.stage_services, "_toolbox_for_stage", return_value=self.toolbox):
            llm_step._run_agent_step(
                {**state, "agent_tools": mode != "plain"}, self.toolbox,
                stage=stage, label=label, artifact_stage="analysis",
                tool_calls_relpath="07_analysis/offline_tool_calls.yaml",
                system=prompts.SYSTEM_PROMPT, user=user, fallback="", client=client,
                native_tool_loop=mode == "native",
                submit_stage=(lambda _delivery: True) if mode == "native" else None,
            )
        self.assertEqual(len(client.requests), 1)
        return user, client.requests[0]

    def test_memory_tail_preserves_base_and_virtualizes_paths_in_all_modes(self):
        artifact = f"{self.run_root}/07_analysis/observed.txt"
        self.put(artifact, "REUSABLE_ARTIFACT_CONTENT\n")
        memory = self.memory_for_reads({"path": artifact})
        state = {**self.state, "flow_memory": memory}
        saved = deepcopy(state)
        for mode in ("plain", "tools", "native"):
            with self.subTest(mode=mode):
                user, request = self._run_request(mode, state)
                expected_system, expected_user = llm_step._virtual_paths(
                    state, prompts.SYSTEM_PROMPT, user)
                self.assertEqual(request[0], {"role": "system", "content": expected_system})
                sent = request[1]["content"]
                self.assertTrue(sent.startswith(expected_user))
                self.assertIn("REUSABLE_ARTIFACT_CONTENT", sent)
                self.assertIn("{{run-dir}}/07_analysis/observed.txt", sent)
                self.assertNotIn(self.run_root, sent)
                self.assertEqual(sent.count("REUSABLE_ARTIFACT_CONTENT"), 1)
                self.assertLessEqual(len(sent) - len(expected_user), 4000)
        self.assertEqual(state, saved)

    def test_dynamic_memory_does_not_change_cross_stage_shared_prefix(self):
        memory = self.memory_for_reads({"path": self.reference, "start_line": 2, "end_line": 2})
        sent = []
        for stage, label in (("analyze_workload", "analysis"),
                             ("design_contract", "isa"), ("implement", "implementation")):
            _, request = self._run_request("tools", {**self.state, "flow_memory": memory},
                                           stage=stage, label=label)
            sent.append(request)
        self.assertTrue(all(request[0] == sent[0][0] for request in sent))
        prefixes = [request[1]["content"].split("## Stage:", 1)[0] for request in sent]
        self.assertTrue(prefixes[0].startswith("Task:"))
        self.assertTrue(all(prefix == prefixes[0] for prefix in prefixes))
        self.assertTrue(all("REUSED_RANGE_TWO" in request[1]["content"] for request in sent))

    def _history(self, state, records, *, accepted=True):
        with stage_history.collect_executions() as events:
            for index, (stage, label, kind, text) in enumerate(records):
                extension = "yaml" if kind == "tools" else "txt"
                path = f"{self.run_root}/09_implementation/llm_raw/{label}_{index:03d}.{extension}"
                self.put(path, text)
                stage_history.record_execution(state, self.toolbox, stage, label, path, kind=kind)
        return stage_history.derive_history(state, {}, events, self.toolbox, accepted=accepted)

    def test_history_and_file_memory_share_one_bounded_virtual_tail(self):
        reads = []
        for index in range(8):
            path = f"docs/large-memory-{index}.txt"
            self.put(path, f"MEMORY_RANGE_{index}\n" + "\n".join(
                f"row {row} " + "x" * 80 for row in range(16)) + "\n")
            reads.append({"path": path})
        state = {**self.state, "flow_memory": self.memory_for_reads(*reads)}
        history = self._history(state, [("implement", "stage3", "model", "HISTORY_MARKER " + "h" * 3000)])
        state["stage_history"] = history
        self.assertLessEqual(len(stage_history.render_history(
            state, self.toolbox, "implement", label="stage3", max_chars=900)), 900)
        original = self._prompt("stage3")
        augmented = llm_step._with_run_memory(state, self.toolbox, "implement", original, label="stage3")
        self.assertTrue(augmented.startswith(original))
        self.assertLessEqual(len(augmented) - len(original), 4000)
        self.assertIn("HISTORY_MARKER", augmented)
        self.assertIn("MEMORY_RANGE_", augmented)
        for mode in ("plain", "tools", "native"):
            with self.subTest(mode=mode):
                user, request = self._run_request(mode, state, stage="implement", label="stage3")
                base_system, base_user = llm_step._virtual_paths(state, prompts.SYSTEM_PROMPT, user)
                self.assertEqual(request[0]["content"], base_system)
                self.assertTrue(request[1]["content"].startswith(base_user))
                self.assertNotIn(self.run_root, request[1]["content"])
                self.assertIn("{{run-dir}}", request[1]["content"])
                self.assertEqual(request[1]["content"].count("HISTORY_MARKER"), 1)
                self.assertLessEqual(len(request[1]["content"]) - len(base_user), 4000)

    def test_effective_request_audits_the_same_bounded_tail_sent_to_fake_sdk(self):
        from agents.cfu_design_agent.runtime.llm import OpenAiTextClient

        artifact = f"{self.run_root}/07_analysis/observed.txt"
        self.put(artifact, "AUDITED_OBSERVED_CONTENT\n")
        state = {**self.state, "flow_memory": self.memory_for_reads({"path": artifact})}
        state["stage_history"] = self._history(
            state, [("implement", "stage3", "model", "AUDITED_HISTORICAL_STRATEGY")])
        sent = []

        def create(**body):
            sent.append(deepcopy(body))
            return SimpleNamespace(
                id="offline-response", usage=None,
                choices=[SimpleNamespace(finish_reason="stop",
                                         message=SimpleNamespace(content="offline answer", tool_calls=None))],
            )

        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = OpenAiTextClient(config={
            "api_key": "offline-test-key", "model": "offline-context-model", "stream": False,
            "api_mode": "chat", "context_window_tokens": 100000, "output_reserve_tokens": 256,
        })
        base = self._prompt("stage3")
        with mock.patch("openai.OpenAI", return_value=sdk), \
             mock.patch.object(llm_step, "_record_llm_output", return_value="offline-raw.txt"), \
             mock.patch.object(llm_step, "_write_tool_calls_artifact"), \
             request_audit.use_request_audit(self.root / self.run_root, self.root):
            llm_step._run_agent_step(
                state, self.toolbox, stage="implement", label="stage3", artifact_stage="implementation",
                tool_calls_relpath="09_implementation/offline_tool_calls.yaml",
                system=prompts.SYSTEM_PROMPT, user=base, fallback="", client=client,
            )
        events = [json.loads(line) for line in
                  (self.root / self.run_root / request_audit.AUDIT_FILENAME).read_text().splitlines()]
        requests = [event for event in events if event["event"] == "request"]
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(requests[0]["payload"]["messages"], sent[0]["messages"])
        sent_user = sent[0]["messages"][1]["content"]
        self.assertIn("AUDITED_OBSERVED_CONTENT", sent_user)
        self.assertIn("AUDITED_HISTORICAL_STRATEGY", sent_user)
        self.assertNotIn(self.run_root, sent_user)
        self.assertEqual(requests[0]["metadata"]["context_budget"]["window_source"], "config")
        self.assertEqual(requests[0]["metadata"]["context_budget"]["reserved_output_tokens"], 256)
        results = [event for event in events if event["event"] == "result"]
        self.assertIsNone(results[0]["result"]["usage"]["cached_input_tokens"])

    def test_in_node_reads_and_history_are_reused_without_parent_mutation(self):
        raw = f"{self.run_root}/07_analysis/llm_raw/analysis_000.txt"
        self.put(raw, "IN_NODE_EXECUTION_NOTE")
        before = deepcopy(self.state)
        with flow_memory.collect_tool_transcripts(), stage_history.collect_executions():
            self.tools.run("read_file", {"path": self.reference, "start_line": 2, "end_line": 2})
            stage_history.record_execution(self.state, self.toolbox, "analyze_workload", "analysis", raw)
            prompt = llm_step._with_run_memory(self.state, self.toolbox, "analyze_workload",
                                               "stable base", label="analysis")
            self.assertTrue(prompt.startswith("stable base"))
            self.assertIn("REUSED_RANGE_TWO", prompt)
            self.assertIn("IN_NODE_EXECUTION_NOTE", prompt)
            self.assertNotIn("UNOBSERVED", prompt)
            self.assertEqual(self.state, before)
        self.assertEqual(llm_step._with_run_memory(self.state, self.toolbox, "analyze_workload",
                                                  "stable base", label="analysis"), "stable base")
        self.assertEqual(self.state, before)


if __name__ == "__main__":
    unittest.main()
