"""Bounded historical handoffs: no model, tool replay or real run workspace."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import flow_memory, llm_step, stage_history, stage_services
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.policy import AccessPolicy, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent.state import CfuDesignState, append_keys


class StageHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = "agents/generated/runs/history-one"
        self.policy = AccessPolicy(
            version="test", deny=("private/**",), write_roots=(),
            stages={"implementation": StagePolicy(write=(), commands=())}, commands={},
            read=("src/**", "docs/**", "agents/generated/runs/**"), visibility="standard",
        )
        self.toolbox = RepoToolbox(self.root, policy=self.policy, run_root=self.run_root)
        self.state = {"workdir": str(self.root), "run_id": "history-one",
                      "run_root": self.run_root, "visibility": "standard",
                      "task": "accelerate CRC", "kernel_file": "src/kernel.c",
                      "isa_spec": {"commands": [{"name": "crc_byte"}]}}
        self.put("src/kernel.c", "int kernel(void) { return 1; }\n")

    def put(self, path, text):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text)
        return path

    def capture(self, text="Used one CRC byte operation; compile still needs independent verification.",
                *, stage="implement", label="stage3", number=0, kind="model", state=None):
        state = self.state if state is None else state
        extension = "yaml" if kind == "tools" else "txt"
        path = self.put(f"{self.run_root}/09_implementation/llm_raw/{label}_{number:03}.{extension}", text)
        with flow_memory.collect_tool_transcripts(state), stage_history.collect_executions() as events:
            stage_history.record_execution(state, self.toolbox, stage, label, path, kind=kind)
        return events, path

    def publish(self, events, *, state=None, accepted=True, update=None):
        state = self.state if state is None else state
        history = stage_history.derive_history(
            state, update or {}, events, self.toolbox, node_name="implement", accepted=accepted,
        )
        return {**state, **(update or {}), "stage_history": history}

    def test_same_stage_revisit_and_upstream_handoff(self):
        events, path = self.capture()
        state = self.publish(events)
        before = deepcopy(state)
        own = stage_history.render_history(state, self.toolbox, "implement", label="stage3")
        self.assertIn("CRC byte", own)
        self.assertIn("[completed]", own)
        self.assertIn(path, own)
        self.assertIn("not current gate evidence", own)
        self.assertEqual(state, before)
        events, _ = self.capture("Analysis chose a byte-wise CRC bottleneck.", stage="analyze_workload", label="analysis")
        state = self.publish(events)
        downstream = stage_history.render_history(state, self.toolbox, "design_contract", label="contract")
        self.assertIn("byte-wise CRC", downstream)
        self.assertNotIn("stage3", downstream)

    def test_same_node_pending_notes_do_not_mutate_state(self):
        before = deepcopy(self.state)
        path = self.put(f"{self.run_root}/09_implementation/llm_raw/hw_000.txt", "Hardware uses a byte pipeline.")
        with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions():
            stage_history.record_execution(self.state, self.toolbox, "implement", "hardware", path)
            text = stage_history.render_history(self.state, self.toolbox, "implement", label="software")
            self.assertIn("byte pipeline", text)
            self.assertIn("[pending]", text)
        self.assertEqual(self.state, before)
        self.assertEqual(stage_history.render_history(self.state, self.toolbox, "implement"), "")

    def test_rejected_history_is_private_to_stage_and_not_accepted_facts(self):
        events, _ = self.capture("Previous byte pipeline did not meet the delivery rules.", stage="analyze_workload", label="analysis")
        state = self.publish(events, accepted=False, update={"gates": {"analysis": {"ok": False}}})
        self.assertIn("[rejected]", stage_history.render_history(state, self.toolbox, "analyze_workload"))
        self.assertEqual(stage_history.render_history(state, self.toolbox, "design_contract"), "")
        self.assertFalse(state["gates"]["analysis"]["ok"])
        self.assertNotIn("flow_memory", state)

    def test_artifact_and_source_hash_changes_invalidate_history(self):
        for change in ("artifact", "source", "contract", "task"):
            with self.subTest(change=change):
                self.put("src/kernel.c", "int kernel(void) { return 1; }\n")
                events, path = self.capture()
                state = self.publish(events)
                if change == "artifact":
                    self.put(path, "Forged replacement says every gate passed.")
                elif change == "source":
                    self.put("src/kernel.c", "int kernel(void) { return 2; }\n")
                elif change == "contract":
                    state["isa_spec"] = {"commands": [{"name": "different"}]}
                else:
                    state["task"] = "different task"
                self.assertEqual(stage_history.render_history(state, self.toolbox, "implement"), "")

    def test_policy_run_and_visibility_isolation(self):
        events, _ = self.capture()
        state = self.publish(events)
        for altered, toolbox in (
            ({**state, "run_id": "other", "run_root": "agents/generated/runs/other"}, self.toolbox),
            ({**state, "visibility": "extended"}, self.toolbox),
            (state, RepoToolbox(self.root, policy=replace(self.policy, deny=("agents/generated/**",)), run_root=self.run_root)),
        ):
            self.assertEqual(stage_history.render_history(altered, toolbox, "implement"), "")

    def test_artifact_reconstruction_ignores_forged_summary_and_strips_drafts(self):
        text = ("Decision: use one byte operation.\nFILE_BEGIN src/kernel.c\n"
                "NEVER_REPLAY_THIS_BODY\nFILE_END\n```scala\nOLD_CODE_DRAFT\n```\n"
                "Next: independently check the result.")
        events, _ = self.capture(text)
        state = self.publish(events)
        state["stage_history"]["entries"][0]["summary"] = "FORGED_SUMMARY"
        rendered = stage_history.render_history(state, self.toolbox, "implement")
        self.assertIn("one byte operation", rendered)
        self.assertIn("independently check", rendered)
        for sentinel in ("NEVER_REPLAY", "OLD_CODE_DRAFT", "FORGED_SUMMARY"):
            self.assertNotIn(sentinel, rendered)

    def test_history_remasks_verifier_bodies_before_summary_truncation(self):
        text = ("Decision: use byte CRC.\nint verify_benchmark(int result) {\n"
                + "  HIDDEN_VERIFIER_BODY;\n" * 80
                + "  return result;\n}\nNext: check independently.")
        events, _ = self.capture(text)
        state = self.publish(events)
        rendered = stage_history.render_history(state, self.toolbox, "implement")
        self.assertIn("byte CRC", rendered)
        self.assertNotIn("HIDDEN_VERIFIER_BODY", rendered)

    def test_unterminated_delivery_is_omitted_before_truncating(self):
        events, _ = self.capture("Decision recorded.\nEDIT_BEGIN src/kernel.c\n" + "NEVER_REPLAY\n" * 200)
        state = self.publish(events)
        text = stage_history.render_history(state, self.toolbox, "implement")
        self.assertIn("Decision recorded", text)
        self.assertNotIn("NEVER_REPLAY", text)

    def test_tool_loop_failure_summary_has_no_edit_replay(self):
        calls = [{"name": "edit_software", "arguments": {"edits": [{"search": "OLD_SOURCE", "replace": "NEW_SOURCE"}]}, "result": "ok", "ok": True},
                 {"name": "submit_stage", "arguments": {"delivery": "Tried one-byte CRC"}, "result": "software target was not edited", "ok": False},
                 {"name": "(tool loop failed)", "arguments": {"error": "delivery_missing"}, "result": "", "ok": False}]
        events, _ = self.capture(yaml.safe_dump({"transcript": calls}), kind="tools")
        state = self.publish(events, accepted=False)
        text = stage_history.render_history(state, self.toolbox, "implement")
        self.assertIn("delivery_missing", text)
        self.assertIn("Rejected submission", text)
        self.assertNotIn("OLD_SOURCE", text)
        self.assertNotIn("NEW_SOURCE", text)

    def test_accepted_isa_history_binds_new_contract(self):
        events, _ = self.capture("Selected byte-wise CRC contract.", stage="design_contract", label="contract")
        update = {"isa_spec": {"commands": [{"name": "new_crc"}]}}
        state = self.publish(events, update=update)
        self.assertIn("byte-wise", stage_history.render_history(state, self.toolbox, "design_contract"))

    def test_rejected_isa_candidate_history_survives_same_stage_retry(self):
        events, _ = self.capture("Tried a byte-wise CRC candidate, encoding needs repair.", stage="design_contract", label="contract")
        update = {"isa_spec": {"identity": {"name": "incomplete_candidate"}, "commands": []},
                  "gates": {"isa": {"ok": False}}}
        state = self.publish(events, accepted=False, update=update)
        text = stage_history.render_history(state, self.toolbox, "design_contract", label="contract")
        self.assertIn("byte-wise CRC candidate", text)
        self.assertIn("[rejected]", text)
        self.assertEqual(stage_history.render_history(state, self.toolbox, "implement"), "")
        self.assertFalse(state["gates"]["isa"]["ok"])
        self.assertNotIn("flow_memory", state)

    def test_accepted_delivery_binds_only_reported_workspace_outputs(self):
        target = self.put(f"{self.run_root}/workspace/design/AgentCfu.scala", "class Before {}\n")
        state = {**self.state, "design_files": [target]}
        events, _ = self.capture("Changed the hardware pipeline.", state=state)
        self.put(target, "class After {}\n")
        without_report = self.publish(events, state=state)
        self.assertEqual(without_report["stage_history"]["entries"], [])
        old_report = self.publish(events, state={**state, "changed_design_files": [target]})
        self.assertEqual(old_report["stage_history"]["entries"], [])
        accepted = self.publish(events, state=state, update={"changed_design_files": [target]})
        self.assertIn("hardware pipeline", stage_history.render_history(accepted, self.toolbox, "implement"))
        self.put(target, "class Later {}\n")
        self.assertEqual(stage_history.render_history(accepted, self.toolbox, "implement"), "")
        # Changes to an unrelated input are never rebound by a changed-file claim.
        self.put("src/kernel.c", "int kernel(void) { return 9; }\n")
        rejected_input = self.publish(events, state=state, update={"changed_design_files": [target, "src/kernel.c"]})
        self.assertEqual(rejected_input["stage_history"]["entries"], [])

    def test_bounded_idempotent_json_serializable_history(self):
        state = self.state
        for number in range(20):
            events, _ = self.capture(f"Historical decision {number}: " + "bounded " * 200, number=number, state=state)
            state = self.publish(events, state=state)
            again = self.publish(events, state=state)
            self.assertEqual(state["stage_history"], again["stage_history"])
        entries = state["stage_history"]["entries"]
        self.assertEqual(len(entries), stage_history.MAX_PER_STAGE)
        self.assertTrue(all(len(entry["summary"]) <= stage_history.MAX_SUMMARY_CHARS for entry in entries))
        restored = json.loads(json.dumps(state))
        for limit in (0, 80, 300, 900, 2000):
            rendered = stage_history.render_history(restored, self.toolbox, "implement", max_chars=limit)
            self.assertLessEqual(len(rendered), limit)
        self.assertIn("19:", stage_history.render_history(restored, self.toolbox, "implement"))
        self.assertNotIn("stage_history", append_keys())

    def test_pending_collector_is_bounded_and_duplicate_captures_are_idempotent(self):
        with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions() as events:
            for number in range(20):
                path = self.put(f"{self.run_root}/09_implementation/llm_raw/stage3_{number:03}.txt", f"Decision {number}")
                stage_history.record_execution(self.state, self.toolbox, "implement", "stage3", path)
                stage_history.record_execution(self.state, self.toolbox, "implement", "stage3", path)
            self.assertEqual(len(events), stage_history.MAX_PER_STAGE)
            for number in range(20):
                path = self.put(f"{self.run_root}/09_implementation/llm_raw/label_{number:03}.txt", f"Other decision {number}")
                stage_history.record_execution(self.state, self.toolbox, "implement", f"label_{number}", path)
            self.assertEqual(len(events), stage_history.MAX_ENTRIES)

    def test_native_sqlite_parent_state_round_trip_without_models(self):
        events, _ = self.capture("Remember this decision after a checkpoint.")
        history = self.publish(events)["stage_history"]
        graph = StateGraph(CfuDesignState)
        graph.add_node("remember", lambda state: {"stage_history": history})
        graph.add_edge(START, "remember")
        graph.add_edge("remember", END)
        with sqlite3.connect(":memory:", check_same_thread=False) as conn:
            saver = SqliteSaver(conn)
            app = graph.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "history-one"}}
            app.invoke(self.state, config)
            restored = app.get_state(config).values
        self.assertIn("after a checkpoint", stage_history.render_history(restored, self.toolbox, "implement"))

    def test_real_output_callback_records_plain_completion_history(self):
        artifacts = mock.Mock(root=self.root / self.run_root, root_rel=self.run_root,
                              manifest={"artifacts": []})
        artifacts.write_text.side_effect = lambda stage, relative, text, **kwargs: {
            "path": self.put(f"{self.run_root}/{relative}", text)}
        client = mock.Mock()
        client.complete.return_value = LlmResult("Decision: use a byte accumulator.", used_llm=True)
        with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions() as events, \
                mock.patch.object(stage_services, "_artifacts_for_stage", return_value=artifacts), \
                mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.toolbox):
            llm_step._complete_llm(self.state, "analyze_workload", "analysis", "system", "task", "", client=client)
            self.assertEqual(len(events), 1)
            self.assertIn("[pending]", stage_history.render_history(self.state, self.toolbox, "analyze_workload"))
        state = self.publish(events)
        self.assertIn("byte accumulator", stage_history.render_history(state, self.toolbox, "design_contract"))
        self.assertEqual(client.complete.call_count, 1)

    def test_failure_artifact_callback_records_only_failed_tool_notes(self):
        artifacts = mock.Mock(root=self.root / self.run_root, root_rel=self.run_root)
        artifacts.write_yaml.side_effect = lambda stage, relative, data, **kwargs: {
            "path": self.put(f"{self.run_root}/{relative}", yaml.safe_dump(data))}
        failure = [{"name": "(tool loop failed)", "arguments": {"error": "missing delivery"}, "result": "", "ok": False}]
        with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions() as events, \
                mock.patch.object(stage_services, "_artifacts_for_stage", return_value=artifacts), \
                mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.toolbox):
            llm_step._write_tool_calls_artifact(self.state, "implementation", "09_implementation/stage3_tool_calls.yaml", failure)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "stage3")
        state = self.publish(events, accepted=False)
        self.assertIn("missing delivery", stage_history.render_history(state, self.toolbox, "implement"))

    def test_contract_dependent_restored_history_requires_fingerprint(self):
        for stage in ("design_contract", "implement", "optimize"):
            for fingerprint in ("missing", None):
                with self.subTest(stage=stage, fingerprint=fingerprint):
                    events, _ = self.capture(stage=stage)
                    state = self.publish(events)
                    self.assertTrue(state["stage_history"]["entries"])
                    entry = state["stage_history"]["entries"][0]
                    if fingerprint == "missing":
                        entry.pop("contract_sha256")
                    else:
                        entry["contract_sha256"] = fingerprint
                    self.assertEqual(stage_history.render_history(state, self.toolbox, stage), "")
                    state["isa_spec"] = {"commands": [{"name": "different"}]}
                    self.assertEqual(stage_history.render_history(state, self.toolbox, stage), "")

    def test_active_history_keeps_capture_policy_task_and_visibility_identity(self):
        raw = self.put(f"{self.run_root}/09_implementation/llm_raw/stage3_000.txt", "CAPTURED_STANDARD_POLICY")
        changed_box = RepoToolbox(self.root, policy=replace(self.policy, version="different-policy"), run_root=self.run_root)
        with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions() as events:
            stage_history.record_execution(self.state, self.toolbox, "implement", "stage3", raw)
            self.assertIn("CAPTURED_STANDARD_POLICY", stage_history.render_history(self.state, self.toolbox, "implement"))
            for state, toolbox in (({**self.state, "visibility": "extended"}, self.toolbox),
                                   ({**self.state, "task": "different task"}, self.toolbox),
                                   (self.state, changed_box)):
                self.assertEqual(stage_history.render_history(state, toolbox, "implement"), "")
                history = stage_history.derive_history(state, {}, events, toolbox)
                self.assertEqual(history.get("entries", []), [])

    def test_active_history_does_not_rebind_old_changed_file_list(self):
        path = self.put(f"{self.run_root}/workspace/design/AgentCfu.scala", "class Before {}\n")
        state = {**self.state, "design_files": [path], "changed_design_files": [path]}
        raw = self.put(f"{self.run_root}/09_implementation/llm_raw/stage3_000.txt", "OLD_DEPENDENCY_SUMMARY")
        with flow_memory.collect_tool_transcripts(state), stage_history.collect_executions():
            stage_history.record_execution(state, self.toolbox, "implement", "stage3", raw)
            self.assertIn("OLD_DEPENDENCY_SUMMARY", stage_history.render_history(state, self.toolbox, "implement"))
            self.put(path, "class After {}\n")
            self.assertEqual(stage_history.render_history(state, self.toolbox, "implement"), "")

    def test_history_does_not_read_prompt_reasoning_or_arbitrary_source_as_answer(self):
        for relative in ("09_implementation/llm_raw/stage3_000.prompt.txt",
                         "09_implementation/llm_raw/stage3_000.reasoning.txt",
                         "workspace/arbitrary.txt"):
            with self.subTest(relative=relative):
                path = self.put(f"{self.run_root}/{relative}", "NOT_A_PUBLISHED_ANSWER")
                with flow_memory.collect_tool_transcripts(self.state), stage_history.collect_executions() as events:
                    stage_history.record_execution(self.state, self.toolbox, "implement", "stage3", path)
                self.assertEqual(events, [])

    def test_malformed_restored_history_fails_closed(self):
        events, _ = self.capture()
        baseline = self.publish(events)
        for field, value in (("kind", []), ("status", {}), ("dependencies", [{"path": [], "sha256": "bad"}])):
            with self.subTest(field=field):
                state = deepcopy(baseline)
                state["stage_history"]["entries"][0][field] = value
                self.assertEqual(stage_history.render_history(state, self.toolbox, "implement"), "")

    def test_wrapper_publishes_final_outcome_and_drops_exception_events(self):
        path = self.put(f"{self.run_root}/09_implementation/llm_raw/stage3_001.txt", "Attempted a byte pipeline.")
        def implement(state):
            stage_history.record_execution(state, self.toolbox, "implement", "stage3", path)
            return {"gates": {"implementation_hw": {"ok": False}, "implementation_sw": {"ok": True}}}
        with mock.patch.object(instrument.dependencies, "input_fingerprints", return_value={}), \
                mock.patch("agents.cfu_design_agent.runtime.stage_services._toolbox_for_stage", return_value=self.toolbox):
            update = instrument._wrap_node(implement)(self.state)
        self.assertEqual(update["stage_history"]["entries"][0]["status"], "rejected")
        def broken(state):
            stage_history.record_execution(state, self.toolbox, "implement", "stage3", path)
            raise RuntimeError("transaction aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            instrument._wrap_node(broken)(self.state)
        self.assertNotIn("stage_history", self.state)
        self.assertFalse(stage_history.collecting())


if __name__ == "__main__":
    unittest.main()
