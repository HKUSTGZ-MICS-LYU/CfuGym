"""Bounded exact ranges and same-node observations, without a real model/build."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest

from agents.cfu_design_agent.runtime import agent_tools, flow_memory
from agents.cfu_design_agent.runtime.policy import AccessPolicy, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox


class FlowMemoryRangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = "agents/generated/runs/ranges-unit"
        self.policy = AccessPolicy(
            version="test", deny=(), write_roots=(), commands={},
            stages={"analysis": StagePolicy(write=(), commands=())},
            read=("src/**", "docs/**", "agents/generated/runs/**"), visibility="standard",
        )
        self.toolbox = RepoToolbox(self.root, policy=self.policy, run_root=self.run_root)
        self.tools = agent_tools.ReadOnlyToolset(self.toolbox)
        self.state = {"workdir": str(self.root), "run_root": self.run_root,
                      "run_id": "ranges-unit", "visibility": "standard", "isa_spec": {"revision": 1}}
        self.path = "src/kernel.c"
        self.lines = [f"line_{index:03d}: " + "x" * 65 for index in range(1, 101)]
        self.put(self.path, "\n".join(self.lines) + "\n")

    def put(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def read(self, first, last, path=None):
        return self.tools.run("read_file", {"path": path or self.path,
                                          "start_line": first, "end_line": last})

    def derive(self, calls, update=None, state=None):
        return flow_memory.derive_memory(state or self.state, update or {}, calls,
                                         self.toolbox, node_name="implement")

    def render(self, memory, maximum=4000, state=None):
        return flow_memory.render_memory(memory, state or self.state, self.toolbox,
                                         "implement", max_chars=maximum)

    def test_disjoint_ranges_keep_body_beyond_600_without_unobserved_gap(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(2, 20)
            self.read(60, 80)
        memory = self.derive(calls)
        entry = memory["entries"][0]
        self.assertEqual(len(entry["ranges"]), 2)
        self.assertGreater(len(entry["ranges"][0]["excerpt"]), 600)
        self.assertIn("line_020", entry["ranges"][0]["excerpt"])
        self.assertIn("line_080", entry["ranges"][1]["excerpt"])
        self.assertLessEqual(len(entry["excerpt"]), 600)
        text = self.render(memory, 6000)
        self.assertIn("observed lines 2-20", text)
        self.assertIn("observed lines 60-80", text)
        self.assertNotIn("line_001", text)
        self.assertNotIn("line_050", text)
        self.assertNotIn("line_081", text)

    def test_repeat_read_and_rederive_are_idempotent_without_reordering(self):
        self.put("docs/guide.md", "stable guide\n")
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(2, 8)
            self.read(1, 1, "docs/guide.md")
            self.read(50, 55)
            captured = deepcopy(calls)
            self.read(2, 8)
            self.assertEqual(calls, captured)
        first = self.derive(calls)
        current = {**self.state, "flow_memory": first}
        self.assertEqual(self.derive(calls, state=current), first)
        self.assertEqual([entry["path"] for entry in first["entries"]],
                         [self.path, "docs/guide.md"])
        self.assertEqual(self.render(first), self.render(first))

    def test_repeated_oversized_read_keeps_the_same_retained_window(self):
        self.put(self.path, "\n".join(f"large_{index:03d} " + "x" * 90 for index in range(250)))
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.tools.run("read_file", {"path": self.path})
            before = deepcopy(calls)
            self.assert_bounded([item["entry"] for item in calls])
            self.tools.run("read_file", {"path": self.path})
            self.assertEqual(calls, before)
            first = self.derive(calls)
            again = self.derive(calls, state={**self.state, "flow_memory": first})
            self.assertEqual(again, first)

    def test_new_state_reference_does_not_expand_observed_coverage(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(40, 42)
        old = self.derive(calls)
        current = {**self.state, "flow_memory": old, "kernel_file": self.path}
        memory = self.derive((), state=current)
        self.assertEqual(memory["entries"], old["entries"])
        self.assertNotIn("line_001", self.render(memory))
        reference = self.derive((), {"kernel_file": self.path})
        self.assertNotIn("ranges", reference["entries"][0])
        self.assertNotIn("line_040", self.render(reference))

    def test_same_node_view_is_fresh_without_mutating_or_consuming_state(self):
        before = deepcopy(self.state)
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(50, 55)
            captures = deepcopy(calls)
            view = flow_memory.memory_for_prompt(self.state, self.toolbox)
            self.assertIn("line_050", self.render(view))
            view["entries"][0]["ranges"][0]["excerpt"] = "modified view"
            next_view = flow_memory.memory_for_prompt(self.state, self.toolbox)
            self.assertIn("line_050", self.render(next_view))
            self.assertEqual(calls, captures)
            self.assertEqual(self.state, before)
        self.assertEqual(flow_memory.memory_for_prompt(self.state, self.toolbox)["entries"], [])

    def test_read_edit_read_rollback_keeps_only_restored_byte_version(self):
        self.put(self.path, "ORIGINAL_BYTES\n")
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 1)
            self.put(self.path, "TEMPORARY_REJECTED_BYTES\n")
            self.read(1, 1)
            changed = flow_memory.memory_for_prompt(self.state, self.toolbox)
            self.assertIn("TEMPORARY_REJECTED_BYTES", self.render(changed))
            self.assertNotIn("ORIGINAL_BYTES", self.render(changed))
            self.put(self.path, "ORIGINAL_BYTES\n")
            restored = flow_memory.memory_for_prompt(self.state, self.toolbox)
            self.assertIn("ORIGINAL_BYTES", self.render(restored))
            self.assertNotIn("TEMPORARY_REJECTED_BYTES", self.render(restored))
        failed = {"errors": ["delivery rejected"], "hardware_plan": "UNACCEPTED_PLAN",
                  "gates": {"implementation_hw": {"ok": False}}}
        memory = self.derive(calls, failed)
        self.assertIn("ORIGINAL_BYTES", self.render(memory, state={**self.state, **failed}))
        self.assertNotIn("TEMPORARY_REJECTED_BYTES", str(memory))
        self.assertNotIn("UNACCEPTED_PLAN", str(memory))
        self.assertEqual(memory["facts"], [])

    def test_contract_change_does_not_rebind_active_or_published_hardware_read(self):
        hardware = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.put(hardware, "OLD_CONTRACT_SEMANTICS\n")
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 1, hardware)
            self.read(2, 3)
            self.assertEqual(calls[0]["entry"]["contract_sha256"],
                             flow_memory._digest(self.state["isa_spec"]))
            changed = {**self.state, "isa_spec": {"revision": 2}}
            self.assertNotIn("OLD_CONTRACT_SEMANTICS",
                             str(flow_memory.memory_for_prompt(changed, self.toolbox)))
            self.assertIn("line_002", str(flow_memory.memory_for_prompt(changed, self.toolbox)))
        published = self.derive(calls, {"isa_spec": {"revision": 2}})
        self.assertNotIn("OLD_CONTRACT_SEMANTICS", str(published))
        self.assertIn("line_002", str(published))

    def test_active_reads_obey_policy_identity_and_file_changes(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 3)
            changed_policy = replace(self.policy, version="new-policy")
            box = RepoToolbox(self.root, policy=changed_policy, run_root=self.run_root)
            self.assertEqual(flow_memory.memory_for_prompt(self.state, box)["entries"], [])
            self.put(self.path, "modified outside the observed lines\n")
            self.assertEqual(flow_memory.memory_for_prompt(self.state, self.toolbox)["entries"], [])
        self.assertEqual(self.derive(calls)["entries"], [])

    def test_storage_and_active_collector_obey_all_budgets(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            for index in range(28):
                path = f"src/source_{index}.c"
                self.put(path, "\n".join(f"row_{line} " + "x" * 1700 for line in range(6)))
                for line in range(1, 7):
                    self.read(line, line, path)
                entries = [call["entry"] for call in calls]
                self.assert_bounded(entries)
            memory = self.derive(calls)
        self.assert_bounded(memory["entries"])
        self.assertLessEqual(len(self.render(memory)), 4000)
        before = deepcopy(memory)
        self.assertEqual(flow_memory.validate_memory(memory, self.state, self.toolbox), memory)
        self.assertEqual(memory, before)

    def assert_bounded(self, entries):
        self.assertLessEqual(len(entries), flow_memory.MAX_ENTRIES)
        ranges = [part for entry in entries for part in entry.get("ranges", [])]
        self.assertLessEqual(len(ranges), flow_memory.MAX_RANGES)
        self.assertLessEqual(sum(len(part["excerpt"]) for part in ranges),
                             flow_memory.MAX_TOTAL_EXCERPT_CHARS)
        for path in {entry["path"] for entry in entries}:
            parts = [part for entry in entries if entry["path"] == path
                     for part in entry.get("ranges", [])]
            self.assertLessEqual(len(parts), flow_memory.MAX_RANGES_PER_FILE)
            self.assertLessEqual(sum(len(part["excerpt"]) for part in parts),
                                 flow_memory.MAX_FILE_EXCERPT_CHARS)
            for part in parts:
                self.assertLessEqual(len(part["excerpt"]), flow_memory.MAX_RANGE_CHARS)

    def test_render_prioritizes_exact_source_and_labels_provided_subrange(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(10, 40)
        update = {"workload_analysis": "ANALYSIS_FACT " + "z" * 2000,
                  "gates": {"analysis": {"ok": True}}}
        memory = self.derive(calls, update)
        current = {**self.state, **update}
        text = self.render(memory, 900, current)
        self.assertIn("line_010", text)
        self.assertIn("observed lines 10-40", text)
        provided = re.search(r"provided lines (\d+)-(\d+)", text)
        self.assertIsNotNone(provided)
        self.assertEqual(int(provided[1]), 10)
        self.assertLess(int(provided[2]), 40)
        for number, content in re.findall(r"(?m)^(\d+): (.*)$", text):
            self.assertEqual(content, self.lines[int(number) - 1])
        for maximum in (0, 4, 130, 300, 500, 900, 1200, 2000, 4000, 8000):
            self.assertLessEqual(len(self.render(memory, maximum, current)), maximum)
        self.assertIn("ANALYSIS_FACT", self.render(memory, 8000, current))

    def test_partial_tool_line_and_oversized_line_do_not_become_exact_ranges(self):
        self.put(self.path, "first\n" + "X" * 5000 + "\nlast\n")
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 3)
        memory = self.derive(calls)
        coverage = [part["coverage"] for part in memory["entries"][0]["ranges"]]
        self.assertEqual([(part["first_line"], part["last_line"]) for part in coverage],
                         [(1, 1), (3, 3)])
        self.assertNotIn("XXX", str(memory))
        short = agent_tools.ReadOnlyToolset(self.toolbox, max_read_chars=len("     1\tfirst\n     2\tXX"))
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            short.run("read_file", {"path": self.path})
        self.assertEqual(calls[0]["entry"]["ranges"][0]["coverage"]["last_line"], 1)

    def test_legacy_v2_prefix_is_not_expanded_to_its_unretained_coverage(self):
        memory = self.derive((), {"kernel_file": self.path})
        entry = memory["entries"][0]
        entry.update(origin="tool", coverage={"first_line": 10, "last_line": 40, "full": False},
                     excerpt="\n".join(f"{n}: {self.lines[n - 1]}" for n in range(10, 41))[:600])
        clean = flow_memory.validate_memory(memory, self.state, self.toolbox)
        self.assertNotIn("ranges", clean["entries"][0])
        self.assertNotIn("line_040", str(clean))
        self.assertIn("provided legacy excerpt prefix only", self.render(clean))

    def test_forged_range_downgrades_only_itself_and_never_claims_full_read(self):
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(2, 3)
            self.read(50, 51)
        memory = self.derive(calls)
        for mutation in ("wrong bytes", "false full read", "bad range type"):
            with self.subTest(mutation=mutation):
                forged = deepcopy(memory)
                part = forged["entries"][0]["ranges"][0]
                if mutation == "wrong bytes":
                    part["excerpt"] = "2: forged source"
                elif mutation == "false full read":
                    part["coverage"]["full"] = True
                else:
                    part["coverage"]["first_line"] = True
                text = self.render(forged)
                self.assertNotIn("line_002", text)
                self.assertNotIn("forged source", text)
                self.assertIn("line_050", text)

    def test_rejected_fact_cannot_inherit_an_old_passing_gate_or_reappear_later(self):
        state = {**self.state, "workload_analysis": "ACCEPTED_ANALYSIS",
                 "workload_spec": {"name": "ACCEPTED_WORKLOAD"},
                 "gates": {"analysis": {"ok": True}, "workload_spec": {"ok": True}}}
        state["flow_memory"] = self.derive((), state=state)
        for update in (
            {"workload_analysis": "REJECTED_ANALYSIS", "errors": ["bad"]},
            {"workload_analysis": "REJECTED_ANALYSIS"},
            {"workload_analysis": "REJECTED_ANALYSIS", "errors": ["bad"],
             "gates": {**state["gates"], "analysis": {"ok": True}}},
        ):
            with self.subTest(update=update):
                before = deepcopy(state)
                with flow_memory.collect_tool_transcripts(state) as calls:
                    self.read(1, 2)
                memory = self.derive(calls, update, state)
                current = {**state, **update, "flow_memory": memory}
                self.assertNotIn("REJECTED_ANALYSIS", self.render(memory, state=current))
                self.assertIn("ACCEPTED_WORKLOAD", self.render(memory, state=current))
                self.assertIn("line_001", self.render(memory, state=current))
                self.assertEqual(state, before)
                self.assertEqual(memory["fact_sha256"]["workload_analysis"],
                                 flow_memory._digest("ACCEPTED_ANALYSIS"))
                for _ in range(2):
                    view = flow_memory.memory_for_prompt(current, self.toolbox)
                    self.assertNotIn("REJECTED_ANALYSIS", self.render(view, state=current))
                    current["flow_memory"] = self.derive((), {"iteration": 1}, current)
                accepted = {"gates": {**state["gates"], "analysis": {"ok": True}}}
                fresh = self.derive((), accepted, current)
                self.assertIn("REJECTED_ANALYSIS", self.render(fresh, state={**current, **accepted}))

    def test_rejected_first_fact_keeps_a_bounded_null_constraint(self):
        state = {**self.state, "gates": {"analysis": {"ok": True}}}
        update = {"workload_analysis": "NEVER_ACCEPTED", "errors": ["bad"]}
        memory = self.derive((), update, state)
        self.assertIsNone(memory["fact_sha256"]["workload_analysis"])
        current = {**state, **update, "flow_memory": memory}
        for _ in range(2):
            memory = flow_memory.memory_for_prompt(current, self.toolbox)
            self.assertNotIn("NEVER_ACCEPTED", self.render(memory, state=current))
            self.assertIsNone(memory["fact_sha256"]["workload_analysis"])
            current["flow_memory"] = memory
        forged = deepcopy(memory)
        forged["fact_sha256"]["unknown_key"] = "unbounded prose" * 1000
        forged["facts"] = [{"source": "state.workload_analysis (accepted analysis gate)",
                            "role": "analysis", "sha256": flow_memory._digest("NEVER_ACCEPTED"),
                            "excerpt": "NEVER_ACCEPTED"}]
        clean = flow_memory.validate_memory(forged, current, self.toolbox)
        self.assertLessEqual(len(clean["fact_sha256"]), 7)
        self.assertNotIn("unknown_key", clean["fact_sha256"])
        self.assertNotIn("NEVER_ACCEPTED", str(clean))

    def test_failed_delivery_cannot_replace_approved_conformance_fact(self):
        state = {**self.state, "conformance_delivery": {"oracle_sha": "accepted_oracle"}}
        state["flow_memory"] = self.derive((), state=state)
        update = {"conformance_delivery": {"oracle_sha": "rejected_oracle"}, "errors": ["bad"]}
        memory = self.derive((), update, state)
        current = {**state, **update, "flow_memory": memory}
        self.assertNotIn("rejected_oracle", self.render(memory, state=current))
        self.assertNotIn("rejected_oracle", str(flow_memory.memory_for_prompt(current, self.toolbox)))

    def test_restored_dependent_ranges_require_a_valid_contract_fingerprint(self):
        hardware = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.put(hardware, "CONTRACT_BOUND_HARDWARE\n")
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 1, hardware)
            self.read(2, 3)
        memory = self.derive(calls)
        for revision in (1, 2):
            for fingerprint in ("missing", None, "not-a-sha"):
                with self.subTest(revision=revision, fingerprint=fingerprint):
                    forged = deepcopy(memory)
                    entry = next(item for item in forged["entries"] if item["path"] == hardware)
                    if fingerprint == "missing":
                        entry.pop("contract_sha256")
                    else:
                        entry["contract_sha256"] = fingerprint
                    current = {**self.state, "isa_spec": {"revision": revision}, "flow_memory": forged}
                    text = self.render(forged, state=current)
                    self.assertNotIn("CONTRACT_BOUND_HARDWARE", text)
                    self.assertIn("line_002", text)
                    view = flow_memory.memory_for_prompt(current, self.toolbox)
                    self.assertNotIn("CONTRACT_BOUND_HARDWARE", str(view))

    def test_old_scaffold_observation_is_invalid_when_tool_name_selects_new_overlay(self):
        tracked = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        overlay = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.put(tracked, "TRACKED_SCAFFOLD_BYTES\n")
        tracked_hash = flow_memory._file_hash(self.toolbox, tracked)
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            self.read(1, 1, tracked)
            memory = self.derive(calls)
            legacy = deepcopy(memory)
            legacy["entries"][0].pop("ranges")
            self.put(overlay, "CURRENT_OVERLAY_BYTES\n")
            self.assertEqual(flow_memory._file_hash(self.toolbox, tracked), tracked_hash)
            for old in (memory, legacy):
                current = {**self.state, "flow_memory": old}
                self.assertEqual(flow_memory.validate_memory(old, current, self.toolbox)["entries"], [])
                self.assertEqual(self.render(old, state=current), "")
                self.assertEqual(flow_memory.memory_for_prompt(current, self.toolbox)["entries"], [])
            result = self.read(1, 1, tracked)
            self.assertIn("CURRENT_OVERLAY_BYTES", result)
            view = flow_memory.memory_for_prompt(self.state, self.toolbox)
            self.assertEqual([entry["path"] for entry in view["entries"]], [overlay])
            self.assertIn("CURRENT_OVERLAY_BYTES", self.render(view))
            self.assertNotIn("TRACKED_SCAFFOLD_BYTES", self.render(view))

    def test_unreadable_overlay_does_not_shadow_a_still_direct_tool_read(self):
        tracked = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        overlay = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.put(tracked, "DIRECT_TRACKED_BYTES\n")
        self.put(overlay, "DENIED_OVERLAY_BYTES\n")
        policy = replace(self.policy, deny=(overlay,))
        toolbox = RepoToolbox(self.root, policy=policy, run_root=self.run_root)
        tools = agent_tools.ReadOnlyToolset(toolbox)
        with flow_memory.collect_tool_transcripts(self.state) as calls:
            result = tools.run("read_file", {"path": tracked})
        self.assertIn("DIRECT_TRACKED_BYTES", result)
        memory = flow_memory.derive_memory(self.state, {}, calls, toolbox)
        self.assertIn("DIRECT_TRACKED_BYTES", flow_memory.render_memory(memory, self.state, toolbox, "implement"))
        self.assertNotIn("DENIED_OVERLAY_BYTES", str(memory))

    def test_raw_body_only_ranges_are_remasked_using_full_file(self):
        source = "int verify_benchmark(int result) {\n  SECRET_ORACLE;\n}\nvisible_neighbor;\n"
        self.put(self.path, source)
        memory = self.derive((), {"kernel_file": self.path})
        memory["entries"][0].update(origin="tool", ranges=[
            {"coverage": {"first_line": 2, "last_line": 2, "full": False},
             "excerpt": "2:   SECRET_ORACLE;"},
            {"coverage": {"first_line": 4, "last_line": 4, "full": False},
             "excerpt": "4: visible_neighbor;"},
        ])
        text = self.render(memory)
        self.assertNotIn("SECRET_ORACLE", text)
        self.assertIn("visible_neighbor", text)
        self.assertEqual((self.root / self.path).read_text(), source)


if __name__ == "__main__":
    unittest.main()
