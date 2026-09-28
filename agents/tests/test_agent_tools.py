"""Read-only agent tools and the bounded tool-calling loop."""

from __future__ import annotations

import shutil
import unittest
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.runtime.llm import LlmResult, OpenAiTextClient
from agents.cfu_design_agent.runtime.tools import RepoToolbox, design_workspace_paths
from agents.tests.common import REPO_ROOT


def _tool_call_response(name: str, arguments: str, call_id: str = "call-1"):
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
        usage=None,
    )


def _multi_tool_call_response(count: int):
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(name="read_file", arguments='{"path": "a"}'),
            )
            for index in range(count)
        ],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
        usage=None,
    )


def _tool_calls_with_content_response(text: str):
    """A model that answers and asks for another tool in the same message."""
    message = SimpleNamespace(
        content=text,
        tool_calls=[
            SimpleNamespace(
                id="late-call",
                function=SimpleNamespace(name="read_file", arguments='{"path": "a"}'),
            )
        ],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
        usage=None,
    )


def _final_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage=None,
    )


class _FakeChatClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                if not outer._responses:
                    raise AssertionError("no scripted response left")
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=Completions())


class ToolLoopTest(unittest.TestCase):
    def _client(self) -> OpenAiTextClient:
        return OpenAiTextClient(config={"model": "m", "api_key": "k", "api_mode": "chat"})

    def test_tool_round_then_final_answer(self) -> None:
        client = self._client()
        fake = _FakeChatClient([
            _tool_call_response("read_file", '{"path": "bench/x.c"}'),
            _final_response("FINAL ANSWER"),
        ])
        executed: list[tuple[str, dict]] = []

        def runner(name, arguments):
            executed.append((name, arguments))
            return "1\tint x;"

        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "system",
                "user",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=runner,
                max_rounds=3,
            )

        self.assertEqual(result.text, "FINAL ANSWER")
        self.assertEqual(executed, [("read_file", {"path": "bench/x.c"})])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "read_file")
        self.assertEqual(result.tool_calls[0]["arguments"], {"path": "bench/x.c"})
        self.assertTrue(result.tool_calls[0]["ok"])
        self.assertEqual(result.tool_rounds, 1)
        # second request must include the tool observation
        roles = [message["role"] for message in fake.requests[1]["messages"]]
        self.assertIn("tool", roles)

    def test_loop_is_bounded_and_withdraws_tools(self) -> None:
        client = self._client()
        fake = _FakeChatClient([
            _tool_call_response("read_file", '{"path": "a"}'),
            _final_response("DONE"),
        ])

        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=1,
            )

        self.assertEqual(result.text, "DONE")
        self.assertEqual(result.tool_rounds, 1)
        # first request offered tools, the forced final request did not
        self.assertIn("tools", fake.requests[0])
        self.assertNotIn("tools", fake.requests[1])

    def test_tools_payload_rejection_degrades_to_plain_completion(self) -> None:
        client = self._client()
        outer = self

        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if "tools" in kwargs:
                    raise RuntimeError("tools not supported")
                return _final_response("PLAIN")

        fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=3,
            )
        self.assertEqual(result.text, "PLAIN")
        self.assertEqual(result.tool_calls, ())

    def test_tools_rejection_degrades_with_a_tools_unavailable_note(self) -> None:
        client = self._client()
        requests: list[dict] = []

        class Completions:
            def create(self, **kwargs):
                requests.append(kwargs)
                if "tools" in kwargs:
                    raise RuntimeError("tools not supported")
                return _final_response("PLAIN")

        fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=3,
            )
        self.assertEqual(result.text, "PLAIN")
        self.assertTrue(result.degraded_reason)
        # The tools request is retried once before the loop gives up.
        self.assertEqual(sum(1 for request in requests if "tools" in request), 2)
        # The degraded plain call must tell the model that tools are gone.
        final_user = requests[-1]["messages"][-1]["content"]
        self.assertIn("TOOL CALLING UNAVAILABLE", final_user)

    def test_budget_reminder_precedes_the_forced_final_turn(self) -> None:
        client = self._client()
        fake = _FakeChatClient([
            _tool_call_response("read_file", '{"path": "a"}'),
            _final_response("DELIVERED"),
        ])
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=1,
            )
        self.assertEqual(result.text, "DELIVERED")
        final_messages = fake.requests[1]["messages"]
        self.assertNotIn("tools", fake.requests[1])
        self.assertEqual(final_messages[-1]["role"], "user")
        self.assertIn("Tool budget exhausted", final_messages[-1]["content"])
        self.assertIn("EDIT_BEGIN", final_messages[-1]["content"])

    def test_post_withdrawal_tool_calls_keep_the_model_text(self) -> None:
        # Some gateways keep emitting tool_calls once the history holds tool
        # messages. The loop must not execute them or fall back to the
        # deterministic text; the model's content is the final answer.
        client = self._client()
        fake = _FakeChatClient([
            _tool_call_response("read_file", '{"path": "a"}'),
            _tool_calls_with_content_response("FINAL TEXT"),
        ])
        executed: list[str] = []
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: executed.append(name) or "obs",
                max_rounds=1,
                fallback="FALLBACK",
            )
        self.assertEqual(result.text, "FINAL TEXT")
        self.assertEqual(executed, ["read_file"])

    def test_per_round_call_cap_skips_extra_calls_and_answers_every_id(self) -> None:
        client = self._client()
        fake = _FakeChatClient([
            _multi_tool_call_response(6),
            _final_response("DONE"),
        ])
        executed: list[str] = []
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: executed.append(name) or "obs",
                max_rounds=1,
                max_calls_per_round=4,
            )
        self.assertEqual(result.text, "DONE")
        self.assertEqual(len(executed), 4)
        tools = [m for m in fake.requests[1]["messages"] if m.get("role") == "tool"]
        # Every call id must get a response, including the skipped ones.
        self.assertEqual(len(tools), 6)
        self.assertEqual(len([m for m in tools if m["content"].startswith("skipped:")]), 2)

    def test_total_call_cap_withdraws_tools_early(self) -> None:
        client = self._client()
        fake = _FakeChatClient([
            _tool_call_response("read_file", '{"path": "a"}'),
            _final_response("DONE"),
        ])
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=5,
                max_total_calls=1,
            )
        self.assertEqual(result.text, "DONE")
        self.assertIn("tools", fake.requests[0])
        # The cap was reached after one call, so tools are withdrawn next round.
        self.assertNotIn("tools", fake.requests[1])

    def test_tool_intent_without_a_call_is_nudged_once(self) -> None:
        # Observed failure: the model replied "I will first verify ..." with no
        # tool call, and the loop accepted it as the final answer.
        client = self._client()
        deferral = (
            "I will first verify the live CfuBus/SpinalHDL handshake idioms before "
            "selecting the repair."
        )
        fake = _FakeChatClient([
            _final_response(deferral),
            _tool_call_response("read_file", '{"path": "a"}'),
            _final_response("FINAL"),
        ])
        executed: list[str] = []
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: executed.append(name) or "obs",
                max_rounds=3,
            )
        self.assertEqual(result.text, "FINAL")
        self.assertEqual(executed, ["read_file"])
        nudged = [m for m in fake.requests[1]["messages"] if str(m.get("content", "")).startswith("You announced")]
        self.assertEqual(len(nudged), 1)

    def test_tool_intent_nudge_is_bounded(self) -> None:
        client = self._client()
        deferral = "Let me read the complete depthconv.c file first."
        fake = _FakeChatClient([_final_response(deferral), _final_response(deferral)])
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=3,
            )
        self.assertEqual(result.text, deferral)
        self.assertEqual(len(fake.requests), 2, "only one corrective re-ask")

    def test_missing_deliverable_is_nudged_when_required(self) -> None:
        # A rung that requires a block must not accept a plan-only answer while
        # tools are still available.
        client = self._client()
        plan = "Plan: zero-extend the input lane before the signed multiply."
        fake = _FakeChatClient([
            _final_response(plan),
            _final_response("FILE_BEGIN x\nbody\nFILE_END"),
        ])
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete_with_tools(
                "s",
                "u",
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                tool_runner=lambda name, args: "obs",
                max_rounds=2,
                is_deliverable=lambda text: "FILE_BEGIN" in text,
            )
        self.assertIn("FILE_BEGIN", result.text)
        nudged = [
            m
            for m in fake.requests[1]["messages"]
            if str(m.get("content", "")).startswith("Your reply contained no final deliverable")
        ]
        self.assertEqual(len(nudged), 1)


class ReadOnlyToolsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"agent-tools-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = REPO_ROOT / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        toolbox = RepoToolbox(
            REPO_ROOT, dry_run=False, stage="implementation", run_root=self.run_root
        )
        toolbox.write_text(
            f"{self.run_root}/workspace/cfu_benchmark/main.c",
            "int main(void) {\n  return 1;\n}\n",
        )
        self.toolset = ReadOnlyToolset(toolbox)
        self.target = f"{self.run_root}/workspace/cfu_benchmark/main.c"

    def test_observation_tool_is_not_exposed_outside_analysis(self) -> None:
        index = {"observations": {"obs_loop_x": {"id": "obs_loop_x"}}}
        tools = ReadOnlyToolset(self.toolset.toolbox, observation_index=index)
        self.assertNotIn("inspect_observation", [
            schema["function"]["name"] for schema in tools.schemas()
        ])
        self.assertIn("unknown tool", tools.run("inspect_observation", {"id": "obs_loop_x"}))

    def test_read_file_numbers_lines_and_slices(self) -> None:
        whole = self.toolset.run("read_file", {"path": self.target})
        self.assertIn("int main(void)", whole)
        self.assertIn("return 1;", whole)
        sliced = self.toolset.run(
            "read_file", {"path": self.target, "start_line": 2, "end_line": 2}
        )
        self.assertIn("return 1;", sliced)
        self.assertNotIn("int main", sliced)

    def test_list_dir_and_search_text(self) -> None:
        listing = self.toolset.run("list_dir", {"path": f"{self.run_root}/workspace/cfu_benchmark"})
        self.assertIn("main.c", listing)
        hits = self.toolset.run(
            "search_text", {"pattern": "return\\s+1", "path": f"{self.run_root}/workspace"}
        )
        self.assertIn("main.c:2:", hits)

    def test_paths_outside_the_stage_policy_are_denied(self) -> None:
        out = self.toolset.run("read_file", {"path": "README.md"})
        self.assertIn("denied by access policy", out)

    def test_unknown_tool_and_bad_glob_are_reported(self) -> None:
        self.assertIn("unknown tool", self.toolset.run("nope", {}))
        self.assertIn("glob must be relative", self.toolset.run("list_dir", {"glob": "../*"}))

    def test_tracked_design_target_reads_the_live_workspace_overlay(self) -> None:
        # The prompt shows, and an edit targets, the run-workspace overlay under
        # the tracked repo path. A repair tool read must therefore return the
        # model's previous change, not the pristine scaffold.
        toolbox = RepoToolbox(
            REPO_ROOT, dry_run=False, stage="implementation", run_root=self.run_root
        )
        toolbox.seed_design_workspace()
        tracked = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        overlay = design_workspace_paths(self.run_root)[tracked]
        (REPO_ROOT / overlay).write_text(
            "// CFU_AGENT_REPAIR_MARKER\n" + (REPO_ROOT / overlay).read_text()
        )
        toolset = ReadOnlyToolset(toolbox)
        self.assertIn("CFU_AGENT_REPAIR_MARKER", toolset.run("read_file", {"path": tracked}))
        self.assertIn(
            "CFU_AGENT_REPAIR_MARKER",
            toolset.run(
                "search_text",
                {"pattern": "CFU_AGENT_REPAIR_MARKER", "path": "src/main/scala/vexiiriscv/soc/mico"},
            ),
        )

    def test_analysis_keeps_reading_the_tracked_scaffold(self) -> None:
        # analysis may read the tracked scaffold but not the design workspace, so
        # the overlay redirect must not turn a valid read into a policy denial.
        RepoToolbox(
            REPO_ROOT, dry_run=False, stage="implementation", run_root=self.run_root
        ).seed_design_workspace()
        analysis = ReadOnlyToolset(
            RepoToolbox(REPO_ROOT, dry_run=False, stage="analysis", run_root=self.run_root)
        )
        out = analysis.run(
            "read_file",
            {"path": "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala", "end_line": 1},
        )
        self.assertNotIn("denied by access policy", out)
        self.assertIn("SPDX", out)


if __name__ == "__main__":
    unittest.main()
