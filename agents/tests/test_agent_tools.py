"""Read-only agent tools and the bounded tool-calling loop."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent.runtime import context_budget, failures, llm, processes
from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.runtime.llm import OpenAiTextClient
from agents.cfu_design_agent.runtime.policy import load_policy
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

    def test_malformed_calls_get_feedback_and_do_not_poison_next_request(self):
        for name, arguments, call_id in (('', '{}', 'bad'), ('read_file', '{}', ''),
                                         ('read_file', '{broken', 'bad'), ('read_file', '[]', 'bad')):
            with self.subTest(name=name, arguments=arguments, call_id=call_id):
                fake = _FakeChatClient([
                    _tool_call_response(name, arguments, call_id),
                    _tool_call_response('read_file', '{"path":"fixed"}', 'fixed'),
                    _final_response('DONE'),
                ])
                runner = mock.Mock(return_value='body')
                with mock.patch('openai.OpenAI', return_value=fake):
                    result = self._client().complete_with_tools('s', 'u',
                        tools=[{'type':'function','function':{'name':'read_file'}}],
                        tool_runner=runner, max_rounds=1, max_total_calls=1)
                runner.assert_called_once_with('read_file', {'path':'fixed'})
                self.assertEqual(result.text, 'DONE')
                self.assertIn('Total tool retries used: 1/10', str(fake.requests[1]['messages']))
                llm._responses_items(fake.requests[1]['messages'])

    def test_mixed_malformed_and_valid_calls_keep_completed_batches_contiguous(self):
        response = _multi_tool_call_response(3)
        response.choices[0].message.tool_calls[1].function.name = ''
        fake = _FakeChatClient([response, _final_response('DONE')])
        runner = mock.Mock(return_value='body')
        with mock.patch('openai.OpenAI', return_value=fake):
            result = self._client().complete_with_tools('s','u',
                tools=[{'type':'function','function':{'name':'read_file'}}], tool_runner=runner)
        self.assertEqual(result.text, 'DONE')
        self.assertEqual(runner.call_count, 2)
        context_budget._tool_batches(fake.requests[1]['messages'])
        llm._responses_items(fake.requests[1]['messages'])
        self.assertEqual([m['role'] for m in fake.requests[1]['messages']][-3:], ['tool','tool','user'])

    def test_tool_failure_retries_are_total_not_reset_by_success(self):
        fake = _FakeChatClient([
            _tool_call_response('read_file', '{}', f'call-{i}') for i in range(4)])
        runner = mock.Mock(side_effect=[ValueError('first'), 'body', 'usage error: second', ValueError('third')])
        with mock.patch('openai.OpenAI', return_value=fake):
            with self.assertRaises(failures.StageDeliveryFailure) as caught:
                self._client().complete_with_tools('s', 'u',
                    tools=[{'type':'function','function':{'name':'read_file'}}],
                    tool_runner=runner, max_tool_retries=2)
        self.assertEqual(caught.exception.terminal_reason, 'tool_retries_exhausted')
        self.assertEqual(len(caught.exception.tool_calls), 4)
        self.assertEqual(runner.call_count, 4)

    def test_default_ten_retries_allow_a_corrected_call_after_round_cap(self):
        fake = _FakeChatClient([
            *[_tool_call_response('read_file', '{}', f'call-{i}') for i in range(11)],
            _final_response('DONE')])
        runner = mock.Mock(side_effect=[ValueError('bad')] * 10 + ['body'])
        with mock.patch('openai.OpenAI', return_value=fake):
            result = self._client().complete_with_tools('s', 'u',
                tools=[{'type':'function','function':{'name':'read_file'}}],
                tool_runner=runner, max_rounds=1, max_total_calls=1)
        self.assertEqual(result.text, 'DONE')
        self.assertEqual(runner.call_count, 11)
        self.assertIn('10/10', str(fake.requests[-2]['messages']))

    def test_zero_retry_budget_and_cancellation_do_not_replay_tools(self):
        for error in (ValueError('bad'), processes.ValidationCancelled('cancelled')):
            with self.subTest(error=type(error).__name__):
                fake = _FakeChatClient([_tool_call_response('read_file', '{}')])
                runner = mock.Mock(side_effect=error)
                expected = processes.ValidationCancelled if isinstance(error, processes.ValidationCancelled) else failures.StageDeliveryFailure
                with mock.patch('openai.OpenAI', return_value=fake), self.assertRaises(expected):
                    self._client().complete_with_tools('s','u',
                        tools=[{'type':'function','function':{'name':'read_file'}}],
                        tool_runner=runner, max_tool_retries=0)
                runner.assert_called_once()
                self.assertEqual(len(fake.requests), 1)

    def test_loop_is_bounded_withdraws_tools_and_adds_budget_reminder(self) -> None:
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
        self.assertEqual(result.tool_rounds, 1)
        self.assertIn("tools", fake.requests[0])
        self.assertNotIn("tools", fake.requests[1])
        final_messages = fake.requests[1]["messages"]
        self.assertEqual(final_messages[-1]["role"], "user")
        self.assertIn("Tool budget exhausted", final_messages[-1]["content"])
        self.assertIn("EDIT_BEGIN", final_messages[-1]["content"])

    def test_tools_payload_rejection_fails_closed_without_plain_completion(self) -> None:
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
            with self.assertRaisesRegex(RuntimeError, "tool-enabled model request failed.*tools not supported"):
                client.complete_with_tools(
                    "s", "u", tools=[{"type": "function", "function": {"name": "read_file"}}],
                    tool_runner=lambda name, args: "obs", max_rounds=3,
                )
        self.assertEqual(len(requests), 1)
        self.assertIn("tools", requests[0])

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

    def test_empty_answer_is_retried_once(self) -> None:
        # Observed failure (run 20261003-095209): after five productive tool
        # rounds the endpoint returned a 200 with no content and no tool call,
        # and the loop raised immediately, killing the whole run.
        client = self._client()
        fake = _FakeChatClient([_final_response(""), _final_response("FINAL")])
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
        self.assertEqual(executed, [])
        self.assertEqual(len(fake.requests), 2)
        nudged = [m for m in fake.requests[1]["messages"]
                  if "previous reply was empty" in str(m.get("content", ""))]
        self.assertEqual(len(nudged), 1)
        self.assertEqual(nudged[0]["role"], "user")

    def test_empty_answer_retry_is_bounded(self) -> None:
        client = self._client()
        fake = _FakeChatClient([_final_response(""), _final_response("")])
        with mock.patch("openai.OpenAI", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "empty stage answer"):
                client.complete_with_tools(
                    "s",
                    "u",
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                    tool_runner=lambda name, args: "obs",
                    max_rounds=3,
                )
        self.assertEqual(len(fake.requests), 2, "at most one empty-answer re-ask")


class ReadOnlyToolsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temp.cleanup)
        cls.workdir = Path(cls._temp.name)
        policy_dest = cls.workdir / "agents/specs/agent_access_policy.yaml"
        policy_dest.parent.mkdir(parents=True, exist_ok=True)
        policy_dest.write_bytes((REPO_ROOT / "agents/specs/agent_access_policy.yaml").read_bytes())
        mico = cls.workdir / "src/main/scala/vexiiriscv/soc/mico"
        mico.mkdir(parents=True, exist_ok=True)
        (mico / "AgentCfu.scala").write_text("class AgentCfu extends AgentCfuBase\n")
        (mico / "AgentCfuFiber.scala").write_text("class AgentCfuFiber\n")
        (mico / "AgentCfuLib.scala").write_text("abstract class AgentCfuBase\n")
        for denied_dir in ("docs", "target", "ext/SpinalHDL"):
            (cls.workdir / denied_dir).mkdir(parents=True, exist_ok=True)
            (cls.workdir / denied_dir / "file.txt").write_text("x\n")
        (cls.workdir / "README.md").write_text("readme\n")
        cls.policy = load_policy(cls.workdir)

    def setUp(self) -> None:
        self.run_id = f"agent-tools-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = self.workdir / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        toolbox = RepoToolbox(
            self.workdir, policy=self.policy, dry_run=False, stage="implementation", run_root=self.run_root
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

    def test_ir_views_schema_is_narrowed_for_ablation(self) -> None:
        index = {"observations": {"obs_loop_x": {"id": "obs_loop_x"}}}
        toolbox = RepoToolbox(self.workdir, policy=self.policy, dry_run=True, stage="analysis")

        def view_enum(toolset):
            schema = next(item for item in toolset.schemas()
                          if item["function"]["name"] == "inspect_observation")
            return (schema["function"]["parameters"]["properties"]["view"]["enum"],
                    schema["function"]["description"])

        enum, description = view_enum(ReadOnlyToolset(toolbox, observation_index=index, ir_views=False))
        self.assertEqual(enum, ["summary", "asm"])
        self.assertNotIn("LLVM IR", description)
        enum, description = view_enum(ReadOnlyToolset(toolbox, observation_index=index))
        self.assertEqual(enum, ["summary", "asm", "ir", "features"])
        self.assertIn("LLVM IR", description)

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
        sub_listing = self.toolset.run("list_dir", {"path": self.run_root})
        self.assertIn("{{run-dir}}/workspace/", sub_listing)
        self.assertNotIn("(no visible entries)", sub_listing)

    def test_run_dir_placeholder_is_accepted_and_echoed_back(self) -> None:
        text = self.toolset.run("read_file", {"path": "{{run-dir}}/workspace/cfu_benchmark/main.c"})
        self.assertIn("int main(void)", text)
        missing = self.toolset.run("read_file", {"path": "{{run-dir}}/workspace/nope.c"})
        self.assertIn("{{run-dir}}/workspace/nope.c", missing)
        self.assertNotIn(self.run_root, missing)

    def test_list_command_logs_finds_the_runs_build_and_simulation_logs(self) -> None:
        logs = self.run_dir / "10_validation/logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "sim_003.log").write_text(
            "command: sbt runMain MiCoSocSim --load-elf x.elf\n"
            "command_id: soc_sim\n"
            "cwd: /repo\n"
            "returncode: 1\n"
            "timed_out: False\n\n=== STDOUT ===\nEMBENCH_VERIFY_FAIL result=7\n")
        cost = self.run_dir / "11_cost/logs"
        cost.mkdir(parents=True, exist_ok=True)
        (cost / "cost_000.log").write_text(
            "command: yosys -s cost.ys\ncommand_id: yosys_cost\ncwd: /repo\n"
            "returncode: 0\n\n=== STDOUT ===\ncells: 42\n")
        listing = self.toolset.run("list_command_logs", {})
        self.assertIn("2 command log(s)", listing)
        self.assertIn("{{run-dir}}/10_validation/logs/sim_003.log", listing)
        self.assertIn("rung=sim rc=1", listing)
        self.assertIn("sbt runMain MiCoSocSim --load-elf x.elf", listing)
        filtered = self.toolset.run("list_command_logs", {"rung": "cost", "limit": 1})
        self.assertIn("1 command log(s)", filtered)
        self.assertIn("cost_000.log", filtered)
        self.assertNotIn("sim_003.log", filtered)
        missing = self.toolset.run("list_command_logs", {"rung": "nope"})
        self.assertIn("no command log", missing)
        self.assertTrue(self.toolset.run("list_command_logs", {"limit": "many"})
                        .startswith("usage error:"))

    def test_the_platform_library_is_discoverable_without_a_repo_wide_search(self) -> None:
        tools = ReadOnlyToolset(
            RepoToolbox(self.workdir, policy=self.policy, dry_run=True, stage="implementation")
        )
        listing = tools.run("list_dir", {"path": "src/main/scala/vexiiriscv/soc/mico"})
        self.assertIn("AgentCfuLib.scala", listing)
        hits = tools.run("search_text", {"pattern": "class AgentCfuBase", "path": "."})
        self.assertIn("AgentCfuLib.scala:", hits)
        self.assertNotIn("usage error", hits)

    def test_a_missing_path_says_so_and_suggests_where_it_lives(self) -> None:
        out = self.toolset.run("search_text", {"pattern": "x", "path": "vexiiriscv"})
        self.assertTrue(out.startswith("usage error: no such path"), out)
        self.assertIn("did you mean: src/main/scala/vexiiriscv", out)
        missing = self.toolset.run("read_file", {"path": "nowhere/AgentCfu.scala"})
        self.assertIn("no such file", missing)
        self.assertNotIn("denied by access policy", missing)

    def test_invisible_trees_and_policy_denials_are_reported(self) -> None:
        for path in ("docs", "target"):
            denied = self.toolset.run("list_dir", {"path": path})
            self.assertIn("denied by access policy", denied)
        self.assertIn(
            "denied by access policy",
            self.toolset.run("search_text", {"pattern": "x", "path": "ext/SpinalHDL"}),
        )
        self.assertIn(
            "denied by access policy",
            self.toolset.run("read_file", {"path": "README.md"}),
        )
        self.assertIn("unknown tool", self.toolset.run("nope", {}))
        self.assertIn("glob must be relative", self.toolset.run("list_dir", {"glob": "../*"}))

    def test_tracked_design_target_reads_the_live_workspace_overlay_in_every_stage(self) -> None:
        toolbox = RepoToolbox(
            self.workdir, policy=self.policy, dry_run=False, stage="implementation", run_root=self.run_root
        )
        toolbox.seed_design_workspace()
        tracked = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
        overlay = design_workspace_paths(self.run_root)[tracked]
        (self.workdir / overlay).write_text(
            "// CFU_AGENT_REPAIR_MARKER\n" + (self.workdir / overlay).read_text()
        )
        for stage in ("analysis", "isa", "implementation"):
            tools = ReadOnlyToolset(
                RepoToolbox(self.workdir, policy=self.policy, dry_run=False, stage=stage, run_root=self.run_root)
            )
            out = tools.run("read_file", {"path": tracked, "end_line": 1})
            self.assertIn("CFU_AGENT_REPAIR_MARKER", out, stage)
        self.assertIn(
            "CFU_AGENT_REPAIR_MARKER",
            tools.run(
                "search_text",
                {"pattern": "CFU_AGENT_REPAIR_MARKER", "path": "src/main/scala/vexiiriscv/soc/mico"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
