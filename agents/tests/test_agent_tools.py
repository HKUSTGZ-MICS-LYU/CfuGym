"""Read-only agent tools and the bounded tool-calling loop."""

from __future__ import annotations

import shutil
import unittest
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.llm import LlmResult, OpenAiTextClient
from agents.cfu_design_agent.tools import RepoToolbox
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


if __name__ == "__main__":
    unittest.main()
