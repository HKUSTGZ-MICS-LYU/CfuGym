"""--test-api branch behavior (with a mocked client, no network)."""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest

from agents.cfu_design_agent import cli
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.tests.common import REPO_ROOT


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.model = "fake"
        self.base_url = "https://example.com/v1"
        self.api_mode = "chat"
        self.api_key = "sk-fake"
        self.thinking = False
        self.reasoning_effort = None

    @property
    def available(self) -> bool:
        return True

    def complete(self, system, user, *, fallback):
        return LlmResult("OK", used_llm=True)


class _UnavailableClient(_FakeClient):
    @property
    def available(self) -> bool:
        return False


class _RaisingClient(_FakeClient):
    def complete(self, system, user, *, fallback):
        raise RuntimeError("boom")


def _args(**overrides):
    base = {
        "llm_config": "default",
        "model": "",
        "dry_run": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class ApiTest(unittest.TestCase):
    def test_success_branch(self) -> None:
        original = cli.OpenAiTextClient
        cli.OpenAiTextClient = _FakeClient
        try:
            self.assertEqual(cli.run_api_test(REPO_ROOT, _args()), 0)
        finally:
            cli.OpenAiTextClient = original

    def test_missing_key_branch(self) -> None:
        original = cli.OpenAiTextClient
        cli.OpenAiTextClient = _UnavailableClient
        try:
            self.assertEqual(cli.run_api_test(REPO_ROOT, _args()), 1)
        finally:
            cli.OpenAiTextClient = original

    def test_raising_client_branch(self) -> None:
        original = cli.OpenAiTextClient
        cli.OpenAiTextClient = _RaisingClient
        try:
            self.assertEqual(cli.run_api_test(REPO_ROOT, _args()), 1)
        finally:
            cli.OpenAiTextClient = original


class RunSummaryTest(unittest.TestCase):
    """The end-of-flow headline: cycles, speedup, and Yosys cost."""

    def _capture(self, state: dict) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli._print_run_summary(state)
        return buffer.getvalue()

    def test_reports_cycles_speedup_and_cost(self) -> None:
        text = self._capture(
            {
                "gates": {
                    "sim": {
                        "ok": True,
                        "reference_cycles": 7853884,
                        "cfu_cycles": 4712330,
                        "speedup_x100": 166,
                    }
                },
                "cost_results": [
                    {
                        "parsed": {
                            "cost": {
                                "top": "AgentCfu",
                                "cells": 3845,
                                "wire_bits": 4402,
                                "memories": 0,
                                "memory_bits": 0,
                            }
                        }
                    }
                ],
            }
        )
        self.assertIn("cycles_before = 7853884", text)
        self.assertIn("cycles_after  = 4712330", text)
        self.assertIn("speedup       = 1.66x", text)
        self.assertIn("cells=3845", text)
        self.assertIn("wire_bits=4402", text)

    def test_reports_na_when_skipped_or_unavailable(self) -> None:
        text = self._capture(
            {
                "gates": {"sim": {"ok": False, "skipped": True, "reason": "--skip-sim"}},
                "cost_results": [{"parsed": {}}],
            }
        )
        self.assertIn("cycles_before = n/a", text)
        self.assertIn("speedup       = n/a (--skip-sim)", text)
        self.assertIn("yosys_cost    = n/a", text)

    def test_reports_per_stage_llm_input_tokens(self) -> None:
        text = self._capture(
            {
                "token_usage": {"calls": 3, "input": 900, "output": 60, "total": 960},
                "token_usage_by_stage": {
                    "implement_hw": {"calls": 2, "input": 800, "output": 50, "total": 850},
                    "design_contract": {"calls": 1, "input": 100, "output": 10, "total": 110},
                },
            }
        )
        self.assertIn("llm_tokens:", text)
        self.assertIn("TOTAL", text)
        self.assertIn("calls=3", text)
        # Sorted by input tokens, largest first: the input-heavy stage is on top
        # and the TOTAL line comes last.
        self.assertLess(text.index("implement_hw"), text.index("design_contract"))
        self.assertLess(text.index("design_contract"), text.index("TOTAL"))

    def test_token_block_is_absent_without_usage(self) -> None:
        self.assertNotIn("llm_tokens:", self._capture({}))

    def test_reports_every_optimization_round_with_its_deltas(self) -> None:
        text = self._capture(
            {
                "loop_optimize": True,
                "gates": {"sim": {"ok": True, "speedup_x100": 210, "reference_cycles": 100, "cfu_cycles": 48}},
                "optimization_rounds": [
                    {"round": 0, "target": "baseline", "improved": True, "measured": True,
                     "speedup_x100": 200, "cost": {"cells": 1200}, "reason": "initial ladder"},
                    {"round": 1, "target": "implement_hw", "improved": True, "measured": True,
                     "speedup_x100": 250, "cost": {"cells": 1300}, "reason": "widen"},
                    {"round": 2, "target": "implement_sw", "improved": False, "measured": True,
                     "speedup_x100": 210, "cost": {"cells": 1300}, "reason": "block the loop"},
                ],
            }
        )
        self.assertIn("loop:", text)
        self.assertIn("round 0: baseline speedup=2.00x cells=1200", text)
        self.assertIn("round 1: implement_hw speedup=2.50x cells=1300 (+0.50x, +100 cells) (best)", text)
        self.assertIn("round 2: implement_sw speedup=2.10x cells=1300 (-0.40x, +0 cells) (not improved)", text)
        self.assertIn("WARNING: the final round is not the best (round 1)", text)

    def test_an_optimization_round_without_a_result_says_so(self) -> None:
        text = self._capture(
            {
                "loop_optimize": True,
                "optimization_rounds": [
                    {"round": 0, "target": "baseline", "improved": True, "measured": True,
                     "speedup_x100": 200, "cost": {"cells": 1200}, "reason": "initial ladder"},
                    {"round": 1, "target": "implement_hw", "improved": False, "measured": False,
                     "speedup_x100": 0, "cost": {}, "reason": "the SoC build failed"},
                ],
            }
        )
        self.assertIn("round 1: implement_hw not measured", text)
        self.assertNotIn("WARNING: the final round is not the best", text)

    def test_reports_the_profile_cache_status(self) -> None:
        text = self._capture(
            {
                "gates": {
                    "profile": {
                        "ok": True,
                        "profile_cache": {
                            "status": "hit",
                            "key": "96b5fabf72de8cb69ebf9b50aa",
                        },
                    }
                }
            }
        )
        self.assertIn("profile_cache = hit (key 96b5fabf72de)", text)


class StreamFlagTest(unittest.TestCase):
    def test_stream_llm_is_tristate_and_can_be_disabled(self) -> None:
        import os
        from unittest import mock

        parser = cli.build_parser()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CFU_AGENT_STREAM_LLM", None)
            # Unset keeps the config value; the client then defaults to streaming.
            self.assertIsNone(parser.parse_args(["--task", "x"]).stream_llm)
        self.assertTrue(parser.parse_args(["--task", "x", "--stream-llm"]).stream_llm)
        self.assertFalse(parser.parse_args(["--task", "x", "--no-stream-llm"]).stream_llm)

    def test_env_default_can_disable_streaming(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"CFU_AGENT_STREAM_LLM": "off"}):
            self.assertIs(cli.build_parser().parse_args(["--task", "x"]).stream_llm, False)
        with mock.patch.dict(os.environ, {"CFU_AGENT_STREAM_LLM": "on"}):
            self.assertIs(cli.build_parser().parse_args(["--task", "x"]).stream_llm, True)


class IrFeaturesFlagTest(unittest.TestCase):
    def test_ir_evidence_is_opt_in_and_passed_to_state(self) -> None:
        from unittest import mock

        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["--task", "x"]).ir_features, "off")
        self.assertEqual(parser.parse_args(["--task", "x", "--ir-features", "auto"]).ir_features, "auto")
        self.assertEqual(parser.parse_args(["--task", "x", "--ir-features", "require"]).ir_features, "require")
        self.assertEqual(parser.parse_args(["--task", "x", "--ir-features", "approx"]).ir_features, "approx")
        captured: dict = {}

        def _fake_run(state, **kwargs):
            captured.update(state)
            return {"status": "planned"}

        with mock.patch.object(cli, "run_agent", _fake_run), \
                mock.patch.object(cli, "export_graph_visualization", lambda path: {}), \
                mock.patch.object(cli, "enable_progress", lambda: None):
            cli.main(["--task", "x", "--ir-features", "auto"])
        self.assertEqual(captured.get("ir_features"), "auto")


class RepairRoutingFlagTest(unittest.TestCase):
    def test_default_is_auto_and_can_be_pinned(self) -> None:
        import os
        from unittest import mock

        parser = cli.build_parser()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CFU_AGENT_REPAIR_ROUTING", None)
            self.assertEqual(parser.parse_args(["--task", "x"]).repair_routing, "auto")
        with mock.patch.dict(os.environ, {"CFU_AGENT_REPAIR_ROUTING": "fixed"}):
            self.assertEqual(
                cli.build_parser().parse_args(["--task", "x"]).repair_routing, "fixed"
            )
        self.assertEqual(
            parser.parse_args(["--task", "x", "--repair-routing", "fixed"]).repair_routing,
            "fixed",
        )

    def test_main_passes_the_choice_into_the_state(self) -> None:
        from unittest import mock

        captured: dict = {}

        def _fake_run(state, **kwargs):
            captured.update(state)
            return {"status": "planned"}

        with mock.patch.object(cli, "run_agent", _fake_run), \
                mock.patch.object(cli, "export_graph_visualization", lambda path: {}), \
                mock.patch.object(cli, "enable_progress", lambda: None):
            cli.main(["--task", "x", "--repair-routing", "fixed"])
        self.assertEqual(captured.get("repair_routing"), "fixed")


class SoftwareChangeFlagTest(unittest.TestCase):
    def test_default_is_auto_and_can_be_pinned(self) -> None:
        import os
        from unittest import mock

        parser = cli.build_parser()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CFU_AGENT_SOFTWARE_CHANGE", None)
            self.assertEqual(parser.parse_args(["--task", "x"]).software_change, "auto")
        with mock.patch.dict(os.environ, {"CFU_AGENT_SOFTWARE_CHANGE": "always"}):
            self.assertEqual(
                cli.build_parser().parse_args(["--task", "x"]).software_change, "always"
            )
        self.assertEqual(
            parser.parse_args(["--task", "x", "--software-change", "always"]).software_change,
            "always",
        )

    def test_main_passes_the_choice_into_the_state(self) -> None:
        from unittest import mock

        captured: dict = {}

        def _fake_run(state, **kwargs):
            captured.update(state)
            return {"status": "planned"}

        with mock.patch.object(cli, "run_agent", _fake_run), \
                mock.patch.object(cli, "export_graph_visualization", lambda path: {}), \
                mock.patch.object(cli, "enable_progress", lambda: None):
            cli.main(["--task", "x", "--software-change", "always"])
        self.assertEqual(captured.get("software_change"), "always")


class LoopFlagTest(unittest.TestCase):
    def test_loop_is_off_by_default_and_can_be_enabled(self) -> None:
        import os
        from unittest import mock

        parser = cli.build_parser()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CFU_AGENT_LOOP", None)
            self.assertFalse(parser.parse_args(["--task", "x"]).loop_optimize)
        self.assertTrue(parser.parse_args(["--task", "x", "--loop"]).loop_optimize)
        with mock.patch.dict(os.environ, {"CFU_AGENT_LOOP": "on"}):
            self.assertTrue(cli.build_parser().parse_args(["--task", "x"]).loop_optimize)

    def test_main_passes_the_flag_into_the_state(self) -> None:
        from unittest import mock

        captured: dict = {}

        def _fake_run(state, **kwargs):
            captured.update(state)
            return {"status": "planned"}

        with mock.patch.object(cli, "run_agent", _fake_run), \
                mock.patch.object(cli, "export_graph_visualization", lambda path: {}), \
                mock.patch.object(cli, "enable_progress", lambda: None):
            cli.main(["--task", "x", "--loop", "--max-iters", "3"])
        self.assertTrue(captured.get("loop_optimize"))
        self.assertEqual(captured.get("max_iters"), 3)


if __name__ == "__main__":
    unittest.main()
