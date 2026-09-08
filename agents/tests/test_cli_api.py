"""--test-api branch behavior (with a mocked client, no network)."""

from __future__ import annotations

import argparse
import unittest

from agents.cfu_design_agent import cli
from agents.cfu_design_agent.llm import LlmResult
from agents.tests.common import REPO_ROOT


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.model = "fake"
        self.base_url = "https://example.com/v1"
        self.api_mode = "chat"
        self.api_key = "sk-fake"

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


if __name__ == "__main__":
    unittest.main()
