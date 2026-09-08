"""Named LLM config loading and client config resolution."""

from __future__ import annotations

from pathlib import Path
import unittest

from agents.cfu_design_agent.llm import OpenAiTextClient
from agents.cfu_design_agent.llm_config import (
    LlmConfigError,
    effective_config,
    load_config,
)
from agents.tests.common import REPO_ROOT


class LlmConfigTest(unittest.TestCase):
    def test_lists_named_configs(self) -> None:
        import json
        import tempfile

        from agents.cfu_design_agent.llm_config import list_configs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "agents" / "llm_configs"
            root.mkdir(parents=True)
            (root / "alpha.json").write_text(json.dumps({"name": "alpha"}))
            (root / "beta.json").write_text(json.dumps({"name": "beta"}))
            (root / "ignore.txt").write_text("x")
            names = list_configs(Path(directory))
        self.assertEqual(names, ["alpha", "beta"])

    def test_load_default_resolves_to_env_fallback(self) -> None:
        # effective_config returns an env-only config when default.json is absent.
        config = effective_config(REPO_ROOT, "default")
        self.assertIn(config["api_mode"], {"auto", "responses", "chat"})
        self.assertIsInstance(config["model"], str)
        self.assertIn("api_key", config)
        self.assertEqual(config["name"], "default")

    def test_invalid_mode_rejected(self) -> None:
        from agents.cfu_design_agent.llm_config import resolve_config

        with self.assertRaises(LlmConfigError):
            resolve_config("bad", {"name": "bad", "api_mode": "nope"})

    def test_missing_config_rejected(self) -> None:
        with self.assertRaises(LlmConfigError):
            load_config(REPO_ROOT, "does_not_exist")

    def test_effective_config_default_missing_uses_env(self) -> None:
        import tempfile

        from agents.cfu_design_agent.llm_config import effective_config

        with tempfile.TemporaryDirectory() as directory:
            cfg = effective_config(Path(directory), "default")
        self.assertEqual(cfg["name"], "default")
        self.assertIn("api_mode", cfg)
        self.assertIn("api_key", cfg)

    def test_client_uses_config_fields(self) -> None:
        client = OpenAiTextClient(
            config={
                "model": "my-model",
                "api_key": "sk-test",
                "base_url": "https://example.com/v1",
                "api_mode": "chat",
            }
        )
        self.assertEqual(client.model, "my-model")
        self.assertEqual(client.api_key, "sk-test")
        self.assertEqual(client.base_url, "https://example.com/v1")
        self.assertEqual(client.api_mode, "chat")
        self.assertTrue(client.available)


if __name__ == "__main__":
    unittest.main()
