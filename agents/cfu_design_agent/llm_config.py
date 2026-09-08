"""Load and resolve named LLM configurations from agents/llm_configs/*.json.

Precedence for each field (highest wins):

- api_key:   config "api_key" (inline) > config "api_key_env" var > OPENAI_API_KEY
- model:     config "model" > CFU_AGENT_MODEL > DEFAULT_MODEL
- base_url:  config "base_url" > OPENAI_BASE_URL
- api_mode:  CFU_AGENT_LLM_API > config "api_mode" > "auto"
- temperature / timeout: from config, with defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .llm import DEFAULT_MODEL

VALID_API_MODES = ("auto", "responses", "chat")


class LlmConfigError(ValueError):
    """Raised when an LLM config is missing or invalid."""


def config_dir(workdir: Path) -> Path:
    return workdir / "agents" / "llm_configs"


def list_configs(workdir: Path) -> list[str]:
    root = config_dir(workdir)
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())


def load_config(workdir: Path, name: str) -> dict[str, Any]:
    """Load and resolve a named config into an effective client config."""
    path = config_dir(workdir) / f"{name}.json"
    if not path.exists():
        raise LlmConfigError(f"LLM config not found: {name} (available: {', '.join(list_configs(workdir)) or 'none'})")
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError as exc:
        raise LlmConfigError(f"LLM config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LlmConfigError(f"LLM config must be a JSON object: {path}")
    return resolve_config(name, data)


def resolve_config(name: str, data: dict[str, Any]) -> dict[str, Any]:
    api_key = data.get("api_key") or None
    if not api_key:
        key_env = data.get("api_key_env") or "OPENAI_API_KEY"
        api_key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")

    model = data.get("model") or os.environ.get("CFU_AGENT_MODEL") or DEFAULT_MODEL
    base_url = data.get("base_url") or os.environ.get("OPENAI_BASE_URL") or None
    api_mode = os.environ.get("CFU_AGENT_LLM_API") or data.get("api_mode") or "auto"

    if api_mode not in VALID_API_MODES:
        raise LlmConfigError(f"invalid api_mode {api_mode!r} in config {name!r}; choose from {VALID_API_MODES}")

    temperature = data.get("temperature")
    timeout = data.get("timeout")

    return {
        "name": name,
        "description": data.get("description", ""),
        "model": model,
        "base_url": base_url,
        "api_key": api_key or None,
        "api_mode": api_mode,
        "temperature": temperature if isinstance(temperature, (int, float)) else 0,
        "timeout": int(timeout) if isinstance(timeout, (int, float)) else 120,
    }


def effective_config(workdir: Path, name: str) -> dict[str, Any]:
    """Return the effective config for ``name``.

    ``name`` defaults to ``default``. If no ``default.json`` exists, an
    environment-only config is built so the framework does not depend on a
    specific config file. An explicitly named config that is missing still
    raises ``LlmConfigError``.
    """
    selected = (name or "default").strip() or "default"
    if selected == "default" and not (config_dir(workdir) / "default.json").exists():
        return env_only_config("default")
    return load_config(workdir, selected)


def env_only_config(name: str) -> dict[str, Any]:
    return resolve_config(
        name,
        {
            "model": os.environ.get("CFU_AGENT_MODEL", ""),
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "api_key_env": "OPENAI_API_KEY",
            "api_mode": os.environ.get("CFU_AGENT_LLM_API", "auto"),
            "description": "Environment-only config (no JSON file).",
        },
    )
