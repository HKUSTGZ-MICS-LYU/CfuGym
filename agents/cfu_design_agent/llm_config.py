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

from .llm import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, VALID_REASONING_EFFORTS

VALID_API_MODES = ("auto", "responses", "chat")


class LlmConfigError(ValueError):
    """Raised when an LLM config is missing or invalid."""


def _coerce_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled", ""}:
            return False
    raise LlmConfigError(f"{field} must be a boolean, got {value!r}")


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

    # Thinking is off unless the config opts in. ``reasoning_effort`` (or its
    # ``effort`` alias) also implies thinking; ``thinking_body`` overrides the
    # gateway-specific switch shape sent as chat ``extra_body``.
    effort = data.get("reasoning_effort")
    if effort is None:
        effort = data.get("effort")
    if isinstance(effort, str):
        effort = effort.strip().lower() or None
    elif effort is not None:
        raise LlmConfigError(f"reasoning_effort must be a string, got {type(effort).__name__}")
    if effort is not None and effort not in VALID_REASONING_EFFORTS:
        raise LlmConfigError(
            f"invalid reasoning_effort {effort!r} in config {name!r}; "
            f"choose from {VALID_REASONING_EFFORTS}"
        )

    thinking_raw = data.get("thinking")
    if thinking_raw is None:
        thinking = effort is not None
    else:
        thinking = _coerce_bool(thinking_raw, "thinking")
    if thinking and effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if not thinking:
        effort = None

    thinking_body = data.get("thinking_body")
    if thinking_body is not None and not isinstance(thinking_body, dict):
        raise LlmConfigError(f"thinking_body must be a JSON object, got {type(thinking_body).__name__}")

    return {
        "name": name,
        "description": data.get("description", ""),
        "model": model,
        "base_url": base_url,
        "api_key": api_key or None,
        "api_mode": api_mode,
        "temperature": temperature if isinstance(temperature, (int, float)) else 0,
        "timeout": int(timeout) if isinstance(timeout, (int, float)) else 120,
        "thinking": thinking,
        "reasoning_effort": effort,
        "thinking_body": dict(thinking_body) if isinstance(thinking_body, dict) else None,
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
            "thinking": os.environ.get("CFU_AGENT_THINKING", ""),
            "reasoning_effort": os.environ.get("CFU_AGENT_REASONING_EFFORT", ""),
            "description": "Environment-only config (no JSON file).",
        },
    )
