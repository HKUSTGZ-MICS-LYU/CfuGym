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

from .llm import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_SUMMARY,
    VALID_REASONING_EFFORTS,
)

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

    # OpenAI/Azure reasoning models expose only a summary through the Responses
    # API, and only when it is requested; default to "auto" while thinking is on.
    summary_raw = data.get("reasoning_summary")
    if summary_raw is not None and not isinstance(summary_raw, str):
        raise LlmConfigError(
            f"reasoning_summary must be a string, got {type(summary_raw).__name__}"
        )
    summary = (summary_raw or "").strip()
    if not thinking:
        summary = ""
    elif not summary:
        summary = DEFAULT_REASONING_SUMMARY

    context_options: dict[str, int] = {}
    # ``CFU_AGENT_OUTPUT_RESERVE`` overrides the file's completion budget so a
    # reasoning model can be given more room without editing every config. The
    # value is also the context-budget output reservation, so it stays subject to
    # the same ``< context_window_tokens`` bound.
    env_reserve = (os.environ.get("CFU_AGENT_OUTPUT_RESERVE") or "").strip()
    configured: dict[str, Any] = {field: data.get(field)
                                  for field in ("context_window_tokens", "output_reserve_tokens")}
    if env_reserve:
        configured["output_reserve_tokens"] = env_reserve
    for field, value in configured.items():
        if value is None:
            continue
        if field == "output_reserve_tokens" and isinstance(value, str):
            try:
                value = int(value, 10)
            except ValueError as exc:
                raise LlmConfigError(
                    f"CFU_AGENT_OUTPUT_RESERVE must be a positive integer, got {value!r}"
                ) from exc
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LlmConfigError(f"{field} must be a positive integer")
        context_options[field] = value
    if (context_options.get("context_window_tokens") is not None
            and context_options.get("output_reserve_tokens", 4096) >= context_options["context_window_tokens"]):
        raise LlmConfigError("output_reserve_tokens must be smaller than context_window_tokens")

    # Token streaming: config may opt out; unset keeps the client default (on).
    stream_raw = data.get("stream")
    stream = None if stream_raw in (None, "") else _coerce_bool(stream_raw, "stream")

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
        "reasoning_summary": summary or None,
        "stream": stream,
        "thinking_body": dict(thinking_body) if isinstance(thinking_body, dict) else None,
        **context_options,
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
            "reasoning_summary": os.environ.get("CFU_AGENT_REASONING_SUMMARY", ""),
            "description": "Environment-only config (no JSON file).",
        },
    )
