"""Small OpenAI SDK adapter used by the CFU design graph.

Token usage is recorded into a module-level ledger so that the surrounding
graph can report per-stage and cumulative token consumption. The stage active
during a call is the one set by :func:`set_stage` (the graph node wrapper sets
it before each node runs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-5"

# Usage ledger: each LLM call appends a dict of its token counts, tagged with
# the stage that was active when the call was made.
_USAGE_LOG: list[dict[str, int]] = []
_CURRENT_STAGE = ""


@dataclass
class LlmResult:
    text: str
    used_llm: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def set_stage(stage: str) -> None:
    global _CURRENT_STAGE
    _CURRENT_STAGE = stage or ""


def current_stage() -> str:
    return _CURRENT_STAGE


def reset_usage_log() -> None:
    _USAGE_LOG.clear()


def record_usage(
    stage: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    _USAGE_LOG.append(
        {
            "stage": stage,
            "input": int(input_tokens or 0),
            "output": int(output_tokens or 0),
            "total": int(total_tokens or 0),
        }
    )


def usage_totals() -> dict[str, int]:
    """Cumulative token usage across all recorded calls."""
    return {
        "calls": len(_USAGE_LOG),
        "input": sum(entry["input"] for entry in _USAGE_LOG),
        "output": sum(entry["output"] for entry in _USAGE_LOG),
        "total": sum(entry["total"] for entry in _USAGE_LOG),
    }


def stage_usage(stage: str) -> dict[str, int]:
    matched = [entry for entry in _USAGE_LOG if entry.get("stage") == stage]
    return {
        "calls": len(matched),
        "input": sum(entry["input"] for entry in matched),
        "output": sum(entry["output"] for entry in matched),
        "total": sum(entry["total"] for entry in matched),
    }


def _read(value, *keys):
    """Read ``value`` as a pydantic model or dict, trying ``keys`` in order."""
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        return 0
    for key in keys:
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate
    return 0


def _extract_usage(response, kind: str) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    if kind == "responses":
        inp = int(_read(usage, "input_tokens", "input") or 0)
        out = int(_read(usage, "output_tokens", "output") or 0)
    else:
        inp = int(_read(usage, "prompt_tokens") or 0)
        out = int(_read(usage, "completion_tokens") or 0)
    total = int(_read(usage, "total_tokens", "total") or 0)
    if not total:
        total = inp + out
    return inp, out, total


class OpenAiTextClient:
    """OpenAI-compatible text client.

    ``CFU_AGENT_LLM_API`` selects the endpoint style:

    - ``auto`` (default): try the Responses API, then fall back to chat
      completions if the endpoint does not support it.
    - ``responses``: use ``client.responses.create`` only.
    - ``chat``: use ``client.chat.completions.create`` only.
    """

    def __init__(self, model: str | None = None, config: dict | None = None):
        cfg = config or {}
        self.config_name = cfg.get("name")
        self.model = (
            model
            or cfg.get("model")
            or os.environ.get("CFU_AGENT_MODEL")
            or DEFAULT_MODEL
        )
        self.api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        self.temperature = cfg.get("temperature")
        self.timeout = cfg.get("timeout")
        mode = cfg.get("api_mode") or os.environ.get("CFU_AGENT_LLM_API") or "auto"
        self.api_mode = str(mode).lower() if isinstance(mode, str) else "auto"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, *, fallback: str) -> LlmResult:
        if not self.available:
            return LlmResult(fallback, used_llm=False)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        try:
            if self.api_mode == "chat":
                return self._chat(client, system, user)
            if self.api_mode == "responses":
                return self._responses(client, system, user)
        except Exception:
            if self.api_mode != "auto":
                raise
            # auto: try Responses first, then fall back to chat completions.
        try:
            return self._responses(client, system, user)
        except Exception as responses_error:
            try:
                return self._chat(client, system, user)
            except Exception as chat_error:
                raise RuntimeError(
                    f"LLM call failed in both responses and chat modes. "
                    f"responses: {responses_error}; chat: {chat_error}"
                ) from responses_error

    def _responses(self, client, system: str, user: str) -> LlmResult:
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout:
            kwargs["timeout"] = self.timeout
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        inp, out, total = _extract_usage(response, "responses")
        record_usage(current_stage(), inp, out, total)
        return LlmResult(
            response.output_text,
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
        )

    def _chat(self, client, system: str, user: str) -> LlmResult:
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout:
            kwargs["timeout"] = self.timeout
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        inp, out, total = _extract_usage(response, "chat")
        record_usage(current_stage(), inp, out, total)
        return LlmResult(
            response.choices[0].message.content or "",
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
        )
