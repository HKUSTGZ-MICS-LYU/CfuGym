"""Small OpenAI SDK adapter used by the CFU design graph."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "gpt-5"


@dataclass
class LlmResult:
    text: str
    used_llm: bool


class OpenAiTextClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("CFU_AGENT_MODEL", DEFAULT_MODEL)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, *, fallback: str) -> LlmResult:
        if not self.available:
            return LlmResult(fallback, used_llm=False)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return LlmResult(response.output_text, used_llm=True)

