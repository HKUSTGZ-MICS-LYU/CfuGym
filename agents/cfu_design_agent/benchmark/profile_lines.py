"""Shared sanitized line parser for cycle-profile observations.

This only tokenizes observation lines. Workload adapters decide which fields
represent cycles, calls, or benchmark metadata.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..runtime.tools import filter_output


_PROFILE_LINE_RE = re.compile(r"^CFU_AGENT_PROFILE[ \t]+(?P<body>[^\n]*)", re.MULTILINE)


def profile_lines(text: str) -> Iterator[dict[str, int]]:
    """Yield numeric key/value fields from each complete CFU profile line."""
    for match in _PROFILE_LINE_RE.finditer(filter_output(text)):
        values: dict[str, int] = {}
        for item in match.group("body").split():
            if "=" not in item:
                continue
            key, raw = item.split("=", 1)
            try:
                values[key] = int(raw)
            except ValueError:
                continue
        yield values
