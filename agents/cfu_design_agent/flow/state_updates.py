"""State update semantics shared with the LangGraph Annotated reducers."""

from __future__ import annotations

import operator
from typing import Annotated, Any, get_args, get_origin, get_type_hints


def append_state_keys(state_type: type) -> tuple[str, ...]:
    """Discover list keys whose TypedDict reducer is ``operator.add``."""
    hints = get_type_hints(state_type, include_extras=True)
    return tuple(
        name for name, hint in hints.items()
        if get_origin(hint) is Annotated and operator.add in get_args(hint)[1:]
    )


def merge_update(state: dict[str, Any], update: Any, append_keys: tuple[str, ...]) -> None:
    """Apply a partial update using the reducers declared by the state schema."""
    if not isinstance(update, dict):
        return
    for key, value in update.items():
        if key in append_keys and isinstance(value, list):
            state[key] = [*(state.get(key) or []), *value]
        else:
            state[key] = value
