"""Typed state shared across the CFU design LangGraph.

The state is one flat set of LangGraph channels, but it is *declared* in four
groups so its shape is readable:

* :class:`~.config.RunConfig` — the run's inputs (knobs, budgets, permissions);
* :class:`~.evidence.Evidence` — what the run observed or produced;
* :class:`~.ledger.Ledger` — gate verdicts, budgets and accumulators;
* :class:`~.optimization.Optimization` — the ``--loop`` ISA revisions and rounds.

Composition is deliberate rather than cosmetic: ``TypedDict`` inheritance keeps
every inherited key in ``__annotations__`` and in ``get_type_hints``, so
LangGraph still registers the same channels and reducers. The names are
re-exported here, and this module stays the single import site for callers.
"""

from __future__ import annotations

from .config import RunConfig
from .evidence import Evidence
from .ledger import Ledger
from .optimization import Optimization
from .schemas import (
    AgentMode,
    BenchmarkProject,
    CommandResult,
    InputKind,
    ProfileRegion,
    ProfileResult,
    RepoContext,
    Visibility,
    WorkloadSpec,
)


class CfuDesignState(RunConfig, Evidence, Ledger, Optimization, total=False):
    """Everything one run carries between nodes."""


def append_keys() -> frozenset[str]:
    """The keys whose LangGraph channels concatenate rather than replace.

    Derived from the annotations, so a new accumulator cannot be forgotten by a
    caller that has to merge two nodes' updates by hand.
    """
    import typing

    hints = typing.get_type_hints(CfuDesignState, include_extras=True)
    return frozenset(key for key, value in hints.items()
                     if getattr(value, "__metadata__", ()))


def append_history(state: CfuDesignState, line: str) -> list[str]:
    return [line]
