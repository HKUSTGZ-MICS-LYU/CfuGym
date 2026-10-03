"""Declarative per-node LangGraph policies.

One place to answer "what does the framework do when this node raises, hangs, or
fails transiently", instead of scattering that decision across node bodies. The
table is applied by ``flow/wiring.py::register_langgraph`` at build time.

Deliberate omissions, so the next reader does not "fix" them:

* **No blanket ``error_handler``.** Most nodes already convert the failures they
  expect into an explicit failed gate. A graph-wide handler would turn an
  unexpected bug (``AttributeError``, ``TypeError``) into a repairable rung and
  spend the model/EDA budget on a framework defect. Only the
  benchmark-preparation rungs declare one, and that handler re-raises anything
  outside its accepted set.
* **No node-level ``timeout``.** Every external command already carries its own
  budget in ``runtime/checks.py``; a second, coarser cap here would only add a
  number nobody measured.
* **No ``retry_policy`` on LLM or EDA nodes.** A node retry re-runs the whole
  node, so it would re-spend tokens and simulation minutes. Retries are limited
  to cheap, idempotent, non-model nodes.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import RetryPolicy

from . import ladder

#: Cheap, idempotent nodes that may be retried: they call no model and no
#: toolchain, so a transient filesystem/parse error is worth one more attempt.
_IDEMPOTENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.2, jitter=True)

NODE_OPTIONS: dict[str, dict[str, Any]] = {
    "load_context": {"retry_policy": _IDEMPOTENT_RETRY},
    "route_input": {"retry_policy": _IDEMPOTENT_RETRY},
    "final_report": {"retry_policy": _IDEMPOTENT_RETRY},
    "prepare_c_project_profile": {"error_handler": ladder.prepare_failure_handler},
    "create_benchmark_project": {"error_handler": ladder.prepare_failure_handler},
    "prepare_embench_benchmark": {"error_handler": ladder.prepare_failure_handler},
}


def node_options() -> dict[str, dict[str, Any]]:
    """Per-node ``add_node`` options, keyed by flow node name."""
    return {name: dict(options) for name, options in NODE_OPTIONS.items()}
