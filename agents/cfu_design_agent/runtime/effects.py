"""External effects as LangGraph tasks.

A node that re-runs after a resume must not re-issue an external command that
already ran. ``@task`` records the result in the checkpoint and restores it
instead of re-executing the body, which is what ``runtime/persistence.py``
previously hand-rolled with its operation journal.

Two deliberate properties:

* **Serialisable arguments.** A task's arguments and result are checkpointed, so
  they are argv/stage/rung/timeout strings and a plain result dict. The run
  state is supplied out of band by the node wrapper (``flow/instrument.py``)
  rather than being passed in, so the checkpoint never has to serialise a
  toolbox or an access policy.
* **A plain path that still works.** ``@task`` raises outside a runnable
  context, and node functions are also called directly (tests, repair helpers),
  so the dispatcher falls back to the plain implementation whenever the wrapper
  has not installed a run state.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from typing import Any, Iterator

from langgraph.func import task

from . import checks, stage_services

#: The state of the node currently executing; ``None`` outside a graph run.
_RUN_STATE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cfu_effect_run_state", default=None
)


@contextmanager
def use_state(state: dict[str, Any]) -> Iterator[None]:
    """Publish the executing node's state for the duration of that node."""
    token = _RUN_STATE.set(state)
    try:
        yield
    finally:
        _RUN_STATE.reset(token)


def current_state() -> dict[str, Any] | None:
    """The state published for the currently executing node, if any."""
    return _RUN_STATE.get()


def inside_run() -> bool:
    """True only where a task can actually be created.

    Both halves matter: the node wrapper publishing a state, *and* a runnable
    context for ``@task``. A caller that publishes a state outside a graph (a
    test, or a helper reused outside the flow) must fall back to the plain call
    rather than raise ``Called get_config outside of a runnable context``.
    """
    if _RUN_STATE.get() is None:
        return False
    from langgraph.config import get_config

    try:
        get_config()
    except RuntimeError:
        return False
    return True


@task
def _command_task(argv: list[str], stage: str, rung: str, command_id: str,
                  timeout: int, mutate: bool, log_stage: str) -> dict[str, Any]:
    state = _RUN_STATE.get()
    if state is None:  # pragma: no cover - guarded by run_command
        raise RuntimeError("command effect ran without a node state")
    toolbox = stage_services._toolbox_for_stage(state, stage)
    return checks._run_check(state, toolbox, list(argv), rung=rung,
                             command_id=command_id, timeout=timeout, mutate=mutate,
                             log_stage=log_stage)


def run_command(state: dict[str, Any], toolbox: Any, argv: list[str], *,
                stage: str, rung: str, command_id: str = "", timeout: int = 3600,
                mutate: bool = False, log_stage: str = "") -> dict[str, Any]:
    """Run one ladder command, restoring a recorded result on resume.

    ``stage`` names the policy stage the command runs under. It is the only
    label the checkpointed task needs, because the task rebuilds the toolbox
    from the state; the toolbox argument serves the plain path unchanged.

    ``log_stage`` names the artifact stage that owns the persisted command log
    (``validation``/``cost`` by default, ``profile`` for the profile stage's own
    builds and simulations). Every executed command writes its complete
    stdout/stderr there, so the run keeps a trace the agent can read back.
    """
    if not inside_run():
        return checks._run_check(state, toolbox, list(argv), rung=rung,
                                 command_id=command_id, timeout=timeout, mutate=mutate,
                                 log_stage=log_stage)
    return _command_task(list(argv), stage, rung, command_id, timeout, mutate, log_stage).result()
