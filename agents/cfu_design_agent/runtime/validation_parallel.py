"""A bounded two-lane race, without LangGraph child-task barriers."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
import threading
from typing import Any, Callable

from . import processes

_FAILURE: ContextVar[Callable[[], None] | None] = ContextVar('validation_failure', default=None)


def command_failed() -> None:
    """Called by plain checks immediately after a nonzero command exit."""
    callback = _FAILURE.get()
    if callback is not None:
        callback()


@contextmanager
def _failure_handler(callback):
    handle = _FAILURE.set(callback)
    try:
        yield
    finally:
        _FAILURE.reset(handle)


@dataclass
class PairResult:
    updates: dict[str, dict[str, Any]]
    cancelled: list[str]
    failure: str = ''
    cancelled_logs: dict[str, dict] = field(default_factory=dict)


def run_pair(workers: dict[str, Callable[[], dict]], *, parallel: bool = True) -> PairResult:
    """Run independent workers; only the caller may publish state or spend budget.

    Workers must cooperate at command/artifact boundaries. All live subprocesses
    are killed on first failure; a worker that ignores cancellation fails loudly
    after the bounded drain rather than letting the repair overlap it.
    """
    from . import effects

    tokens = {name: processes.CancellationToken() for name in workers}
    lock = threading.Lock()
    first_failure: list[str] = []

    def failed(name: str) -> None:
        with lock:
            if first_failure:
                return
            first_failure.append(name)
        for sibling, token in tokens.items():
            if sibling != name:
                token.cancel()

    def invoke(name: str, worker: Callable[[], dict]) -> dict:
        with processes.use_token(tokens[name]), effects.plain_commands(), _failure_handler(lambda: failed(name)):
            try:
                update = worker()
                processes.check_cancelled()
                gate = (update.get('gates') or {}).get(name)
                if not isinstance(gate, dict):
                    raise RuntimeError(f'{name} validation worker returned no rung gate')
                if not gate.get('ok') and not gate.get('skipped'):
                    failed(name)
                return update
            except processes.ValidationCancelled:
                raise
            except BaseException:
                failed(name)
                raise

    updates: dict[str, dict] = {}
    cancelled: list[str] = []
    cancelled_logs: dict[str, dict] = {}

    def record_cancelled(name, error):
        cancelled.append(name)
        if error.result:
            cancelled_logs[name] = {key: error.result[key] for key in
                                    ('log_path', 'full_log_path', 'returncode', 'command_id')
                                    if key in error.result}

    if not parallel:
        for name, worker in workers.items():
            try:
                updates[name] = copy_context().run(invoke, name, worker)
            except processes.ValidationCancelled as error:
                record_cancelled(name, error)
        return PairResult(updates, cancelled, first_failure[0] if first_failure else '', cancelled_logs)

    executor = ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix='candidate-validation')
    # Each Context may only be entered by one thread at a time.
    futures = {executor.submit(copy_context().run, invoke, name, worker): name
               for name, worker in workers.items()}
    pending = set(futures)
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                name = futures[future]
                try:
                    updates[name] = future.result()
                except processes.ValidationCancelled as error:
                    record_cancelled(name, error)
            if first_failure and pending:
                done, pending = wait(pending, timeout=3.0)
                if pending:
                    raise processes.ProcessTerminationError(
                        'validation worker did not stop after process-tree cancellation')
                for future in done:
                    name = futures[future]
                    try:
                        updates[name] = future.result()
                    except processes.ValidationCancelled as error:
                        record_cancelled(name, error)
        return PairResult(updates, cancelled, first_failure[0] if first_failure else '', cancelled_logs)
    except BaseException as error:
        # Do not turn a harness/provider/programming exception into a design gate.
        # One cleanup fault must not prevent cancellation of the other lane.
        cleanup_errors = []
        for token in tokens.values():
            try:
                token.cancel()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        _, remaining = wait(futures, timeout=3.0)
        if remaining:
            raise processes.ProcessTerminationError('validation workers remain active after abort') from error
        if cleanup_errors:
            raise processes.ProcessTerminationError('cannot confirm validation process termination') from cleanup_errors[0]
        raise
    finally:
        executor.shutdown(wait=False)
