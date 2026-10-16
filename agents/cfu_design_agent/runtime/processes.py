"""Managed command process groups and out-of-band validation cancellation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import signal
import subprocess
import threading
import time
from typing import Iterator


class ValidationCancelled(RuntimeError):
    """A sibling failed; keep any executed command's partial output for audit."""

    def __init__(self, message='validation sibling failed'):
        super().__init__(message)
        self.stdout = ''
        self.stderr = ''
        self.returncode = None
        self.started = False
        self.result = None


class ProcessTerminationError(RuntimeError):
    """The harness could not confirm bounded process-tree termination."""


TERM_GRACE_SECONDS = 0.5
KILL_REAP_SECONDS = 1.0


def _group_alive(pgid: int) -> bool:
    # Linux may retain orphaned zombies until init reaps them. They cannot run or
    # hold pipes; do not confuse them with live descendants after SIGKILL.
    try:
        entries = os.scandir('/proc')
    except OSError:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(f'/proc/{entry.name}/stat') as handle:
                    fields = handle.read().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    return True
            except (OSError, ValueError, IndexError):
                continue
    return False


def terminate_tree(process: subprocess.Popen) -> None:
    """TERM the complete session group, then KILL and reap within bounded grace."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise ProcessTerminationError(f'cannot TERM process group {process.pid}') from exc
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while _group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ProcessTerminationError(f'cannot KILL process group {process.pid}') from exc
    try:
        process.wait(timeout=KILL_REAP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise ProcessTerminationError(f'cannot reap process {process.pid}') from exc
    deadline = time.monotonic() + KILL_REAP_SECONDS
    while _group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_alive(process.pid):
        raise ProcessTerminationError(f'process group {process.pid} remains alive after KILL')


class CancellationToken:
    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._cleanup_lock = threading.Lock()
        self._error: BaseException | None = None

    def raise_if_cancelled(self) -> None:
        if self._error is not None:
            raise self._error
        if self.event.is_set():
            raise ValidationCancelled('validation sibling failed')

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)
            cancelled = self.event.is_set()
        if cancelled:
            self.cancel()
            self.raise_if_cancelled()

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel(self) -> None:
        self.event.set()
        # Concurrent failure/registration can request cleanup simultaneously.
        # Serialize only cleanup, never command execution or metadata writes.
        with self._cleanup_lock:
            with self._lock:
                processes = list(self._processes)
            for process in processes:
                try:
                    terminate_tree(process)
                except BaseException as exc:
                    self._error = exc
                    raise
                finally:
                    self.unregister(process)
            if self._error is not None:
                raise self._error


_TOKEN: ContextVar[CancellationToken | None] = ContextVar('validation_cancel_token', default=None)


def current_token() -> CancellationToken | None:
    return _TOKEN.get()


def check_cancelled() -> None:
    token = current_token()
    if token is not None:
        token.raise_if_cancelled()


@contextmanager
def use_token(token: CancellationToken) -> Iterator[None]:
    handle = _TOKEN.set(token)
    try:
        token.raise_if_cancelled()
        yield
    finally:
        _TOKEN.reset(handle)


@contextmanager
def record_cancelled_output() -> Iterator[None]:
    """Bounded artifact-only cleanup; never execute commands in this scope."""
    handle = _TOKEN.set(None)
    try:
        yield
    finally:
        _TOKEN.reset(handle)


def run(args, *, cwd=None, input=None, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=None, check=False, **kwargs):
    """The run_command subprocess.run seam, with tree-safe timeout/cancellation."""
    check_cancelled()
    token = current_token()
    process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE if input is not None else None,
                               stdout=stdout, stderr=stderr, text=text,
                               start_new_session=True, **kwargs)
    cleanup_attempted = False
    communicated = False

    def cleanup():
        nonlocal cleanup_attempted
        cleanup_attempted = True
        terminate_tree(process)

    try:
        if token is not None:
            token.register(process)
        check_cancelled()
        try:
            output, errors = process.communicate(input=input, timeout=timeout)
            communicated = True
        except subprocess.TimeoutExpired as exc:
            cleanup()
            try:
                output, errors = process.communicate(timeout=KILL_REAP_SECONDS)
                communicated = True
            except subprocess.TimeoutExpired as drain_error:
                raise ProcessTerminationError('command pipes remain open after tree termination') from drain_error
            check_cancelled()
            raise subprocess.TimeoutExpired(args, timeout, output=output, stderr=errors) from exc
        check_cancelled()
        # A shell can return while a background child outlives it with closed
        # pipes. Do not unregister a still-live group and leak that child.
        if _group_alive(process.pid):
            cleanup()
        result = subprocess.CompletedProcess(args, process.returncode, output, errors)
        if check:
            result.check_returncode()
        return result
    except ValidationCancelled as exc:
        if not communicated:
            try:
                output, errors = process.communicate(timeout=KILL_REAP_SECONDS)
            except subprocess.TimeoutExpired as drain_error:
                raise ProcessTerminationError('cancelled command pipes remain open') from drain_error
        exc.stdout, exc.stderr = output or '', errors or ''
        exc.returncode = process.returncode
        exc.started = True
        raise
    except BaseException:
        # A cancelled token has already killed its registered group. Unexpected
        # errors must also clean up, including children after the leader exits.
        if not cleanup_attempted and (token is None or not token.event.is_set()):
            cleanup()
        raise
    finally:
        if token is not None:
            token.unregister(process)
