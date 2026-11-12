"""Managed command process groups and out-of-band validation cancellation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import os
import select
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


# The Vivado supervisor may need 3s TERM grace plus 2s to reap its workers.
TERM_GRACE_SECONDS = 5.5
KILL_REAP_SECONDS = 1.0
# Best effort only: never wait indefinitely for a forking process tree.
TREE_SCAN_SECONDS = 0.5
TREE_SCAN_LIMIT = 4096

# CPython builds can omit pidfd wrappers even on a capable Linux kernel.
# These syscall numbers are shared by the listed Linux architectures only.
_PIDFD_SYSCALLS = (hasattr(os, 'uname') and os.uname().sysname == 'Linux'
                   and os.uname().machine in {'x86_64', 'aarch64', 'riscv64'})


def _syscall(number, *args):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    if _PIDFD_SYSCALLS:
        return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    raise OSError('pidfd unavailable')


def _pidfd_send_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
    elif _PIDFD_SYSCALLS:
        _syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
    else:
        raise OSError('pidfd unavailable')


def _proc_identity(pid: int):
    """Linux stat: state, parent, and start ticks (the command may contain ')')."""
    try:
        with open(f'/proc/{pid}/stat') as handle:
            fields = handle.read().rsplit(')', 1)[1].split()
        return fields[0], int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def remember_process(process: subprocess.Popen) -> None:
    """Record the leader's identity before anyone can reap and reuse its PID."""
    identity = _proc_identity(process.pid)
    process._cfu_start_ticks = identity[2] if identity else None
    process._cfu_pidfd = None
    process._cfu_tracking_expected = True
    if identity:
        try:
            fd = _pidfd_open(process.pid)
            current = _proc_identity(process.pid)
            if current and current[1:] == identity[1:]:
                process._cfu_pidfd = fd
            else:
                os.close(fd)
        except OSError:
            pass


def forget_process(process: subprocess.Popen) -> None:
    fd = getattr(process, '_cfu_pidfd', None)
    if isinstance(fd, int):
        os.close(fd)
        process._cfu_pidfd = None


def _term_escaped_descendants(process: subprocess.Popen, errors=None) -> list[int]:
    """Freeze and TERM identifiable descendants, including across sessions.

    This is Linux best effort, not a containment boundary. A process already
    reparented, created after the bounded scan, hidden by procfs permissions, or
    running without pidfds cannot safely be identified. The caller still kills
    the original process group. Never signal a cross-session numeric PID.
    """
    errors = errors if errors is not None else []
    start = getattr(process, '_cfu_start_ticks', None)
    root_fd = getattr(process, '_cfu_pidfd', None)
    if not isinstance(start, int) or not isinstance(root_fd, int):
        if getattr(process, '_cfu_tracking_expected', False) is True:
            errors.append('pidfd or procfs identity unavailable; cross-session cleanup unconfirmed')
        return []
    deadline = time.monotonic() + TREE_SCAN_SECONDS
    handles = []
    root = None

    def acquire(pid, expected):
        try:
            fd = _pidfd_open(pid)
        except ProcessLookupError:
            return None
        except OSError as exc:
            errors.append(f'cannot open descendant pidfd {pid}: {exc}')
            return None
        current = _proc_identity(pid)
        if not current or current[1:] != expected[1:]:
            os.close(fd)
            return None
        return fd

    def send(fd, sig):
        try:
            _pidfd_send_signal(fd, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            errors.append(f'cannot signal pidfd: {exc}')
            return False

    try:
        identity = _proc_identity(process.pid)
        if not identity or identity[2] != start:
            errors.append('leader identity no longer matches; descendants cannot be verified')
            return []
        try:
            root = os.dup(root_fd)
        except OSError as exc:
            errors.append(f'cannot duplicate leader pidfd: {exc}')
            return []
        if not send(root, signal.SIGSTOP):
            return []
        parents = {process.pid}
        seen = {process.pid}
        while parents and len(seen) < TREE_SCAN_LIMIT and time.monotonic() < deadline:
            try:
                with os.scandir('/proc') as entries:
                    pids = [int(entry.name) for entry in entries if entry.name.isdigit()]
            except OSError as exc:
                errors.append(f'cannot scan procfs: {exc}')
                break
            children = set()
            for pid in pids:
                if time.monotonic() >= deadline or len(seen) >= TREE_SCAN_LIMIT:
                    break
                if pid in seen:
                    continue
                info = _proc_identity(pid)
                if not info or info[1] not in parents or info[0] == 'Z':
                    continue
                fd = acquire(pid, info)
                if fd is not None:
                    if send(fd, signal.SIGSTOP):
                        handles.append(fd)
                        seen.add(pid)
                        children.add(pid)
                    else:
                        os.close(fd)
            parents = children
        if parents and (len(seen) >= TREE_SCAN_LIMIT or time.monotonic() >= deadline):
            errors.append('descendant scan limit reached; cleanup unconfirmed')
    finally:
        for fd in reversed(handles):
            send(fd, signal.SIGTERM)
            send(fd, signal.SIGCONT)
        if root is not None:
            # Resume even if procfs scanning fails; the normal group path follows.
            send(root, signal.SIGCONT)
            os.close(root)
    return handles


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
    """Best-effort cross-session cleanup, then bounded group termination."""
    errors = []
    handles = _term_escaped_descendants(process, errors)
    try:
        # A reaped leader's numeric PID/PGID can be reused. Prefer a possible
        # orphan over signalling an unrelated group.
        if isinstance(process.returncode, int):
            if errors:
                raise ProcessTerminationError('; '.join(errors))
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f'cannot TERM process group {process.pid}: {exc}')

        def live_handles():
            poller = select.poll()
            for fd in handles:
                poller.register(fd, select.POLLIN)
            ready = {fd for fd, _ in poller.poll(0)}
            return [fd for fd in handles if fd not in ready]

        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while (_group_alive(process.pid) or live_handles()) and time.monotonic() < deadline:
            time.sleep(0.01)
        for fd in live_handles():
            try:
                _pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(f'cannot KILL descendant: {exc}')
        if not isinstance(process.returncode, int) and _group_alive(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(f'cannot KILL process group {process.pid}: {exc}')
        try:
            process.wait(timeout=KILL_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            errors.append(f'cannot reap process {process.pid}')
        deadline = time.monotonic() + KILL_REAP_SECONDS
        while (_group_alive(process.pid) or live_handles()) and time.monotonic() < deadline:
            time.sleep(0.01)
        if _group_alive(process.pid) or live_handles():
            errors.append(f'process tree {process.pid} remains alive after KILL')
        if errors:
            raise ProcessTerminationError('; '.join(errors))
    finally:
        for fd in handles:
            os.close(fd)


class CancellationToken:
    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._cleanup_lock = threading.Lock()
        self._error: BaseException | None = None

    def raise_if_cancelled(self) -> None:
        if self.event.is_set():
            # cancel publishes the event while holding this lock, so observing
            # cancellation also waits for its registered process cleanup.
            with self._cleanup_lock:
                if self._error is not None:
                    raise self._error
                raise ValidationCancelled('validation sibling failed')
        if self._error is not None:
            raise self._error

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
        # Publish the event only while owning cleanup: a worker that observes
        # it must not unregister/close its pidfd before this snapshot completes.
        with self._cleanup_lock:
            self.event.set()
            with self._lock:
                processes = list(self._processes)
            errors = []
            for process in processes:
                try:
                    terminate_tree(process)
                except Exception as exc:
                    errors.append(f'{process.pid}: {exc}')
                finally:
                    self.unregister(process)
            if errors:
                self._error = ProcessTerminationError('; '.join(errors))
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
    registered_once = False
    communicated = False

    def cleanup():
        nonlocal cleanup_attempted
        cleanup_attempted = True
        if token is None:
            terminate_tree(process)
            return
        # Timeout and sibling cancellation must not concurrently wait on the
        # same Popen or use a pidfd that the other side has just closed.
        with token._cleanup_lock:
            with token._lock:
                registered = process in token._processes
            if registered or not token.event.is_set() or not registered_once:
                try:
                    terminate_tree(process)
                finally:
                    token.unregister(process)
            elif token._error is not None:
                raise token._error

    try:
        remember_process(process)
        if token is not None:
            token.register(process)
            registered_once = True
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
        # Serialize even if cancellation was published: an exception can occur
        # before registration, when the canceller cannot yet see this Popen.
        if not cleanup_attempted:
            cleanup()
        raise
    finally:
        if token is not None:
            # The canceller may already hold a snapshot containing this Popen.
            # Do not close its pidfd while that snapshot is being cleaned.
            with token._cleanup_lock:
                token.unregister(process)
                forget_process(process)
        else:
            forget_process(process)
