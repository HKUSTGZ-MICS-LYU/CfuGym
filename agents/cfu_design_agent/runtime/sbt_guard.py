"""Host-owned, cancellable workspace serialization for every agent sbt command.

Never unlink sbt's own lock/socket files: a live process may still own them.
The separate flock is released by closing its descriptor, including on failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from . import processes


POLL_SECONDS = 0.05
_JAVA_OPTIONS = "-Dsbt.server.autostart=false -Dsbt.supershell=false -Dsbt.ci=true"


def is_sbt(args) -> bool:
    return bool(args) and Path(args[0]).name == "sbt"


@contextmanager
def command_guard(args, workdir: Path, timeout: float):
    """Wait up to the command timeout before starting sbt, then hold its lock.

    Waiting has its own bounded budget; queued commands retain their full execution
    timeout. This lock is independent of the optional experiment PATH shim's flock,
    so using both cannot recursively acquire the same lock. Other tools are no-ops.
    """
    if not is_sbt(args):
        yield None
        return

    processes.check_cancelled()
    lock_path = Path(workdir).resolve() / "agents/generated/.sbt.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    deadline = time.monotonic() + timeout
    waiting = False
    try:
        while True:
            processes.check_cancelled()
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waiting:
                    print(f"[sbt] waiting for workspace lock: {lock_path}", file=sys.stderr, flush=True)
                    waiting = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        args, timeout, stderr=f"timed out waiting for sbt workspace lock: {lock_path}"
                    )
                time.sleep(min(POLL_SECONDS, remaining))
        processes.check_cancelled()
        if waiting:
            print("[sbt] workspace lock acquired", file=sys.stderr, flush=True)
        env = dict(os.environ)
        # JAVA_TOOL_OPTIONS reaches both sbt launchers without changing the
        # policy-validated command argv or relying on a launcher-specific option.
        env["JAVA_TOOL_OPTIONS"] = (env.get("JAVA_TOOL_OPTIONS", "") + " " + _JAVA_OPTIONS).strip()
        yield env
    finally:
        os.close(fd)
