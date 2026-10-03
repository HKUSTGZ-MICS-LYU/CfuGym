"""Run-scoped, fail-closed JSONL audit of application-level LLM requests.

Call ``record_request`` immediately before the SDK call, passing the exact
application request body (not SDK client options, credentials, or HTTP headers).
Call ``record_result`` on both the successful and exceptional paths. No manifest
entry is written: this append-only ledger is independent of run artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping


AUDIT_FILENAME = "request_audit.jsonl"
_CURRENT_AUDIT: ContextVar["RequestAudit | None"] = ContextVar("request_audit", default=None)
_MISSING = object()
# Client objects are stage-local; metadata discovery is cached for the entire
# run without persisting credentials or guessing a model's capacity.
_MODEL_METADATA: dict[tuple[str, str, str], dict[str, Any]] = {}
_PREVIOUS_PAYLOAD: dict[tuple[str, str, str, str], str] = {}
# App request fields such as max_output_tokens are legitimate; transport and
# credential fields belong to SDK client construction, never this ledger.
_FORBIDDEN_KEYS = frozenset({
    "api_key", "apikey", "authorization", "proxy_authorization",
    "headers", "http_headers", "default_headers", "extra_headers",
    "http_client", "async_http_client", "http_options", "httpx_client",
    "access_token", "bearer_token", "password", "client_secret", "cookies",
})


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower().replace("-", "_") in _FORBIDDEN_KEYS:
                raise RequestAuditError(f"transport or secret field is forbidden: {key}")
            _reject_secrets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_secrets(child)




class RequestAuditError(RuntimeError):
    """An active request audit cannot safely persist an event."""


def _absolute_parts(path: os.PathLike[str] | str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise RequestAuditError("audit paths must be absolute")
    parts = raw.split("/")[1:]
    if any(part in ("", ".", "..") for part in parts):
        raise RequestAuditError("audit path contains an empty or traversal component")
    return tuple(parts)


def _open_directory(parts: tuple[str, ...]) -> int:
    """Traverse from / via directory handles, never following any symlink."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in parts:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


class RequestAudit:
    """A run-root-bound recorder; each append locks and rechecks the ledger."""

    def __init__(self, run_root: Path | str, workdir: Path | str):
        self._root_parts = _absolute_parts(run_root)
        work_parts = _absolute_parts(workdir)
        if len(self._root_parts) <= len(work_parts) or self._root_parts[:len(work_parts)] != work_parts:
            raise RequestAuditError("run_root must be strictly inside workdir")
        try:
            fd = _open_directory(self._root_parts)
            os.close(fd)
        except OSError as exc:
            raise RequestAuditError("run_root must be an existing, non-symlink directory") from exc
        self.path = Path(run_root) / AUDIT_FILENAME
        self.run_id = Path(run_root).name
        self.model_metadata: dict[tuple[str, str], dict[str, Any]] = {}

    def _append(self, entry: dict[str, Any]) -> int:
        # Serialize before opening: unsupported values must not leave half a line.
        _reject_secrets(entry)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        entry.setdefault("run_id", self.run_id)
        try:
            json.dumps(entry, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RequestAuditError("audit event is not strict JSON") from exc
        try:
            directory = _open_directory(self._root_parts)
            try:
                flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
                fd = os.open(AUDIT_FILENAME, flags, 0o600, dir_fd=directory)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                            or info.st_uid != os.getuid() or info.st_nlink != 1):
                        raise RequestAuditError("audit file must be an unlinked-to-elsewhere owned regular file with mode 0600")
                    # Read the final event under the same lock. Re-parsing the
                    # entire growing prompt ledger on every append is quadratic
                    # and can exhaust memory on a long tool loop.
                    seq = 0
                    if info.st_size:
                        if os.pread(fd, 1, info.st_size - 1) != b"\n":
                            raise RequestAuditError("audit file has a partial final line")
                        end = info.st_size - 1
                        start = end
                        chunks = []
                        while start:
                            lower = max(0, start - 8192)
                            block = os.pread(fd, start - lower, lower)
                            newline = block.rfind(b"\n")
                            chunks.insert(0, block[newline + 1:] if newline >= 0 else block)
                            start = lower
                            if newline >= 0:
                                break
                        try:
                            item = json.loads(b"".join(chunks))
                        except (UnicodeDecodeError, ValueError) as exc:
                            raise RequestAuditError("audit file has invalid final JSONL event") from exc
                        if not isinstance(item, dict) or type(item.get("seq")) is not int or item["seq"] < 1:
                            raise RequestAuditError("audit file sequence is invalid")
                        seq = item["seq"]
                    entry["seq"] = seq + 1
                    if entry.get("event") == "request":
                        entry["request_id"] = f"req-{seq + 1:08d}"
                    data = (json.dumps(entry, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
                    # A short write can leave an ambiguous partial event; in
                    # particular, do not treat it as a successful SDK preflight.
                    written = os.write(fd, data)
                    if written != len(data):
                        raise RequestAuditError("short audit write")
                    os.fsync(fd)
                    return seq + 1
                finally:
                    os.close(fd)
            finally:
                os.close(directory)
        except OSError as exc:
            raise RequestAuditError("cannot safely write request audit") from exc

    def record_request(self, payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str:
        """Append the exact application-level request before its SDK invocation."""
        if not isinstance(payload, Mapping) or not isinstance(metadata or {}, Mapping):
            raise RequestAuditError("payload and metadata must be mappings")
        _reject_secrets(payload)
        _reject_secrets(metadata or {})
        entry = {"event": "request", "payload": dict(payload), "metadata": dict(metadata or {})}
        # The request id equals the globally ordered event sequence number.
        seq = self._append(entry)
        return f"req-{seq:08d}"

    def record_result(
        self, request_id: str, *, result: Any = _MISSING,
        error: Any = _MISSING, metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one JSON-compatible result or error for a recorded request."""
        if not isinstance(request_id, str) or not request_id.startswith("req-") or not request_id[4:].isdigit():
            raise RequestAuditError("invalid audit request id")
        if (result is _MISSING) == (error is _MISSING):
            raise RequestAuditError("supply exactly one of result and error")
        if not isinstance(metadata or {}, Mapping):
            raise RequestAuditError("metadata must be a mapping")
        entry = {"event": "result" if error is _MISSING else "error", "request_id": request_id,
                 "metadata": dict(metadata or {})}
        entry["result" if error is _MISSING else "error"] = result if error is _MISSING else error
        self._append(entry)


@contextmanager
def use_request_audit(run_root: Path | str, workdir: Path | str) -> Iterator[RequestAudit]:
    """Activate a recorder in this context only, restoring any outer recorder."""
    recorder = RequestAudit(run_root, workdir)
    token = _CURRENT_AUDIT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT_AUDIT.reset(token)


def record_event(event: str, metadata: Mapping[str, Any]) -> None:
    """Record a preflight/compaction/discovery event without pretending it was sent."""
    recorder = _CURRENT_AUDIT.get()
    if recorder is not None:
        recorder._append({"event": event, "metadata": dict(metadata)})


def model_metadata(key: tuple[str, str]) -> dict[str, Any] | None:
    recorder = _CURRENT_AUDIT.get()
    return _MODEL_METADATA.get((str(recorder.path.parent), *key)) if recorder is not None else None


def remember_model_metadata(key: tuple[str, str], metadata: Mapping[str, Any]) -> None:
    recorder = _CURRENT_AUDIT.get()
    if recorder is not None:
        _MODEL_METADATA[(str(recorder.path.parent), *key)] = dict(metadata)


def previous_payload(key: tuple[str, str, str], serialized: str) -> str | None:
    """Remember a previous request only for diagnostic prefix comparison."""
    recorder = _CURRENT_AUDIT.get()
    if recorder is None:
        return None
    scoped = (str(recorder.path.parent), *key)
    previous = _PREVIOUS_PAYLOAD.get(scoped)
    _PREVIOUS_PAYLOAD[scoped] = serialized
    return previous


def active() -> bool:
    """True when a run-scoped recorder already covers this context."""
    return _CURRENT_AUDIT.get() is not None


def record_request(payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str | None:
    """Return an audit request id, or None outside an active run context."""
    recorder = _CURRENT_AUDIT.get()
    return recorder.record_request(payload, metadata) if recorder is not None else None


def record_result(
    request_id: str | None, *, result: Any = _MISSING,
    error: Any = _MISSING, metadata: Mapping[str, Any] | None = None,
) -> None:
    """Record a request outcome, or do nothing outside an active run context."""
    recorder = _CURRENT_AUDIT.get()
    if recorder is not None:
        recorder.record_result(request_id, result=result, error=error, metadata=metadata)
