#!/usr/bin/env python3
"""Run the fixed whole-MiCoSoc Vivado flow in a new run-scoped directory.

Example (all input files, including headers and memory data, are explicit):
  python3 skills/cfu-designer/scripts/vivado_ppa_report.py \
    --run-dir agents/generated/runs/ID --rtl-root .../rtl \
    --rtl .../rtl/MiCoSoc.v --out-dir .../11_cost/evaluations/E/vivado

The subprocess deliberately INHERITS its coordinator's session/process group.
A group cancellation therefore reaches Vivado and its workers. Direct signals
and timeouts are additionally forwarded to the observed Linux process tree;
no setsid(), start_new_session or detached process group is created here.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

# Fixed checkout import, not input-controlled plugin loading. Bypass the agent
# package's eager graph imports so this standalone tool needs only stdlib.
import importlib.util
_RUNTIME = Path(__file__).resolve().parents[3] / 'agents/cfu_design_agent/runtime/vivado.py'
_SPEC = importlib.util.spec_from_file_location('_cfugym_vivado_runtime', _RUNTIME)
assert _SPEC is not None and _SPEC.loader is not None
vivado = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vivado)


def _process_info(pid: int) -> tuple[int, str, str] | None:
    """Return ppid, kernel starttime and state (guards PID reuse during cleanup)."""
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return int(fields[1]), fields[19], fields[0]
    except (OSError, ValueError, IndexError):
        return None


def _descendants(root: int) -> dict[int, str]:
    children: dict[int, list[tuple[int, str]]] = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            pid = int(entry.name)
            info = _process_info(pid)
            if info is not None:
                children.setdefault(info[0], []).append((pid, info[1]))
    result: dict[int, str] = {}
    queue = [root]
    while queue:
        parent = queue.pop()
        for pid, started in children.get(parent, []):
            result[pid] = started
            queue.append(pid)
    return result


def _pidfd_call(number: int, *args: int) -> int:
    # Python 3.10 / glibc 2.35 may lack wrappers even on a recent kernel.
    # Linux x86-64 and arm64 share these syscall numbers; other architectures
    # must provide the Python APIs rather than using a potentially wrong ABI.
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise OSError('pidfd is unavailable on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(number, *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _open_pidfd(pid: int) -> int:
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    return _pidfd_call(434, pid, 0)  # Linux pidfd_open


def _send_pidfd(fd: int, sig: int) -> None:
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
    else:
        _pidfd_call(424, fd, sig, 0, 0)  # Linux pidfd_send_signal


def _signal_tracked(tracked: dict[int, str], sig: int) -> None:
    errors = []
    for pid, started in reversed(list(tracked.items())):
        try:
            # Open first, then recheck the start time; signal the opened
            # identity, never a numeric PID that may have been recycled.
            fd = _open_pidfd(pid)
            try:
                info = _process_info(pid)
                if info is not None and info[1] == started and info[2] != 'Z':
                    _send_pidfd(fd, sig)
            finally:
                os.close(fd)
        except ProcessLookupError:
            pass
        except OSError as exc:
            # One inaccessible PID must not prevent signalling other workers.
            errors.append(f'{pid}: {exc}')
    if errors:
        raise OSError('could not signal Vivado descendants: ' + '; '.join(errors))


def _living(tracked: dict[int, str]) -> bool:
    return any((info := _process_info(pid)) is not None and info[1] == started and info[2] != 'Z'
               for pid, started in tracked.items())


def _reap_adopted(tracked: dict[int, str], root_pid: int | None) -> None:
    """Reap this supervisor's adopted zombies, never the Popen-owned root."""
    for pid, started in _descendants(os.getpid()).items():
        info = _process_info(pid)
        if pid == root_pid or info is None or info != (os.getpid(), started, 'Z'):
            continue
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        if waited:
            tracked.pop(pid, None)


def _run_process_worker(argv: Sequence[str], cwd: Path, log_path: Path, timeout: float,
                        grace_seconds: float = 3.0) -> dict[str, Any]:
    """Run in an exclusive exec-created subreaper, never in the caller process."""
    if sys.platform != 'linux':
        raise RuntimeError('Vivado process-tree cleanup requires Linux /proc')
    # Refuse to start Vivado unless race-free signalling is available. Checking
    # only during cleanup could leave an already launched process behind.
    fd = _open_pidfd(os.getpid())
    try:
        _send_pidfd(fd, 0)
    finally:
        os.close(fd)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    received: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: received.append(signum))
    process: subprocess.Popen | None = None
    tracked: dict[int, str] = {}
    issue = None
    start = time.monotonic()
    try:
        with log_path.open('x', encoding='utf-8') as log:
            # Retain the coordinator's process group for group cancellation.
            process = subprocess.Popen(list(argv), cwd=str(cwd), stdout=log,
                                       stderr=subprocess.STDOUT, shell=False)
            while True:
                tracked.update(_descendants(os.getpid()))
                # The main Popen owns its own wait status. Adopted workers must
                # be reaped even before that main process exits: some vendors
                # wait for their helper PIDs to disappear from /proc.
                _reap_adopted(tracked, process.pid)
                if received:
                    issue = f'cancelled by signal {received[0]}'
                    break
                if time.monotonic() - start >= timeout:
                    issue = f'timeout after {timeout:g} seconds'
                    break
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            tracked.update(_descendants(os.getpid()))
            if received and issue is None:
                issue = f'cancelled by signal {received[0]}'
            if issue is not None or _living(tracked):
                if issue is None:
                    issue = 'vendor launcher exited while child workers were still running'
                deadline = time.monotonic() + grace_seconds
                while time.monotonic() < deadline:
                    tracked.update(_descendants(os.getpid()))
                    _signal_tracked(tracked, received[0] if received else signal.SIGTERM)
                    _reap_adopted(tracked, process.pid if process.poll() is None else None)
                    if not _living(tracked):
                        break
                    time.sleep(0.05)
                tracked.update(_descendants(os.getpid()))
                _signal_tracked(tracked, signal.SIGKILL)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired as exc:
                raise OSError('Vivado process could not be reaped after termination') from exc
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                tracked.update(_descendants(os.getpid()))
                _reap_adopted(tracked, None)
                if not _descendants(os.getpid()):
                    break
                _signal_tracked(tracked, signal.SIGKILL)
                time.sleep(0.05)
            if _descendants(os.getpid()):
                raise OSError('Vivado descendants could not be confirmed reaped')
        return {'returncode': process.returncode, 'execution_issue': issue,
                'cancel_signal': received[0] if received else None,
                'elapsed_seconds': time.monotonic() - start}
    finally:
        cleanup_errors = []
        try:
            tracked.update(_descendants(os.getpid()))
        except OSError as exc:
            cleanup_errors.append(f'process scan: {exc}')
        try:
            _signal_tracked(tracked, signal.SIGKILL)
        except OSError as exc:
            cleanup_errors.append(str(exc))
        if process is not None:
            try:
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(f'main process reap: {exc}')
        try:
            _reap_adopted(tracked, process.pid if process and process.poll() is None else None)
            if _descendants(os.getpid()):
                cleanup_errors.append('descendants remain after cleanup')
        except OSError as exc:
            cleanup_errors.append(f'descendant reap: {exc}')
        if cleanup_errors:
            raise OSError('Vivado cleanup not confirmed: ' + '; '.join(cleanup_errors))


def run_process(argv: Sequence[str], cwd: Path, log_path: Path, timeout: float,
                grace_seconds: float = 3.0) -> dict[str, Any]:
    """Run a private supervisor without changing the caller's child ownership.

    Its only children are the requested vendor tree, so rapidly reparented
    orphans cannot be confused with unrelated concurrent Popen instances.
    Both processes inherit the existing coordinator group; neither calls setsid.
    """
    request = json.dumps({'argv': list(argv), 'cwd': str(cwd), 'log_path': str(log_path),
                          'timeout': timeout, 'grace_seconds': grace_seconds})
    received: list[int] = []
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda signum, frame: received.append(signum))
    worker = None
    try:
        worker = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                   '--internal-process-worker'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pending_input = request
        forwarded = False
        deadline = time.monotonic() + timeout + grace_seconds + 6.0
        while True:
            if received and not forwarded and worker.poll() is None:
                worker.send_signal(received[0])
                forwarded = True
            try:
                stdout, stderr = worker.communicate(input=pending_input, timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                if time.monotonic() >= deadline:
                    worker.terminate()
                    try:
                        worker.communicate(timeout=grace_seconds + 3.0)
                    except subprocess.TimeoutExpired as exc:
                        worker.kill()
                        worker.communicate()
                        raise OSError('Vivado supervisor did not confirm descendant cleanup') from exc
                    raise OSError('Vivado supervisor exceeded its bounded cleanup deadline')
        if worker.returncode != 0:
            raise OSError(f'Vivado supervisor failed ({worker.returncode}): {stderr[-1000:]}')
        response = json.loads(stdout)
        if 'error' in response:
            raise OSError(response.get('errno') or 0, response['error'])
        result = response['result']
    finally:
        if worker is not None and worker.poll() is None:
            # Give the private supervisor a chance to terminate and reap its
            # tree even if the caller itself exits via an unexpected exception.
            worker.terminate()
            try:
                worker.wait(timeout=grace_seconds + 3.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    # A signal may arrive after the worker produced a successful result but
    # before its response is delivered to the caller. Never report that as a
    # passed run merely because the worker finished first.
    if received:
        result['cancel_signal'] = received[0]
        result['execution_issue'] = f'cancelled by signal {received[0]}'
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dir', required=True, type=Path)
    result.add_argument('--rtl-root', required=True, type=Path)
    result.add_argument('--rtl', action='append', type=Path, required=True,
                        help='Ordered complete hardware fileset; repeat for HDL/headers/memory data')
    result.add_argument('--sha256', action='append', default=[], help='One hash per --rtl, in order')
    result.add_argument('--out-dir', required=True, type=Path, help='Must not already exist')
    result.add_argument('--part', default=vivado.DEFAULT_PART)
    result.add_argument('--target-frequency-mhz', type=float, default=100.0)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--timeout', type=float, default=3600.0)
    result.add_argument('--snapshot-sha256')
    result.add_argument('--snapshot-hardware-input-sha256')
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vivado.validate_inputs(args.run_dir, args.rtl_root, args.rtl, args.out_dir,
                                 part=args.part, target_frequency_mhz=args.target_frequency_mhz,
                                 threads=args.threads)
    vivado._positive(args.timeout, 'timeout')
    files, root, out = cfg['files'], cfg['rtl_root'], cfg['out_dir']
    digests = [vivado.sha256_file(path) for path in files]
    if args.sha256 and args.sha256 != digests:
        raise ValueError('Frozen fileset hash mismatch (or wrong number of --sha256 values)')
    # Exclusive directory reservation: never replace reports, checkpoints or sources.
    out.mkdir(parents=True, exist_ok=False)
    staged_root = out / 'inputs'
    staged_root.mkdir()
    staged = []
    entries = []
    for path, digest in zip(files, digests):
        relative = path.relative_to(root)
        dest = staged_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if vivado.sha256_file(dest) != digest or vivado.sha256_file(path) != digest:
            raise ValueError('Hardware fileset changed while staging')
        staged.append(dest)
        entries.append({'path': relative.as_posix(), 'sha256': digest})
    manifest = out / 'files.txt'
    manifest.write_text(''.join(str(path) + '\n' for path in staged), encoding='utf-8')
    fixed_tcl = Path(__file__).resolve().with_name('vivado_ppa.tcl')
    argv = ['vivado', '-mode', 'batch', '-notrace', '-source', str(fixed_tcl),
            '-log', str(out / 'vivado.log'), '-journal', str(out / 'vivado.jou'),
            '-tclargs', str(manifest), str(out), str(cfg['target_frequency_mhz']),
            str(cfg['threads']), cfg['part']]
    provenance: dict[str, Any] = {
        'flow': 'whole_soc_synth_opt_place_physopt_route', 'generator_top': 'MiCoSoc',
        'input_kind': 'hardware_rtl', 'flatten_hierarchy': 'none', 'dont_touch': False,
        'clock_port': 'socCtrl_systemClk', 'asynchronous_exceptions': {
            'from': ['socCtrl_asyncReset', 'system_peripheral_uart_logic_uart_rxd'],
            'to': ['system_peripheral_uart_logic_uart_txd']},
        'run_dir': str(cfg['run_dir']), 'rtl_root': str(root), 'files': entries,
        'fileset_sha256': hashlib_fileset(entries),
        'snapshot_sha256': args.snapshot_sha256,
        'hardware_input_sha256': args.snapshot_hardware_input_sha256,
        'tcl_sha256': vivado.sha256_file(fixed_tcl),
        'driver_sha256': vivado.sha256_file(Path(__file__).resolve()),
        'argv': argv, 'cwd': str(out), 'threads': cfg['threads'], 'timeout_seconds': args.timeout,
        'requested_part': cfg['part'], 'requested_frequency_mhz': cfg['target_frequency_mhz'],
    }
    (out / 'launch.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    try:
        provenance.update(run_process(argv, out, out / 'console.log', args.timeout))
    except OSError as exc:
        provenance.update(returncode=None, execution_issue=f'{type(exc).__name__}: {exc}')
    texts: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    for name in ('metrics.tsv', 'utilization_soc.rpt', 'utilization_cfu.rpt', 'power.rpt',
                 'utilization_post_synth.rpt', 'utilization_cfu_post_synth.rpt'):
        path = out / name
        if path.is_file() and path.resolve().is_relative_to(out):
            artifacts[name] = vivado.sha256_file(path)
            texts[name] = path.read_text(encoding='utf-8', errors='replace')
        else:
            texts[name] = ''
    provenance['artifacts'] = artifacts
    try:
        metrics = vivado.parse_metrics_tsv(texts['metrics.tsv'])
    except ValueError as exc:
        metrics = {}
        provenance['execution_issue'] = str(exc)
    # Even failed pre-synth launches describe the requested target explicitly.
    metrics.setdefault('part', cfg['part'])
    metrics.setdefault('target_frequency_mhz', str(cfg['target_frequency_mhz']))
    summary = vivado.make_summary(metrics, texts['utilization_soc.rpt'], texts['utilization_cfu.rpt'],
                                  texts['power.rpt'], provenance=provenance,
                                  returncode=provenance.get('returncode'),
                                  execution_issue=provenance.get('execution_issue'),
                                  soc_synthesis_area=texts['utilization_post_synth.rpt'],
                                  cfu_synthesis_area=texts['utilization_cfu_post_synth.rpt'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'summary.md').write_text(vivado.render_markdown(summary), encoding='utf-8')
    return summary


def hashlib_fileset(entries: list[dict[str, str]]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main() -> int:
    args = parser().parse_args()
    try:
        summary = execute(args)
    except (OSError, ValueError) as exc:
        print(f'Vivado PPA input/output error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': summary['status'], 'phase': summary['phase'],
                      'summary': str(args.out_dir / 'summary.json')}))
    signum = summary['provenance'].get('cancel_signal')
    if signum:
        return 128 + signum
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--internal-process-worker']:
        try:
            request = json.load(sys.stdin)
            result = _run_process_worker(request['argv'], Path(request['cwd']),
                                         Path(request['log_path']), request['timeout'],
                                         request['grace_seconds'])
            print(json.dumps({'result': result}))
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}',
                              'errno': getattr(exc, 'errno', None)}))
    else:
        raise SystemExit(main())
