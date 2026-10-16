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
import shutil
import signal
import subprocess
import sys
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


def _signal_tracked(tracked: dict[int, str], sig: int) -> None:
    for pid, started in reversed(list(tracked.items())):
        info = _process_info(pid)
        if info is not None and info[1] == started and info[2] != 'Z':
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def _living(tracked: dict[int, str]) -> bool:
    return any((info := _process_info(pid)) is not None and info[1] == started and info[2] != 'Z'
               for pid, started in tracked.items())


def run_process(argv: Sequence[str], cwd: Path, log_path: Path, timeout: float,
                grace_seconds: float = 3.0) -> dict[str, Any]:
    """Run argv without a shell and reap workers on direct/group cancel/timeout.

    Linux child-subreaper adoption catches workers whose vendor launcher exits
    first. Existing coordinator process groups are never replaced or signalled
    indiscriminately. SIGKILL of the *coordinator group* still reaches every
    inherited-group worker even when Python cannot run a handler.
    """
    if sys.platform != 'linux':
        raise RuntimeError('Vivado process-tree cleanup requires Linux /proc')
    libc = ctypes.CDLL(None, use_errno=True)
    previous_subreaper = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER')
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    received: list[int] = []
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda signum, frame: received.append(signum))
    process: subprocess.Popen | None = None
    tracked: dict[int, str] = {}
    issue = None
    start = time.monotonic()
    # Do not ever use start_new_session/process_group/preexec_fn here.
    try:
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(list(argv), cwd=str(cwd), stdout=log,
                                       stderr=subprocess.STDOUT, shell=False)
            while True:
                tracked.update(_descendants(os.getpid()))
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
            # Group signals can make wait() return in the same iteration that
            # the handler fires; preserve cancellation even if no worker lives.
            if received and issue is None:
                issue = f'cancelled by signal {received[0]}'
            if issue is not None or _living(tracked):
                # A successful launcher leaving a live worker is not success.
                if issue is None:
                    issue = 'vendor launcher exited while child workers were still running'
                _signal_tracked(tracked, received[0] if received else signal.SIGTERM)
                deadline = time.monotonic() + grace_seconds
                while _living(tracked) and time.monotonic() < deadline:
                    tracked.update(_descendants(os.getpid()))
                    _signal_tracked(tracked, signal.SIGTERM)
                    time.sleep(0.05)
                _signal_tracked(tracked, signal.SIGKILL)
            process.wait()
        return {'returncode': process.returncode, 'execution_issue': issue,
                'cancel_signal': received[0] if received else None,
                'elapsed_seconds': time.monotonic() - start}
    finally:
        tracked.update(_descendants(os.getpid()))
        _signal_tracked(tracked, signal.SIGKILL)
        if process is not None:
            process.wait()
        # Only reap descendants we observed; never consume unrelated children.
        for pid in tracked:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        libc.prctl(36, previous_subreaper.value, 0, 0, 0)


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
    raise SystemExit(main())
