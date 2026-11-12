"""Small real-process tests; no model, simulator, or toolchain is launched."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import processes
from agents.experiments.harness import execution


def has_pidfd():
    try:
        fd = processes._pidfd_open(os.getpid())
        processes._pidfd_send_signal(fd, 0)
        os.close(fd)
        return True
    except OSError:
        return False


HAS_PIDFD = has_pidfd()

# Root -> new-session child -> new-session grandchild, all still parented.
ROOT = r'''
import subprocess, sys, time
child = r"""
import os, subprocess, sys, time
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'], start_new_session=True)
open(sys.argv[1], 'w').write(str(os.getpid()) + ' ' + str(p.pid))
time.sleep(8)
"""
subprocess.Popen([sys.executable, '-c', child, sys.argv[1]], start_new_session=True)
time.sleep(8)
'''


@unittest.skipUnless(HAS_PIDFD, 'Linux procfs and pidfds required')
class ProcessTreeCleanupTest(unittest.TestCase):
    def run_case(self, outer):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pids = root / 'pids'
            # This independent new-session process must never be signalled.
            other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'],
                                     start_new_session=True)
            try:
                if outer:
                    result = execution.run_logged([sys.executable, '-c', ROOT, str(pids)],
                                                  root / 'out', root / 'err', root,
                                                  os.environ.copy(), 1)
                    self.assertTrue(result['timed_out'])
                else:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        processes.run([sys.executable, '-c', ROOT, str(pids)], timeout=1)
                self.assertTrue(pids.exists(), 'test descendants failed to start')
                for pid in map(int, pids.read_text().split()):
                    identity = processes._proc_identity(pid)
                    self.assertTrue(identity is None or identity[0] == 'Z', (pid, identity))
                self.assertIsNone(other.poll())
            finally:
                other.kill()
                other.wait(timeout=2)

    def test_outer_sweep_timeout_cleans_nested_sessions(self):
        self.run_case(True)

    def test_internal_command_timeout_cleans_nested_sessions(self):
        self.run_case(False)

    def test_cross_session_supervisor_gets_time_to_clean_up(self):
        self.assertGreaterEqual(processes.TERM_GRACE_SECONDS, 5.0)
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'cleaned'
            child = ("import signal,time,sys\n"
                     "def stop(*args):\n time.sleep(.3); open(sys.argv[1], 'w').write('done'); sys.exit(0)\n"
                     "signal.signal(signal.SIGTERM, stop)\n"
                     "print('ready', flush=True)\n"
                     "time.sleep(8)\n")
            root = ("import subprocess,sys,time\n"
                    "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]], "
                    "start_new_session=True, stdout=subprocess.PIPE)\n"
                    "p.stdout.readline()\nprint('started',flush=True)\ntime.sleep(8)\n")
            with mock.patch.object(processes, 'TERM_GRACE_SECONDS', 0.8):
                with self.assertRaises(subprocess.TimeoutExpired):
                    processes.run([sys.executable, '-c', root, child, str(marker)], timeout=1)
            self.assertEqual(marker.read_text(), 'done')

    def test_reused_root_identity_is_not_signalled(self):
        proc = mock.Mock(pid=123, _cfu_start_ticks=10, _cfu_pidfd=42)
        with mock.patch.object(processes, '_proc_identity', return_value=('S', 1, 11)), \
             mock.patch.object(processes, '_pidfd_open') as opened:
            processes._term_escaped_descendants(proc)
        opened.assert_not_called()

    def test_missing_pidfd_is_reported_after_group_cleanup(self):
        proc = mock.Mock(pid=123, returncode=None, _cfu_start_ticks=10,
                         _cfu_pidfd=None, _cfu_tracking_expected=True)
        with mock.patch.object(processes.os, 'killpg') as kill, \
             mock.patch.object(processes, '_group_alive', return_value=False):
            with self.assertRaisesRegex(processes.ProcessTerminationError, 'pidfd'):
                processes.terminate_tree(proc)
        kill.assert_called_once_with(123, signal.SIGTERM)
        proc.wait.assert_called_once()

    def test_kill_failure_does_not_skip_other_descendants(self):
        proc = mock.Mock(pid=123, returncode=None)
        poller = mock.Mock()
        poller.poll.return_value = []
        def send(fd, sig):
            if fd == 101:
                raise PermissionError('denied')
        with mock.patch.object(processes, '_term_escaped_descendants', return_value=[101, 102]), \
             mock.patch.object(processes, '_group_alive', return_value=False), \
             mock.patch.object(processes.os, 'killpg'), mock.patch.object(processes.os, 'close'), \
             mock.patch.object(processes.select, 'poll', return_value=poller), \
             mock.patch.object(processes, 'TERM_GRACE_SECONDS', 0), \
             mock.patch.object(processes, 'KILL_REAP_SECONDS', 0), \
             mock.patch.object(processes, '_pidfd_send_signal', side_effect=send) as signal_child:
            with self.assertRaisesRegex(processes.ProcessTerminationError, 'denied'):
                processes.terminate_tree(proc)
        self.assertEqual([c.args[0] for c in signal_child.call_args_list], [101, 102])

    def test_sweep_signals_cancel_worker_and_nested_sessions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for sig in (signal.SIGINT, signal.SIGTERM):
                with self.subTest(signal=sig):
                    case = root / str(sig)
                    case.mkdir()
                    pidfile = case / 'pids'
                    coordinator = '''
import os, sys
from pathlib import Path
from agents.experiments.harness import cli, execution
from agents.tests.test_process_tree_cleanup import ROOT
root = Path(sys.argv[1]); pids = root / 'pids'
def worker(_):
    execution.run_logged([sys.executable, '-c', ROOT, str(pids)], root / 'out', root / 'err',
                         root, os.environ.copy(), 30)
    return []
try:
    cli._run_jobs([1], worker, 1)
except KeyboardInterrupt:
    sys.exit(130)
'''
                    outer = subprocess.Popen([sys.executable, '-c', coordinator, str(case)],
                                             start_new_session=True, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True)
                    try:
                        deadline = time.monotonic() + 8
                        while not pidfile.exists() and outer.poll() is None and time.monotonic() < deadline:
                            time.sleep(.02)
                        self.assertTrue(pidfile.exists(), 'nested child did not start')
                        nested = list(map(int, pidfile.read_text().split()))
                        outer.send_signal(sig)
                        stdout, stderr = outer.communicate(timeout=12)
                        self.assertEqual(outer.returncode, 130, stdout + stderr)
                        for pid in nested:
                            identity = processes._proc_identity(pid)
                            self.assertTrue(identity is None or identity[0] == 'Z', (pid, identity))
                    finally:
                        if outer.poll() is None:
                            os.killpg(outer.pid, signal.SIGKILL)
                            outer.communicate(timeout=3)

    def test_outer_keyboard_interrupt_cleans_before_reraising(self):
        proc = mock.Mock(pid=123, returncode=None)
        proc.communicate.side_effect = KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(execution.subprocess, 'Popen', return_value=proc), \
                 mock.patch.object(processes, 'remember_process') as remember, \
                 mock.patch.object(processes, 'terminate_tree') as terminate, \
                 mock.patch.object(processes, 'forget_process') as forget:
                with self.assertRaises(KeyboardInterrupt):
                    execution.run_logged(['fake'], root / 'out', root / 'err', root, {}, 3)
        remember.assert_called_once_with(proc)
        terminate.assert_called_once_with(proc)
        forget.assert_called_once_with(proc)

    def test_exception_before_registration_after_cancel_cleans_spawned_process(self):
        token = processes.CancellationToken()
        proc = mock.Mock(pid=123, returncode=None)
        def fail_during_remember(_):
            token.cancel()
            raise RuntimeError('registration failed')
        with mock.patch.object(processes.subprocess, 'Popen', return_value=proc), \
             mock.patch.object(processes, 'remember_process', side_effect=fail_during_remember), \
             mock.patch.object(processes, 'terminate_tree') as terminate, \
             mock.patch.object(processes, 'forget_process'):
            with processes.use_token(token):
                with self.assertRaisesRegex(RuntimeError, 'registration failed'):
                    processes.run(['fake'])
        terminate.assert_called_once_with(proc)

    def test_cancel_waits_for_cleanup_before_worker_observes_event(self):
        token = processes.CancellationToken()
        proc = mock.Mock(pid=123)
        token.register(proc)
        entered = threading.Event()
        release = threading.Event()
        observed = threading.Event()
        errors = []
        def terminate(_):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test cleanup did not resume')
        def observe():
            try:
                token.raise_if_cancelled()
            except processes.ValidationCancelled:
                observed.set()
            except Exception as exc:
                errors.append(exc)
        with mock.patch.object(processes, 'terminate_tree', side_effect=terminate):
            cancel_thread = threading.Thread(target=token.cancel)
            cancel_thread.start()
            try:
                self.assertTrue(entered.wait(2))
                observer = threading.Thread(target=observe)
                observer.start()
                self.assertFalse(observed.wait(.1), 'worker observed unfinished cleanup')
                self.assertIn(proc, token._processes)
            finally:
                release.set()
                cancel_thread.join(timeout=3)
                if 'observer' in locals():
                    observer.join(timeout=3)
            self.assertFalse(cancel_thread.is_alive())
            self.assertFalse(observer.is_alive())
            self.assertTrue(observed.is_set())
            self.assertFalse(errors)
            self.assertNotIn(proc, token._processes)

    def test_cancel_continues_after_one_process_fails(self):
        token = processes.CancellationToken()
        first, second = mock.Mock(pid=1), mock.Mock(pid=2)
        token.register(first)
        token.register(second)
        def terminate(proc):
            if proc is first:
                raise processes.ProcessTerminationError('denied')
        with mock.patch.object(processes, 'terminate_tree', side_effect=terminate) as cleanup:
            with self.assertRaisesRegex(processes.ProcessTerminationError, 'denied'):
                token.cancel()
        self.assertEqual({call.args[0] for call in cleanup.call_args_list}, {first, second})


if __name__ == '__main__':
    unittest.main()
