"""Fast sbt launch/locking contracts; no Java, EDA, or model calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import processes, sbt_guard, tools


class SbtGuardTest(unittest.TestCase):
    def box(self, root):
        return tools.RepoToolbox(Path(root), policy=SimpleNamespace(check_command=lambda *a: None))

    @contextmanager
    def held_lock(self, root):
        path = Path(root) / "agents/generated/.sbt.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield path

    def test_non_sbt_does_not_create_lock_or_change_environment(self):
        with tempfile.TemporaryDirectory() as root:
            with sbt_guard.command_guard(["make", "compile"], Path(root), 1) as env:
                self.assertIsNone(env)
            self.assertFalse((Path(root) / "agents").exists())
            with mock.patch.object(tools.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                self.box(root).run_command(["make", "compile"], command_id="fake")
            self.assertNotIn("env", run.call_args.kwargs)
            self.assertIsNone(run.call_args.kwargs["input"])

    def test_absolute_sbt_preserves_java_options_and_closes_stdin(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict(os.environ, {"JAVA_TOOL_OPTIONS": "-Xmx1g"}), \
                mock.patch.object(tools.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            args = ["/some/env/bin/sbt", "compile"]
            self.box(root).run_command(args, command_id="fake", timeout=7)
            self.assertEqual(run.call_args.args[0], args)
            self.assertEqual(run.call_args.kwargs["input"], "")
            self.assertEqual(run.call_args.kwargs["timeout"], 7)
            options = run.call_args.kwargs["env"]["JAVA_TOOL_OPTIONS"]
            self.assertIn("-Xmx1g", options)
            self.assertIn("-Dsbt.server.autostart=false", options)
            self.assertIn("-Dsbt.ci=true", options)
            self.assertEqual(os.environ["JAVA_TOOL_OPTIONS"], "-Xmx1g")

    def test_waits_for_release_before_launching(self):
        waiting = threading.Event()
        release = threading.Event()
        def pause(_):
            waiting.set()
            if not release.wait(2):
                raise AssertionError("test failed to release sbt lock")
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(sbt_guard.time, "sleep", side_effect=pause), \
                mock.patch.object(tools.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run, \
                ThreadPoolExecutor(max_workers=1) as pool:
            with self.held_lock(root):
                future = pool.submit(self.box(root).run_command, ["sbt", "compile"], command_id="fake")
                self.assertTrue(waiting.wait(2))
                run.assert_not_called()
            release.set()
            self.assertEqual(future.result(timeout=2)["returncode"], 0)
            run.assert_called_once()

    def test_wait_timeout_never_launches_or_unlinks_live_lock(self):
        with tempfile.TemporaryDirectory() as root, self.held_lock(root) as path, \
                mock.patch.object(tools.subprocess, "run") as run:
            inode = path.stat().st_ino
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                self.box(root).run_command(["sbt", "compile"], command_id="fake", timeout=0)
            self.assertIn("waiting for sbt workspace lock", caught.exception.stderr)
            self.assertEqual(path.stat().st_ino, inode)
            run.assert_not_called()

    def test_cancelled_wait_never_launches(self):
        token = processes.CancellationToken()
        with tempfile.TemporaryDirectory() as root, self.held_lock(root), \
                processes.use_token(token), \
                mock.patch.object(sbt_guard.time, "sleep", side_effect=lambda _: token.cancel()), \
                mock.patch.object(tools.subprocess, "run") as run:
            with self.assertRaises(processes.ValidationCancelled):
                self.box(root).run_command(["sbt", "compile"], command_id="fake")
            run.assert_not_called()

    def test_lock_releases_on_launch_error_and_command_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            for error in (OSError("missing launcher"), subprocess.TimeoutExpired(["sbt"], 1)):
                with self.subTest(error=type(error).__name__), \
                        mock.patch.object(tools.subprocess, "run", side_effect=error):
                    with self.assertRaises(type(error)):
                        self.box(root).run_command(["sbt", "compile"], command_id="fake")
                # An existing lock file must not be mistaken for a held lock.
                with sbt_guard.command_guard(["sbt"], Path(root), 0):
                    pass

    def test_lock_is_held_until_command_finishes_then_released(self):
        with tempfile.TemporaryDirectory() as root:
            def execute(*args, **kwargs):
                with self.assertRaises(subprocess.TimeoutExpired):
                    with sbt_guard.command_guard(["sbt"], Path(root), 0):
                        self.fail("lock was released before command completion")
                return subprocess.CompletedProcess([], 0, "", "")
            with mock.patch.object(tools.subprocess, "run", side_effect=execute):
                self.box(root).run_command(["sbt", "compile"], command_id="fake")
            with sbt_guard.command_guard(["sbt"], Path(root), 0):
                pass

    def test_lock_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "agents/generated/.sbt.lock"
            path.parent.mkdir(parents=True)
            code = ("import fcntl,sys; f=open(sys.argv[1],'w'); "
                    "fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.read()")
            with subprocess.Popen([sys.executable, "-c", code, str(path)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as holder:
                try:
                    self.assertEqual(holder.stdout.readline().strip(), "locked")
                    with self.assertRaises(subprocess.TimeoutExpired):
                        with sbt_guard.command_guard(["sbt"], Path(root), 0):
                            self.fail("another process still holds the lock")
                finally:
                    holder.communicate(input="", timeout=2)
            with sbt_guard.command_guard(["sbt"], Path(root), 0):
                pass

    def test_real_fake_launcher_receives_eof_and_properties(self):
        with tempfile.TemporaryDirectory() as root:
            launcher = Path(root) / "sbt"
            launcher.write_text("#!/bin/sh\nif read -r value; then exit 9; fi\nprintf '%s' \"$JAVA_TOOL_OPTIONS\"\n")
            launcher.chmod(0o755)
            result = self.box(root).run_command([str(launcher), "compile"], command_id="fake", timeout=2)
            self.assertEqual(result["returncode"], 0, result)
            self.assertIn("-Dsbt.server.autostart=false", result["stdout_tail"])

    def test_dry_run_has_no_lock_or_process_side_effects(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(tools.subprocess, "run") as run:
            result = self.box(root).run_command(["sbt", "compile"], command_id="fake", execute=False)
            self.assertTrue(result["skipped"])
            self.assertFalse((Path(root) / "agents").exists())
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
