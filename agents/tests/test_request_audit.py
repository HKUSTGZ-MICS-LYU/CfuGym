"""Disposable tests of run-scoped, fail-closed request auditing."""

from __future__ import annotations

from contextvars import copy_context
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.cfu_design_agent.runtime import request_audit


class RequestAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.run_root = self.workdir / "agents" / "generated" / "runs" / "test-1"
        self.run_root.mkdir(parents=True)
        self.path = self.run_root / request_audit.AUDIT_FILENAME

    def entries(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_exact_body_results_errors_and_global_sequence_across_attempts(self):
        body = {"model": "local", "input": [{"role": "user", "content": "x\n🙂"}],
                "tools": [{"type": "function", "name": "look", "parameters": {"type": "object"}}],
                "stream": True, "extra_body": {"thinking": {"type": "enabled"}}}
        with request_audit.use_request_audit(self.run_root, self.workdir):
            first = request_audit.record_request(body, {"stage": "analysis", "attempt": 1})
            self.assertEqual(first, "req-00000001")
            request_audit.record_result(first, result={"usage": {"total_tokens": 8}})
            second = request_audit.record_request({"model": "local", "input": "retry"})
            request_audit.record_result(second, error={"type": "TimeoutError", "message": "timed out"})
        with request_audit.use_request_audit(self.run_root, self.workdir):
            third = request_audit.record_request({"model": "local", "input": "next"})
        self.assertEqual((second, third), ("req-00000003", "req-00000005"))
        entries = self.entries()
        self.assertEqual([entry["seq"] for entry in entries], list(range(1, 6)))
        self.assertEqual([entry["event"] for entry in entries], ["request", "result", "request", "error", "request"])
        self.assertEqual(entries[0]["payload"], body)
        self.assertEqual(entries[0]["metadata"], {"stage": "analysis", "attempt": 1})
        self.assertEqual(entries[1]["request_id"], first)
        self.assertEqual(entries[3]["request_id"], second)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertFalse((self.run_root / "run.yaml").exists())

    def test_context_scoping_nested_and_inactive(self):
        self.assertIsNone(request_audit.record_request({"input": "inactive"}))
        request_audit.record_result(None, error="ignored")
        other = self.workdir / "other"
        other.mkdir()
        with request_audit.use_request_audit(self.run_root, self.workdir):
            self.assertEqual(request_audit.record_request({"input": "outer"}), "req-00000001")
            isolated = copy_context()
            with request_audit.use_request_audit(other, self.workdir):
                self.assertEqual(request_audit.record_request({"input": "inner"}), "req-00000001")
            self.assertEqual(request_audit.record_request({"input": "outer again"}), "req-00000002")
        self.assertIsNone(request_audit.record_request({"input": "inactive again"}))
        self.assertEqual(isolated.run(request_audit.record_request, {"input": "captured"}), "req-00000003")
        self.assertEqual(len(self.entries()), 3)
        self.assertEqual(len((other / request_audit.AUDIT_FILENAME).read_text().splitlines()), 1)

    def test_refuses_path_traversal_symlink_components_and_escaped_root(self):
        outside = self.workdir / "outside"
        outside.mkdir()
        (self.workdir / "alias").symlink_to(outside, target_is_directory=True)
        for root in (self.workdir / "alias", self.workdir / "outside" / ".." / "outside",
                     self.workdir / "agents" / "generated" / "runs" / ".." / "test-1"):
            with self.subTest(root=root), self.assertRaises(request_audit.RequestAuditError):
                with request_audit.use_request_audit(root, self.workdir):
                    pass
        with self.assertRaises(request_audit.RequestAuditError):
            with request_audit.use_request_audit(self.workdir, self.workdir):
                pass
        with self.assertRaises(request_audit.RequestAuditError):
            with request_audit.use_request_audit(outside, self.run_root):
                pass

    def test_refuses_symlink_destination_and_wrong_permissions(self):
        outside = self.workdir / "outside.txt"
        outside.write_text("untouched")
        self.path.symlink_to(outside)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaises(request_audit.RequestAuditError):
                request_audit.record_request({"model": "safe"})
        self.assertEqual(outside.read_text(), "untouched")
        self.path.unlink()
        self.path.write_text("")
        os.chmod(self.path, 0o644)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaisesRegex(request_audit.RequestAuditError, "0600"):
                request_audit.record_request({"model": "safe"})
        self.assertEqual(self.path.read_text(), "")

    def test_refuses_hardlinked_audit_destination(self):
        outside = self.workdir / "hardlink.txt"
        outside.write_text("untouched")
        os.chmod(outside, 0o600)
        os.link(outside, self.path)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaisesRegex(request_audit.RequestAuditError, "owned regular file"):
                request_audit.record_request({"model": "safe"})
        self.assertEqual(outside.read_text(), "untouched")

    def test_fail_closed_on_corrupt_log_invalid_json_and_write_failure(self):
        self.path.write_text('{"seq":1}\nunfinished')
        os.chmod(self.path, 0o600)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaisesRegex(request_audit.RequestAuditError, "partial final line"):
                request_audit.record_request({"model": "safe"})
        self.path.write_text("")
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaisesRegex(request_audit.RequestAuditError, "strict JSON"):
                request_audit.record_request({"model": object()})
            with patch.object(request_audit.os, "fsync", side_effect=OSError("disk full")):
                with self.assertRaises(request_audit.RequestAuditError):
                    request_audit.record_request({"model": "safe"})
        self.assertEqual(self.entries()[0]["seq"], 1)

    def test_rejects_secret_and_http_transport_fields_before_writing(self):
        with request_audit.use_request_audit(self.run_root, self.workdir):
            for body, metadata in (
                ({"model": "local", "api_key": "secret"}, {}),
                ({"model": "local", "extra_body": {"Authorization": "secret"}}, {}),
                ({"model": "local"}, {"http_client": "unsafe"}),
            ):
                with self.subTest(body=body, metadata=metadata), self.assertRaisesRegex(
                    request_audit.RequestAuditError, "forbidden"
                ):
                    request_audit.record_request(body, metadata)
        self.assertFalse(self.path.exists())

    def test_refuses_unrepresentable_result_and_requires_one_outcome(self):
        with request_audit.use_request_audit(self.run_root, self.workdir):
            rid = request_audit.record_request({"input": "hi"})
            with self.assertRaises(request_audit.RequestAuditError):
                request_audit.record_result(rid, result=object())
            with self.assertRaises(request_audit.RequestAuditError):
                request_audit.record_result(rid)
            with self.assertRaises(request_audit.RequestAuditError):
                request_audit.record_result(rid, result=None, error="oops")
            with self.assertRaisesRegex(request_audit.RequestAuditError, "forbidden"):
                request_audit.record_result(rid, error={"api_key": "must not leak"})
            request_audit.record_result(rid, result=None)
        self.assertEqual([entry["event"] for entry in self.entries()], ["request", "result"])


if __name__ == "__main__":
    unittest.main()
