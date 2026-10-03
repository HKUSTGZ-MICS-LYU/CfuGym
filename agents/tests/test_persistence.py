"""Disposable per-run SQLite and write-ahead side-effect journal tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from agents.cfu_design_agent.runtime.persistence import (
    PendingOperationError, PersistenceRiskError, RunWriterBusyError,
    open_run_persistence,
)
from agents.cfu_design_agent.runtime.policy import PolicyError


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rid = "example-001"
        self.run_root = self.root / "agents/generated/runs" / self.rid

    def open(self, **kwargs):
        return open_run_persistence(self.root, self.rid, **kwargs)

    def _manifest(self):
        return yaml.safe_load((self.run_root / "run.yaml").read_text())

    def _save(self, manifest):
        (self.run_root / "run.yaml").write_text(yaml.safe_dump(manifest))

    def _artifact(self, manifest, name, data, *, kind="changed_source", source=None):
        digest = hashlib.sha256(data).hexdigest()
        path = self.run_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rel = path.relative_to(self.root).as_posix()
        immutable = self.run_root / "versions" / f"{path.name}.{digest}"
        immutable.parent.mkdir(exist_ok=True)
        immutable.write_bytes(data)
        ref = {"path": rel, "immutable_path": immutable.relative_to(self.root).as_posix(),
               "sha256": digest, "kind": kind}
        if source:
            ref["inputs"] = [source]
        manifest["artifacts"].append(ref)
        return ref

    def test_sqlite_saver_roundtrip_and_close(self):
        with self.open() as run:
            self.assertEqual(run.config["configurable"]["thread_id"], self.rid)
            saver = run.saver
            config = run.config
            saver.put(config, {"v": 4, "ts": "t", "id": "1", "channel_values": {},
                               "channel_versions": {}, "versions_seen": {}}, {}, {})
            self.assertTrue(run.checkpoint_path.is_file())
        with self.assertRaises(Exception):
            saver.conn.execute("SELECT 1")
        with self.open(resume=True) as reopened:
            self.assertIsNotNone(reopened.saver.get_tuple(reopened.config))

    def test_reject_existing_nonempty_and_missing_resume(self):
        with self.assertRaises(PersistenceRiskError):
            with self.open(resume=True):
                pass
        self.run_root.mkdir(parents=True)
        (self.run_root / "keep.txt").write_text("precious")
        with self.assertRaises(FileExistsError):
            with self.open():
                pass
        self.assertEqual((self.run_root / "keep.txt").read_text(), "precious")

    def test_invalid_run_ids_and_path_escape(self):
        for bad in ("../bad", "bad.name", "", "a" * 129):
            with self.subTest(bad=bad), self.assertRaises(PolicyError):
                with open_run_persistence(self.root, bad, resume=True):
                    pass
        with self.assertRaises(PolicyError):
            with self.open(out_dir="../../outside"):
                pass

    def test_lock_nonblocking_and_reusable(self):
        with self.open() as run:
            with self.assertRaises(RunWriterBusyError):
                with self.open(resume=True):
                    pass
        with self.open(resume=True):
            pass

    def test_journal_pending_blocks_resume_and_duplicate_side_effect(self):
        with self.open() as run:
            with run.journal.external_operation("build", "build-1", {"argv": ["tool"]}):
                raw = json.loads(run.journal.path.read_text())
                self.assertEqual(raw["operations"]["build-1"]["status"], "pending")
            self.assertEqual(json.loads(run.journal.path.read_text())["operations"]["build-1"]["status"], "committed")
            with self.assertRaises(PersistenceRiskError):
                with run.journal.external_operation("build", "build-1"):
                    pass
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with run.journal.external_operation("sim", "sim-1"):
                    raise RuntimeError("failed")
        with self.assertRaisesRegex(PendingOperationError, "Reconcile manually"):
            with self.open(resume=True):
                pass

    def test_manifest_schema_and_immutable_source_hash(self):
        with self.open():
            pass
        source = self.root / "src/kernel.c"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original")
        name = source.relative_to(self.root).as_posix()
        manifest = self._manifest()
        ref = self._artifact(manifest, "12_review/changed.c", b"original", source=name)
        self._save(manifest)
        with self.open(resume=True, candidate_source=name, candidate_sha256=ref["sha256"]):
            pass
        # The mutable latest alias is not the snapshot; changing it cannot falsify
        # an unchanged immutable version, but modifying the source itself can.
        (self.root / ref["path"]).write_bytes(b"new latest")
        with self.open(resume=True, candidate_source=name, candidate_sha256=ref["sha256"]):
            pass
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(PersistenceRiskError, "changed"):
            with self.open(resume=True):
                pass
        source.write_bytes(b"original")
        (self.root / ref["immutable_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(PersistenceRiskError, "immutable artifact hash mismatch"):
            with self.open(resume=True):
                pass
        (self.root / ref["immutable_path"]).write_bytes(b"original")
        manifest["run_id"] = "other"
        self._save(manifest)
        with self.assertRaisesRegex(PersistenceRiskError, "run_id/schema"):
            with self.open(resume=True):
                pass

    def test_latest_candidate_snapshot_integrity(self):
        with self.open():
            pass
        source = self.root / "src/kernel.c"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"v2")
        src = "src/kernel.c"
        manifest = self._manifest()
        old = self._artifact(manifest, "12_review/old", b"v1", kind="validated_source", source=src)
        snap = self._artifact(manifest, "12_review/new", b"v2", kind="validated_source", source=src)
        candidate = {"files": [{"source": src, "snapshot": snap["path"], "sha256": snap["sha256"]}]}
        self._artifact(manifest, "12_review/candidates/000/candidate.yaml",
                       yaml.safe_dump(candidate).encode(), kind="yaml")
        self._save(manifest)
        with self.open(resume=True, candidate_source=src, candidate_sha256=snap["sha256"]):
            pass
        with self.assertRaisesRegex(PersistenceRiskError, "candidate source/hash"):
            with self.open(resume=True, candidate_source=src, candidate_sha256=old["sha256"]):
                pass
        (self.root / snap["immutable_path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(PersistenceRiskError, "immutable artifact hash mismatch"):
            with self.open(resume=True):
                pass

    def test_dependency_missing_is_error_not_memory_fallback(self):
        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
            with self.assertRaises(ImportError):
                with self.open():
                    pass
        self.assertFalse(self.run_root.exists())


if __name__ == "__main__":
    unittest.main()
