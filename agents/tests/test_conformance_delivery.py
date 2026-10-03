"""Fixed, policy-approved conformance delivery integrity and transaction tests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from agents.cfu_design_agent import conformance
from agents.cfu_design_agent.runtime import conformance_delivery as delivery_mod
from agents.cfu_design_agent.runtime.policy import AccessPolicy, PolicyError, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox


class ConformanceDeliveryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = "agents/generated/runs/delivery-unit"
        self.state = {"workdir": str(self.root), "run_root": self.run,
                      "isa_spec": {"commands": [{"name": "add", "function_id": 1}]}}
        policy = AccessPolicy(version="1", deny=(), write_roots=(self.run,),
                              stages={"conformance": StagePolicy(write=(self.run + "/**",), commands=())},
                              commands={}, read=("agents/generated/runs/**",))
        self.toolbox = RepoToolbox(self.root, policy=policy, stage="conformance", run_root=self.run)
        self.paths = delivery_mod.delivery_paths(self.state)
        self.header = self.run + "/workspace/include/cfu_intrinsics.h"
        self.oracle = {"commands": [{"name": "add", "reference": "a + b",
                                    "cases": [{"args": [1, 2]}]}]}
        self.text = yaml.safe_dump(self.oracle)

    def session(self, **kwargs):
        return delivery_mod.ConformanceDelivery(self.state, self.toolbox, self.header, **kwargs)

    def bytes(self):
        return {path: (self.root / path).read_bytes() if (self.root / path).is_file() else None
                for path in self.paths.values()}

    def approved(self):
        session = self.session()
        record, problems = session.write(self.text)
        self.assertEqual(problems, [])
        self.state["conformance_delivery"] = deepcopy(record)
        session.commit()
        return record

    def update_metadata(self, modify):
        full = self.root / self.paths["oracle_path"]
        record = yaml.safe_load(full.read_text())
        modify(record)
        full.write_text(yaml.safe_dump(record))

    def test_paths_are_fixed_and_invalid_run_roots_fail_closed(self):
        self.assertEqual(self.paths, {
            "oracle_path": self.run + "/09_implementation/conformance/oracle.yaml",
            "source_path": self.run + "/09_implementation/conformance/cfu_conformance.c"})
        for bad in ("", "/tmp/run", self.run + "/../other", self.run + "/nested"):
            with self.subTest(root=bad), self.assertRaises(ValueError):
                delivery_mod.delivery_paths({"run_root": bad})

    def test_write_pending_current_requires_external_host_approval(self):
        session = self.session()
        record, problems = session.write(self.text)
        self.assertEqual(problems, [])
        self.assertEqual(record["oracle"], self.oracle)
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["header_path"], self.header)
        self.assertEqual(record["source_sha"], hashlib.sha256(
            conformance.render_conformance(self.state["isa_spec"], self.oracle).encode()).hexdigest())
        canonical = json.dumps(self.state["isa_spec"], sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode()
        self.assertEqual(record["isa_sha"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(session.current(), (record, []))
        self.assertEqual(session.record, record)
        self.assertTrue(session.changed)
        self.assertNotIn("conformance_delivery", self.state)
        _, errors = delivery_mod.inspect_delivery(self.state, self.toolbox)
        self.assertTrue(any("approval record is missing" in error for error in errors))
        self.state["conformance_delivery"] = record
        self.assertEqual(delivery_mod.inspect_delivery(self.state, self.toolbox), (record, []))

    def test_unchanged_approved_delivery_is_reusable_and_not_changed(self):
        self.approved()
        session = self.session()
        self.assertEqual(session.current()[1], [])
        self.assertFalse(session.changed)
        self.assertEqual(session.write(self.text)[1], [])
        self.assertFalse(session.changed)

    def test_invalid_or_empty_oracle_rejected_without_mutation(self):
        self.approved()
        before = self.bytes()
        session = self.session()
        for text in ("", "[]", "commands: []", "commands:\n - name: add\n   reference: 0\n   cases: []"):
            with self.subTest(text=text):
                self.assertTrue(session.write(text)[1])
                self.assertEqual(self.bytes(), before)
                self.assertFalse(session.changed)

    def test_owner_parser_validator_and_renderer_are_patchable(self):
        session = self.session()
        with mock.patch.object(conformance, "parse_oracle", return_value=({}, ["parse rejected"])) as parse:
            self.assertEqual(session.write(self.text)[1], ["parse rejected"])
            parse.assert_called_once_with(self.text)
        with mock.patch.object(conformance, "validate_oracle", return_value=["lint rejected"]):
            self.assertEqual(session.write(self.text)[1], ["lint rejected"])
        with mock.patch.object(conformance, "render_conformance", wraps=conformance.render_conformance) as render:
            self.assertEqual(session.write(self.text)[1], [])
            self.assertGreaterEqual(render.call_count, 1)

    def test_frozen_and_diagnosis_locked_writes_never_mutate(self):
        self.approved()
        before = self.bytes()
        frozen = self.session(editable=False)
        self.assertEqual(frozen.current()[1], [])
        self.assertIn("frozen", frozen.write(self.text)[1][0])
        locked = self.session(diagnosis_required=True)
        self.assertFalse(locked.unlocked)
        self.assertIn("diagnosis", locked.write(self.text)[1][0])
        self.assertEqual(self.bytes(), before)
        locked.diagnosis = {"root_cause": "parent validated"}
        self.assertTrue(locked.unlocked)
        self.assertEqual(locked.write(self.text)[1], [])

    def test_source_tamper_cannot_be_approved_by_metadata_hash_alone(self):
        self.approved()
        full = self.root / self.paths["source_path"]
        full.write_text("int main(void) { return 0; }\n")
        self.update_metadata(lambda record: record.update(source_sha=hashlib.sha256(full.read_bytes()).hexdigest()))
        _, errors = delivery_mod.inspect_delivery(self.state, self.toolbox)
        self.assertTrue(any("regeneration" in error for error in errors))
        self.assertTrue(any("host state" in error for error in errors))

    def test_source_missing_and_source_hash_tamper_fail(self):
        self.approved()
        full = self.root / self.paths["source_path"]
        original = full.read_bytes()
        full.unlink()
        self.assertTrue(any("source is missing" in error for error in
                            delivery_mod.inspect_delivery(self.state, self.toolbox)[1]))
        full.write_bytes(original + b"// tamper\n")
        self.assertTrue(any("source hash mismatch" in error for error in
                            delivery_mod.inspect_delivery(self.state, self.toolbox)[1]))

    def test_stale_actual_isa_and_generator_rejected(self):
        self.approved()
        self.state["isa_spec"]["metadata"] = {"changed": True}
        self.assertTrue(any("ISA hash" in error for error in
                            delivery_mod.inspect_delivery(self.state, self.toolbox)[1]))
        self.state["isa_spec"].pop("metadata")
        with mock.patch.object(delivery_mod, "_generator_sha", return_value="0" * 64):
            self.assertTrue(any("generator hash" in error for error in
                                delivery_mod.inspect_delivery(self.state, self.toolbox)[1]))

    def test_schema_hashes_legacy_empty_and_non_mapping_rejected(self):
        self.approved()
        original = (self.root / self.paths["oracle_path"]).read_bytes()
        for raw in ("[]", "oracle: {}", "schema_version: 1\noracle: {commands: []}"):
            (self.root / self.paths["oracle_path"]).write_text(raw)
            self.assertTrue(delivery_mod.inspect_delivery(self.state, self.toolbox)[1])
        (self.root / self.paths["oracle_path"]).write_bytes(original)
        self.update_metadata(lambda record: record.update(schema_version=True, oracle_sha="bad"))
        errors = delivery_mod.inspect_delivery(self.state, self.toolbox)[1]
        self.assertTrue(any("schema_version" in error for error in errors))
        self.assertTrue(any("oracle_sha" in error for error in errors))

    def test_metadata_arbitrary_paths_not_followed_or_returned(self):
        self.approved()
        self.update_metadata(lambda record: record.update(oracle_path="secret.yaml", source_path="secret.c",
                                                         header_path="/tmp/secret.h"))
        with mock.patch.object(self.toolbox, "normalize", wraps=self.toolbox.normalize) as normalize:
            record, errors = delivery_mod.inspect_delivery(self.state, self.toolbox)
        self.assertEqual(record["oracle_path"], self.paths["oracle_path"])
        self.assertEqual(record["source_path"], self.paths["source_path"])
        self.assertTrue(errors)
        self.assertEqual({call.args[0] for call in normalize.call_args_list}, set(self.paths.values()))

    def test_missing_or_mismatched_host_hashes_rejected(self):
        record = self.approved()
        for field in ("isa_sha", "oracle_sha", "source_sha", "generator_sha"):
            self.state["conformance_delivery"] = {**record, field: "0" * 64}
            self.assertTrue(any("host state" in error and field in error for error in
                                delivery_mod.inspect_delivery(self.state, self.toolbox)[1]))

    def test_policy_denial_other_run_and_symlink_fail_closed(self):
        self.approved()
        with mock.patch.object(self.toolbox, "is_read_allowed", return_value=False), self.assertRaises(PolicyError):
            delivery_mod.inspect_delivery(self.state, self.toolbox)
        other = {**self.state, "run_root": "agents/generated/runs/other"}
        with self.assertRaises(PolicyError):
            delivery_mod.inspect_delivery(other, self.toolbox)
        full = self.root / self.paths["source_path"]
        full.unlink()
        other_file = self.root / "target.c"
        other_file.write_text("private")
        full.symlink_to(other_file)
        with self.assertRaises(PolicyError):
            delivery_mod.inspect_delivery(self.state, self.toolbox)

    def test_second_write_failure_restores_previous_pair_and_pending(self):
        session = self.session()
        first, errors = session.write(self.text)
        self.assertEqual(errors, [])
        before = self.bytes()
        real_write = self.toolbox.write_text
        def fail_source(path, text):
            result = real_write(path, text)
            if path == self.paths["source_path"]:
                raise OSError("second write failed after mutation")
            return result
        revised = deepcopy(self.oracle)
        revised["commands"][0]["cases"].append({"args": [3, 4]})
        with mock.patch.object(self.toolbox, "write_text", side_effect=fail_source), self.assertRaises(OSError):
            session.write(yaml.safe_dump(revised))
        self.assertEqual(self.bytes(), before)
        self.assertEqual(session.current(), (first, []))
        session.rollback()
        self.assertEqual(self.bytes(), {path: None for path in self.paths.values()})
        self.assertFalse(session.changed)

    def test_second_write_failure_restores_absent_files(self):
        session = self.session()
        real_write = self.toolbox.write_text
        def fail_source(path, text):
            if path == self.paths["source_path"]:
                raise OSError("failure")
            return real_write(path, text)
        with mock.patch.object(self.toolbox, "write_text", side_effect=fail_source), self.assertRaises(OSError):
            session.write(self.text)
        self.assertFalse(session.changed)
        self.assertEqual(self.bytes(), {path: None for path in self.paths.values()})

    def test_rollback_exact_bytes_and_only_fixed_pair(self):
        for path in self.paths.values():
            full = self.root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(b"old\r\nbytes\xff")
        before = self.bytes()
        manifest = self.root / self.run / "manifest.yaml"
        manifest.write_text("untouched: true")
        session = self.session()
        self.assertEqual(session.write(self.text)[1], [])
        session.rollback()
        self.assertEqual(self.bytes(), before)
        self.assertEqual(manifest.read_text(), "untouched: true")
        self.assertFalse(session.changed)
        self.assertTrue(session.write(self.text)[1])
        with self.assertRaises(RuntimeError):
            session.commit()

    def test_inspection_io_failure_propagates_and_invalid_header_never_mutates(self):
        self.approved()
        with mock.patch.object(Path, "read_bytes", side_effect=OSError("read fault")), self.assertRaises(OSError):
            delivery_mod.inspect_delivery(self.state, self.toolbox)
        before = self.bytes()
        invalid = delivery_mod.ConformanceDelivery(self.state, self.toolbox, "/tmp/arbitrary.h")
        self.assertTrue(invalid.write(self.text)[1])
        self.assertEqual(self.bytes(), before)

    def test_first_write_failure_after_mutation_restores_pair(self):
        self.approved()
        session = self.session()
        before = self.bytes()
        real_write = self.toolbox.write_text
        def fail_first(path, text):
            real_write(path, text)
            raise OSError("first write fault")
        with mock.patch.object(self.toolbox, "write_text", side_effect=fail_first), self.assertRaises(OSError):
            session.write(self.text)
        self.assertEqual(self.bytes(), before)
        self.assertFalse(session.changed)
        self.assertEqual(session.current()[1], [])

    def test_commit_keeps_bytes_and_closes_transaction(self):
        session = self.session()
        self.assertEqual(session.write(self.text)[1], [])
        session.commit()
        before = self.bytes()
        session.rollback()
        self.assertEqual(self.bytes(), before)
        self.assertTrue(session.write(self.text)[1])
        with self.assertRaises(ValueError):
            self.session().commit()


if __name__ == "__main__":
    unittest.main()
