"""Offline ISA draft transactions and native contract-stage micro integrations.

All artifacts live under TemporaryDirectory; no generated-run fixtures, models,
commands, simulator or graph execution are involved.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from agents.tests.common import REPO_ROOT, VALID_ISA
from agents.cfu_design_agent.runtime import failures, isa_tools, llm_step, parsing, stage_services
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.policy import PolicyError
from agents.cfu_design_agent.runtime.stage_executor import _RunDirView
from agents.cfu_design_agent.stages import contract


class _NativeClient:
    """Prearranged model rounds consumed by the real native StageExecutor."""
    available = True
    model = "offline-isa-editing"
    _stage_round_client = object()  # No SDK construction, including localhost.

    def __init__(self, turns, observe=None):
        self.turns = list(turns)
        self.seen = []
        self.observe = observe

    def model_round(self, sdk, messages, *, allow_tools, tools, response_format):
        if self.observe:
            self.observe()
        self.seen.append((deepcopy(messages), deepcopy(tools)))
        if not self.turns:
            raise AssertionError("unexpected native model round")
        calls = self.turns.pop(0)
        return SimpleNamespace(
            text="", reasoning="", reasoning_source="", finish_reason="stop",
            extra_keys=(), input_tokens=1, output_tokens=1, total_tokens=2,
            message=SimpleNamespace(tool_calls=[
                SimpleNamespace(id=f"call_{len(self.seen)}_{i}", function=SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)))
                for i, (name, arguments) in enumerate(calls)
            ]),
        )


class IsaEditingTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="isa-editing-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        policy = self.root / "agents/specs/agent_access_policy.yaml"
        policy.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "agents/specs/agent_access_policy.yaml", policy)
        self.base = yaml.safe_load(VALID_ISA)
        self.base["software"]["intrinsic_names"] = ["cfu_compute"]
        self.state = {
            "workdir": str(self.root), "run_id": "isa-editing", "out_dir": "agents/generated",
            "run_root": "agents/generated/runs/isa-editing", "task": "extend weighted-average CFU",
            "isa_spec": deepcopy(self.base), "isa_spec_text": parsing._yaml_str(self.base),
            "design_contract": parsing._yaml_str(self.base), "isa_revision": 2,
            "workload_spec": {}, "context": {}, "agent_tools": True,
            "dry_run": False, "apply_patch": True, "max_tool_rounds": 8,
            "optimization_plan": {"parent_round": 4, "focus": "isa", "reason": "reduce dispatch",
                                  "hypothesis": {"nested": {"instruction": "cfu_finish"},
                                                 "counterexample": "unseen uint8 lanes"}},
        }
        stage_services._write_isa_artifact(self.state, self.base, [])
        self.canonical = self.root / self.state["run_root"] / "08_isa/cfu_isa.yaml"
        self.canonical_before = self.canonical.read_bytes()

    def session(self, stage="extend_isa", *, state=None, seed=None):
        state = state if state is not None else self.state
        return isa_tools.IsaDraftSession(
            state, stage_services._toolbox_for_stage(state, "isa"),
            seed=seed if seed is not None else self.base, stage=stage)

    def extension(self):
        result = deepcopy(self.base)
        result["commands"].append({
            "function_id": 2, "name": "finish", "operand": "rs1, rs2", "result": "rd",
            "semantics": "return sum of rs1 and rs2 modulo 2**32", "response": "value",
            "resources": [{"storage": "cpu", "access": "read", "role": "rs1"},
                          {"storage": "cpu", "access": "read", "role": "rs2"},
                          {"storage": "cpu", "access": "write", "role": "rd"}],
        })
        result["software"]["intrinsic_names"].append("cfu_finish")
        return result

    @staticmethod
    def replace_with(session, candidate):
        return session.edit([{"search": session.text, "replace": parsing._yaml_str(candidate)}])

    def assertCanonicalUnchanged(self):
        self.assertEqual(self.canonical.read_bytes(), self.canonical_before)

    def test_check_reports_wrong_command_shapes_without_framework_error(self):
        for invalid in (123, True, "compute", {}, None):
            with self.subTest(commands=invalid):
                session = self.session()
                candidate = self.extension()
                candidate["commands"] = invalid
                self.replace_with(session, candidate)
                self.assertTrue(session.check().startswith("ISA check failed"))
                ok, reason = session.submit("invalid draft")
                self.assertFalse(ok)
                self.assertIn("commands", reason)
                self.assertCanonicalUnchanged()

    def missing_submit_turns(self):
        return [[("edit_isa", {"edits": [{"search": parsing._yaml_str(self.base),
                                         "replace": parsing._yaml_str(self.extension())}]})],
                [], [], []]

    def test_native_missing_submit_retains_draft_and_routes_answer_repair_then_recovers(self):
        self.state["max_answer_repairs"] = 2
        rejected, _ = self.native_extend(self.missing_submit_turns())
        self.assertFalse(rejected["gates"]["isa"]["ok"])
        self.assertEqual(rejected["gates"]["isa"]["next"], "extend_isa")
        self.assertEqual(rejected["answer_repairs_used"], 1)
        self.assertEqual(rejected["stage_jobs"], "extend_isa")
        self.assertNotIn("isa_spec", rejected)
        self.assertNotIn("isa_revision", rejected)
        self.assertEqual(self.state["isa_spec"], self.base)
        self.assertCanonicalUnchanged()
        record = rejected["isa_draft"]
        self.assertEqual(record["stage"], "extend_isa")
        snapshot = (self.root / record["immutable_path"]).read_text()
        self.assertEqual(record["sha256"], isa_tools.IsaDraftSession.digest(snapshot))
        self.assertEqual(yaml.safe_load(snapshot), self.extension())
        self.assertNotIn("submission_error", record)
        self.assertIn("did not call submit_isa", str(rejected["isa_spec_errors"]))
        audit_path = rejected["llm_raw_outputs"][0]
        audit = yaml.safe_load((self.root / audit_path).read_text())
        self.assertEqual(audit["kind"], "model_delivery_failure")
        self.assertEqual(audit["terminal_reason"], "delivery_missing")
        self.assertIn("Host failure audit, not a model final answer", audit["note"])
        self.assertEqual(audit["draft"], record)
        self.assertEqual([call["name"] for call in audit["tool_calls"]], ["edit_isa"])
        self.assertFalse((self.canonical.parent / "llm_raw").exists())
        self.state.update(rejected)
        restored = self.session()
        self.assertEqual(restored.record, record)
        self.assertEqual(yaml.safe_load(restored.text), self.extension())
        with self.assertRaises(ValueError):
            restored.accepted_text()
        accepted, client = self.native_extend([
            [("check_isa", {})], [("submit_isa", {"delivery": "checked restored draft, no new edits"})],
        ])
        self.assertEqual(accepted["isa_spec_errors"], [])
        self.assertTrue(accepted["gates"]["isa"]["ok"])
        self.assertEqual(accepted["isa_revision"], 3)
        self.assertEqual(accepted["isa_draft"], {})
        self.assertEqual(accepted["isa_spec"], self.extension())
        self.assertEqual(yaml.safe_load(self.canonical.read_text()), self.extension())
        self.assertEqual(len(client.seen), 2)

    def test_native_missing_submit_with_zero_answer_budget_stops_without_accepting(self):
        self.state["max_answer_repairs"] = 0
        rejected, _ = self.native_extend(self.missing_submit_turns())
        self.assertFalse(rejected["gates"]["isa"]["ok"])
        self.assertEqual(rejected["gates"]["isa"]["next"], "review_and_route")
        self.assertEqual(rejected["stopped_at"], "isa")
        self.assertEqual(rejected["stage_jobs"], "review")
        self.assertNotIn("answer_repairs_used", rejected)
        self.assertNotIn("isa_spec", rejected)
        self.assertNotIn("isa_revision", rejected)
        self.assertTrue(rejected["isa_draft"]["sha256"])
        self.assertIn("answer-repair budget is exhausted", str(rejected["errors"]))
        self.assertEqual(self.state["isa_spec"], self.base)
        self.assertCanonicalUnchanged()

    def test_infrastructure_io_and_programming_errors_propagate_unchanged(self):
        for exc in (failures.StageToolsUnavailable("native tools unavailable"),
                    OSError("artifact device unavailable"), ValueError("invalid execution request")):
            with self.subTest(exception=type(exc).__name__):
                with mock.patch.object(llm_step, "_llm", return_value=_NativeClient([])), \
                     mock.patch.object(llm_step, "_tools_enabled", return_value=True), \
                     mock.patch.object(llm_step, "_run_agent_step", side_effect=exc), \
                     self.assertRaises(type(exc)) as raised:
                    contract.extend_isa(self.state)
                self.assertIs(raised.exception, exc)
                self.assertFalse((self.canonical.parent / "drafts/extend_isa_failure_000.yaml").exists())
                self.assertFalse((self.canonical.parent / "extensions/attempt_000.yaml").exists())
                self.assertCanonicalUnchanged()

    def test_plain_step_supports_yaml_json_edit_without_forcing_json_mode(self):
        seed = parsing._yaml_str(self.base)
        for delivery in (parsing._yaml_str(self.extension()), json.dumps(self.extension()),
                         self.edit_delivery(f"{self.state['run_root']}/08_isa/drafts/extend_isa.yaml",
                                            seed.rstrip("\n"), parsing._yaml_str(self.extension()).rstrip("\n"))):
            with self.subTest(delivery=delivery[:20]):
                session = self.session()
                with mock.patch.object(llm_step, "_complete_llm", return_value=(
                        LlmResult(delivery, used_llm=True), "offline-plain-output")) as complete:
                    result, raw, record = contract._isa_step(
                        self.state, object(), False, "extend_isa", "isa_extension", "plain prompt", seed,
                        session=session)
                self.assertIsNone(complete.call_args.kwargs["response_format"])
                self.assertEqual(yaml.safe_load(result.text), self.extension())
                self.assertEqual(yaml.safe_load(session.text), self.extension())
                self.assertEqual(raw, "offline-plain-output")
                self.assertEqual(record, session.record)
                self.assertTrue(record["sha256"])
                self.assertCanonicalUnchanged()

    def test_virtual_run_path_read_edit_roundtrip_preserves_actual_disk_path(self):
        seed = deepcopy(self.base)
        root = self.state["run_root"]
        seed["software"]["notes"] = f"inspect {root}/08_isa/cfu_isa.yaml"
        session = self.session(seed=seed)
        tools = _RunDirView(isa_tools.IsaToolset(session.toolbox, session=session), root)
        visible = tools.run("read_isa", {})
        self.assertIn("inspect {{run-dir}}/08_isa/cfu_isa.yaml", visible)
        self.assertNotIn(root, visible)
        result = tools.run("edit_isa", {"edits": [{
            "search": "inspect {{run-dir}}/08_isa/cfu_isa.yaml",
            "replace": "inspect {{run-dir}}/08_isa/revisions/003.yaml",
        }]})
        self.assertIn("applied 1 ISA draft edit", result)
        expected = f"inspect {root}/08_isa/revisions/003.yaml"
        self.assertEqual(yaml.safe_load(session.text)["software"]["notes"], expected)
        for path in (session.record["path"], session.record["immutable_path"]):
            self.assertEqual(yaml.safe_load((self.root / path).read_text())["software"]["notes"], expected)
        self.assertNotIn("{{run-dir}}", session.text)
        self.assertCanonicalUnchanged()

    def test_ambiguous_virtual_search_rejects_whole_batch_without_save(self):
        seed = deepcopy(self.base)
        root = self.state["run_root"]
        seed["software"]["notes"] = (
            f"inspect {root}/08_isa/cfu_isa.yaml | inspect {{{{run-dir}}}}/08_isa/cfu_isa.yaml")
        session = self.session(seed=seed)
        tools = _RunDirView(isa_tools.IsaToolset(session.toolbox, session=session), root)
        visible = tools.run("read_isa", {})
        search = "inspect {{run-dir}}/08_isa/cfu_isa.yaml"
        self.assertEqual(visible.count(search), 2)
        self.assertEqual(session.text.count(search), 1)  # Raw uniqueness is insufficient.
        before = session.text
        result = tools.run("edit_isa", {"edits": [
            {"search": "name: u8_wavg_agent", "replace": "name: must_not_leak"},
            {"search": search, "replace": "inspect {{run-dir}}/08_isa/revisions/003.yaml"},
        ]})
        self.assertIn("virtual SEARCH must match exactly once", result)
        self.assertEqual(session.text, before)
        self.assertEqual(session.record, {})
        self.assertEqual(session.edits, 0)
        self.assertFalse((self.root / session.path).exists())
        self.assertCanonicalUnchanged()

    def test_precise_edit_read_ranges_and_atomic_batch(self):
        session = self.session("design_contract")
        self.assertIn("1: spec_version:", session.read(1, 2))
        before = session.text
        for edits in (
            [{"search": "name: u8_wavg_agent", "replace": "name: edited"},
             {"search": "missing unique text", "replace": "x"}],
            [{"search": "draft", "replace": "approved"}],
            [{"search": "", "replace": "x"}],
            [{"search": "name: u8_wavg_agent", "replace": "name: u8_wavg_agent"}],
            [{"search": "name: u8_wavg_agent", "replace": "x", "path": "other"}],
        ):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                session.edit(edits)
            self.assertEqual(session.text, before)
            self.assertEqual(session.record, {})
            self.assertFalse((self.root / session.path).exists())
        session.edit([{"search": "name: u8_wavg_agent", "replace": "name: edited"},
                      {"search": "name: edited", "replace": "name: edited_twice"}])
        self.assertEqual(session.edits, 2)
        self.assertIn("name: edited_twice", session.text)
        self.assertEqual(session.record["sha256"], session.digest(session.text))
        self.assertCanonicalUnchanged()
        for args in ((True,), (0,), (1, False), (2, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                session.read(*args)

    def test_check_reports_syntax_schema_and_duplicate_encodings(self):
        session = self.session()
        for text, expected in (("commands: [", "syntax"), ("{}", "identity.name is required")):
            with self.subTest(text=text):
                session.apply_delivery(text)
                self.assertIn("ISA check failed", session.check())
                self.assertIn(expected, session.check())
                self.assertFalse(session.submit("ready")[0])
        bad = self.extension()
        bad["commands"][1]["function_id"] = 1
        session.apply_delivery(parsing._yaml_str(bad))
        self.assertIn("duplicates", session.check())
        self.assertCanonicalUnchanged()

    def test_extension_protects_normative_fields_but_allows_advisory(self):
        for field, mutate in (
            ("commands[0].semantics", lambda c: c["commands"][0].update(semantics="different")),
            ("datapath.xlen", lambda c: c["datapath"].update(xlen=64)),
            ("software.intrinsic_names", lambda c: c["software"].update(intrinsic_names=["cfu_finish"])),
            ("integration.bus_style", lambda c: c["integration"].update(bus_style="different")),
            ("assumptions.input_domain", lambda c: c["assumptions"].update(input_domain="small only")),
        ):
            with self.subTest(field=field):
                session = self.session()
                candidate = self.extension()
                mutate(candidate)
                self.replace_with(session, candidate)
                self.assertIn(field, session.check())
                self.assertFalse(session.submit("ready")[0])
                self.assertTrue(session.record)
        session = self.session()
        candidate = self.extension()
        candidate["software"].update(kernel_plan="new schedule", extension_kernel_plan="use finish",
                                     notes="advisory", rationale="reduce dispatch")
        candidate["datapath"]["implementation"] = {"pipeline": 4, "xlen": 128}
        candidate["integration"].update(notes="same wiring", rationale="advisory")
        self.replace_with(session, candidate)
        self.assertIn("ISA check passed", session.check())
        self.assertEqual(session.submit("appended finish and scheduling advice"), (True, ""))
        self.assertEqual(yaml.safe_load(session.accepted_text()), candidate)
        self.assertCanonicalUnchanged()

    def test_extension_requires_new_command_even_with_advisory_changes(self):
        session = self.session()
        candidate = deepcopy(self.base)
        candidate["software"]["kernel_plan"] = "new prose, no instruction"
        self.replace_with(session, candidate)
        self.assertIn("append at least one new command", session.check())
        self.assertFalse(session.submit("only planning changed")[0])

    def test_formatting_only_initial_draft_is_not_a_submission(self):
        session = self.session("design_contract")
        session.apply_delivery(json.dumps(self.base, indent=4))
        self.assertNotEqual(session.text, session.seed_text)
        self.assertIn("ISA check passed", session.check())
        self.assertIn("unchanged", session.submit("formatted JSON")[1])
        self.assertFalse(session.submit(" ")[0])
        self.assertCanonicalUnchanged()

    def test_edit_and_rollback_invalidate_submitted_hash_without_publishing(self):
        session = self.session()
        self.replace_with(session, self.extension())
        self.assertTrue(session.submit("ready")[0])
        accepted_hash = session.submitted_sha256
        self.assertEqual(accepted_hash, session.digest(session.text))
        session.edit([{"search": "name: finish", "replace": "name: finish_v2"}])
        self.assertNotEqual(session.digest(session.text), accepted_hash)
        self.assertEqual(session.submitted_sha256, "")
        self.assertIsNone(session.accepted)
        with self.assertRaises(ValueError):
            session.accepted_text()
        self.assertTrue(session.submit("renamed new command only")[0])
        record = dict(session.record)
        session.rollback()
        self.assertEqual(session.record, record)
        self.assertTrue((self.root / record["immutable_path"]).is_file())
        with self.assertRaises(ValueError):
            session.accepted_text()
        self.assertCanonicalUnchanged()

    def test_hash_guard_detects_unrecorded_text_change(self):
        session = self.session()
        self.replace_with(session, self.extension())
        self.assertTrue(session.submit("ready")[0])
        session.text += "\n# changed outside edit\n"
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            session.accepted_text()

    def test_restore_uses_immutable_snapshot_not_mutable_alias(self):
        session = self.session()
        self.replace_with(session, self.extension())
        record = dict(session.record)
        snapshot_text = session.text
        self.assertTrue(session.submit("ready")[0])
        (self.root / session.path).write_text("tampered alias", encoding="utf-8")
        restored = self.session(state={**self.state, "isa_draft": record})
        self.assertEqual(restored.text, snapshot_text)
        self.assertEqual(restored.record, record)
        self.assertIsNone(restored.accepted)
        with self.assertRaises(ValueError):
            restored.accepted_text()
        self.assertCanonicalUnchanged()

    def test_restore_ignores_other_base_or_stage_and_rejects_hash_mismatch(self):
        session = self.session()
        self.replace_with(session, self.extension())
        record = dict(session.record)
        for changes in ({"stage": "design_contract"}, {"base_sha256": "other-base"}):
            restored = self.session(state={**self.state, "isa_draft": {**record, **changes}})
            self.assertEqual(restored.text, restored.seed_text)
            self.assertEqual(restored.record, {})
        with self.assertRaisesRegex(ValueError, "hash"):
            self.session(state={**self.state, "isa_draft": {**record, "sha256": "0" * 64}})
        (self.root / record["immutable_path"]).write_text("corrupt snapshot", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.session(state={**self.state, "isa_draft": record})

    def test_current_run_and_snapshot_paths_fail_closed(self):
        session = self.session()
        self.replace_with(session, self.extension())
        for target in (
            "agents/generated/runs/another/08_isa/drafts/versions/extend_isa.yaml.bad",
            f"{self.state['run_root']}/08_isa/cfu_isa.yaml",
            f"{self.state['run_root']}/08_isa/drafts/versions/extend_isa.yaml../../../../cfu_isa.yaml",
            f"{self.state['run_root']}/08_isa/drafts/versions/extend_isa.yaml.bad/../other",
            "/tmp/outside-isa-draft",
        ):
            with self.subTest(target=target), self.assertRaises(PolicyError):
                self.session(state={**self.state, "isa_draft": {**session.record, "immutable_path": target}})
        with self.assertRaises(PolicyError):
            isa_tools.IsaDraftSession({**self.state, "run_root": "agents/generated/runs/other"},
                                     session.toolbox, seed=self.base, stage="extend_isa")
        with self.assertRaises(ValueError):
            self.session("implement")

    def test_constructor_rejects_draft_symlink_to_accepted_contract(self):
        draft = self.canonical.parent / "drafts/extend_isa.yaml"
        draft.parent.mkdir(parents=True)
        draft.symlink_to(self.canonical)
        with self.assertRaises(PolicyError):
            self.session()
        self.assertCanonicalUnchanged()
        self.assertFalse((draft.parent / "versions").exists())

    def test_edit_rechecks_versions_directory_symlink_after_constructor(self):
        session = self.session()
        draft = self.root / session.path
        draft.parent.mkdir(parents=True)
        other = self.root / "agents/generated/runs/another/08_isa/drafts/versions"
        other.mkdir(parents=True)
        (draft.parent / "versions").symlink_to(other, target_is_directory=True)
        before = session.text
        with self.assertRaises(PolicyError):
            self.replace_with(session, self.extension())
        self.assertEqual(list(other.iterdir()), [])
        self.assertEqual(session.text, before)
        self.assertEqual(session.record, {})
        self.assertEqual(session.edits, 0)
        self.assertFalse(draft.exists())
        self.assertCanonicalUnchanged()

    @staticmethod
    def edit_delivery(path, search, replace):
        return (f"EDIT_BEGIN {path}\n<<<<<<< SEARCH\n{search}\n=======\n"
                f"{replace}\n>>>>>>> REPLACE\nEDIT_END")

    def test_plain_yaml_json_and_optional_edit_delivery(self):
        for text in (parsing._yaml_str(self.extension()), json.dumps(self.extension())):
            with self.subTest(format=text[:1]):
                session = self.session()
                session.apply_delivery(text)
                self.assertEqual(session.candidate(), (self.extension(), []))
                self.assertTrue(session.submit("complete compatible contract")[0])
                self.assertCanonicalUnchanged()
        session = self.session("design_contract")
        delivery = self.edit_delivery(session.path, "  name: u8_wavg_agent", "  name: edited")
        session.apply_delivery(delivery)
        self.assertEqual(yaml.safe_load(session.text)["identity"]["name"], "edited")
        for target in (f"{self.state['run_root']}/08_isa/cfu_isa.yaml", "../outside.yaml",
                       "agents/generated/runs/other/08_isa/drafts/design_contract.yaml"):
            with self.subTest(target=target), self.assertRaises(PolicyError):
                session.apply_delivery(self.edit_delivery(target, "edited", "forbidden"))
        self.assertCanonicalUnchanged()

    def test_tool_schemas_offer_draft_edits_not_shell_or_contract_writes(self):
        session = self.session()
        tools = isa_tools.IsaToolset(session.toolbox, session=session)
        schemas = tools.schemas()
        names = {item["function"]["name"] for item in schemas}
        self.assertTrue({"read_isa", "edit_isa", "check_isa"} <= names)
        self.assertLessEqual(names, {"read_isa", "edit_isa", "check_isa", "read_file", "list_dir",
                                     "search_text", "list_command_logs"})
        edit_schema = next(s["function"]["parameters"] for s in schemas
                           if s["function"]["name"] == "edit_isa")
        self.assertEqual(set(edit_schema["properties"]), {"edits"})
        self.assertFalse(edit_schema["additionalProperties"])
        self.assertNotIn("path", edit_schema["properties"]["edits"]["items"]["properties"])
        self.assertIn("unsupported arguments", tools.run("edit_isa", {"path": str(self.canonical)}))
        for name in ("run_command", "write_file", "apply_patch", "submit_isa"):
            self.assertIn("error", tools.run(name, {}))
        self.assertCanonicalUnchanged()

    def native_extend(self, turns):
        client = _NativeClient(turns, self.assertCanonicalUnchanged)
        # Only the provider factory is replaced: real prompt, session, toolset,
        # llm_step, native executor, submit gate, artifact writes and stage commit.
        real_record_output = llm_step._record_llm_output

        def record_before_commit(*args, **kwargs):
            self.assertCanonicalUnchanged()  # Native submit has returned already.
            return real_record_output(*args, **kwargs)

        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(llm_step, "_tools_enabled", return_value=True), \
             mock.patch.object(llm_step, "_record_llm_output", side_effect=record_before_commit):
            update = contract.extend_isa(self.state)
        self.assertFalse(client.turns)
        return update, client

    def test_real_native_initial_contract_commits_only_submitted_draft(self):
        candidate = deepcopy(self.base)
        candidate["identity"]["name"] = "initial_authored_contract"
        seed = contract._isa_from_template(self.state["isa_spec_text"], {}, "")
        client = _NativeClient([
            [("edit_isa", {"edits": [{"search": parsing._yaml_str(seed),
                                      "replace": parsing._yaml_str(candidate)}]})],
            [("check_isa", {})], [("submit_isa", {"delivery": "authored minimal exact contract"})],
        ], self.assertCanonicalUnchanged)
        real_record_output = llm_step._record_llm_output

        def record_before_commit(*args, **kwargs):
            self.assertCanonicalUnchanged()
            return real_record_output(*args, **kwargs)

        with mock.patch.object(llm_step, "_llm", return_value=client), \
             mock.patch.object(llm_step, "_tools_enabled", return_value=True), \
             mock.patch.object(llm_step, "_record_llm_output", side_effect=record_before_commit):
            update = contract.design_contract(self.state)
        self.assertFalse(client.turns)
        self.assertEqual(update["isa_spec_errors"], [])
        self.assertEqual(update["isa_spec"], candidate)
        self.assertEqual(update["isa_revision"], 0)
        self.assertEqual(update["isa_draft"], {})
        self.assertEqual(yaml.safe_load(self.canonical.read_text()), candidate)

    def test_real_native_extend_edits_checks_submits_then_commits_revision(self):
        seed = parsing._yaml_str(self.base)
        candidate = self.extension()
        update, client = self.native_extend([
            [("read_isa", {})],
            [("edit_isa", {"edits": [{"search": seed, "replace": parsing._yaml_str(candidate)}]})],
            [("check_isa", {})],
            [("submit_isa", {"delivery": "appended finish command and cfu_finish name"})],
        ])
        self.assertEqual(update["isa_spec_errors"], [])
        self.assertEqual(update["isa_spec"], candidate)
        self.assertEqual(update["isa_revision"], 3)
        self.assertEqual(update["isa_draft"], {})
        self.assertEqual(yaml.safe_load(self.canonical.read_text()), candidate)
        self.assertNotEqual(self.canonical.read_bytes(), self.canonical_before)
        self.assertEqual(self.state["isa_spec"], self.base)
        revision = self.canonical.parent / "revisions/003.yaml"
        self.assertEqual(yaml.safe_load(revision.read_text()), candidate)
        names = {s["function"]["name"] for s in client.seen[0][1]}
        self.assertIn("submit_isa", names)
        self.assertNotIn("submit_stage", names)
        messages = str(client.seen[0][0])
        for value in ("unseen uint8 lanes", "cfu_finish", "reduce dispatch", "parent_round"):
            self.assertIn(value, messages)
        audit = yaml.safe_load((self.canonical.parent / "isa_tool_calls.yaml").read_text())
        self.assertIn("submit_isa", str(audit))

    def test_native_rejected_normative_mutation_can_be_locally_fixed_and_submitted(self):
        bad = self.extension()
        old_semantics = self.base["commands"][0]["semantics"]
        bad["commands"][0]["semantics"] = "forbidden approximate semantics"
        update, client = self.native_extend([
            [("edit_isa", {"edits": [{"search": parsing._yaml_str(self.base),
                                       "replace": parsing._yaml_str(bad)}]})],
            [("check_isa", {})],
            [("submit_isa", {"delivery": "first try"})],
            [("edit_isa", {"edits": [{"search": "forbidden approximate semantics", "replace": old_semantics}]}),
             ("check_isa", {})],
            [("submit_isa", {"delivery": "restored old semantics, retained new command"})],
        ])
        self.assertEqual(update["isa_spec_errors"], [])
        self.assertEqual(update["isa_spec"], self.extension())
        self.assertEqual(update["isa_revision"], 3)
        self.assertIn("commands[0].semantics", str(client.seen[3][0]))
        self.assertEqual(yaml.safe_load(self.canonical.read_text()), self.extension())

    def test_stage_rejection_preserves_accepted_alias_and_draft_for_next_call(self):
        bad = self.extension()
        bad["commands"][0]["semantics"] = "bad old-command mutation"
        captured = {}

        def rejected_step(state, toolbox, **kwargs):
            session = kwargs["toolset_kwargs"]["session"]
            self.replace_with(session, bad)
            captured["record"] = dict(session.record)
            self.assertFalse(kwargs["submit_stage"]("invalid attempt")[0])
            self.assertCanonicalUnchanged()
            return LlmResult("invalid attempt", used_llm=True), "offline-raw", []

        with mock.patch.object(llm_step, "_llm", return_value=_NativeClient([])), \
             mock.patch.object(llm_step, "_tools_enabled", return_value=True), \
             mock.patch.object(llm_step, "_run_agent_step", side_effect=rejected_step):
            rejected = contract.extend_isa(self.state)
        self.assertNotIn("isa_spec", rejected)
        self.assertNotIn("isa_revision", rejected)
        self.assertEqual(rejected["isa_draft"], captured["record"])
        self.assertIn("commands[0].semantics", str(rejected["isa_spec_errors"]))
        self.assertCanonicalUnchanged()
        attempt = yaml.safe_load((self.canonical.parent / "extensions/attempt_000.yaml").read_text())
        self.assertFalse(attempt["accepted"])
        self.assertEqual(attempt["draft"]["immutable_path"], captured["record"]["immutable_path"])
        self.state.update(rejected)
        restored = self.session()
        self.assertEqual(yaml.safe_load(restored.text), bad)
        update, _ = self.native_extend([
            [("edit_isa", {"edits": [{"search": "bad old-command mutation",
                                      "replace": self.base["commands"][0]["semantics"]}]})],
            [("check_isa", {})], [("submit_isa", {"delivery": "fixed old command only"})],
        ])
        self.assertEqual(update["isa_spec_errors"], [])
        self.assertEqual(update["isa_spec"], self.extension())
        self.assertEqual(update["isa_revision"], 3)


if __name__ == "__main__":
    unittest.main()
