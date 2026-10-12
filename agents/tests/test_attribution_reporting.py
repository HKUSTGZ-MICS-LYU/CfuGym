"""Exclusive attribution/report contracts: temp artifacts, fake requests only."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from agents.cfu_design_agent import cli, graph
from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import failures, llm, provenance, request_usage, stage_executor, stage_services
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.stages.review import report
from agents.experiments.harness import classification, collection, metrics, parsing, reporting
from agents.tests.common import REPO_ROOT


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def audit_events(count=1, *, missing=False):
    events = [{"event": "request", "request_id": "metadata", "metadata": {"endpoint": "models.retrieve"}}]
    for index in range(count):
        rid = f"req-{index}"
        events.append({"event": "request", "request_id": rid, "metadata": {"endpoint": "chat", "stage": "implement"}})
        if missing and index == count - 1:
            events.append({"event": "error", "request_id": rid, "error": {"type": "APIError"}})
            continue
        events.extend([
            {"event": "result", "request_id": rid, "result": {"status": "received", "usage": {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cached_input_tokens": 40}}},
            {"event": "result", "request_id": rid, "result": {"status": "completed", "input_tokens": 101,
                "output_tokens": 11, "cached_input_tokens": 41, "total_tokens": 112}},
        ])
    return events


def put_audit(path, events):
    put(path, "".join(json.dumps(event) + "\n" for event in events))


def measured(**overrides):
    return {"ok": True, "terminal_reason": "passed", "correctness_ok": True, "timing_ok": True,
            "performance_valid": True, "reference_cycles": 100, "cfu_cycles": 50,
            "speedup_x100": 200, **overrides}


class RequestAttributionTest(unittest.TestCase):
    def test_final_usage_is_counted_once_per_request_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            put_audit(path, audit_events(4))
            usage = request_usage.load_request_usage(path)
            self.assertEqual((usage["calls"], usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
                              usage["cached_input_tokens"]), (4, 404, 44, 448, 164))
            self.assertTrue(usage["usage_complete"])
            self.assertEqual(usage["by_stage"]["implement"]["calls"], 4)

    def test_received_known_usage_survives_later_partial_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            events = audit_events()
            events[-1]["result"] = {"status": "completed", "input_tokens": 105, "output_tokens": None}
            events.append({"event": "error", "request_id": "req-0", "error": {"type": "ParserError"}})
            put_audit(path, events)
            usage = request_usage.load_request_usage(path)
            self.assertEqual(usage["calls"], 1)
            self.assertEqual(usage["input_tokens"], 105)
            self.assertEqual(usage["output_tokens"], 10)
            self.assertEqual(usage["total_tokens"], 115)

    def test_failed_request_unknown_usage_is_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            put_audit(path, audit_events(2, missing=True))
            warnings = []
            usage = request_usage.load_request_usage(path, warnings)
            self.assertEqual(usage["calls"], 2)
            self.assertIsNone(usage["input_tokens"])
            self.assertIsNone(usage["total_tokens"])
            self.assertEqual(usage["known_usage"]["total_tokens"], 112)
            self.assertEqual(usage["missing_usage_requests"], ["req-1"])
            self.assertIn("usage missing", ";".join(warnings))

    def test_missing_cache_does_not_erase_known_input_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            events = audit_events()
            for event in events[2:]:
                event["result"].get("usage", event["result"]).pop("cached_input_tokens", None)
            put_audit(path, events)
            usage = request_usage.load_request_usage(path)
            self.assertEqual(usage["total_tokens"], 112)
            self.assertIsNone(usage["cached_input_tokens"])

    def test_malformed_trailing_event_marks_totals_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            put_audit(path, audit_events())
            with path.open("a") as handle:
                handle.write('{"event":')
            usage = request_usage.load_request_usage(path)
            self.assertIsNone(usage["total_tokens"])
            self.assertEqual(usage["known_usage"]["total_tokens"], 112)
            self.assertFalse(usage["usage_complete"])

    def test_old_attempt_requests_are_not_charged_to_reused_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request_audit.jsonl"
            events = audit_events(2)
            for event in events:
                event["timestamp"] = "2026-10-04T00:00:00+00:00" if event.get("request_id") == "req-0" else "2026-10-05T00:00:00+00:00"
            put_audit(path, events)
            usage = request_usage.load_request_usage(path, started_at="2026-10-05T00:00:00+00:00")
            self.assertEqual(usage["calls"], 1)

    def test_terra_six_and_luna_three_crashed_runs_recover_cost_without_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, repeats in (("terra", 6), ("luna", 3)):
                records = []
                for repeat in range(repeats):
                    run, fw = root / model / str(repeat) / "run", root / model / str(repeat) / "fw"
                    put(run / "stderr.log", "openai.APIError: upstream error: 500\n")
                    put(fw / "run.yaml", "status: initialized\n")
                    put_audit(fw / "request_audit.jsonl", audit_events(2, missing=repeat == 0))
                    record = collection.collect_record("e", {"model": model}, str(repeat), run, fw, {"exit_code": 1})
                    self.assertEqual(record["token_calls"], 2)
                    self.assertEqual(record["token_usage_source"], "request_audit")
                    records.append(record)
                self.assertEqual(sum(record["token_calls"] for record in records), repeats * 2)
                self.assertEqual(sum(record["request_usage"]["known_usage"]["total_tokens"] for record in records), (repeats * 2 - 1) * 112)
                self.assertIsNone(records[0]["token_total"])

    def test_no_audit_preserves_report_fallback_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "run/stdout.log", "status=failed\ntoken_usage:\n  calls       = 3\n  input_tokens= 100\n  output_tokens=20\n  total_tokens=120\n")
            record = collection.collect_record("e", {}, "r", root / "run", root / "fw", {"exit_code": 1})
            self.assertEqual(record["token_total"], 120)
            self.assertEqual(record["token_usage_source"], "cli/report")
            self.assertIn("request_audit.jsonl missing", ";".join(record["token_usage_warnings"]))

    def test_infra_cost_is_retained_without_entering_pass_denominator(self):
        record = {"status": "failed", "infra": {"infra": True}, "token_calls": 2, "token_total": None,
                  "request_usage": {"known_usage": {"total_tokens": 112}}}
        row = metrics.cell_metrics([record], bootstrap_iters=2)
        self.assertEqual(row["runs"], 0)
        self.assertEqual(row["infra_cost_calls"], 2)
        self.assertEqual(row["infra_cost_known_tokens"], 112)


class CandidateProvenanceTest(unittest.TestCase):
    def test_current_requires_matching_candidate_and_passed_terminal(self):
        candidate = {"input_sha256": "new", "candidate_hashes": {"hardware": "hw"}, "evidence_epoch": {}}
        for reason in ("passed", "speedup_floor", "assertion"):
            with self.subTest(reason=reason):
                sim = measured(terminal_reason=reason, performance_provenance=candidate)
                view = provenance.current_performance(sim, candidate)
                self.assertEqual(view["performance_valid"], reason != "assertion")
                self.assertEqual(provenance.current_performance(sim, {**candidate, "input_sha256": "other"})["performance_valid"], False)

    def test_compiled_product_bytes_change_invalidates_source_identical_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"workdir": str(root), "run_root": "agents/generated/runs/fake",
                     "benchmark_project": {"kind": "generated_benchmark", "elf_path": "products/candidate.elf"}}
            elf = root / "products/candidate.elf"
            asm = elf.with_suffix(".asm")
            rtl = provenance.commands._soc_rtl_path(state)
            for path, text in ((elf, "ELF-v1"), (asm, "ASM-v1"), (rtl, "RTL-v1")):
                put(path, text)
            original = provenance.candidate_provenance(state)
            self.assertEqual(set(original["compiled_products"]), {"elf", "rtl", "asm"})
            sim = measured(performance_provenance=original)
            self.assertTrue(provenance.current_performance(sim, original)["performance_valid"])
            for path, text in ((elf, "ELF-v2"), (rtl, "RTL-v2"), (asm, "ASM-v2")):
                with self.subTest(product=path.name):
                    put(path, text)
                    changed = provenance.candidate_provenance(state)
                    self.assertEqual(changed["input_sha256"], original["input_sha256"])
                    self.assertFalse(provenance.current_performance(sim, changed)["performance_valid"])
                    self.assertEqual(changed["compiled_products"]["elf"]["path"], "products/candidate.elf")

    def test_legacy_source_only_proof_cannot_certify_known_compiled_products(self):
        proof = {"input_sha256": "same", "candidate_hashes": {}}
        current = {**proof, "compiled_products": {"elf": {"path": "candidate.elf", "sha256": "known"}}}
        self.assertFalse(provenance.current_performance(measured(performance_provenance=proof), current)["performance_valid"])
        self.assertTrue(provenance.current_performance(measured(performance_provenance={**proof, "compiled_products": {}}),
                                                       {**proof, "compiled_products": {}})["performance_valid"])

    def test_embench_without_run_binding_does_not_guess_absolute_products(self):
        state = {"benchmark_project": {"kind": "embench_benchmark", "benchmark": "nsichneu"}}
        self.assertEqual(provenance.compiled_products(state), {})
        self.assertEqual(provenance.candidate_provenance(state)["compiled_products"], {})

    def test_compiled_product_path_outside_workdir_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"workdir": str(root), "benchmark_project": {"elf_path": "../outside.elf"}}
            with self.assertRaisesRegex(ValueError, "compiled product escapes workdir"):
                provenance.compiled_products(state)

    def test_legacy_gate_input_hash_does_not_upgrade_unstamped_measurement(self):
        sim = measured(input_sha256="same")
        candidate = {"input_sha256": "same", "compiled_products": {}}
        self.assertFalse(provenance.current_performance(sim, candidate)["performance_valid"])
        self.assertEqual(provenance.observed_performance(sim)["speedup_x100"], 200)

    def test_rewind_epoch_change_invalidates_identical_bytes(self):
        proof = {"input_sha256": "same", "candidate_hashes": {}, "evidence_epoch": {"sim": 1}}
        self.assertFalse(provenance.current_performance(measured(performance_provenance=proof),
            {**proof, "evidence_epoch": {"sim": 2}})["performance_valid"])

    def test_old_qr_tar_wiki_build_failure_is_historical_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("qr", "tar", "wiki"):
                run, fw = root / name / "run", root / name / "fw"
                put(run / "stdout.log", "status=failed\nstopped_at=sw_build\n")
                gates = {"sim": measured(), "sw_build": {"ok": False, "errors": ["current compile failure"]}}
                put(fw / "report.md", "# Report\nStatus: failed\n## Gates\n" +
                    "\n".join(f"- {key}: {gate}" for key, gate in gates.items()) +
                    "\n## Validation Ladder\n- stopped at: sw_build\n- repairs used: 1/2 (iteration=1)\n")
                snapshot = fw / "10_validation/versions/sim.yaml.valid"
                text = yaml.safe_dump(measured())
                put(snapshot, text)
                record = collection.collect_record("e", {"benchmark": name}, name, run, fw, {"exit_code": 1})
                self.assertFalse(record["performance_valid"])
                self.assertIsNone(record["correctness_ok"])
                self.assertIsNone(record["timing_ok"])
                self.assertIsNone(record["speedup_x100"])
                observed = record["last_observed_valid_performance"]
                self.assertEqual(observed["speedup_x100"], 200)
                self.assertEqual(observed["evidence"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
                self.assertFalse(record["current_performance"]["performance_valid"])

    def test_initialized_api_failure_does_not_promote_historical_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "run/stderr.log", "openai.APITimeoutError: Request timed out\n")
            put(root / "fw/run.yaml", "status: initialized\n")
            put(root / "fw/10_validation/embench.yaml", yaml.safe_dump(measured()))
            record = collection.collect_record("e", {}, "r", root / "run", root / "fw", {"exit_code": 1})
            self.assertFalse(record["performance_valid"])
            self.assertIsNone(record["speedup_x100"])
            self.assertIsNone(record["correctness_ok"])
            self.assertEqual(record["last_observed_valid_performance"]["speedup_x100"], 200)

    def test_archived_record_preserves_snapshot_when_framework_artifacts_are_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "run/stdout.log", "status=failed\nstopped_at=hw_build\n")
            archive_path = root / "run/record.json"
            archived = {"status": "failed", "stopped_at": "hw_build", "terminal": True,
                        "gates": {"sim": measured()}, "speedup_x100": 200}
            text = json.dumps(archived)
            put(archive_path, text)
            record = collection.collect_record("e", {}, "r", root / "run", root / "missing-fw", {"exit_code": 1})
            observed = record["last_observed_valid_performance"]
            self.assertEqual(observed["speedup_x100"], 200)
            self.assertEqual(observed["evidence"]["source"], "archived_record")
            self.assertEqual(observed["evidence"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertFalse(record["performance_valid"])
            self.assertIsNone(record["speedup_x100"])

    def test_aggregation_of_old_archives_drops_only_current_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = {"status": "failed", "terminal": True, "stopped_at": "sw_build", "benchmark": "qr",
                       "gates": {"sim": measured()}, "performance_valid": True, "correctness_ok": True,
                       "timing_ok": True, "speedup_x100": 200}
            put(root / "runs/run/record.json", json.dumps(archive))
            records = reporting.load_records(root)
            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["performance_valid"])
            self.assertIsNone(records[0]["speedup_x100"])
            self.assertEqual(records[0]["last_observed_valid_performance"]["speedup_x100"], 200)
            self.assertEqual(json.loads((root / "runs/run/record.json").read_text()), archive)

    def test_failed_latest_sim_retains_earlier_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "10_validation/versions/sim.yaml.earlier", yaml.safe_dump(measured()))
            put(root / "10_validation/sim.yaml", yaml.safe_dump(measured(terminal_reason="assertion", correctness_ok=False)))
            observed = provenance.load_last_observed_performance(root)
            self.assertEqual(observed["speedup_x100"], 200)
            self.assertIn("earlier", observed["evidence"]["snapshot"])

    def test_missing_candidate_hash_never_claims_current_performance(self):
        view = provenance.performance_view({"gates": {"sim": measured()}})
        self.assertFalse(view["current_performance"]["performance_valid"])
        self.assertEqual(view["last_observed_valid_performance"]["speedup_x100"], 200)

    def test_report_parses_separate_current_and_historical_evidence(self):
        state = {"status": "failed", "run_root": "not-existing", "stopped_at": "hw_build",
                 "gates": {"sim": measured()}, "token_usage": {"calls": 1, "input": 2, "output": 3, "total": 5}}
        parsed = parsing.parse_report(report.render_report(state), [])
        self.assertFalse(parsed["current_performance"]["performance_valid"])
        self.assertEqual(parsed["last_observed_valid_performance"]["speedup_x100"], 200)


class TypedFailureTest(unittest.TestCase):
    def test_xgb_probe_crash_is_infra_not_prior_timeout_assertion(self):
        sim = measured(ok=False, terminal_reason="sim_harness_crash", performance_valid=False,
                       errors=["OLD liveness timeout; defer; design assertion", "cycle1 probe bookkeeping desync"])
        record = {"status": "failed", "stopped_at": "sim", "gates": {"sim": sim}}
        self.assertEqual(classification.classify_failure(record)["failure_class"], "infra_error")
        self.assertTrue(classification.detect_infra("status=failed\n", "", False, 1, record["gates"], stopped_at="sim")["infra"])

    def test_liveness_defer_design_assert_are_not_infra(self):
        for reason, expected in (("liveness_timeout", "sim_timeout"), ("assertion", "sim_assertion")):
            gate = {"ok": False, "terminal_reason": reason, "errors": ["design defer; streaming assertion"]}
            record = {"status": "failed", "stopped_at": "sim", "gates": {"sim": gate}}
            self.assertEqual(classification.classify_failure(record)["failure_class"], expected)
            self.assertFalse(classification.detect_infra("status=failed\n", "", False, 1, record["gates"], stopped_at="sim")["infra"])

    def test_reference_driver_failure_is_precondition(self):
        gate = {"ok": False, "terminal_reason": "driver_reference_failure", "errors": ["missing baseline"]}
        record = {"status": "failed", "stopped_at": "sim", "gates": {"sim": gate}}
        self.assertEqual(classification.classify_failure(record)["failure_class"], "precondition")
        self.assertFalse(classification.detect_infra("status=failed\n", "", False, 1, record["gates"])["infra"])

    def test_model_delivery_type_wins_over_missing_status_and_infra_words(self):
        typed = failures.termination(failures.StageDeliveryFailure("empty delivery; old timeout", stage="implement"))
        infra = classification.detect_infra("", "", False, 1, {}, typed)
        self.assertFalse(infra["infra"])
        self.assertEqual(classification.classify_failure({"status": "failed", "termination": typed})["failure_class"], "no_deliverable")

    def test_profile_build_or_reference_is_precondition_not_llm_or_infra(self):
        gate = {"ok": False, "terminal_reason": "profile_precondition", "failure_kind": "precondition_failure",
                "errors": ["reference executable did not pass its profile preflight"]}
        record = {"status": "failed", "stopped_at": "profile", "gates": {"profile": gate}}
        self.assertEqual(classification.classify_failure(record)["failure_class"], "precondition")
        self.assertFalse(classification.detect_infra("status=failed\n", "", False, 1, record["gates"], stopped_at="profile")["infra"])

    def test_model_failure_flag_alone_is_not_endpoint_exception(self):
        gate = {"ok": False, "model_failure": True, "errors": ["malformed oracle answer"]}
        self.assertFalse(classification.detect_infra("status=failed\n", "", False, 1, {"hw_conformance": gate})["infra"])

    def test_no_model_answer_without_transport_evidence_is_delivery_failure(self):
        gate = {"ok": False, "model_failure": True, "errors": ["the conformance oracle request got no model answer"]}
        self.assertFalse(classification.detect_infra("status=failed\n", "", False, 1, {"hw_conformance": gate})["infra"])

    def test_native_rejection_is_typed_rolls_back_and_not_delivered(self):
        class Tools:
            def __init__(self, *args, **kwargs):
                self.rollback = mock.Mock()
                instance.append(self)
            def schemas(self):
                return [{"type": "function", "function": {"name": "read_file"}}]
            def run(self, *args):
                return "body"
        class Client:
            available = True
            model = "fake"
            _stage_round_client = object()
            def _chat_round(self, *args, **kwargs):
                return SimpleNamespace(text="", reasoning="", reasoning_source="", finish_reason="stop", extra_keys=(),
                    input_tokens=7, output_tokens=3, total_tokens=10,
                    message=SimpleNamespace(tool_calls=[SimpleNamespace(id="submit", function=SimpleNamespace(
                        name="submit_stage", arguments=json.dumps({"delivery": "nothing applied"}))) ]))
        instance = []
        record_output, write_calls = mock.Mock(), mock.Mock()
        executor = stage_executor.StageExecutor(complete=mock.Mock(), record_output=record_output,
            write_tool_calls=write_calls, has_deliverable=lambda text: True, toolset_factory=Tools)
        request = stage_executor.StageExecutionRequest(stage="implement", label="stage3", artifact_stage="implementation",
            tool_calls_relpath="09_implementation/tools.yaml", system="s", user="u", fallback="", tools_enabled=True,
            native_tool_loop=True, submit_stage=lambda text: (False, "missing oracle"))
        with self.assertRaises(failures.StageDeliveryFailure) as caught:
            executor.execute({}, object(), request, client=Client())
        self.assertEqual(caught.exception.stage, "implement")
        self.assertEqual(len(caught.exception.tool_calls), 2)
        self.assertFalse(any(call["ok"] for call in caught.exception.tool_calls))
        record_output.assert_not_called()
        instance[0].rollback.assert_called_once()
        self.assertFalse(write_calls.call_args.args[-1][-1]["ok"])


class PartialReportTest(unittest.TestCase):
    def setUp(self):
        llm.reset_usage_log()
    def tearDown(self):
        llm.reset_usage_log()
        instrument.disable_progress()

    def test_node_exception_attaches_observed_state_without_wrapping_type(self):
        error = TimeoutError("provider stalled")
        def fake_node(state):
            raise error
        state = {"run_id": "temp", "gates": {"profile": {"ok": True}}, "errors": ["old error"]}
        with self.assertRaises(TimeoutError) as caught:
            instrument._wrap_node(fake_node)(state)
        self.assertIs(caught.exception, error)
        self.assertEqual(error.partial_state, state)
        self.assertEqual(error.failed_stage, "fake_node")

    def test_native_get_state_wins_over_initial_input_on_exception(self):
        error = TimeoutError("provider failed")
        error.partial_state = {"gates": {"old": {"ok": True}}}
        error.failed_stage = "implement"
        app = SimpleNamespace(checkpointer=object(), invoke=mock.Mock(side_effect=error),
            get_state=mock.Mock(return_value=SimpleNamespace(values={"run_id": "temp", "gates": {"profile": {"ok": True}}})))
        with self.assertRaises(TimeoutError) as caught:
            graph._invoke_agent_graph(app, {"status": "initialized"}, {"configurable": {"thread_id": "temp"}}, stream=False)
        self.assertIs(caught.exception, error)
        self.assertEqual(error.partial_state_source, "native_checkpoint")
        self.assertIn("profile", error.partial_state["gates"])
        app.get_state.assert_called_once()

    def test_real_native_checkpointer_recovers_last_established_node_state(self):
        from langgraph.checkpoint.memory import InMemorySaver

        def established(state):
            return {"gates": {"profile": {"ok": True}}, "errors": ["prior observed error"]}
        def abort(state):
            raise TimeoutError("fake SDK aborted")
        builder = graph.StateGraph(graph.CfuDesignState)
        builder.add_node("established", instrument._wrap_node(established))
        builder.add_node("abort", instrument._wrap_node(abort))
        builder.add_edge(graph.START, "established")
        builder.add_edge("established", "abort")
        builder.add_edge("abort", graph.END)
        app = builder.compile(checkpointer=InMemorySaver())
        with self.assertRaises(TimeoutError) as caught:
            graph._invoke_agent_graph(app, {"status": "initialized"},
                {"configurable": {"thread_id": "exclusive-native-recovery"}}, stream=False)
        recovered = caught.exception.partial_state
        self.assertEqual(caught.exception.partial_state_source, "native_checkpoint")
        self.assertEqual(caught.exception.failed_stage, "abort")
        self.assertEqual(recovered["gates"]["profile"], {"ok": True})
        self.assertEqual(recovered["errors"], ["prior observed error"])
        self.assertNotIn("sim", recovered["gates"])

    def test_cli_sdk_failure_persists_report_termination_trace_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "exclusive-partial"
            run_root = f"agents/generated/runs/{run_id}"
            fw = root / run_root
            put(fw / "run.yaml", "status: initialized\nartifacts: []\n")
            put_audit(fw / "request_audit.jsonl", audit_events(2, missing=True))
            def fail_run(state, **kwargs):
                llm.record_usage("implement", 101, 11, 112, 41)
                error = TimeoutError("SDK failed after one successful call")
                failures.attach_partial_state(error, {**state, "run_root": run_root, "errors": ["historical error"],
                    "gates": {"profile": {"ok": True}}}, source="node_entry", stage="implement")
                raise error
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(cli, "run_agent", side_effect=fail_run), \
                 mock.patch.object(cli, "export_graph_visualization", return_value={}), \
                 mock.patch.object(stage_services, "_policy", return_value=load_policy(REPO_ROOT)), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = cli.main(["--task", "offline", "--workdir", str(root), "--run-id", run_id])
            self.assertEqual(rc, 1)
            self.assertIn("status=failed", stdout.getvalue())
            self.assertIn("termination=INFRASTRUCTURE_EXCEPTION", stdout.getvalue())
            self.assertIn("TimeoutError", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            text = (fw / "report.md").read_text()
            parsed = parsing.parse_report(text, [])
            self.assertEqual(parsed["termination"]["exception_type"], "TimeoutError")
            self.assertEqual(parsed["gates"]["profile"], {"ok": True})
            self.assertIn("historical error", text)
            self.assertIn("input tokens: n/a", text)
            self.assertIn("Traceback", (fw / "12_review/exception.txt").read_text())
            manifest = yaml.safe_load((fw / "run.yaml").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["request_usage"]["calls"], 2)
            self.assertEqual(manifest["token_usage"]["total"], 112)
            self.assertTrue((fw / "12_review/termination.json").exists())

    def test_cli_debug_prints_original_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            error = ValueError("bad state")
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(cli, "run_agent", side_effect=error), \
                 mock.patch.object(cli, "export_graph_visualization", return_value={}), \
                 mock.patch.object(report, "write_partial_report", return_value={"status": "failed"}), \
                 mock.patch.dict("os.environ", {"CFU_AGENT_DEBUG": "1"}), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(cli.main(["--task", "offline", "--workdir", tmp]), 1)
            self.assertIn("Traceback", stderr.getvalue())
            self.assertIn("ValueError: bad state", stderr.getvalue())

    def test_report_uses_provided_usage_not_unrelated_process_counter(self):
        llm.record_usage("unrelated", 1000, 1000, 2000)
        text = report.render_report({"token_usage": {"calls": 1, "input": 2, "output": 3, "total": 5}})
        self.assertIn("- total tokens: 5", text)
        self.assertNotIn("- total tokens: 2000", text)


if __name__ == "__main__":
    unittest.main()
