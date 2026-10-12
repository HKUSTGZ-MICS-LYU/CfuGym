"""Isolated read-only observation-index tests; no compiler or real run needed."""

from __future__ import annotations

import unittest
from pathlib import Path

from agents.cfu_design_agent.ir.observation import build_observation_index, inspect_observation
from agents.cfu_design_agent.runtime.policy import PolicyError

RUN = "agents/generated/runs/evidence-test"
ASM = f"{RUN}/workspace/embench_crc32/reference/embench_main.asm"
IR = f"{RUN}/07_analysis/ir/crc_32.ll"
FEATURES = f"{RUN}/07_analysis/ir/crc_32.features.xml"


class ReadOnlyToolbox:
    def __init__(self, files=None, run_root=RUN):
        self.files = files or {}
        self.run_root = run_root
        self.stage = "analysis"
        self.workdir = Path("/tmp/workspace")
        self.reads = []

    def read_text(self, path, *, limit):
        self.reads.append((path, limit))
        return self.files.get(path, "")[:limit]


class ObservationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.point = {"id": "obs_loop_crc", "kind": "loop", "file": "crc_32.c",
                      "function": "crc_32", "line": 12, "ordinal": 1}
        self.kwargs = dict(run_root=RUN, benchmark="crc32",
                           observation={"points": [self.point], "applied": ["obs_loop_crc"]},
                           profiles={"profile": {"obs_loop_crc": {"cycles": 125, "calls": 4}}},
                           ir_result={"ir_files": [IR]})
        self.asm_text = ("00000000 <crc_32>:\n  addi a0,a0,1\n  ret\n"
                         "00000008 <other>:\n  nop\n"
                         "00000010 <crc_32.constprop.0>:\n  sub a0,a0,1\n  ret\n")
        self.ir_text = ("define i32 @other(i32 %x) !dbg !20 {\n"
                        "  %a = add i32 %x, 1, !dbg !11\n}\n"
                        "define i32 @crc_32(i32 %x) !dbg !20 {\n"
                        "  %one = add i32 %x, 1, !dbg !11\n"
                        "  br label %head, !dbg !14, !llvm.loop !31\n"
                        "head:\n"
                        "  %two = sub i32 %one, 2, !dbg !11\n"
                        "  ret i32 %two, !dbg !12\n}\n"
                        "!10 = !DILocation(line: 12, scope: !21)\n"
                        "!11 = !DILocation(line: 12, column: 3, scope: !21)\n"
                        "!12 = !DILocation(line: 13, scope: !21)\n"
                        "!14 = !DILocation(line: 14, scope: !21)\n"
                        "!20 = distinct !DISubprogram(name: \"crc_32\", file: !22)\n"
                        "!21 = !DILexicalBlock(scope: !20, file: !22)\n"
                        "!22 = !DIFile(filename: \"crc_32.c\", directory: \"/tmp\")\n"
                        "!31 = distinct !{!31, !11}\n")

    def index(self, **changes):
        return build_observation_index(**{**self.kwargs, **changes})

    def test_pure_index_applied_only_and_exact_stem(self):
        kwargs = {**self.kwargs, "observation": {"points": [self.point,
            {"id": "not_applied", "kind": "function", "file": "crc_32.c", "function": "other"}],
            "applied": ["obs_loop_crc"]}, "ir_result": {"ir_files": [
                f"{RUN}/07_analysis/ir/crc_32_extra.ll", IR.replace(RUN, "agents/generated/runs/foreign")]}}
        indexed = build_observation_index(**kwargs)
        self.assertEqual(list(indexed["observations"]), ["obs_loop_crc"])
        row = indexed["observations"]["obs_loop_crc"]
        self.assertEqual((row["cycles"], row["calls"], row["asm_path"], row["ir_path"]),
                         (125, 4, ASM, None))
        self.assertEqual(self.index()["observations"]["obs_loop_crc"]["ir_path"], IR)

    def test_features_view_targets_the_loop_and_does_not_read_other_artifacts(self):
        xml_text = (
            '<ir_features schema_version="1" provenance="static_ir">'
            '<function name="crc_32">'
            '<loop id="loop:crc_32:head" source_line="12" function="crc_32">'
            '<data_movement><totals bytes_total="unknown" trip_count="64"/></data_movement>'
            '</loop>'
            '<loop id="loop:crc_32:other" source_line="99" function="crc_32">'
            '<data_movement><totals bytes_total="unknown" trip_count="1"/></data_movement>'
            '</loop>'
            '</function></ir_features>'
        )
        index = self.index(ir_result={"ir_files": [IR], "feature_files": [FEATURES]})
        self.assertEqual(index["observations"]["obs_loop_crc"]["features_path"], FEATURES)
        toolbox = ReadOnlyToolbox({FEATURES: xml_text, IR: self.ir_text, ASM: self.asm_text})
        output = inspect_observation(index, toolbox, "obs_loop_crc", "features")
        self.assertIn("loop/function-focused excerpt (static_ir)", output)
        self.assertIn('trip_count="64"', output)
        self.assertNotIn('trip_count="1"', output)
        self.assertIn("target match unknown", output)
        self.assertEqual([p for p, _ in toolbox.reads], [FEATURES])
        summary = inspect_observation(index, toolbox, "obs_loop_crc")
        self.assertIn("Feature XML: " + FEATURES, summary)
        self.assertIn("view='features'", summary)

    def test_features_missing_unparseable_or_unmatched_do_not_fabricate(self):
        index = self.index(ir_result={"ir_files": [IR], "feature_files": [FEATURES]})
        output = inspect_observation(index, ReadOnlyToolbox(), "obs_loop_crc", "features")
        self.assertIn("Feature XML unavailable: missing or empty", output)
        bad = inspect_observation(index, ReadOnlyToolbox({FEATURES: "<ir_features>"}), "obs_loop_crc", "features")
        self.assertIn("unparseable feature XML", bad)
        other = ("<ir_features schema_version='1'><loop id='x' source_line='77' "
                 "function='crc_32'><a/></loop></ir_features>")
        fallback = inspect_observation(index, ReadOnlyToolbox({FEATURES: other}), "obs_loop_crc", "features")
        self.assertIn("whole document (no loop matched this observation)", fallback)
        no_func_attr = ("<ir_features schema_version='1'><loop id='y' source_line='12'>"
                        "<data_movement/></loop><loop id='z' source_line='55'>"
                        "<data_movement/></loop></ir_features>")
        line_match = inspect_observation(index, ReadOnlyToolbox({FEATURES: no_func_attr}),
                                         "obs_loop_crc", "features")
        self.assertIn("loop/function-focused excerpt (static_ir)", line_match)
        self.assertIn('id="y"', line_match)
        self.assertNotIn('id="z"', line_match)
        unrecorded = inspect_observation(self.index(), ReadOnlyToolbox(), "obs_loop_crc", "features")
        self.assertIn("Feature XML unavailable: not recorded", unrecorded)

    def test_a_missing_reference_disassembly_explains_itself(self):
        # Observed 2026-09-30: a cached profile run has no build output, so the
        # view said "missing or empty" with no hint that the IR view carries the
        # evidence. That reads as "this evidence does not exist at all".
        toolbox = ReadOnlyToolbox({IR: self.ir_text})
        asm_view = inspect_observation(self.index(), toolbox, "obs_loop_crc", "asm")
        self.assertIn("no reference disassembly in this run's workspace", asm_view)
        self.assertIn("use view='ir'", asm_view)

    def test_the_exported_ir_and_feature_files_are_reachable(self):
        # `collect_ir_evidence` writes 07_analysis/ir/<stem>.ll and
        # <stem>.features.xml; the analysis index must find exactly those.
        features = ('<function name="crc_32" kind="loop" hotspot_match="true">'
                    '<summary loop_carried_edges="1" /></function>')
        toolbox = ReadOnlyToolbox({IR: self.ir_text, FEATURES: features})
        index = self.index(ir_result={"ir_files": [IR], "feature_files": [FEATURES],
                                      "status": "approximate", "mode": "auto"})
        summary = inspect_observation(index, toolbox, "obs_loop_crc")
        self.assertIn("LLVM IR candidate: " + IR, summary)
        self.assertIn("Feature XML: " + FEATURES, summary)
        ir_view = inspect_observation(index, toolbox, "obs_loop_crc", "ir")
        self.assertIn("candidate instruction", ir_view)
        features_view = inspect_observation(index, toolbox, "obs_loop_crc", "features")
        self.assertIn("loop_carried_edges", features_view)

    def test_summary_and_missing_cache_asm_do_not_fabricate(self):
        toolbox = ReadOnlyToolbox({IR: self.ir_text})
        summary = inspect_observation(self.index(), toolbox, "obs_loop_crc")
        self.assertIn("cycles=125 calls=4", summary)
        self.assertIn("ASM reference candidate: " + ASM + " — no reference disassembly", summary)
        self.assertIn("Hot-spot ASM instruction counts: no reference disassembly", summary)
        self.assertIn("LLVM IR candidate: " + IR + " — not inspected in this view", summary)
        self.assertEqual([p for p, _ in toolbox.reads], [ASM])
        toolbox.reads.clear()
        ir_view = inspect_observation(self.index(), toolbox, "obs_loop_crc", "ir")
        self.assertIn("2 candidate instruction", ir_view)
        self.assertEqual([p for p, _ in toolbox.reads], [IR])

    def test_summary_counts_hotspot_riscv_mnemonics_per_function_candidate(self):
        asm = (
            "00000000 <crc_32>:\n"
            "  00000000: 00550513 addi a0,a0,5\n"
            "  00000004: 00052583 lw a1,0(a0)\n"
            "  00000008: 00b52023 sw a1,0(a0)\n"
            "  0000000c: fea594e3 bne a1,a0,0x0\n"
            "  00000010: 00008067 ret\n"
            "00000014 <crc_32.constprop.0>:\n"
            "  00150513 addi a0,a0,1\n"
            "  c.addi a0,1\n"
            "0000001c <other>:\n"
            "  add a0,a1,a2\n"
        )
        toolbox = ReadOnlyToolbox({ASM: asm})
        summary = inspect_observation(self.index(), toolbox, "obs_loop_crc")
        self.assertIn("crc_32: total=5 (addi=1, bne=1, lw=1, ret=1, sw=1)", summary)
        self.assertIn("crc_32.constprop.0: total=2 (addi=1, c.addi=1)", summary)
        self.assertNotIn("other:", summary)
        self.assertIn("not exact loop addresses", summary)
        self.assertEqual([p for p, _ in toolbox.reads], [ASM])

    def test_asm_clone_ambiguity_no_loop_address_and_physical_lines(self):
        output = inspect_observation(self.index(), ReadOnlyToolbox({ASM: self.asm_text}),
                                     "obs_loop_crc", "asm")
        self.assertIn("ambiguous clones", output)
        self.assertIn(f"{ASM}:1 <crc_32>:", output)
        self.assertIn(f"{ASM}:6 <crc_32.constprop.0>:", output)
        self.assertNotIn("<other>", output)
        self.assertIn("no source-line link", output)
        self.assertIn("not measured loop addresses", output)

    def test_ir_metadata_only_instruction_refs_in_matching_define(self):
        output = inspect_observation(self.index(), ReadOnlyToolbox({IR: self.ir_text}),
                                     "obs_loop_crc", "ir")
        self.assertIn(f"{IR}:5   %one", output)
        self.assertIn(f"{IR}:8   %two", output)
        self.assertNotIn(f"{IR}:2 ", output)
        self.assertNotIn(f"{IR}:6 ", output)
        self.assertNotIn(f"{IR}:9 ", output)
        self.assertIn("may be discontiguous", output)
        self.assertIn("header-line candidate only, not loop-body", output)

    def test_asm_excerpt_budget_and_approx_target_provenance(self):
        many = "00000000 <crc_32>:\n" + "  addi a0,a0,1\n" * 35 + "00000080 <other>:\n  ret\n"
        toolbox = ReadOnlyToolbox({ASM: many, IR: self.ir_text})
        asm_output = inspect_observation(self.index(), toolbox, "obs_loop_crc", "asm", max_lines=28)
        self.assertIn(f"{ASM}:20   addi", asm_output)
        self.assertNotIn(f"{ASM}:38", asm_output)
        self.assertEqual([p for p, _ in toolbox.reads], [ASM])
        approximate = self.index(ir_result={"ir_files": [IR], "toolchain": {"target_match": False}})
        ir_output = inspect_observation(approximate, toolbox, "obs_loop_crc", "ir")
        self.assertIn("approximate target; not faithful GCC machine code", ir_output)
        self.assertEqual([p for p, _ in toolbox.reads], [ASM, IR])

    def test_ir_run_scoped_metadata_filename_and_same_basename_spoof(self):
        expected = f"{RUN}/workspace/embench_crc32/reference/src/crc_32.c"
        scoped = self.ir_text.replace('filename: "crc_32.c"', f'filename: "{expected}"')
        matching = inspect_observation(self.index(), ReadOnlyToolbox({IR: scoped}), "obs_loop_crc", "ir")
        self.assertIn("2 candidate instruction line", matching)
        absolute = self.ir_text.replace('filename: "crc_32.c"',
                                         f'filename: "/tmp/workspace/{expected}"')
        with_abspath = ReadOnlyToolbox({IR: absolute})
        with_abspath.workdir = Path("/tmp/workspace")
        self.assertIn("2 candidate instruction line", inspect_observation(
            self.index(), with_abspath, "obs_loop_crc", "ir"))
        unrelated = self.ir_text.replace('filename: "crc_32.c"',
                                         'filename: "agents/generated/runs/foreign/workspace/embench_crc32/reference/src/crc_32.c"')
        mismatching = inspect_observation(self.index(), ReadOnlyToolbox({IR: unrelated}), "obs_loop_crc", "ir")
        self.assertIn("unmapped (no !dbg instruction", mismatching)

    def test_ir_absent_line_mismatch_and_foreign_file(self):
        without_line = self.index(observation={"points": [{**self.point, "line": None}],
                                                   "applied": ["obs_loop_crc"]})
        self.assertEqual(without_line["observations"], {})
        no_line = self.index(observation={"points": [{k: v for k, v in self.point.items() if k != "line"}],
                                               "applied": ["obs_loop_crc"]})
        output = inspect_observation(no_line, ReadOnlyToolbox({IR: self.ir_text}), "obs_loop_crc", "ir")
        self.assertIn("unmapped (no source line", output)
        no_ir = self.index(ir_result={})
        output = inspect_observation(no_ir, ReadOnlyToolbox({ASM: self.asm_text}), "obs_loop_crc", "ir")
        self.assertIn("LLVM IR candidate: not recorded", output)
        self.assertNotIn("candidate !dbg", output)
        foreign = self.ir_text.replace('filename: "crc_32.c"', 'filename: "other.c"')
        output = inspect_observation(self.index(), ReadOnlyToolbox({IR: foreign}), "obs_loop_crc", "ir")
        self.assertIn("unmapped (no !dbg instruction", output)

    def test_malformed_unknown_or_unsafe_paths_refused(self):
        for root, name in (("../escape", "crc32"), (RUN, "../evil"), (RUN + "/extra", "crc32")):
            with self.subTest(root=root, name=name), self.assertRaises(ValueError):
                self.index(run_root=root, benchmark=name)
        for filename in ("../../outside.c", "/tmp/out.c", "foo/bar.c", "fake.cpp"):
            with self.subTest(filename=filename):
                result = self.index(observation={"points": [{**self.point, "file": filename}],
                                                 "applied": ["obs_loop_crc"]})
                self.assertEqual(result["observations"], {})
        toolbox = ReadOnlyToolbox()
        with self.assertRaisesRegex(ValueError, "unknown observation ID"):
            inspect_observation(self.index(), toolbox, "unrequested")
        self.assertEqual(toolbox.reads, [])
        index = self.index()
        index["observations"]["obs_loop_crc"]["asm_path"] = f"{RUN}/workspace/embench_crc32/profile/embench_main.asm"
        with self.assertRaisesRegex(ValueError, "artifact path"):
            inspect_observation(index, toolbox, "obs_loop_crc")
        self.assertEqual(toolbox.reads, [])
        with self.assertRaisesRegex(ValueError, "another run"):
            inspect_observation(self.index(), ReadOnlyToolbox(run_root="agents/generated/runs/foreign"), "obs_loop_crc")
        with self.assertRaisesRegex(ValueError, "another run"):
            inspect_observation(self.index(), ReadOnlyToolbox(run_root=""), "obs_loop_crc")
        wrong_stage = ReadOnlyToolbox()
        wrong_stage.stage = "profile"
        with self.assertRaisesRegex(ValueError, "analysis toolbox"):
            inspect_observation(self.index(), wrong_stage, "obs_loop_crc")
        self.assertEqual(wrong_stage.reads, [])

    def test_policy_denial_propagates(self):
        class Denied(ReadOnlyToolbox):
            def read_text(self, path, *, limit):
                raise PolicyError("analysis read forbidden")

        for view in ("asm", "summary", "ir"):
            with self.subTest(view=view), self.assertRaisesRegex(PolicyError, "read forbidden"):
                inspect_observation(self.index(), Denied(), "obs_loop_crc", view)
        self.assertEqual(inspect_observation(self.index(), Denied(), "obs_loop_crc", "features").splitlines()[0],
                         "Observation obs_loop_crc: loop crc_32.c:12 function=crc_32 ordinal=1")

    def test_ir_views_and_lines_removed_under_ablation(self):
        # --no-llvm-ir (allow_ir=False): the ir/features views are gone and the
        # summary carries no IR content; asm/summary evidence stays.
        index = self.index(ir_result={"ir_files": [IR], "feature_files": [FEATURES]})
        toolbox = ReadOnlyToolbox({FEATURES: "<ir_features>", IR: self.ir_text, ASM: self.asm_text})
        for view in ("ir", "features"):
            output = inspect_observation(index, toolbox, "obs_loop_crc", view, allow_ir=False)
            self.assertIn("IR views are disabled", output)
        self.assertEqual(toolbox.reads, [])
        summary = inspect_observation(index, toolbox, "obs_loop_crc", allow_ir=False)
        self.assertNotIn("LLVM IR", summary)
        self.assertNotIn("Feature XML", summary)
        self.assertNotIn("not recorded", summary)
        self.assertIn("ASM reference candidate", summary)
        asm = inspect_observation(index, toolbox, "obs_loop_crc", "asm", allow_ir=False)
        self.assertIn("ASM symbols", asm)

    def test_no_cycle_data_duplicate_ids_and_limits(self):
        blank = self.index(profiles={"profile": {"obs_loop_crc": {"cycles": -1, "calls": "12"}}})
        result = inspect_observation(blank, ReadOnlyToolbox(), "obs_loop_crc")
        self.assertIn("cycles=unavailable calls=unavailable", result)
        doubled = self.index(observation={"points": [self.point, self.point], "applied": ["obs_loop_crc"]})
        self.assertEqual(doubled["observations"], {})
        long_asm = "00000000 <crc_32>:\n" + "  add a0,a0,a0\n" * 50000
        output = inspect_observation(self.index(), ReadOnlyToolbox({ASM: long_asm}),
                                     "obs_loop_crc", "asm", max_lines=3)
        self.assertLessEqual(len(output.splitlines()), 8)
        self.assertLessEqual(len(output), 12000)
        self.assertIn("truncated", output)
        self.assertLessEqual(len(inspect_observation(self.index(), ReadOnlyToolbox({IR: self.ir_text}),
                                                      "obs_loop_crc", "ir", max_lines=1000).splitlines()), 100)


if __name__ == "__main__":
    unittest.main()
