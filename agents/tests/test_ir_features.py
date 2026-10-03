"""Feature-extractor tests: opt analysis, typed ops, dependencies, XML output."""

from __future__ import annotations

import shutil
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

from agents.cfu_design_agent.ir.features import (
    extract_ir_features, render_features_xml, summarize_features,
)

# Nested loops mirroring md5 shape: constant inner trip, SCEV addrecs,
# xor reduction, global constant table, parameter stream, debug locations.
NESTED = '''; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@tab = internal global [64 x i32] zeroinitializer, align 4

declare void @rand_beebs(i32)

define i32 @crc_32(i32* %arg, i32 %n) !dbg !20 {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ], !dbg !30
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %outer.latch ], !dbg !31
  %p = getelementptr inbounds i32, i32* %arg, i32 %i, !dbg !32
  %v = load i32, i32* %p, align 4, !dbg !33
  call void @rand_beebs(i32 %n)
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ], !dbg !40
  %x = phi i32 [ %acc, %outer ], [ %x.next, %inner ], !dbg !41
  %q = getelementptr inbounds [64 x i32], [64 x i32]* @tab, i32 0, i32 %j, !dbg !42
  %w = load i32, i32* %q, align 4, !dbg !43
  %x.next = xor i32 %x, %w, !dbg !44
  %j.next = add nuw nsw i32 %j, 1, !dbg !45
  %done = icmp eq i32 %j.next, 64, !dbg !46
  br i1 %done, label %outer.latch, label %inner, !dbg !47

outer.latch:
  %acc.next = add i32 %acc, %x.next, !dbg !50
  %i.next = add i32 %i, 1, !dbg !51
  %odone = icmp eq i32 %i.next, %n, !dbg !52
  br i1 %odone, label %exit, label %outer, !dbg !53

exit:
  ret i32 %acc.next
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!20 = distinct !DISubprogram(name: "crc_32", scope: !1, file: !1, line: 10, type: !21, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DISubroutineType(types: !22)
!22 = !{}
!30 = !DILocation(line: 20, column: 3, scope: !20)
!31 = !DILocation(line: 20, column: 3, scope: !20)
!32 = !DILocation(line: 21, column: 5, scope: !20)
!33 = !DILocation(line: 21, column: 11, scope: !20)
!40 = !DILocation(line: 25, column: 5, scope: !20)
!41 = !DILocation(line: 25, column: 5, scope: !20)
!42 = !DILocation(line: 26, column: 9, scope: !20)
!43 = !DILocation(line: 26, column: 15, scope: !20)
!44 = !DILocation(line: 27, column: 9, scope: !20)
!45 = !DILocation(line: 28, column: 9, scope: !20)
!46 = !DILocation(line: 29, column: 9, scope: !20)
!47 = !DILocation(line: 29, column: 9, scope: !20)
!50 = !DILocation(line: 32, column: 3, scope: !20)
!51 = !DILocation(line: 33, column: 3, scope: !20)
!52 = !DILocation(line: 34, column: 3, scope: !20)
!53 = !DILocation(line: 34, column: 3, scope: !20)
'''

SINGLE_BLOCK = '''define i32 @plain(i32 %x) {
entry:
  %y = add i32 %x, 1
  %c = icmp sgt i32 %y, 3
  %s = select i1 %c, i32 %y, i32 7
  ret i32 %s
}
'''

QUOTED = '''define i32 @"quoted fn"(i32 %"a b") {
"odd label":
  %"r 1" = add i32 %"a b", 1
  ret i32 %"r 1"
}
'''

INVOKE = '''declare i32 @callee()
define i32 @with_invoke() personality i32 0 {
entry:
  %v = invoke i32 @callee() to label %ok unwind label %bad
ok:
  ret i32 %v
bad:
  ret i32 0
}
'''

VECTOR = '''define void @vec(<4 x i8>* %p) {
entry:
  %a = load <4 x i8>, <4 x i8>* %p, align 4
  %b = add <4 x i8> %a, <i8 1, i8 1, i8 1, i8 1>
  %c = icmp slt <4 x i8> %b, %a
  store <4 x i8> %b, <4 x i8>* %p, align 4
  ret void
}
'''


class ExtractorOptTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("opt"), "opt unavailable")
    def test_nested_loops_trip_counts_recurrences_and_hotspot(self) -> None:
        features, audit = extract_ir_features(
            NESTED, source_file="t.c", function="crc_32", hotspot_line=25)
        self.assertEqual(features["analysis_source"], "llvm14_opt+text")
        loops = [f for f in features["functions"] if f["kind"] == "loop"]
        self.assertEqual(len(loops), 2)
        outer = next(f for f in loops if f["depth"] == 1)
        inner = next(f for f in loops if f["depth"] == 2)
        self.assertEqual(outer["trip_count"], "unknown")
        self.assertNotEqual(outer["trip_count_expr"], "unknown")
        self.assertEqual(inner["trip_count"], 64)
        self.assertEqual(inner["trip_multiple"], 64)
        self.assertEqual(inner["source_line"], 25)
        self.assertEqual(inner["hotspot_match"], "true")
        self.assertEqual(outer["hotspot_match"], "false")
        recurrences = {r["phi"]: r["update_class"] for r in inner["dependencies"]["recurrences"]}
        self.assertIn("xor_reduction", recurrences.values())
        self.assertIn("induction", recurrences.values())
        self.assertTrue(any(e["kind"] == "loop_carried"
                            for e in inner["dependencies"]["edges"]))
        self.assertGreater(inner["dependencies"]["summary"]["static_depth"], 1)

    @unittest.skipUnless(shutil.which("opt"), "opt unavailable")
    def test_data_movement_strides_and_upper_bound(self) -> None:
        features, _ = extract_ir_features(NESTED, source_file="t.c", function="crc_32")
        loops = [f for f in features["functions"] if f["kind"] == "loop"]
        inner = next(f for f in loops if f["depth"] == 2)
        accesses = inner["data_movement"]["accesses"]
        tab = next(a for a in accesses if a["base"] == "@tab")
        self.assertEqual(tab["base_kind"], "global")
        self.assertEqual(tab["access"], "load")
        self.assertEqual(tab["element_bits"], 32)
        self.assertIn(tab["stride_confidence"], ("scev_constant_step", "scev_pointer_step",
                                                 "induction_phi"))
        self.assertEqual(tab["stride_bytes"], 4)
        totals = inner["data_movement"]["totals"]
        self.assertEqual(totals["trip_count"], 64)
        self.assertEqual(totals["bytes_touched_static"], 4)  # one i32 load
        self.assertEqual(totals["bytes_total"], "unknown")
        self.assertEqual(totals["bytes_upper_bound"], 4 * 64)
        self.assertEqual(totals["estimate_kind"], "static_upper_bound")
        outer = next(f for f in loops if f["depth"] == 1)
        outer_accesses = outer["data_movement"]["accesses"]
        stream = next(a for a in outer_accesses if a["base"] == "%arg")
        self.assertEqual(stream["base_kind"], "argument")
        self.assertEqual(outer["data_movement"]["totals"]["bytes_upper_bound"], "unknown")

    @unittest.skipUnless(shutil.which("opt"), "opt unavailable")
    def test_operations_calls_and_dependencies(self) -> None:
        features, _ = extract_ir_features(NESTED, source_file="t.c", function="crc_32")
        loops = [f for f in features["functions"] if f["kind"] == "loop"]
        outer = next(f for f in loops if f["depth"] == 1)
        ops = {(o["class"], o["type"]): o for o in outer["operations"]["ops"]}
        add = ops.get(("integer_add", "i32"))
        self.assertIsNotNone(add)
        self.assertEqual(add["bits"], 32)
        self.assertEqual(add["lanes"], 1)
        icmp = ops.get(("compare_icmp", "i32"))
        self.assertIsNotNone(icmp)
        self.assertIn(icmp.get("predicate", ""), ("eq", "slt", "sgt"))
        calls = {c["callee"]: c for c in outer["operations"]["calls"]}
        self.assertEqual(calls["rand_beebs"]["count"], 1)
        self.assertFalse(calls["rand_beebs"]["intrinsic"])
        inner = next(f for f in loops if f["depth"] == 2)
        summary = inner["dependencies"]["summary"]
        self.assertGreaterEqual(summary["ssa_edges"], 4)
        self.assertGreaterEqual(summary["loop_carried_edges"], 2)
        self.assertFalse(summary["unknown_base_objects"])
        self.assertTrue(summary["static_depth_not_cycles"])

    @unittest.skipUnless(shutil.which("opt"), "opt unavailable")
    def test_vector_types_predicates_and_constant_operands(self) -> None:
        features, _ = extract_ir_features(VECTOR, source_file="vec.c", function="vec")
        func = features["functions"][0]
        ops = {o["class"]: o for o in func["operations"]["ops"]}
        self.assertEqual(ops["load"]["lanes"], 4)
        self.assertEqual(ops["load"]["bits"], 8)
        self.assertEqual(ops["integer_add"]["lanes"], 4)
        self.assertEqual(ops["compare_icmp"]["predicate"], "slt")
        self.assertGreaterEqual(ops["integer_add"]["constant_operands"], 1)

    def test_opt_failure_falls_back_to_text_only(self) -> None:
        def failing(argv, **kwargs):
            return {"argv": argv, "returncode": 127, "stderr": "boom", "stdout": b""}
        features, audit = extract_ir_features(
            NESTED, source_file="t.c", function="crc_32", runner=failing)
        self.assertEqual(features["analysis_source"], "text_only")
        self.assertEqual(audit["analysis_source"], "text_only")
        loops = [f for f in features["functions"] if f["kind"] == "loop"]
        self.assertGreaterEqual(len(loops), 2)
        self.assertEqual(features["functions"][0]["trip_count"], "unknown")

    def test_text_only_uses_no_opt_at_all(self) -> None:
        def unexpected(argv, **kwargs):
            raise AssertionError(f"executed {argv}")
        features, _ = extract_ir_features(
            NESTED, source_file="t.c", function="crc_32", opt_binary="", runner=unexpected)
        self.assertEqual(features["analysis_source"], "text_only")


class RobustnessTest(unittest.TestCase):
    def test_single_block_function_scope_features(self) -> None:
        features, _ = extract_ir_features(
            SINGLE_BLOCK, source_file="p.c", opt_binary="")
        func = features["functions"][0]
        self.assertEqual(func["kind"], "function_scope")
        self.assertEqual(func["trip_count"], "unknown")
        self.assertIn("not_computable", func["data_movement"]["totals"]["estimate_kind"])

    def test_quoted_names_parse(self) -> None:
        features, _ = extract_ir_features(QUOTED, source_file="q.c", opt_binary="")
        self.assertEqual(features["functions"][0]["name"], "quoted fn")
        ops = {o["class"]: o for o in features["functions"][0]["operations"]["ops"]}
        self.assertIn("integer_add", ops)

    def test_invoke_recorded_as_unsupported_without_crash(self) -> None:
        features, _ = extract_ir_features(INVOKE, source_file="i.c", opt_binary="")
        # Text fallback still reports the function; constructs stay documented.
        self.assertTrue(features["functions"])

    def test_garbage_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_ir_features("not ir at all", opt_binary="")


class RenderTest(unittest.TestCase):
    def _features(self):
        features, _ = extract_ir_features(
            NESTED, source_file="t.c", function="crc_32", opt_binary="")
        return features

    def test_xml_parses_and_is_deterministic(self) -> None:
        features = self._features()
        xml_text = render_features_xml(features)
        self.assertEqual(xml_text, render_features_xml(features))
        root = ET.fromstring(xml_text)
        self.assertEqual(root.tag, "ir_features")
        self.assertEqual(root.get("provenance"), "static_ir")
        self.assertIn(root.get("analysis_source"), ("text_only", "llvm14_opt+text"))
        loop_nodes = [f for f in root.iter("function") if f.get("kind") == "loop"]
        self.assertTrue(loop_nodes)
        for node in loop_nodes:
            self.assertTrue(node.get("id", "").startswith("loop:"))
            self.assertIn(node.get("source_link"), ("debug_location", "unmapped"))

    def test_unknown_never_zero_for_unproven_totals(self) -> None:
        features = self._features()
        root = ET.fromstring(render_features_xml(features))
        for totals in root.iter("totals"):
            self.assertEqual(totals.get("bytes_total"), "unknown")
            if totals.get("trip_count") == "unknown":
                self.assertEqual(totals.get("bytes_upper_bound"), "unknown")

    def test_caps_enforced_and_marked(self) -> None:
        features = self._features()
        for func in features["functions"]:
            func["dependencies"]["edges"] = (
                [{"from": f"%{i}", "to": f"%{i + 1}", "kind": "ssa"}
                 for i in range(500)])
        xml_text = render_features_xml(features)
        self.assertLessEqual(len(xml_text.encode()), 200_000)
        root = ET.fromstring(xml_text)
        self.assertEqual(root.get("truncated"), "true")
        for func in root.iter("function"):
            self.assertLessEqual(len(list(func.iter("edge"))), 240)

    def test_over_budget_output_stays_well_formed(self) -> None:
        # Shrink the byte budget instead of rendering hundreds of clones: the
        # whole-element drop loop is the contract under test, not its scale.
        import agents.cfu_design_agent.ir.features as features_module

        features = self._features()
        template = features["functions"][0]
        features["functions"] = [
            {**template, "id": f"loop:t:{index}", "name": f"f{index}",
             "dependencies": {**template["dependencies"],
                              "edges": template["dependencies"]["edges"]}}
            for index in range(40)
        ]
        budget = 8_000
        with mock.patch.object(features_module, "MAX_OUTPUT_BYTES", budget):
            xml_text = render_features_xml(features)
        self.assertLessEqual(len(xml_text.encode()), budget)
        root = ET.fromstring(xml_text)
        self.assertEqual(root.get("truncated"), "true")
        self.assertTrue(list(root.iter("function")))

    def test_summary_is_bounded_and_caveated(self) -> None:
        text = summarize_features(self._features())
        self.assertLessEqual(len(text.splitlines()), 40)
        self.assertIn("static", text)
        self.assertIn("bytes_total=unknown", text)
        self.assertIn("not measured cycles", text.lower())


if __name__ == "__main__":
    unittest.main()
