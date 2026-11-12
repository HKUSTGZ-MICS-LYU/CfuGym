"""Fast, tempfile-only Embench measurement and structural integrity contracts."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from agents.cfu_design_agent.benchmark import cfu_execution, embench, embench_integrity as integrity, embench_patchers
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT


SOURCE = '''#include "support.h"
#define LOCAL_SCALE_FACTOR 17
#define SIZE 4
static const unsigned input[SIZE] = {1, 2, 3, 4};
static unsigned output[SIZE];
typedef unsigned word;
struct Pair { word left; word right; };
static int benchmark_body(unsigned lsf, unsigned gsf);
static int calculate(int n) { return n + 1; }
void initialise_benchmark(void) { output[0] = 0; }
void warm_caches(int heat) { benchmark_body(1, heat); }
int benchmark(void) { return benchmark_body(LOCAL_SCALE_FACTOR, GLOBAL_SCALE_FACTOR); }
static int benchmark_body(unsigned lsf, unsigned gsf) {
    const char *message = "literal { } #define SIZE 1";
    (void)message;
    for (unsigned i = 0; i < lsf * gsf; ++i) output[0] = calculate(input[0]);
    return output[0];
}
int verify_benchmark(int result) { return result == 2 && output[0] == 2; }
'''


def replace_body(source: str, name: str, body: str) -> str:
    parsed = integrity._parse(source)
    obj = next(o for o in parsed.objects if o.name == name)
    brace = next(t for t in parsed.tokens if obj.start <= t.start < obj.end and t.value == "{")
    return source[:brace.start] + "{ " + body + " }" + source[obj.end:]


class IntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workdir = Path(cls._temporary.name)
        cls.run_root = "agents/generated/runs/integrity-test"
        cls.name = "fixture"
        source = cls.workdir / embench.benchmark_dir(cls.name) / "kernel.c"
        source.parent.mkdir(parents=True)
        source.write_text(SOURCE)
        support = cls.workdir / embench.EMBENCH_ROOT / "support"
        support.mkdir(parents=True)
        for fname in embench.SUPPORT_FILES:
            (support / fname).write_text((REPO_ROOT / embench.EMBENCH_ROOT / "support" / fname).read_text())
        board = cls.workdir / embench.BOARD_ROOT
        board.mkdir(parents=True)
        for fname in embench.BOARD_FILES:
            (board / fname).write_text((REPO_ROOT / embench.BOARD_ROOT / fname).read_text())
        cls.project = embench.prepare_embench_benchmark(
            RepoToolbox(cls.workdir, policy=load_policy(REPO_ROOT), stage="prepare_benchmark", run_root=cls.run_root),
            {"benchmark_name": cls.name})
        cls.metadata = cls.workdir / cls.run_root / "04_benchmark/embench.yaml"
        cls.metadata.parent.mkdir(parents=True)
        cls.metadata_text = yaml.safe_dump(cls.project)
        cls.metadata.write_text(cls.metadata_text)
        cls.target = f"{embench.config_workspace(cls.run_root, cls.name, 'cfu')}/src/kernel.c"
        cls.original = (cls.workdir / cls.target).read_text()

    def check(self, text: str | None = None, extra: dict[str, str] | None = None) -> list[str]:
        pending = dict(extra or {})
        if text is not None:
            pending[self.target] = text
        return integrity.validate_candidate(self.workdir, self.run_root, self.name, pending or None)

    def test_unchanged_and_pending_computational_bodies_allowed(self) -> None:
        self.assertEqual(self.check(), [])
        text = replace_body(self.original, "benchmark_body", "return calculate(input[0]);")
        text = replace_body(text, "calculate", "return n + 2;")
        self.assertEqual(self.check(text), [])
        self.assertEqual((self.workdir / self.target).read_text(), self.original)

    def test_comments_whitespace_and_literals_do_not_confuse_scanner(self) -> None:
        text = self.original.replace("static const unsigned input", "/* #define SIZE 0 */\nstatic\nconst unsigned input")
        text = replace_body(text, "calculate", 'const char *s = "} #define benchmark(x) 0 {"; return s[0] + n;')
        self.assertEqual(self.check(text), [])
        text = self.original.replace("{1, 2, 3, 4}", "{1, 2, 3, 5}")
        self.assertTrue(self.check(text))

    def test_fresh_helpers_scratch_macros_and_intrinsics_allowed(self) -> None:
        text = '#include <stdint.h>\n#include "cfu_intrinsics.h"\n#define CFU_UNROLL 4\n' + self.original
        text += '\nstatic word cfu_scratch[4];\nstatic struct Pair cfu_pair;\nstatic int cfu_helper(int value) { return value * CFU_UNROLL; }\n'
        header = '#ifndef CFU_INTRINSICS_H\n#define CFU_INTRINSICS_H\n#include <stdint.h>\n#define CFU_OP 1\nstatic inline uint32_t cfu_compute(uint32_t a) { return a; }\n#endif\n'
        header_rel = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/include/cfu_intrinsics.h"
        self.assertEqual(self.check(text, {header_rel: header}), [])

    def test_new_host_entry_and_timing_authority_references_rejected(self) -> None:
        bodies = [
            "extern void start_trigger(void); start_trigger(); return 1;",
            "extern void stop_trigger(void); void (*finish)(void) = stop_trigger; finish(); return 1;",
            "(void)verify_benchmark(0); return 1;",
            "(void)benchmark(); return 1;",
            "initialise_benchmark(); return 1;",
        ]
        for body in bodies:
            with self.subTest(body=body):
                changed = replace_body(self.original, "calculate", body)
                self.assertTrue(any("authority" in error for error in self.check(changed)))
        alias = self.original + "\n#define AGENT_TRIGGER start_trigger\n"
        self.assertTrue(any("authority" in error for error in self.check(alias)))
        header = self.target.rsplit("/src/", 1)[0] + "/include/cfu_intrinsics.h"
        text = "static inline int innocent(void) { extern void start_trigger(void); start_trigger(); return 0; }"
        self.assertTrue(any("authority" in error for error in self.check(extra={header: text})))

    def test_every_entry_body_immutable(self) -> None:
        for name in integrity.ENTRY_FUNCTIONS:
            with self.subTest(name=name):
                self.assertTrue(self.check(replace_body(self.original, name, "return 1;")))

    def test_declarations_initializers_types_and_interfaces_immutable(self) -> None:
        substitutions = [
            ("input[SIZE] = {1, 2, 3, 4}", "input[SIZE] = {4, 3, 2, 1}"),
            ("output[SIZE]", "output[1]"),
            ("typedef unsigned word", "typedef int word"),
            ("word left; word right;", "word left; unsigned short right;"),
            ("calculate(int n)", "calculate(unsigned n)"),
            ("#define SIZE 4", "#define SIZE 1"),
            ("#define LOCAL_SCALE_FACTOR 1", "#define LOCAL_SCALE_FACTOR 0"),
            ("benchmark_body(unsigned lsf, unsigned gsf)", "benchmark_body(unsigned lsf, unsigned gsf, int fake)"),
        ]
        for before, after in substitutions:
            with self.subTest(before=before):
                self.assertIn(before, self.original)
                self.assertTrue(self.check(self.original.replace(before, after)))

    def test_missing_and_duplicate_objects_and_functions_rejected(self) -> None:
        changes = [
            self.original.replace("static unsigned output[SIZE];", ""),
            self.original + "\nstatic unsigned output[SIZE];\n",
            self.original + "\nint benchmark(void) { return 1; }\n",
            self.original.replace("int verify_benchmark", "int verify_replaced"),
            self.original.replace("#define SIZE 4", "#define SIZE 4\n#define SIZE 4"),
        ]
        for text in changes:
            with self.subTest(text=text[-70:]):
                self.assertTrue(self.check(text))

    def test_macro_shadowing_variants_rejected(self) -> None:
        directives = [
            "#define benchmark() 1", "#define verify_benchmark(x) 1",
            "#define input cfu_fake", "#undef SIZE",
            "#define GLOBAL_SCALE_FACTOR 0", "#define unsigned signed",
            "#define bench\\\nmark() 1", "%:define benchmark() 1",
            "#/**/define /**/ benchmark /**/() 1", "#define calculate(x) 0",
        ]
        for directive in directives:
            with self.subTest(directive=directive):
                self.assertTrue(self.check(self.original + "\n" + directive + "\n"))

    def test_host_preprocessing_flags_reserved_in_source_and_intrinsics(self) -> None:
        rel = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/include/cfu_intrinsics.h"
        for name in ("USE_AGENT_CFU", "EMBENCH_OBS", "EMBENCH_NAME", "HAVE_BOARDSUPPORT_H"):
            for command in (f"#define {name} 0", f"#undef {name}"):
                with self.subTest(name=name, command=command):
                    self.assertTrue(self.check(self.original + "\n" + command + "\n"))
                    self.assertTrue(self.check(extra={rel: command + "\n"}))

    def test_missing_or_legacy_metadata_fails_closed_and_requires_new_run(self) -> None:
        try:
            cases = (
                ({}, "start a new run"),
                ({**self.project, "integrity": {"version": 0}}, "start a new run"),
                ({**self.project, "measurement": {"protocol": "legacy"}}, "start a new run"),
                ({**self.project, "integrity": {**self.project["integrity"], "protected_files": {}}},
                 "missing required harness hashes"),
            )
            for data, expected in cases:
                with self.subTest(expected=expected):
                    self.metadata.write_text(yaml.safe_dump(data))
                    self.assertIn(expected, "\n".join(self.check()))
            self.metadata.unlink()
            self.assertIn("start a new run", "\n".join(self.check()))
        finally:
            self.metadata.write_text(self.metadata_text)

    def test_intrinsics_header_cannot_shadow_source_or_harness(self) -> None:
        rel = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/include/cfu_intrinsics.h"
        for name in ("input", "SIZE", "benchmark", "verify_benchmark", "start_trigger", "malloc_beebs"):
            with self.subTest(name=name):
                header = f"#ifndef CFU_INTRINSICS_H\n#define CFU_INTRINSICS_H\n#define {name}(...) 1\n#endif\n"
                self.assertTrue(self.check(extra={rel: header}))

    def test_preprocessor_wrappers_constructors_and_pragmas_rejected(self) -> None:
        changes = [
            "#if 0\n" + self.original + "\n#endif\n",
            self.original.replace("static unsigned output[SIZE];", "#ifndef USE_AGENT_CFU\nstatic unsigned output[SIZE];\n#endif"),
            self.original + "\n#pragma GCC optimize (\"O0\")\n",
            self.original + '\n__attribute__((constructor)) static void cfu_init(void) { output[0] = 2; }\n',
            self.original + '\n#define CFU_ATTR __attribute__((constructor))\nstatic void CFU_ATTR cfu_init(void) {}\n',
            self.original + '\n#define CFU_PRAGMA _Pragma("GCC optimize(\\\"O0\\\")")\n',
            self.original + '\nstatic void (*cfu_init)(void) __attribute__((section(".init_array"))) = initialise_benchmark;\n',
            self.original + '\n#include "untrusted.h"\n',
            self.original + '\nasm(".global benchmark; .set benchmark, cfu_fast");\n',
            self.original + '\nstatic int cfu_fake(void) __asm__("benchmark");\n',
            self.original + '\nvoid *malloc_beebs(unsigned size) { return output; }\n',
        ]
        for text in changes:
            with self.subTest(tail=text[-100:]):
                self.assertTrue(self.check(text))

    def test_conditional_optimization_inside_computational_body_allowed(self) -> None:
        text = replace_body(self.original, "calculate", "\n#ifdef USE_AGENT_CFU\nreturn n + 1;\n#else\nreturn n + 1;\n#endif\n")
        self.assertEqual(self.check(text), [])

    def test_full_transaction_and_disk_tampering_checks(self) -> None:
        scala = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.assertEqual(self.check(extra={scala: "new Scala bytes"}), [])
        sample_ref = next(iter(self.project["integrity"]["reference_sources"]))
        sample_prot = next(iter(self.project["integrity"]["protected_files"]))
        for rel in (sample_ref, sample_prot):
            with self.subTest(path=rel):
                path = self.workdir / rel
                text = path.read_text()
                self.assertEqual(self.check(extra={scala: "new Scala bytes", rel: text}), [])
                self.assertTrue(self.check(extra={scala: "new Scala bytes", rel: text + "/* tampered */"}))
                try:
                    path.write_text(text + "/* tampered */\n")
                    self.assertTrue(self.check())
                finally:
                    path.write_text(text)
        unknown = self.target.replace("kernel.c", "untrusted.c")
        self.assertTrue(self.check(extra={unknown: "int fake;"}))
        sibling = self.target.replace("embench_fixture", "embench_other")
        self.assertTrue(self.check(extra={sibling: "int fake;"}))

    def test_missing_required_source_harness_or_unknown_candidate_rejected(self) -> None:
        rel = next(iter(self.project["integrity"]["protected_files"]))
        path = self.workdir / rel
        backup = path.with_suffix(".backup")
        path.rename(backup)
        try:
            self.assertTrue(self.check())
        finally:
            backup.rename(path)
        cfu = self.workdir / self.target
        original = cfu.read_text()
        cfu_backup = cfu.with_suffix(".backup")
        cfu.rename(cfu_backup)
        try:
            self.assertTrue(self.check(original))
            self.assertTrue(self.check(extra={"unknown.c": original}))
        finally:
            cfu_backup.rename(cfu)


class RealEmbenchIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temp.cleanup)
        cls.workdir = Path(cls._temp.name)
        import shutil
        for subdir in ("src", "support"):
            shutil.copytree(REPO_ROOT / embench.EMBENCH_ROOT / subdir,
                            cls.workdir / embench.EMBENCH_ROOT / subdir)
        board = cls.workdir / embench.BOARD_ROOT
        board.mkdir(parents=True)
        for fname in embench.BOARD_FILES:
            shutil.copyfile(REPO_ROOT / embench.BOARD_ROOT / fname, board / fname)
        cls.run_root = "agents/generated/runs/real-integrity-test"
        cls.policy = load_policy(REPO_ROOT)
        cls._prepared: dict[str, tuple[dict, dict[str, str]]] = {}

    def prepare(self, name: str) -> tuple[dict, dict[str, str]]:
        if name not in self._prepared:
            project = embench.prepare_embench_benchmark(RepoToolbox(
                self.workdir, policy=self.policy, stage="prepare_benchmark", run_root=self.run_root),
                {"benchmark_name": name})
            sources = {str(p.relative_to(self.workdir)): p.read_text()
                       for p in (self.workdir / embench.config_workspace(self.run_root, name, "cfu") / "src").glob("*.c")}
            self._prepared[name] = (project, sources)
        project, sources = self._prepared[name]
        metadata = self.workdir / self.run_root / "04_benchmark/embench.yaml"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(yaml.safe_dump(project))
        return project, sources

    def test_representative_real_benchmarks_prepare_and_validate(self) -> None:
        names = embench.list_benchmarks(self.workdir)
        self.assertEqual(len(names), 19)
        for name in ("crc32", "huffbench", "statemate", "md5sum"):
            with self.subTest(name=name):
                original = {p.name: p.read_bytes() for p in (self.workdir / embench.benchmark_dir(name)).glob("*.c")}
                project, sources = self.prepare(name)
                self.assertEqual(project["measurement"], {"protocol": embench.MEASUREMENT_PROTOCOL,
                    "local_scale_factor": 1, "global_scale_factor": 1, "warmup": False,
                     "cfu_status_protocol": cfu_execution.PROTOCOL})
                cfg = self.workdir / embench.config_workspace(self.run_root, name, "cfu")
                source_paths = [*(cfg / "src").glob("*.c"),
                    *(cfg / "support" / filename for filename in embench_patchers.extra_support_files(name)
                      if filename.endswith(".c"))]
                definitions = [t.value for p in source_paths for t in integrity.scan(p.read_text())
                               if t.directive and t.value.split()[:3] == ["#", "define", "LOCAL_SCALE_FACTOR"]]
                self.assertEqual(definitions, ["# define LOCAL_SCALE_FACTOR 1"])
                main = cfg.joinpath("embench_main.c").read_text()
                self.assertNotIn("warm_caches", [t.value for t in integrity.scan(main) if not t.directive])
                self.assertEqual(integrity.validate_candidate(self.workdir, self.run_root, name), [])
                rel, text = next(iter(sources.items()))
                parsed = integrity._parse(text)
                fn = next(o.name for o in parsed.objects if o.body is not None and o.name not in integrity.ENTRY_FUNCTIONS)
                pending = {rel: replace_body(text, fn, "/* kernel optimization fixture */ return 0;")}
                self.assertEqual(integrity.validate_candidate(self.workdir, self.run_root, name, pending), [])
                if any(o.name == "verify_benchmark" for o in parsed.objects):
                    changed = replace_body(text, "verify_benchmark", "return 1;")
                    self.assertTrue(integrity.validate_candidate(self.workdir, self.run_root, name, {rel: changed}))
                self.assertEqual(original, {p.name: p.read_bytes() for p in (self.workdir / embench.benchmark_dir(name)).glob("*.c")})

    def test_md5_extra_argument_is_preserved_and_protected(self) -> None:
        _, sources = self.prepare("md5sum")
        rel, text = next(iter(sources.items()))
        self.assertIn("benchmark_body (LOCAL_SCALE_FACTOR, GLOBAL_SCALE_FACTOR, MSG_SIZE)", text)
        changed = text.replace("GLOBAL_SCALE_FACTOR, MSG_SIZE", "GLOBAL_SCALE_FACTOR, 1")
        self.assertTrue(integrity.validate_candidate(self.workdir, self.run_root, "md5sum", {rel: changed}))
        changed = text.replace("int len)", "unsigned len)")
        self.assertTrue(integrity.validate_candidate(self.workdir, self.run_root, "md5sum", {rel: changed}))

    def test_xgboost_model_tables_and_headers_protected(self) -> None:
        project, sources = self.prepare("xgboost")
        rel = next(k for k in sources if k.endswith("/xgboost.c"))
        text = sources[rel]
        parsed = integrity._parse(text)
        declaration = next(o for o in parsed.objects if o.body is None and "X_test" in o.signature)
        changed = text[:declaration.start] + "float X_test[1][1] = {{0}};" + text[declaration.end:]
        self.assertTrue(integrity.validate_candidate(self.workdir, self.run_root, "xgboost", {rel: changed}))
        self.assertTrue(any(p.endswith("/src/xgboost.h") for p in project["integrity"]["protected_files"]))
        testbench = next(k for k in sources if k.endswith("/testbench.c"))
        changed = sources[testbench] + "\n#define X_test cfu_scratch\n"
        self.assertTrue(integrity.validate_candidate(self.workdir, self.run_root, "xgboost", {testbench: changed}))

    def test_nsichneu_inline_algorithm_not_frozen(self) -> None:
        _, sources = self.prepare("nsichneu")
        rel, text = next(iter(sources.items()))
        changed = text.replace("if (x < y)", "if ((x - y) < 0)", 1)
        self.assertNotEqual(changed, text)
        self.assertEqual(integrity.validate_candidate(self.workdir, self.run_root, "nsichneu", {rel: changed}), [])


class SourceTransformTest(unittest.TestCase):
    def test_unique_source_scale_definition_required(self) -> None:
        for sources in ({"a.c": "int x;"}, {"a.c": "#define LOCAL_SCALE_FACTOR 7\n", "b.c": "#define LOCAL_SCALE_FACTOR 9\n"},
                        {"a.c": "#define LOCAL_SCALE_FACTOR 7\n#undef LOCAL_SCALE_FACTOR\n"}):
            with self.subTest(sources=sources):
                with self.assertRaises(ValueError):
                    integrity.normalize_local_scale_factor(sources)
        source = '/* #define LOCAL_SCALE_FACTOR 4 */\nconst char *s = "#define LOCAL_SCALE_FACTOR 3";\n# define LOCAL_SCALE_FACTOR 42 /* real */\n'
        normalized = integrity.normalize_local_scale_factor({"a.c": source})["a.c"]
        self.assertIn("#define LOCAL_SCALE_FACTOR 1\n", normalized)
        self.assertIn('"#define LOCAL_SCALE_FACTOR 3"', normalized)

    def test_main_warmup_omitted_not_heat_zero(self) -> None:
        source = 'void main(void) { initialise_benchmark(); warm_caches (123); start_trigger(); int result = benchmark(); stop_trigger(); correct = verify_benchmark (result); }'
        main = embench.instrument_embench_main(source)
        self.assertNotIn("warm_caches", main)
        self.assertIn("initialise_benchmark();", main)
        self.assertIn("start_trigger();", main)
        self.assertIn("stop_trigger();", main)
        self.assertIn("EMBENCH_VERIFY_", main)


if __name__ == "__main__":
    unittest.main()
