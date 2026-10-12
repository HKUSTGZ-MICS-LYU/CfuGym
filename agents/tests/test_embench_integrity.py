"""Fast, tempfile-only Embench measurement and structural integrity contracts."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from agents.cfu_design_agent.benchmark import cfu_execution, embench, embench_integrity as integrity
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT
from agents.tests.test_embench import temporary_embench_repo


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
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workdir = Path(temporary.name)
        self.run_root = "agents/generated/runs/integrity-test"
        self.name = "fixture"
        source = self.workdir / embench.benchmark_dir(self.name) / "kernel.c"
        source.parent.mkdir(parents=True)
        source.write_text(SOURCE)
        support = self.workdir / embench.EMBENCH_ROOT / "support"
        support.mkdir(parents=True)
        for fname in embench.SUPPORT_FILES:
            (support / fname).write_text((REPO_ROOT / embench.EMBENCH_ROOT / "support" / fname).read_text())
        board = self.workdir / embench.BOARD_ROOT
        board.mkdir(parents=True)
        for fname in embench.BOARD_FILES:
            (board / fname).write_text((REPO_ROOT / embench.BOARD_ROOT / fname).read_text())
        self.project = embench.prepare_embench_benchmark(
            RepoToolbox(self.workdir, policy=load_policy(REPO_ROOT), stage="prepare_benchmark", run_root=self.run_root),
            {"benchmark_name": self.name})
        self.metadata = self.workdir / self.run_root / "04_benchmark/embench.yaml"
        self.metadata.parent.mkdir(parents=True)
        self.metadata.write_text(yaml.safe_dump(self.project))
        self.target = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/src/kernel.c"
        self.original = (self.workdir / self.target).read_text()

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
        # Literal spelling in an initializer is protected, unlike a comment.
        text = self.original.replace("{1, 2, 3, 4}", "{1, 2, 3, 5}")
        self.assertTrue(self.check(text))

    def test_fresh_helpers_scratch_macros_and_intrinsics_allowed(self) -> None:
        text = '#include <stdint.h>\n#include "cfu_intrinsics.h"\n#define CFU_UNROLL 4\n' + self.original
        text += '\nstatic word cfu_scratch[4];\nstatic struct Pair cfu_pair;\nstatic int cfu_helper(int value) { return value * CFU_UNROLL; }\n'
        header = '#ifndef CFU_INTRINSICS_H\n#define CFU_INTRINSICS_H\n#include <stdint.h>\n#define CFU_OP 1\nstatic inline uint32_t cfu_compute(uint32_t a) { return a; }\n#endif\n'
        header_rel = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/include/cfu_intrinsics.h"
        self.assertEqual(self.check(text, {header_rel: header}), [])

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
            "#define benchmark() 1", "#define benchmark_body(a,b) 1",
            "#define verify_benchmark(x) 1", "#define input cfu_fake", "#undef SIZE",
            "#define GLOBAL_SCALE_FACTOR 0", "#define unsigned signed",
            "#define start_trigger()", "#define initialise_benchmark()",
            "#define bench\\\nmark() 1", "%:define benchmark() 1",
            "#/**/define /**/ benchmark /**/() 1",
            "#define SIZE(x) 4", "#define calculate(x) 0",
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

    def test_missing_or_legacy_metadata_requires_new_run(self) -> None:
        for data in ({}, {**self.project, "integrity": {"version": 0}},
                     {**self.project, "measurement": {"protocol": "legacy"}}):
            with self.subTest(data=str(data)[:50]):
                self.metadata.write_text(yaml.safe_dump(data))
                problems = self.check()
                self.assertTrue(problems)
                self.assertIn("start a new run", "\n".join(problems))
                self.assertNotIn("re-prepare", "\n".join(problems))
        missing = self.metadata.with_name("missing.yaml")
        self.assertFalse(missing.exists())
        self.metadata.rename(missing)
        self.assertIn("start a new run", "\n".join(self.check()))

    def test_intrinsics_header_cannot_shadow_source_or_harness(self) -> None:
        rel = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/include/cfu_intrinsics.h"
        for name in ("input", "SIZE", "benchmark", "verify_benchmark", "start_trigger", "malloc_beebs"):
            with self.subTest(name=name):
                header = f"#ifndef CFU_INTRINSICS_H\n#define CFU_INTRINSICS_H\n#define {name}(...) 1\n#endif\n"
                self.assertTrue(self.check(extra={rel: header}))
        header_path = self.workdir / rel
        header_path.parent.mkdir(parents=True)
        header_path.write_text("#define benchmark_body(...) 0\n")
        self.assertTrue(self.check())

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

    def test_missing_legacy_and_incomplete_metadata_fail_closed(self) -> None:
        for data in ({}, {"benchmark": self.name}, {**self.project, "integrity": {"version": 0}},
                     {**self.project, "integrity": {**self.project["integrity"], "protected_files": {}}}):
            with self.subTest(data=str(data)[:60]):
                self.metadata.write_text(yaml.safe_dump(data))
                self.assertTrue(self.check())
        self.metadata.write_text("")
        self.assertTrue(self.check())
        missing = self.metadata.with_name("missing.yaml")
        self.assertFalse(missing.exists())
        self.metadata.rename(missing)
        self.assertTrue(self.check())

    def test_full_transaction_ignores_scala_but_checks_pending_harness(self) -> None:
        scala = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.assertEqual(self.check(extra={scala: "new Scala bytes"}), [])
        for rel in list(self.project["integrity"]["reference_sources"]) + list(self.project["integrity"]["protected_files"]):
            with self.subTest(path=rel):
                text = (self.workdir / rel).read_text()
                self.assertEqual(self.check(extra={scala: "new Scala bytes", rel: text}), [])
                self.assertTrue(self.check(extra={scala: "new Scala bytes", rel: text + "/* tampered */"}))
                self.assertEqual((self.workdir / rel).read_text(), text)
        unknown = self.target.replace("kernel.c", "untrusted.c")
        self.assertTrue(self.check(extra={unknown: "int fake;"}))
        sibling = self.target.replace("embench_fixture", "embench_other")
        self.assertTrue(self.check(extra={sibling: "int fake;"}))
        self.assertTrue(self.check(extra={str(self.metadata.relative_to(self.workdir)): "fake metadata"}))
        self.assertTrue(self.check(extra={f"{embench.benchmark_dir(self.name)}/kernel.c": "fake source"}))

    def test_trusted_reference_and_all_config_harness_hashes_checked(self) -> None:
        paths = list(self.project["integrity"]["reference_sources"]) + list(self.project["integrity"]["protected_files"])
        for rel in paths:
            with self.subTest(path=rel):
                path = self.workdir / rel
                original = path.read_text()
                path.write_text(original + "/* tampered */\n")
                self.assertTrue(self.check())
                path.write_text(original)

    def test_missing_required_source_harness_or_unknown_candidate_rejected(self) -> None:
        rel = next(iter(self.project["integrity"]["protected_files"]))
        path = self.workdir / rel
        backup = path.with_suffix(".backup")
        self.assertFalse(backup.exists())
        path.rename(backup)
        self.assertTrue(self.check())
        backup.rename(path)
        cfu = self.workdir / self.target
        original = cfu.read_text()
        cfu.rename(cfu.with_suffix(".backup"))
        self.assertTrue(self.check(original))
        self.assertTrue(self.check(extra={"unknown.c": original}))


class RealEmbenchIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = temporary_embench_repo(self)
        self.run_root = "agents/generated/runs/real-integrity-test"
        self.policy = load_policy(REPO_ROOT)

    def prepare(self, name: str) -> tuple[dict, dict[str, str]]:
        project = embench.prepare_embench_benchmark(RepoToolbox(
            self.workdir, policy=self.policy, stage="prepare_benchmark", run_root=self.run_root),
            {"benchmark_name": name})
        metadata = self.workdir / self.run_root / "04_benchmark/embench.yaml"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(yaml.safe_dump(project))
        sources = {str(p.relative_to(self.workdir)): p.read_text()
                   for p in (self.workdir / embench.config_workspace(self.run_root, name, "cfu") / "src").glob("*.c")}
        return project, sources

    def test_all_real_19_prepare_single_invocation_and_validate(self) -> None:
        names = embench.list_benchmarks(self.workdir)
        self.assertEqual(len(names), 19)
        for name in names:
            with self.subTest(name=name):
                original = {p.name: p.read_bytes() for p in (self.workdir / embench.benchmark_dir(name)).glob("*.c")}
                project, sources = self.prepare(name)
                self.assertEqual(project["measurement"], {"protocol": embench.MEASUREMENT_PROTOCOL,
                    "local_scale_factor": 1, "global_scale_factor": 1, "warmup": False,
                     "cfu_status_protocol": cfu_execution.PROTOCOL})
                for config in embench.CONFIGS:
                    cfg = self.workdir / embench.config_workspace(self.run_root, name, config)
                    definitions = [t.value for p in (cfg / "src").glob("*.c") for t in integrity.scan(p.read_text())
                                   if t.directive and t.value.split()[:3] == ["#", "define", "LOCAL_SCALE_FACTOR"]]
                    self.assertEqual(definitions, ["# define LOCAL_SCALE_FACTOR 1"])
                    main = cfg.joinpath("embench_main.c").read_text()
                    self.assertNotIn("warm_caches", [t.value for t in integrity.scan(main) if not t.directive])
                    for call in ("initialise_board", "initialise_benchmark", "start_trigger", "benchmark", "stop_trigger", "verify_benchmark"):
                        self.assertIn(call, [t.value for t in integrity.scan(main) if not t.directive])
                    self.assertIn("-DGLOBAL_SCALE_FACTOR=1", embench.config_defines(name, config))
                self.assertEqual(integrity.validate_candidate(self.workdir, self.run_root, name), [])
                for rel, text in sources.items():
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
