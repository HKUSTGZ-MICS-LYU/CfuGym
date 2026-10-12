"""Embench workload support: discovery, instrumentation, commands, policy."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.benchmark.embench_loops import discover_loops

from agents.cfu_design_agent.benchmark import embench, embench_instrument
from agents.cfu_design_agent.isa import render_cfu_intrinsics
from agents.cfu_design_agent.runtime.policy import PolicyError, load_policy
from agents.cfu_design_agent.spec import normalize_workload_spec, validate_workload_spec
from agents.cfu_design_agent.runtime.tools import RepoToolbox, infer_command_id
from agents.tests.common import REPO_ROOT
from agents.cfu_design_agent.runtime import stage_services as services
from agents.cfu_design_agent.stages import profile


def temporary_embench_repo(test: unittest.TestCase) -> Path:
    """Real source fixtures copied into disposable storage; never run artifacts."""
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    root = Path(directory.name)
    for subdir in ("src", "support"):
        shutil.copytree(REPO_ROOT / embench.EMBENCH_ROOT / subdir,
                        root / embench.EMBENCH_ROOT / subdir)
    board = root / embench.BOARD_ROOT
    board.mkdir(parents=True)
    for fname in embench.BOARD_FILES:
        shutil.copyfile(REPO_ROOT / embench.BOARD_ROOT / fname, board / fname)
    return root


FIXTURE = """#include <stdio.h>

static int helper(int x);

static int
benchmark_body(unsigned int n)
{
  int i;
  for (i = 0; i < n; ++i) {
    if (i == 3) {
      return i;
    }
  }
  return n;
}

static int
helper(int x)
{
  return benchmark_body(x);
}
"""


class InstrumenterTest(unittest.TestCase):
    def test_matches_definition_not_prototype(self) -> None:
        text, applied, skipped = embench_instrument.instrument_text(
            FIXTURE,
            [{"function": "benchmark_body", "id": "benchmark_body"}],
            rel_path="benchmark.c",
        )
        self.assertEqual(applied, ["benchmark_body"])
        # The marker must be inside benchmark_body, not in helper (the next
        # function after the prototype).
        body = text.split("benchmark_body(unsigned int n)")[1]
        self.assertIn("EMBENCH_OBS_ENTER(benchmark_body);", body.split("}")[0])

    def test_adds_observation_include(self) -> None:
        text, _, _ = embench_instrument.instrument_text(
            FIXTURE, [{"function": "helper", "id": "helper"}], rel_path="benchmark.c"
        )
        self.assertIn(embench_instrument.OBS_INCLUDE, text)

    def test_idempotent(self) -> None:
        once, _, _ = embench_instrument.instrument_text(
            FIXTURE, [{"function": "helper", "id": "helper"}], rel_path="benchmark.c"
        )
        twice, applied, skipped = embench_instrument.instrument_text(
            once, [{"function": "helper", "id": "helper"}], rel_path="benchmark.c"
        )
        self.assertEqual(applied, ["helper"])
        self.assertEqual(twice.count("EMBENCH_OBS_ENTER(helper)"), 1)
        self.assertEqual(skipped, [])

    def test_loop_points_wrap_full_loop_once(self) -> None:
        text, applied, skipped = embench_instrument.instrument_text(
            FIXTURE,
            [{"function": "benchmark_body", "id": "loop1", "kind": "loop", "ordinal": 1}],
            rel_path="benchmark.c",
        )
        self.assertEqual(applied, ["loop1"])
        self.assertEqual(skipped, [])
        self.assertIn("{ EMBENCH_OBS_ENTER(loop1);\nfor (i = 0; i < n; ++i)", text)
        self.assertIn("  }\n}\n  return n;", text)

    def test_missing_definition_reported(self) -> None:
        _, applied, skipped = embench_instrument.instrument_text(
            FIXTURE, [{"function": "nope", "id": "nope"}], rel_path="benchmark.c"
        )
        self.assertEqual(applied, [])
        self.assertTrue(any("not found" in item["reason"] for item in skipped))

    def test_default_plan_finds_loops(self) -> None:
        points = embench_instrument.default_plan({"benchmark.c": FIXTURE})
        self.assertIn("benchmark_body", [p["function"] for p in points])
        self.assertEqual([p["kind"] for p in points], ["function", "loop"])
        self.assertEqual(points[1]["line"], 9)

    def test_default_plan_has_stable_cross_run_ids(self) -> None:
        paths = [f"agents/generated/runs/{run}/workspace/embench_x/profile/src/benchmark.c"
                 for run in ("one", "two")]
        first, second = (embench_instrument.default_plan({path: FIXTURE}) for path in paths)
        self.assertEqual(first, second)
        self.assertEqual(first[1]["file"], "benchmark.c")
        self.assertTrue(first[0]["id"].startswith("obs_fn_"))

    def test_nested_unbraced_do_and_preprocessor_text(self) -> None:
        source = '''#include <stdint.h>
#define HIDDEN for (int z=0;z<5;z++)
int work(int n) {
  // while (0) { not a loop }
  const char *message = "for (;;)";
  for (int i=0;i<n;i++)
    for (int j=0;j<2;j++)
      if (j) continue;
  do { n--; } while(n>0);
  return n;
}
'''
        loops = discover_loops(source, ["work"])
        self.assertEqual([loop.kind for loop in loops], ["for", "for", "do"])
        self.assertEqual([loop.ordinal for loop in loops], [1, 2, 3])
        points = embench_instrument.default_plan({"benchmark.c": source})
        result, report = embench_instrument.instrument_sources({"benchmark.c": source}, {"points": points})
        self.assertEqual(len(report.applied), 4)
        self.assertEqual(report.skipped, [])
        self.assertEqual(result["benchmark.c"].count("EMBENCH_OBS_ENTER("), 4)

    def test_loop_scope_runtime_counts_early_exits(self) -> None:
        compiler = shutil.which("gcc")
        if not compiler:
            self.skipTest("host GCC not installed")
        source = '''#include <stdio.h>
int work(int n) {
  int total = 0;
  for (int i=0; i<n; ++i) {
    if (i == 1) continue;
    if (i == 4) break;
    total += i;
  }
  do { if (n < 0) return -1; --n; } while(n > 0);
  return total;
}
int main(void) { printf("%d %d\\n", work(5), work(-1)); return 0; }
'''
        points = [{"file": "work.c", "function": "work", "kind": "loop", "id": "obs_for", "ordinal": 1},
                  {"file": "work.c", "function": "work", "kind": "loop", "id": "obs_do", "ordinal": 2}]
        text, report = embench_instrument.instrument_sources({"work.c": source}, {"points": points})
        self.assertEqual(report.applied, ["obs_for", "obs_do"])
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir)
            (path / "work.c").write_text(text["work.c"])
            (path / "obs.c").write_text('''#include <stdint.h>
#include <stdio.h>
static int entries = 0, exits = 0;
uint32_t embench_obs_enter_scope(const char *name) { (void)name; ++entries; return 0; }
void embench_obs_exit_scope(const char *name, uint32_t start) { (void)name; (void)start; ++exits; }
int main_dummy(void) { return 0; }
__attribute__((destructor)) static void check(void) { printf("%d %d\\n", entries, exits); }
''')
            binary = path / "work"
            built = subprocess.run([compiler, "-DEMBENCH_OBS", "-I" + str(REPO_ROOT / "sw/embench"),
                                    str(path / "work.c"), str(path / "obs.c"), "-o", str(binary)],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(built.returncode, 0, built.stderr)
            output = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            self.assertEqual(output.stdout.splitlines(), ["5 -1", "4 4"])

    def test_instrumented_matmult_compiles_for_riscv(self) -> None:
        compiler = shutil.which("riscv64-unknown-elf-gcc")
        if not compiler:
            self.skipTest("RISC-V GCC not installed")
        path = REPO_ROOT / "benchmarks/embench-iot/src/matmult-int/matmult-int.c"
        source = path.read_text()
        instrumented, report = embench_instrument.instrument_sources({str(path): source})
        self.assertGreaterEqual(len([p for p in report.points if p["kind"] == "loop"]), 7)
        with tempfile.TemporaryDirectory() as workdir:
            test_file = Path(workdir) / "matmult-int.c"
            test_file.write_text(instrumented[str(path)])
            command = [compiler, "-fsyntax-only", "-march=rv32imc_zicsr_zifencei", "-mabi=ilp32",
                       "-DEMBENCH_OBS", "-DGLOBAL_SCALE_FACTOR=1", "-I" + str(REPO_ROOT / "sw/embench"),
                       "-I" + str(REPO_ROOT / "benchmarks/embench-iot/support"), str(test_file)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)


class EmbenchDiscoveryTest(unittest.TestCase):
    def test_lists_benchmarks(self) -> None:
        names = embench.list_benchmarks(REPO_ROOT)
        self.assertIn("crc32", names)
        self.assertIn("md5sum", names)
        self.assertGreaterEqual(len(names), 19)

    def test_sources_and_verification(self) -> None:
        files = embench.benchmark_source_files(REPO_ROOT, "crc32")
        self.assertEqual(len(files), 1)
        text = (REPO_ROOT / files[0]).read_text()
        self.assertTrue(embench.has_real_verification(text))

    def test_build_command_uses_makefile_hooks(self) -> None:
        cmd = embench.build_command("agents/generated/runs/r1", "crc32", "profile")
        joined = " ".join(cmd)
        self.assertIn("TARGET=vexii_soc", joined)
        self.assertIn("EXTRA_SOURCES=", joined)
        self.assertIn("EXTRA_INCLUDES=", joined)
        self.assertIn("-DEMBENCH_OBS", joined)
        self.assertIn("-std=gnu11", joined)
        # Each configuration owns its source tree so objects never collide.
        self.assertIn("/profile/embench_main", joined)
        self.assertIn("/profile/src/*.c", joined)
        for token in cmd:
            self.assertFalse(token.startswith("/"), token)

    def test_all_build_configs_pin_gnu11(self) -> None:
        for config in embench.CONFIGS:
            with self.subTest(config=config):
                cmd = embench.build_command("agents/generated/runs/r1", "wikisort", config)
                extra_cflags = next(item for item in cmd if item.startswith("EXTRA_CFLAGS="))
                self.assertIn("-std=gnu11", extra_cflags)

    def test_config_defines(self) -> None:
        self.assertNotIn("-DEMBENCH_OBS", embench.config_defines("crc32", "reference"))
        self.assertIn("-DEMBENCH_OBS", embench.config_defines("crc32", "profile"))
        self.assertIn("-DUSE_AGENT_CFU", embench.config_defines("crc32", "cfu"))

    def test_profile_parsing_and_hotspot(self) -> None:
        text = "CFU_AGENT_PROFILE crc32=100\nCFU_AGENT_PROFILE fn_a=60 calls=3\n"
        parsed = embench.parse_profile(text)
        self.assertEqual(parsed["crc32"]["cycles"], 100)
        self.assertEqual(parsed["fn_a"]["calls"], 3)
        self.assertEqual(embench.hotspot_from_profile(parsed, exclude=("crc32",)), "fn_a")

    def test_profile_parsing_strips_terminal_ansi_sequences(self) -> None:
        text = "\x1b[0mCFU_AGENT_PROFILE crc32=15503563\x1b[0m\x1b[0J\n"
        parsed = embench.parse_profile(text)
        self.assertEqual(parsed["crc32"]["cycles"], 15503563)

    def test_profile_parsing_from_sbt_prefixed_simulator_output(self) -> None:
        # A real MiCoSocSim log prefixes *every* guest stdout line with [info].
        # It must retain both the whole-loop baseline and individual loop calls.
        text = (
            "[info] Start MiCoSocSim test simulation with seed 42\n"
            "[info] CFU_AGENT_PROFILE crc32=16223945\n"
            "[info] CFU_AGENT_PROFILE obs_loop_ffaeb24f_crc32pseudo_1=15488349 calls=170\n"
            "[info] EMBENCH_VERIFY_PASS benchmark=crc32 verify=1 result=11433\n"
        )
        # The harness EMBENCH_VERIFY line folds verify/result into the
        # whole-benchmark entry so the gate can compare the reference run's
        # observable result with the cfu run's, independent of verify_benchmark.
        self.assertEqual(embench.parse_profile(text), {
            "crc32": {"cycles": 16223945, "verify": 1, "result": 11433},
            "obs_loop_ffaeb24f_crc32pseudo_1": {"cycles": 15488349, "calls": 170},
        })
        self.assertEqual(embench.hotspot_from_profile(embench.parse_profile(text), exclude=("crc32",)),
                         "obs_loop_ffaeb24f_crc32pseudo_1")
        self.assertEqual(embench.parse_profile("[warn] CFU_AGENT_PROFILE crc32=12\n"), {})
        self.assertEqual(embench.parse_profile("something CFU_AGENT_PROFILE crc32=12\n"), {})

    def test_parse_verify_result(self) -> None:
        text = (
            "[info] EMBENCH_VERIFY_FAIL benchmark=md5sum verify=0 result=42\n"
            "[info] EMBENCH_VERIFY_PASS benchmark=md5sum verify=1 result=12345\n"
        )
        # The LAST line wins: a run's final observed result is the comparison basis.
        self.assertEqual(embench.parse_verify_result(text), {
            "benchmark": "md5sum", "pass": True, "verify": 1, "result": 12345,
        })
        self.assertIsNone(embench.parse_verify_result("no verify line here"))
        self.assertEqual(
            embench.parse_verify_result("EMBENCH_VERIFY_FAIL benchmark=x verify=0 result=-7"),
            {"benchmark": "x", "pass": False, "verify": 0, "result": -7},
        )


class EmbenchTamperTest(unittest.TestCase):
    """The benchmark oracle must not be editable through the cfu copy."""

    def _workspace(self, tmp: Path, name: str = "md5sum") -> tuple[Path, str]:
        run_root = "agents/generated/runs/test-tamper"
        for config in ("reference", "cfu"):
            root = tmp / embench.config_workspace(run_root, name, config)
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "board").mkdir(parents=True, exist_ok=True)
            (root / "support").mkdir(parents=True, exist_ok=True)
            (root / "embench_main.c").write_text("main")
            for fname in ("boardsupport.c", "boardsupport.h", "embench_obs.h"):
                (root / "board" / fname).write_text(fname)
            for fname in ("main.c", "beebsc.c", "beebsc.h", "support.h"):
                (root / "support" / fname).write_text(fname)
        src = (
            "int kernel(int r) { return r; }\n"
            "int verify_benchmark(int r) { return r == RESULT; }\n"
        )
        (tmp / embench.config_workspace(run_root, name, "reference") / "src" / "md5.c").write_text(src)
        (tmp / embench.config_workspace(run_root, name, "cfu") / "src" / "md5.c").write_text(src)
        return tmp, run_root

    def test_identical_copies_report_no_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp, run_root = self._workspace(Path(tmp_dir))
            self.assertEqual(embench.workspace_tampering(tmp, run_root, "md5sum"), [])

    def test_modified_verify_benchmark_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp, run_root = self._workspace(Path(tmp_dir))
            cfu_src = tmp / embench.config_workspace(run_root, "md5sum", "cfu") / "src" / "md5.c"
            cfu_src.write_text(
                "int kernel(int r) { return r; }\n"
                "int verify_benchmark(int r) { return 1; }\n"
            )
            problems = embench.workspace_tampering(tmp, run_root, "md5sum")
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("verify_benchmark()", problems[0])

    def test_modified_kernel_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp, run_root = self._workspace(Path(tmp_dir))
            cfu_src = tmp / embench.config_workspace(run_root, "md5sum", "cfu") / "src" / "md5.c"
            cfu_src.write_text(
                "int kernel(int r) { return r + 1; }\n"
                "int verify_benchmark(int r) { return r == RESULT; }\n"
            )
            self.assertEqual(embench.workspace_tampering(tmp, run_root, "md5sum"), [])

    def test_modified_boardsupport_and_main_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp, run_root = self._workspace(Path(tmp_dir))
            cfu_root = tmp / embench.config_workspace(run_root, "md5sum", "cfu")
            (cfu_root / "board" / "boardsupport.c").write_text("int cycles(void) { return 1; }")
            (cfu_root / "embench_main.c").write_text("int main(void) { return 0; }")
            problems = embench.workspace_tampering(tmp, run_root, "md5sum")
            self.assertEqual(len(problems), 2, problems)
            self.assertTrue(any("boardsupport.c" in p for p in problems))
            self.assertTrue(any("embench_main.c" in p for p in problems))

    def test_missing_reference_copy_skips_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Fixture workspaces without a full config set must not error out.
            self.assertEqual(embench.workspace_tampering(Path(tmp_dir), "r", "md5sum"), [])


class EmbenchPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(REPO_ROOT)

    def test_build_command_is_allowlisted(self) -> None:
        cmd = embench.build_command("agents/generated/runs/r1", "crc32", "reference")
        self.assertEqual(infer_command_id(cmd, self.policy), "embench_build")
        self.policy.check_command("profile", "embench_build", cmd, REPO_ROOT)

    def test_embench_tree_is_readable_but_not_writable(self) -> None:
        src = "benchmarks/embench-iot/src/crc32/crc_32.c"
        self.assertTrue(self.policy.can_read(src))
        for stage in ("implementation", "profile", "validation"):
            self.assertFalse(self.policy.can_write(stage, src), stage)

    def test_workspace_copy_is_writable(self) -> None:
        self.assertTrue(
            self.policy.can_write(
                "profile", "agents/generated/runs/r1/workspace/embench_crc32/src/crc_32.c"
            )
        )


class EmbenchSpecTest(unittest.TestCase):
    def test_spec_inference_and_validation(self) -> None:
        spec, _ = normalize_workload_spec(
            task="accelerate crc32",
            input_mode="embench",
            context={},
            workdir=REPO_ROOT,
            benchmark_name="crc32",
        )
        self.assertEqual(spec["source_kind"], "embench")
        self.assertEqual(spec["benchmark_name"], "crc32")
        self.assertEqual(validate_workload_spec(spec), [])

    def test_missing_name_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_workload_spec(
                task="accelerate something",
                input_mode="embench",
                context={},
                workdir=REPO_ROOT,
            )


class IntrinsicsRenderTest(unittest.TestCase):
    def test_renders_custom0_intrinsics(self) -> None:
        text = render_cfu_intrinsics(
            {
                "identity": {"name": "wavg"},
                "commands": [{"function_id": 1, "name": "compute", "semantics": "mac"}],
            }
        )
        self.assertIn("cfu_enable", text)
        self.assertIn("#define CFU_COMPUTE_FUNC3 1", text)
        self.assertIn("static inline uint32_t cfu_compute(", text)
        self.assertIn("(1 << 12)", text)

    def test_renderer_skips_function_ids_that_do_not_fit_func3(self) -> None:
        # A wider id would overwrite the rs1 field; never emit it.
        text = render_cfu_intrinsics(
            {
                "identity": {"name": "wavg"},
                "commands": [{"function_id": 8, "name": "compute", "semantics": "mac"}],
            }
        )
        self.assertNotIn("(8 << 12)", text)
        self.assertNotIn("cfu_compute", text)


class PrepareWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "embench-prep-00000000"
        self.workdir = temporary_embench_repo(self)
        self.run_dir = self.workdir / f"agents/generated/runs/{self.run_id}"

    def test_prepare_copies_benchmark_support_and_board(self) -> None:
        toolbox = RepoToolbox(
            self.workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{self.run_id}",
        )
        project = embench.prepare_embench_benchmark(
            toolbox, {"benchmark_name": "crc32"}
        )
        ws = self.workdir / project["workspace"]
        for config in ("reference", "profile", "cfu"):
            self.assertTrue((ws / config / "src/crc_32.c").exists(), config)
            self.assertTrue((ws / config / "support/main.c").exists(), config)
            self.assertTrue((ws / config / "board/boardsupport.c").exists(), config)
            self.assertTrue((ws / config / "embench_main.c").exists(), config)
        self.assertEqual(project["verification"], "bench_verify")
        self.assertEqual(set(project["elf_paths"]), {"reference", "profile", "cfu"})

    def test_generated_main_c_carries_the_verify_diagnostic(self) -> None:
        toolbox = RepoToolbox(
            self.workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{self.run_id}",
        )
        project = embench.prepare_embench_benchmark(toolbox, {"benchmark_name": "crc32"})
        main = (self.workdir / project["workspace"] / "reference/embench_main.c").read_text()
        self.assertIn("EMBENCH_VERIFY_", main)
        self.assertIn("EMBENCH_STRINGIFY", main)
        # Embench verify_benchmark() returns non-zero when correct, so a PASS is
        # `correct` non-zero (main returns !correct and start.S passes on 0).
        self.assertIn('correct ? "PASS" : "FAIL"', main)

    def test_main_instrumentation_is_a_noop_without_the_verify_call(self) -> None:
        self.assertEqual(embench.instrument_embench_main("int main(void){return 0;}\n"), (
            "int main(void){return 0;}\n"
        ))
        injected = embench.instrument_embench_main(
            "  correct = verify_benchmark (result);\n  return (!correct);\n"
        )
        self.assertIn("EMBENCH_VERIFY_", injected)

    def test_unknown_benchmark_rejected(self) -> None:
        toolbox = RepoToolbox(
            self.workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{self.run_id}",
        )
        with self.assertRaises(ValueError):
            embench.prepare_embench_benchmark(toolbox, {"benchmark_name": "does-not-exist"})

class EmbenchHotspotTest(unittest.TestCase):
    def test_selected_loop_keeps_source_function(self) -> None:
        from unittest import mock
        from agents.cfu_design_agent import graph

        source = {"workload_spec": {"name": "matmult-int", "kernel_function": "main"},
                  "profile_result": {"hot_spot": "obs_loop_x_Multiply_3", "regions": [],
                                     "hot_spot_point": {"kind": "loop", "id": "obs_loop_x_Multiply_3",
                                                        "function": "Multiply", "file": "matmult-int.c",
                                                        "line": 153, "ordinal": 3, "loop_kind": "for"}},
                  "benchmark_project": {"candidate_hotspots": ["Multiply"]}, "gates": {}}
        with mock.patch.object(services, "_write_hotspot_artifact"):
            updated = profile.extract_hotspot(source)
        self.assertEqual(updated["workload_spec"]["kernel_function"], "Multiply")
        self.assertEqual(updated["workload_spec"]["hot_loop"]["line"], 153)
        self.assertEqual(updated["workload_spec"]["hot_spot"], "obs_loop_x_Multiply_3")
        self.assertIn("hot_loop:", updated["spec_text"])

    def test_instrumented_config_is_preferred(self) -> None:
        from agents.cfu_design_agent import graph

        profiles = {
            "reference": {"crc32": {"cycles": 100}},
            "profile": {"crc32": {"cycles": 100}, "crc32pseudo": {"cycles": 90, "calls": 2}},
        }
        instrumented = profile.embench_instrumented_profile(profiles)
        self.assertIn("crc32pseudo", instrumented)
        self.assertEqual(
            embench.hotspot_from_profile(instrumented, exclude=("crc32",)), "crc32pseudo"
        )

    def test_falls_back_to_reference(self) -> None:
        from agents.cfu_design_agent import graph

        self.assertEqual(
            profile.embench_instrumented_profile({"reference": {"crc32": {"cycles": 1}}}),
            {"crc32": {"cycles": 1}},
        )

class EmbenchPatchRedirectTest(unittest.TestCase):
    def test_benchmark_source_patch_redirects_to_cfu_tree(self) -> None:
        from agents.cfu_design_agent.runtime.tools import repair_patch_paths

        patch = (
            "--- a/benchmarks/embench-iot/src/crc32/crc_32.c\n"
            "+++ b/benchmarks/embench-iot/src/crc32/crc_32.c\n"
            "@@ -1,1 +1,1 @@\n-x\n+y\n"
        )
        rewritten, redirected, rejected = repair_patch_paths(
            patch, "agents/generated/runs/r1", embench_benchmark="crc32"
        )
        self.assertEqual(rejected, [])
        expected = "agents/generated/runs/r1/workspace/embench_crc32/cfu/src/crc_32.c"
        self.assertIn(expected, redirected)
        self.assertIn("+++ b/" + expected, rewritten)

    def test_other_benchmark_source_is_rejected(self) -> None:
        from agents.cfu_design_agent.runtime.tools import repair_patch_paths

        patch = (
            "--- a/benchmarks/embench-iot/src/md5sum/md5.c\n"
            "+++ b/benchmarks/embench-iot/src/md5sum/md5.c\n"
        )
        _, _, rejected = repair_patch_paths(
            patch, "agents/generated/runs/r1", embench_benchmark="crc32"
        )
        self.assertTrue(rejected)

    def test_cfu_tree_is_writable(self) -> None:
        policy = load_policy(REPO_ROOT)
        self.assertTrue(
            policy.can_write(
                "implementation",
                "agents/generated/runs/r1/workspace/embench_crc32/cfu/src/crc_32.c",
            )
        )

    def test_apply_patch_targets_cfu_copy(self) -> None:
        from agents.cfu_design_agent.runtime.tools import RepoToolbox

        run_id = "embench-cfu-patch-0001"
        workdir = temporary_embench_repo(self)
        prep = RepoToolbox(
            workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{run_id}",
        )
        embench.prepare_embench_benchmark(prep, {"benchmark_name": "crc32"})
        cfu = workdir / f"agents/generated/runs/{run_id}/workspace/embench_crc32/cfu/src/crc_32.c"
        original_text = cfu.read_text()
        target = "#define LOCAL_SCALE_FACTOR 1"
        self.assertIn(target, original_text)
        idx = original_text.splitlines().index(target)
        context = original_text.splitlines()[idx - 1 : idx + 2]
        patch = "\n".join(
            [
                "--- a/benchmarks/embench-iot/src/crc32/crc_32.c",
                "+++ b/benchmarks/embench-iot/src/crc32/crc_32.c",
                f"@@ -{idx},3 +{idx},3 @@",
                " " + context[0],
                "-" + context[1],
                "+" + context[1] + " /* agent kernel target */",
                " " + context[2],
                "",
            ]
        )
        toolbox = RepoToolbox(
            workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="implementation",
            run_root=f"agents/generated/runs/{run_id}",
            embench_benchmark="crc32",
            apply_patch=True,
        )
        result = toolbox.apply_patch_text(patch)
        self.assertEqual(result["returncode"], 0, result.get("stderr_tail"))
        self.assertIn("agent kernel target", cfu.read_text())
        original = REPO_ROOT / "benchmarks/embench-iot/src/crc32/crc_32.c"
        self.assertNotIn("agent kernel target", original.read_text())

class FileBlockProtocolTest(unittest.TestCase):
    def test_extract_file_blocks(self) -> None:
        from agents.cfu_design_agent.runtime.tools import extract_file_blocks

        text = "chat\nFILE_BEGIN benchmarks/embench-iot/src/crc32/crc_32.c\nint x;\nFILE_END\ntail"
        blocks = extract_file_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], "benchmarks/embench-iot/src/crc32/crc_32.c")
        self.assertEqual(blocks[0][1], "int x;\n")

    def test_file_block_redirect_and_write(self) -> None:
        from agents.cfu_design_agent.runtime.tools import RepoToolbox

        run_id = "embench-fileblock-0001"
        workdir = temporary_embench_repo(self)
        prep = RepoToolbox(
            workdir, policy=load_policy(REPO_ROOT), dry_run=False,
            stage="prepare_benchmark", run_root=f"agents/generated/runs/{run_id}"
        )
        embench.prepare_embench_benchmark(prep, {"benchmark_name": "crc32"})
        toolbox = RepoToolbox(
            workdir,
            policy=load_policy(REPO_ROOT),
            dry_run=False,
            stage="implementation",
            run_root=f"agents/generated/runs/{run_id}",
            embench_benchmark="crc32",
        )
        resolved = toolbox.resolve_target("benchmarks/embench-iot/src/crc32/crc_32.c")
        self.assertEqual(
            resolved, f"agents/generated/runs/{run_id}/workspace/embench_crc32/cfu/src/crc_32.c"
        )
        with self.assertRaises(ValueError):
            toolbox.resolve_target("src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala")


if __name__ == "__main__":
    unittest.main()
