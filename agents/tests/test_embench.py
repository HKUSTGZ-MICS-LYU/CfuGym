"""Embench workload support: discovery, instrumentation, commands, policy."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent import embench, embench_instrument
from agents.cfu_design_agent.isa import render_cfu_intrinsics
from agents.cfu_design_agent.policy import PolicyError, load_policy
from agents.cfu_design_agent.spec import normalize_workload_spec, validate_workload_spec
from agents.cfu_design_agent.tools import RepoToolbox, infer_command_id
from agents.tests.common import REPO_ROOT


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
        self.assertEqual(applied, [])
        self.assertEqual(twice.count("EMBENCH_OBS_ENTER(helper)"), 1)
        self.assertTrue(any("already" in item["reason"] for item in skipped))

    def test_loop_points_degrade_with_reason(self) -> None:
        _, applied, skipped = embench_instrument.instrument_text(
            FIXTURE,
            [{"function": "benchmark_body", "id": "loop1", "kind": "loop"}],
            rel_path="benchmark.c",
        )
        self.assertEqual(applied, [])
        self.assertTrue(any("loop-level" in item["reason"] for item in skipped))

    def test_missing_definition_reported(self) -> None:
        _, applied, skipped = embench_instrument.instrument_text(
            FIXTURE, [{"function": "nope", "id": "nope"}], rel_path="benchmark.c"
        )
        self.assertEqual(applied, [])
        self.assertTrue(any("not found" in item["reason"] for item in skipped))

    def test_default_plan_finds_loops(self) -> None:
        points = embench_instrument.default_plan({"benchmark.c": FIXTURE})
        self.assertIn("benchmark_body", [p["function"] for p in points])


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
        # Each configuration owns its source tree so objects never collide.
        self.assertIn("/profile/embench_main", joined)
        self.assertIn("/profile/src/*.c", joined)
        for token in cmd:
            self.assertFalse(token.startswith("/"), token)

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


class EmbenchPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(REPO_ROOT)

    def test_build_command_is_allowlisted(self) -> None:
        cmd = embench.build_command("agents/generated/runs/r1", "crc32", "reference")
        self.assertEqual(infer_command_id(cmd, self.policy), "embench_build")
        self.policy.check_command("profile", "embench_build", cmd, REPO_ROOT)

    def test_embench_tree_is_readable_but_not_writable(self) -> None:
        src = "benchmarks/embench-iot/src/crc32/crc_32.c"
        self.assertTrue(self.policy.can_read("prepare_benchmark", src))
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
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def test_prepare_copies_benchmark_support_and_board(self) -> None:
        toolbox = RepoToolbox(
            REPO_ROOT,
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{self.run_id}",
        )
        project = embench.prepare_embench_benchmark(
            toolbox, {"benchmark_name": "crc32"}
        )
        ws = REPO_ROOT / project["workspace"]
        for config in ("reference", "profile", "cfu"):
            self.assertTrue((ws / config / "src/crc_32.c").exists(), config)
            self.assertTrue((ws / config / "support/main.c").exists(), config)
            self.assertTrue((ws / config / "board/boardsupport.c").exists(), config)
            self.assertTrue((ws / config / "embench_main.c").exists(), config)
        self.assertEqual(project["verification"], "bench_verify")
        self.assertEqual(set(project["elf_paths"]), {"reference", "profile", "cfu"})

    def test_unknown_benchmark_rejected(self) -> None:
        toolbox = RepoToolbox(
            REPO_ROOT,
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{self.run_id}",
        )
        with self.assertRaises(ValueError):
            embench.prepare_embench_benchmark(toolbox, {"benchmark_name": "does-not-exist"})

class EmbenchHotspotTest(unittest.TestCase):
    def test_instrumented_config_is_preferred(self) -> None:
        from agents.cfu_design_agent import graph

        profiles = {
            "reference": {"crc32": {"cycles": 100}},
            "profile": {"crc32": {"cycles": 100}, "crc32pseudo": {"cycles": 90, "calls": 2}},
        }
        instrumented = graph.embench_instrumented_profile(profiles)
        self.assertIn("crc32pseudo", instrumented)
        self.assertEqual(
            embench.hotspot_from_profile(instrumented, exclude=("crc32",)), "crc32pseudo"
        )

    def test_falls_back_to_reference(self) -> None:
        from agents.cfu_design_agent import graph

        self.assertEqual(
            graph.embench_instrumented_profile({"reference": {"crc32": {"cycles": 1}}}),
            {"crc32": {"cycles": 1}},
        )

class EmbenchPatchRedirectTest(unittest.TestCase):
    def test_benchmark_source_patch_redirects_to_cfu_tree(self) -> None:
        from agents.cfu_design_agent.tools import repair_patch_paths

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
        from agents.cfu_design_agent.tools import repair_patch_paths

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
        from agents.cfu_design_agent.tools import RepoToolbox

        run_id = "embench-cfu-patch-0001"
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        prep = RepoToolbox(
            REPO_ROOT,
            dry_run=False,
            stage="prepare_benchmark",
            run_root=f"agents/generated/runs/{run_id}",
        )
        embench.prepare_embench_benchmark(prep, {"benchmark_name": "crc32"})
        cfu = REPO_ROOT / f"agents/generated/runs/{run_id}/workspace/embench_crc32/cfu/src/crc_32.c"
        original_text = cfu.read_text()
        target = "#define LOCAL_SCALE_FACTOR 170"
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
            REPO_ROOT,
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
        from agents.cfu_design_agent.tools import extract_file_blocks

        text = "chat\nFILE_BEGIN benchmarks/embench-iot/src/crc32/crc_32.c\nint x;\nFILE_END\ntail"
        blocks = extract_file_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], "benchmarks/embench-iot/src/crc32/crc_32.c")
        self.assertEqual(blocks[0][1], "int x;\n")

    def test_file_block_redirect_and_write(self) -> None:
        from agents.cfu_design_agent.tools import RepoToolbox

        run_id = "embench-fileblock-0001"
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        prep = RepoToolbox(
            REPO_ROOT, dry_run=False, stage="prepare_benchmark", run_root=f"agents/generated/runs/{run_id}"
        )
        embench.prepare_embench_benchmark(prep, {"benchmark_name": "crc32"})
        toolbox = RepoToolbox(
            REPO_ROOT,
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
