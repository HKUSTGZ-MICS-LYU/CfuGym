"""The conformance oracle may reuse the workload's pristine scalar implementation.

The CFU's semantics are a transcription of the workload's scalar code, so the
oracle can `#include` that code instead of copying its constant tables by hand.
Regression (run 20261005-073745): a hand-copied 64-entry MD5 K table held 63
entries, and the rung failed *correct* hardware for three consecutive rounds.

These tests pin the plumbing (include dirs, benchmark defines, available sources)
and that a non-Embench run is offered nothing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.runtime import commands as commands_mod


class ScalarReferencePlumbingTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_root = "agents/generated/runs/unit-0001"
        reference = self.root / self.run_root / "workspace" / "embench_crc32" / "reference"
        (reference / "src").mkdir(parents=True)
        (reference / "support").mkdir()
        (reference / "board").mkdir()
        (reference / "src" / "crc32.c").write_text("int crc32_word(int x) { return x; }\n")
        (reference / "support" / "beebsc.c").write_text("int beebs_heap;\n")
        (reference / "support" / "main.c").write_text("int main(void) { return 0; }\n")
        (reference / "board" / "boardsupport.c").write_text("int board;\n")
        self.header = f"{self.run_root}/workspace/embench_crc32/cfu/include/cfu_intrinsics.h"
        self.state = {
            "workdir": str(self.root),
            "run_root": self.run_root,
            "benchmark_project": {"benchmark": "crc32"},
        }

    def test_scalar_sources_include_the_allocator_but_never_main(self) -> None:
        sources = commands_mod.conformance_scalar_sources(self.state)
        names = [Path(item).name for item in sources]
        self.assertIn("crc32.c", names)
        self.assertIn("beebsc.c", names)
        self.assertNotIn("main.c", names)

    def test_include_dirs_cover_the_reference_tree(self) -> None:
        includes = commands_mod.conformance_scalar_includes(self.state)
        for sub in ("src", "support", "board"):
            self.assertIn(f"{self.run_root}/workspace/embench_crc32/reference/{sub}", includes)
        self.assertEqual(commands_mod.conformance_scalar_defines(self.state)[:1],
                         ["-DHAVE_BOARDSUPPORT_H"])

    def test_both_command_builders_carry_the_scalar_tree(self) -> None:
        syntax = commands_mod._conformance_syntax_command(self.state, self.header)
        self.assertIn(f"{self.run_root}/workspace/embench_crc32/reference/src", syntax)
        self.assertIn("-DEMBENCH_NAME=crc32", syntax)
        self.assertIn("-std=gnu11", syntax)
        argv, _elf = commands_mod._conformance_build_command(self.state, self.header)
        extra_includes = next(item for item in argv if item.startswith("EXTRA_INCLUDES="))
        # The build runs with `-C sw`, so its include paths must be sw-relative.
        self.assertIn("../agents/generated/runs/unit-0001/workspace/embench_crc32/reference/src",
                      extra_includes)
        extra_cflags = next(item for item in argv if item.startswith("EXTRA_CFLAGS="))
        self.assertIn("-DGLOBAL_SCALE_FACTOR=", extra_cflags)
        self.assertIn("-std=gnu11", extra_cflags)

    def test_a_non_benchmark_run_is_offered_nothing(self) -> None:
        bare = {"workdir": str(self.root), "run_root": self.run_root, "benchmark_project": {}}
        self.assertEqual(commands_mod.conformance_scalar_sources(bare), [])
        self.assertEqual(commands_mod.conformance_scalar_includes(bare), [])
        self.assertEqual(commands_mod.conformance_scalar_defines(bare), [])
        argv, _elf = commands_mod._conformance_build_command(bare, self.header)
        extra_includes = next(item for item in argv if item.startswith("EXTRA_INCLUDES="))
        self.assertNotIn("reference", extra_includes)
        self.assertFalse(any(item.startswith("EXTRA_CFLAGS=") for item in argv))

    def test_prompt_offers_the_scalar_tree_and_the_allocator_recipe(self) -> None:
        from agents.cfu_design_agent import prompts
        from agents.tests.test_conformance import CPU_COMMAND, isa

        prompt = prompts.conformance_prompt(
            "task", isa(CPU_COMMAND), {}, scalar_sources=["crc32.c", "beebsc.c"])
        self.assertIn("PRISTINE SCALAR REFERENCE AVAILABLE", prompt)
        self.assertIn("crc32.c", prompt)
        self.assertIn("init_heap_beebs", prompt)
        self.assertIn("Never include `support/main.c`", prompt)
        without = prompts.conformance_prompt("task", isa(CPU_COMMAND), {})
        self.assertNotIn("PRISTINE SCALAR REFERENCE", without)


if __name__ == "__main__":
    unittest.main()
