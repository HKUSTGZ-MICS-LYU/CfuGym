"""Isolated, no-build tests of read-only P0 Embench LLVM IR export."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.cfu_design_agent.ir import export_embench_ir
from agents.cfu_design_agent.ir.export import _approx_headers, _clang_argv, _execute

RUN = "agents/generated/runs/unit-test"
NAME = "crc32"


class IrExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "sw" / "MiCo-Lib/targets").mkdir(parents=True)
        (self.root / "sw/Makefile").write_text("# fixture\n")
        (self.root / "sw/MiCo-Lib/targets/common.mk").write_text("# fixture\n")
        (self.root / "sw/MiCo-Lib/targets/vexii_soc.mk").write_text(
            "MARCH_BASE = $(shell echo $(MARCH) | sed 's/_.*$$//')\n"
        )
        self.cfg = self.root / RUN / "workspace/embench_crc32/reference/src"
        self.cfg.mkdir(parents=True)
        self.source = self.cfg / "crc_32.c"
        self.source.write_text("int value(void) { return 4; }\n")

    def test_off_never_probes_inputs_or_executes(self) -> None:
        with patch("agents.cfu_design_agent.ir.export._execute", side_effect=AssertionError("executed")):
            summary, files = export_embench_ir(self.root, "not/a/run", "!invalid", "off")
        self.assertEqual((summary["status"], files), ("off", {}))

    def test_invalid_source_or_symlink_fails_before_subprocess(self) -> None:
        with patch("agents.cfu_design_agent.ir.export._execute", side_effect=AssertionError("executed")):
            summary, files = export_embench_ir(self.root, "../../escape", NAME, "require")
            self.assertEqual((summary["status"], files), ("failed", {}))
            self.source.unlink()
            self.source.symlink_to(self.root / "sw" / "outside.c")
            summary, files = export_embench_ir(self.root, RUN, NAME, "auto")
        self.assertEqual((summary["status"], files), ("unavailable", {}))

    def test_missing_clang_auto_and_require(self) -> None:
        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value=None):
            auto, files = export_embench_ir(self.root, RUN, NAME, "auto")
            required, _ = export_embench_ir(self.root, RUN, NAME, "require")
        self.assertEqual(auto["status"], "unavailable")
        self.assertEqual(required["status"], "failed")
        self.assertIn("sources", auto["inputs"])
        self.assertIn("clang", auto["reason"])
        self.assertFalse(files)

    def _fake_run(self, argv, *, cwd, limit):
        record = {"argv": argv, "cwd": str(cwd), "returncode": 0, "stderr": ""}
        if "--version" in argv:
            record["stdout"] = b"clang test version 99\n"
        elif Path(argv[0]).name == "make":
            src = "../" + self.source.relative_to(self.root).as_posix()
            record["stdout"] = (
                "riscv64-unknown-elf-gcc -O3 -march=rv32imc_zicsr_zifencei "
                "-mabi=ilp32 -DHAVE_BOARDSUPPORT_H -I./ -c -o obj.o " + src + "\n"
            ).encode()
        else:
            self.assertEqual(argv[1:9], ["-target", "riscv32-unknown-elf",
                              "-march=rv32imc_zicsr_zifencei", "-mabi=ilp32",
                              "-O1", "-g", "-S", "-emit-llvm"])
            self.assertEqual(argv[-1], str(self.source))
            record["stdout"] = b"; ModuleID = 'crc_32.c'\nsource_filename = \"crc_32.c\"\n"
        return record

    def test_artifacts_are_returned_not_written_with_audit(self) -> None:
        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=self._fake_run):
            summary, files = export_embench_ir(self.root, RUN, NAME, "require")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(list(files), ["07_analysis/ir/crc_32.ll"])
        self.assertEqual(len(summary["commands"]), 3)
        self.assertEqual(summary["units"][0]["status"], "ok")
        self.assertEqual(len(summary["units"][0]["ir_sha256"]), 64)
        self.assertFalse((self.cfg / "crc_32.ll").exists())

    def test_clang_failure_is_auditable_auto_vs_require(self) -> None:
        def failing(argv, *, cwd, limit):
            result = self._fake_run(argv, cwd=cwd, limit=limit)
            if "-emit-llvm" in argv:
                result.update(returncode=1, stderr="unsupported target extension", stdout=b"")
            return result
        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=failing):
            auto, files = export_embench_ir(self.root, RUN, NAME, "auto")
            required, _ = export_embench_ir(self.root, RUN, NAME, "require")
        self.assertEqual(auto["status"], "unavailable")
        self.assertEqual(required["status"], "failed")
        self.assertEqual(auto["units"][0]["returncode"], 1)
        self.assertIn("unsupported target", auto["units"][0]["stderr"])
        self.assertEqual(files, {})

    def test_approx_uses_explicit_arch_and_approved_headers(self) -> None:
        headers = ("/home/jzj/riscv/riscv64-unknown-elf/include",
                   "/home/jzj/riscv/lib/gcc/riscv64-unknown-elf/15.1.0/include")
        calls = []

        def approximate(argv, *, cwd, limit):
            calls.append(argv)
            if "-emit-llvm" in argv:
                self.assertIn("-march=rv32imc", argv)
                self.assertNotIn("-march=rv32imc_zicsr_zifencei", argv)
                for header in headers:
                    self.assertEqual(argv[argv.index(header) - 1], "-isystem")
                return {"argv": argv, "cwd": str(cwd), "returncode": 0, "stderr": "",
                        "stdout": b"; approximate LLVM IR\n"}
            return self._fake_run(argv, cwd=cwd, limit=limit)

        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._approx_headers", return_value=headers), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=approximate):
            summary, files = export_embench_ir(self.root, RUN, NAME, "approx")
        self.assertEqual(summary["status"], "approximate")
        self.assertEqual(summary["units"][0]["status"], "approximate")
        self.assertEqual(summary["confidence"], "approximate_static_ir")
        self.assertEqual(summary["requested_arch"], "rv32imc_zicsr_zifencei")
        self.assertEqual(summary["actual_arch"], "rv32imc")
        self.assertIs(summary["target_match"], False)
        self.assertIn("not faithful", summary["limitations"])
        self.assertEqual(summary["toolchain"]["gcc_headers"], list(headers))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(calls), 3)

    def test_approx_header_probe_failure_never_falls_back(self) -> None:
        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._approx_headers", side_effect=ValueError("unsafe GCC headers")), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=self._fake_run):
            summary, files = export_embench_ir(self.root, RUN, NAME, "approx")
        self.assertEqual((summary["status"], files), ("unavailable", {}))
        self.assertIn("unsafe GCC headers", summary["reason"])
        self.assertIs(summary["target_match"], False)

    def test_approx_failure_and_mixed_status(self) -> None:
        self.source.with_name("other.c").write_text("int other(void) { return 3; }\n")

        def mixed(argv, *, cwd, limit):
            if "-emit-llvm" in argv:
                record = {"argv": argv, "cwd": str(cwd), "returncode": 0,
                          "stderr": "", "stdout": b"; approximate LLVM IR\n"}
                if "other.c" in argv[-1]:
                    record.update(returncode=1, stderr="compile failed", stdout=b"")
                return record
            record = self._fake_run(argv, cwd=cwd, limit=limit)
            if Path(argv[0]).name == "make" and "other.c" in " ".join(argv):
                record["stdout"] = record["stdout"].replace(b"crc_32.c", b"other.c")
            return record

        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._approx_headers", return_value=("/trusted/include", "/trusted/internal")), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=mixed):
            partial, files = export_embench_ir(self.root, RUN, NAME, "approx")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(len(files), 1)
        self.assertIn("other.c", partial["reason"])

        def failed(argv, *, cwd, limit):
            record = mixed(argv, cwd=cwd, limit=limit)
            if "-emit-llvm" in argv:
                record.update(returncode=1, stderr="compile failed", stdout=b"")
            return record

        with patch("agents.cfu_design_agent.ir.export._discover_clang", return_value="/usr/bin/clang"), \
             patch("agents.cfu_design_agent.ir.export._approx_headers", return_value=("/trusted/include", "/trusted/internal")), \
             patch("agents.cfu_design_agent.ir.export._execute", side_effect=failed):
            missing, files = export_embench_ir(self.root, RUN, NAME, "approx")
        self.assertEqual((missing["status"], files), ("unavailable", {}))
        self.assertIn("compile failed", missing["reason"])

    def test_approved_header_probe_rejects_unsafe_output(self) -> None:
        with patch("agents.cfu_design_agent.ir.export.Path.is_file", return_value=True), \
             patch("agents.cfu_design_agent.ir.export.Path.is_symlink", return_value=False), \
             patch("agents.cfu_design_agent.ir.export.os.access", return_value=True), \
             patch("agents.cfu_design_agent.ir.export._execute", return_value={
                 "returncode": 0, "stdout": b"/tmp/unsafe\n", "stderr": "",
             }) as execute:
            with self.assertRaisesRegex(ValueError, "unsafe directory"):
                _approx_headers(self.root / "sw", [])
        self.assertEqual(execute.call_args.args[0],
                         ["/home/jzj/riscv/bin/riscv64-unknown-elf-gcc", "-print-sysroot"])

    def test_real_approved_gcc_headers_read_only_smoke(self) -> None:
        """Only run when this host's explicitly approved installed tools exist."""
        gcc = Path("/home/jzj/riscv/bin/riscv64-unknown-elf-gcc")
        clang = Path("/usr/bin/clang")
        if not gcc.is_file() or not clang.is_file():
            self.skipTest("approved RISC-V GCC or trusted Clang not installed")
        audit = []
        headers = _approx_headers(self.root / "sw", audit)
        self.assertEqual(len(audit), 2)
        self.assertEqual(headers, (
            "/home/jzj/riscv/riscv64-unknown-elf/include",
            "/home/jzj/riscv/lib/gcc/riscv64-unknown-elf/15.1.0/include",
        ))
        # A source from the temporary fixture and -o - keep output in memory.
        result = _execute([
            str(clang), "-target", "riscv32-unknown-elf", "-march=rv32imc",
            "-mabi=ilp32", "-O1", "-S", "-emit-llvm",
            "-isystem", headers[0], "-isystem", headers[1],
            "-include", "stdlib.h", "-o", "-", str(self.source),
        ], cwd=self.root / "sw", limit=4096)
        self.assertEqual(result["returncode"], 0, result["stderr"])
        self.assertIn(b"target triple = \"riscv32-unknown-unknown-elf\"", result["stdout"])
        self.assertFalse((self.cfg / "crc_32.ll").exists())

    def test_unrecognized_compile_switch_is_not_forwarded(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported compiler option"):
            _clang_argv(["gcc", "-Xclang", "-load", "/tmp/plugin", "-march=rv32imc_zicsr_zifencei",
                         "-mabi=ilp32"], self.source, self.root / "sw", "/usr/bin/clang")


if __name__ == "__main__":
    unittest.main()
