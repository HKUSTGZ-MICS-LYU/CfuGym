"""Explicit CPU/vector resource contracts and register-aware C intrinsics."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from agents.cfu_design_agent.isa import (
    IsaLintError, render_cfu_intrinsics, render_isa_markdown,
    validate_cfu_resources, validate_isa_extension, validate_isa_spec,
)


def cpu(role, access="read"):
    return {"storage": "cpu", "access": access, "role": role}


def vector(role="vs1", source="cpu_value", register="rs1", value=0):
    return {"storage": "vector", "access": "write" if role == "vd" else "read", "role": role,
            "index": {"source": source, **({"value": value} if source == "constant" else {"register": register})}}


def spec(resources=None, explicit=True):
    command = {"name": "op", "function_id": 1, "subfunction_id": 3, "semantics": "test operation"}
    if explicit:
        command["resources"] = [] if resources is None else resources
    return {"spec_version": "1", "identity": {"name": "test", "opcode": "custom0", "function_id_width": 3},
            "commands": [command], "datapath": {"regDepth": 2}, "integration": {"hardware_component": "AgentCfu"},
            "software": {"intrinsics_header": "cfu.h", "scalar_fallback": True}, "validation": {}}


class CfuResourcesTest(unittest.TestCase):
    def test_valid_resource_combinations(self):
        for resources in ([], [cpu("rs1"), cpu("rs2"), cpu("rd", "write")],
                          [vector(), vector("vs2", register="rs2"), cpu("rd", "write")],
                          [vector(), vector("vd", source="constant", value=1)],
                          [vector(source="legacy_register_number", register="rs2")]):
            self.assertEqual(validate_isa_spec(spec(resources)), [])

    def test_malformed_shapes_and_enums_never_throw(self):
        for resources in (None, {}, 1, True, "cpu", [None], [[]], [True]):
            self.assertTrue(validate_cfu_resources(resources))
        base = cpu("rs1")
        for key in ("storage", "access", "role"):
            for invalid in ([], {}, None, True, 1, "unknown"):
                item = {**base, key: invalid}
                self.assertTrue(validate_cfu_resources([item]), item)
        for key in ("storage", "access", "role"):
            item = dict(base)
            del item[key]
            self.assertTrue(validate_cfu_resources([item]))
        self.assertTrue(validate_cfu_resources([{**base, "extra": 1}]))
        self.assertTrue(validate_cfu_resources([{**base, "index": {}}]))
        for index in (None, [], True, {}, {"source": []}, {"source": "cpu_value", "register": []},
                      {"source": "cpu_value", "register": "rs1", "value": 1},
                      {"source": "constant", "value": True}, {"source": "constant", "value": -1},
                      {"source": "constant", "value": []}, {"source": "legacy_register_number", "register": "rd"}):
            self.assertTrue(validate_cfu_resources([{**vector(), "index": index}]), index)

    def test_duplicate_roles_and_dual_write(self):
        for resources in ([cpu("rs1"), cpu("rs1")], [vector(), vector()],
                          [cpu("rd", "write"), vector("vd")], [vector("vd"), vector("vd")]):
            self.assertTrue(validate_cfu_resources(resources))
        self.assertEqual(validate_cfu_resources([cpu("rs1"), vector()]), [])

    def test_constant_bounds_and_legacy_truncation(self):
        self.assertTrue(validate_cfu_resources([vector(source="constant", value=2)], {"regDepth": 2}))
        self.assertTrue(validate_cfu_resources([vector(source="constant", value=3)], {"regDepth": 3}))
        self.assertEqual(validate_cfu_resources([vector(source="constant", value=2)], {"regDepth": 3}), [])
        for depth in (None, {}, {"regDepth": 0}, {"regDepth": True}, {"regDepth": "unused"}):
            self.assertEqual(validate_cfu_resources([vector(source="constant", value=100)], depth), [])
        self.assertEqual(validate_cfu_resources([vector(source="legacy_register_number", register="rs2")], {"regDepth": 2}), [])

    def test_legacy_intrinsic_body_is_byte_exact(self):
        header = render_cfu_intrinsics(spec(explicit=False))
        expected = '''static inline uint32_t cfu_op(uint32_t rs1, uint32_t rs2)
{
  register uint32_t rd asm("a0") = 0;
  register uint32_t r1 asm("a1") = rs1;
  register uint32_t r2 asm("a2") = rs2;
  __asm__ volatile(
      ".word ((0x0B) | (10 << 7) | (11 << 15) | (12 << 20) | (1 << 12) | (3 << 25))"
      : "+r"(rd) : "r"(r1), "r"(r2) : "memory");
  return rd;
}
'''
        self.assertIn(expected, header)

    def test_resource_intrinsic_parameters_and_destinations(self):
        header = render_cfu_intrinsics(spec([vector(register="rs2"), cpu("rd", "write")]))
        self.assertIn("uint32_t cfu_op(uint32_t rs2)", header)
        self.assertIn("(10 << 7) | (0 << 15) | (12 << 20)", header)
        self.assertNotIn('r1 asm', header)
        self.assertIn("index = CPU RS2 value", header)
        header = render_cfu_intrinsics(spec([vector("vd", source="constant", value=1)]))
        self.assertIn("void cfu_op(void)", header)
        self.assertIn("(0 << 7) | (0 << 15) | (0 << 20)", header)
        self.assertNotIn("return rd", header)
        self.assertIn(':  :  : "memory"', header)
        header = render_cfu_intrinsics(spec([vector(source="legacy_register_number", register="rs1")]))
        self.assertIn("void cfu_op(void)", header)
        self.assertIn('r1 asm("a1") = 0;', header)
        self.assertIn('(0 << 7) | (11 << 15) | (0 << 20)', header)
        self.assertIn(':  : "r"(r1)', header)
        self.assertIn("void cfu_op(void)", render_cfu_intrinsics(spec([])))
        with self.assertRaises(IsaLintError):
            render_cfu_intrinsics(spec([cpu("rd", "write"), vector("vd")]))

    def test_markdown_resources_and_completion(self):
        text = render_isa_markdown(spec([vector(), cpu("rd", "write")]))
        self.assertIn("read vector VS1, index = CPU RS1 value (full-width bounds check)", text)
        self.assertIn("write CPU RD value", text)
        self.assertIn("result to CPU RD", text)
        self.assertIn("none (ack only)", render_isa_markdown(spec([])))
        self.assertIn("CPU rd=x0", render_isa_markdown(spec([])))
        self.assertIn("INVALID", render_isa_markdown(spec([{"storage": []}])))

    def test_extensions_freeze_presence_and_resource_bindings(self):
        old = spec(explicit=False)
        new = deepcopy(old)
        new["commands"][0]["resources"] = []
        self.assertTrue(validate_isa_extension(old, new))
        old = spec([vector()])
        new = deepcopy(old)
        new["commands"][0]["resources"][0]["index"]["register"] = "rs2"
        self.assertTrue(validate_isa_extension(old, new))
        new = deepcopy(old)
        new["commands"].append({"name": "next", "function_id": 2, "semantics": "ack", "resources": []})
        self.assertEqual(validate_isa_extension(old, new), [])

    def test_real_riscv_compilation_and_encoded_register_fields(self):
        gcc = shutil.which("riscv64-unknown-elf-gcc")
        objdump = shutil.which("riscv64-unknown-elf-objdump")
        if not gcc or not objdump:
            self.skipTest("RISC-V compiler/objdump unavailable")
        cases = [([], 0, 0, 0, ""),
                 ([vector(register="rs2"), cpu("rd", "write")], 10, 0, 12, "22"),
                 ([vector(source="legacy_register_number", register="rs1")], 0, 11, 0, ""),
                 ([cpu("rs1"), cpu("rs2"), vector("vd", source="constant", value=0)], 0, 11, 12, "11,22")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for n, (resources, rd, rs1, rs2, args) in enumerate(cases):
                (root / "cfu.h").write_text(render_cfu_intrinsics(spec(resources)))
                returns = rd != 0
                (root / "test.c").write_text('#include "cfu.h"\n' + ('unsigned' if returns else 'void') +
                    f' test(void) {{ {"return " if returns else ""}cfu_op({args}); }}\n')
                obj = root / f"test{n}.o"
                command = [gcc, "-march=rv32im", "-mabi=ilp32", "-O2", "-Wall", "-Werror", "-c", str(root / "test.c"), "-o", str(obj)]
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                dump = subprocess.run([objdump, "-d", str(obj)], text=True, capture_output=True)
                self.assertEqual(dump.returncode, 0, dump.stderr)
                word = 0x0b | rd << 7 | rs1 << 15 | rs2 << 20 | 1 << 12 | 3 << 25
                self.assertRegex(dump.stdout.lower(), rf"\b{word:08x}\b")


if __name__ == "__main__":
    unittest.main()
