"""Sub-operation encoding and conservative append-only ISA compatibility."""
from copy import deepcopy
from pathlib import Path
import unittest

from agents.cfu_design_agent.isa import (
    command_kind, intrinsic_symbols, render_cfu_intrinsics, render_isa_markdown,
    validate_isa_extension, validate_isa_spec,
)

ROOT = Path(__file__).resolve().parents[2]


def spec():
    return {
        "spec_version": "1",
        "identity": {"name": "example", "opcode": "custom0", "function_id_width": 3},
        "commands": [{"name": "compute", "function_id": 1, "semantics": "identity", "operand": "rs1", "result": "rs1"}],
        "software": {"intrinsics_header": "cfu_intrinsics.h", "scalar_fallback": True},
        "integration": {"single_cpu_bus_owner": True},
        "datapath": {"xlen": 32},
    }


def extension():
    result = spec()
    result["commands"].append({"name": "compute_pair", "function_id": 1, "subfunction_id": 127, "kind": "compute", "semantics": "add inputs"})
    return result


class IsaEvolutionTest(unittest.TestCase):
    def test_legacy_and_extension_valid(self):
        self.assertEqual(validate_isa_spec(spec()), [])
        self.assertEqual(validate_isa_spec(extension()), [])
        self.assertEqual(validate_isa_extension(spec(), extension()), [])

    def test_subfunction_range_and_types(self):
        for value in (-1, 128, True, "1", None, [], {}):
            with self.subTest(value=value):
                data = spec()
                data["commands"][0]["subfunction_id"] = value
                self.assertTrue(any("subfunction_id" in e for e in validate_isa_spec(data)))

    def test_duplicate_default_and_explicit_zero(self):
        data = spec()
        data["commands"].append({**data["commands"][0], "name": "other", "subfunction_id": 0})
        self.assertTrue(any("encoding duplicates" in e for e in validate_isa_spec(data)))

    def test_kind_defaults_and_validation(self):
        for func, kind in ((1, "compute"), (2, "control"), (4, "load"), (5, "store"), (7, "compute")):
            self.assertEqual(command_kind({"function_id": func}), kind)
        data = spec()
        for kind in ("compute", "control", "load", "store"):
            data["commands"][0]["kind"] = kind
            self.assertEqual(validate_isa_spec(data), [])
        for kind in ("config", None, [], 7):
            data["commands"][0]["kind"] = kind
            self.assertTrue(any(".kind" in e for e in validate_isa_spec(data)))

    def test_symbol_name_and_macro_collisions(self):
        for first, second in (("compute", "compute"), ("a-b", "a_b"), ("compute", "COMPUTE")):
            data = extension()
            data["commands"][0]["name"] = first
            data["commands"][1]["name"] = second
            self.assertTrue(any("duplicates" in e or "conflicts" in e for e in validate_isa_spec(data)))
        for name in ("enable", "memory-fence", "agent_counter_read", "agent_counter_elapsed"):
            data = spec()
            data["commands"][0]["name"] = name
            self.assertTrue(any("reserved helper" in e for e in validate_isa_spec(data)))
        for name in (None, 123, " ", "!!!", "计算"):
            data = spec()
            data["commands"][0]["name"] = name
            self.assertTrue(any(".name" in e for e in validate_isa_spec(data)))

    def test_encoding_and_markdown(self):
        data = extension()
        header = render_cfu_intrinsics(data)
        self.assertIn('(1 << 12) | (127 << 25)', header)
        self.assertIn('(12 << 20) | (1 << 12))', header)
        self.assertIn('#define CFU_COMPUTE_PAIR_SUBFUNCTION_ID 127', header)
        self.assertEqual(intrinsic_symbols(data), ["cfu_compute", "cfu_compute_pair"])
        word = 0x0B | (10 << 7) | (1 << 12) | (11 << 15) | (12 << 20) | (127 << 25)
        self.assertEqual((word >> 25) & 127, 127)
        self.assertEqual((word >> 15) & 31, 11)
        self.assertEqual((word >> 20) & 31, 12)
        self.assertIn('| 1 | 127 | compute | compute_pair |', render_isa_markdown(data))

    def test_legacy_default_normalization_no_mutation(self):
        old = spec()
        candidate = deepcopy(old)
        candidate["identity"].update(subfunction_id_width=7, opcode="custom0 (0x0B)", status="validated")
        candidate["commands"][0].update(subfunction_id=0, kind="compute")
        before = deepcopy(old)
        self.assertEqual(validate_isa_extension(old, candidate), [])
        self.assertEqual(before, old)

    def test_old_command_fields_cannot_change(self):
        for field, value in (("name", "renamed"), ("function_id", 3), ("subfunction_id", 1), ("kind", "control"), ("operand", "rs2"), ("result", "rs2"), ("semantics", "changed"), ("new_semantic_metadata", 1)):
            with self.subTest(field=field):
                candidate = spec()
                candidate["commands"][0][field] = value
                self.assertTrue(validate_isa_extension(spec(), candidate))
        candidate = extension()
        candidate["commands"].pop(0)
        self.assertTrue(validate_isa_extension(spec(), candidate))

    def test_shared_contract_is_conservatively_frozen(self):
        for field in ("state", "memory", "datapath", "integration", "operations_contract", "assumptions"):
            candidate = spec()
            candidate[field] = {"changed": True}
            self.assertTrue(validate_isa_extension(spec(), candidate), field)
        candidate = spec()
        candidate["identity"]["function_id_width"] = 2
        self.assertTrue(validate_isa_extension(spec(), candidate))
        candidate = spec()
        candidate["software"]["intrinsics_header"] = "other.h"
        self.assertTrue(validate_isa_extension(spec(), candidate))

    def test_intrinsic_names_can_only_grow(self):
        old = spec()
        old["software"]["intrinsic_names"] = ["cfu_compute"]
        candidate = extension()
        candidate["software"]["intrinsic_names"] = ["cfu_compute", "cfu_compute_pair"]
        self.assertEqual(validate_isa_extension(old, candidate), [])
        candidate["software"]["intrinsic_names"] = []
        self.assertTrue(validate_isa_extension(old, candidate))

    def test_subfunction_identity_width(self):
        for value in (3, 8, True, "7", None):
            data = spec()
            data["identity"]["subfunction_id_width"] = value
            self.assertTrue(validate_isa_spec(data))

    def test_bad_container_shapes_return_errors(self):
        data = spec()
        data["software"] = None
        self.assertTrue(validate_isa_spec(data))
        self.assertTrue(validate_isa_extension(None, spec()))
        self.assertTrue(validate_isa_spec(None))
        for value in (None, 3, "wrong"):
            data = spec()
            data["validation"] = {"requirements": value}
            self.assertTrue(validate_isa_extension(spec(), data))
        data = spec()
        data["commands"][0]["function_id"] = []
        self.assertTrue(validate_isa_spec(data))

    def test_hardware_decode_static_contract(self):
        # Static integration guard, not a replacement for Scala elaboration/sim.
        # The read-only platform holds the framework; the agent-owned design file
        # holds only the instruction table.
        param = (ROOT / "src/main/scala/vexiiriscv/Param.scala").read_text()
        design = (ROOT / "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala").read_text()
        platform = (ROOT / "src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala").read_text()
        self.assertIn('instruction = M"-------------------------0001011"', param)
        self.assertIn('functionId = List(14 downto 12)', param)
        self.assertIn('val subOp = io.bus.cmd.raw_insn(31 downto 25).asUInt', platform)
        self.assertIn('subOp === U(operation, 7 bits)', platform)
        for family in ("compute", "config", "load", "store"):
            self.assertIn(f'Instruction("{family}", AgentCfuFunction.{family}', design)
        self.assertIn('protected def instructionTable: Seq[Instruction]', platform)
        self.assertIn('override protected def instructionTable: Seq[Instruction]', design)
        self.assertIn('matches(instruction.function, instruction.subop)', platform)
        self.assertIn('duplicate instruction encoding', platform)
        self.assertIn('when(!supported)', platform)
        self.assertIn('!responsePending && !handlerBusy && selectedReady', platform)
        # A whole-file rewrite of the design cannot drop the platform: the bus shell,
        # dispatcher and resource vocabulary must stay out of AgentCfu.scala.
        self.assertIn('abstract class AgentCfuBase(', platform)
        self.assertIn('object CfuResource {', platform)
        self.assertIn('sealed trait CfuResource', platform)
        for framework in ('io.bus.cmd.ready :=', 'enqueueResponse', 'CfuLsu(',
                          'val vectorRegs', 'def matches(', 'case class Instruction(',
                          'sealed trait CfuResource'):
            self.assertNotIn(framework, design, framework)

    def test_single_vector_slot_is_addressed_statically(self):
        # regDepth = 1 is a valid parameter, but SpinalHDL addresses a Vec with
        # log2Up(size) bits: a one-element Vec needs a zero-width address while the
        # slot index is one bit wide. The platform must special-case it, otherwise
        # elaboration fails with "Vec address width mismatch" for every vector
        # resource. Covered functionally by AgentCfuSingleSlotSim.
        platform = (ROOT / "src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala").read_text()
        self.assertIn("private val singleVectorSlot = p.regDepth == 1", platform)
        # The one-slot path uses a static index; the multi-slot path keeps a dynamic
        # one, but every design-supplied address goes through the checked accessor.
        for conditional in (
            "if (singleVectorSlot) vectorRegs(0) else vectorRegs(slotAddress(addr))",
            "if (singleVectorSlot) vectorRegs(0) := data else vectorRegs(slotAddress(addr)) := data",
            "if (singleVectorSlot) vectorRegs(0)(offset, p.xlen bits) := data",
        ):
            self.assertIn(conditional, platform)
        # An over-wide slot address is reported here instead of surfacing later as an
        # opaque backend "Vec address width mismatch".
        self.assertIn("private def slotAddress(addr: UInt): UInt", platform)
        self.assertIn("vector slot address has", platform)
        # The old unconditional dynamic read must be gone.
        self.assertNotIn("def vectorRead(addr: UInt): Bits = vectorRegs(addr)", platform)

    def test_platform_design_checks_carry_the_read_only_hint(self):
        # Every platform-enforced design check must be actionable from the design file
        # and name the read-only boundary, so a repair knows not to edit this file.
        platform = (ROOT / "src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala").read_text()
        self.assertIn("val READ_ONLY_HINT: String", platform)
        self.assertIn("read-only platform file", platform)
        self.assertIn(
            "protected def designRequire(condition: Boolean, message: => String): Unit", platform
        )
        for check in (
            "duplicate resource role",
            "at most one resource write destination",
            "undeclared resource read",
            "undeclared vector index",
            "vector result width mismatch",
            "constant vector index out of range",
            "defer requires a tick handler",
            "duplicate instruction encoding",
            "vector slot address has",
        ):
            self.assertIn(check, platform)
        # Design-facing checks route through designRequire, not a bare require.
        self.assertNotIn("require(rs.map(resourceRole).distinct.size == rs.size", platform)
        self.assertNotIn("require(declarations.forall(i => i.name.nonEmpty)", platform)


if __name__ == "__main__":
    unittest.main()
