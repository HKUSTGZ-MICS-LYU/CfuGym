"""Fixture-only static contract/actual-decode checks; no elaboration or sweep."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from agents.cfu_design_agent.runtime import instruction_encoding as enc, stage_services
from agents.cfu_design_agent.stages.verify import soc

PATH = 'workspace/design/AgentCfu.scala'


def table(*entries, preamble=''):
    return '''object AgentCfuFunction {
      val compute = 1
      val config = 2
      val load = 4
      val store = 5
    }
    ''' + preamble + '\nclass AgentCfu { override protected def instructionTable: Seq[Instruction] = Seq(\n' + ',\n'.join(entries) + '\n) }'


def instruction(name='op', function='AgentCfuFunction.compute', subop='1', enabled='true', body='ctx.complete(ctx.read(RS1))'):
    sub = '' if subop is None else ', subop = ' + subop
    return f'Instruction("{name}", {function}{sub}, enabled = {enabled}, start = ctx => {{ {body} }})'


def spec(*commands, memory=None):
    return {'identity': {'opcode': 'custom0'}, 'commands': list(commands) or [
        {'name': 'op', 'function_id': 1, 'subfunction_id': 1, 'kind': 'compute'}], 'memory': memory or {}}


def rtl(pairs=((1, 1),), *, complete=True, extras=''):
    lines = ['module AgentCfu;', 'assign subOp = io_bus_cmd_payload_raw_insn[31 : 25];']
    for i, (f, s) in enumerate(pairs):
        lines.append(f"assign selected_{i} = ((io_bus_cmd_payload_function_id == 3'd{f}) && (subOp == 7'h{s:x}));")
    if complete:
        selected = ' || '.join(f'selected_{i}' for i in range(len(pairs)))
        lines.append(f"assign unsupported = (! (1'b0 || {selected}));")
    return '\n'.join(lines + [extras, 'endmodule'])


class InstructionEncodingTest(TestCase):
    def source(self, text, contract=None):
        return enc.validate_sources({PATH: text}, contract or spec())

    def test_default_subop_zero_does_not_match_aha_contract_one(self):
        result = self.source(table(instruction(subop=None)))
        self.assertEqual(result['declarations'][0]['subfunction_id'], 0)
        self.assertIn('encoding_mismatch', result['problems'][0])
        self.assertFalse(result['checked'])

    def test_all_command_families_and_custom_pairs_are_covered(self):
        commands = [{'name': name, 'function_id': f, 'subfunction_id': s, 'kind': kind}
                    for name, f, s, kind in [('calc', 6, 19, 'compute'), ('cfg', 0, 12, 'control'),
                                             ('ld', 3, 77, 'load'), ('st', 7, 99, 'store')]]
        source = table(*(instruction(c['name'], str(c['function_id']), str(c['subfunction_id'])) for c in commands))
        result = self.source(source, spec(*commands))
        self.assertTrue(result['checked'])
        self.assertEqual(result['problems'], [])
        self.assertEqual(len(result['coverage']), 4)

    def test_range_duplicate_and_disabled_checks_include_disabled_entries(self):
        for entries, message in [
            ([instruction(function='8')], 'encoding_range'),
            ([instruction(function='-1')], 'encoding_range'),
            ([instruction(subop='128')], 'encoding_range'),
            ([instruction(subop='-1')], 'encoding_range'),
            ([instruction(), instruction(name='shadow', enabled='false')], 'duplicate_encoding'),
            ([instruction(enabled='false')], 'disabled_instruction'),
        ]:
            with self.subTest(message=message):
                self.assertTrue(any(message in p for p in self.source(table(*entries))['problems']))

    def test_enable_path_uses_effective_host_configuration(self):
        source = table(instruction(enabled='p.withTilelink && p.withLoad'))
        result = self.source(source, spec(memory={'load': False, 'store': False}))
        self.assertFalse(result['config']['withTilelink'])
        self.assertIn('disabled_instruction', result['problems'][0])
        source = table(instruction(body='lsu.cmd.load := True; ctx.complete(ctx.read(RS1))'))
        result = self.source(source, spec(memory={'load': False}))
        self.assertTrue(any('disabled_path' in p for p in result['problems']))

    def test_nonliteral_fields_are_unchecked_not_falsely_checked(self):
        for key, value in [('function', 'computeId'), ('subop', 'operationId'), ('enabled', 'p.someMode')]:
            result = self.source(table(instruction(**{key: value})))
            self.assertFalse(result['checked'])
            self.assertTrue(any(key in u for u in result['unchecked']))
        source = 'class AgentCfu { override protected def instructionTable: Seq[Instruction] = makeTable(p) }'
        self.assertFalse(self.source(source)['checked'])
        self.assertIn('nonliteral table expression', self.source(source)['unchecked'][0])

    def test_only_live_table_not_strings_comments_prototypes_or_old_template(self):
        prefix = '''// Instruction("op", 1, subop=0)
        val docs = "Instruction(\\"op\\", 1, subop=0)"
        val longDocs = """Instruction("op", 1, subop=0)"""
        case class Instruction(name: String, function: Int, subop: Int = 0)
        def oldTemplate = Seq(Instruction("op", 1, subop=0))
        '''
        result = self.source(table(instruction(), preamble=prefix))
        self.assertEqual(result['problems'], [])
        self.assertEqual(len(result['declarations']), 1)
        self.assertTrue(result['checked'])

    def test_agent_owned_symbol_literal_redefinition_is_not_assumed_default(self):
        source = table(instruction()).replace('val compute = 1', 'val compute = 6')
        self.assertIn('encoding_mismatch', self.source(source)['problems'][0])
        source = table(instruction()).replace('val compute = 1', 'val compute = calculateId()')
        self.assertFalse(self.source(source)['checked'])

    def test_empty_literal_table_and_dynamic_membership_are_not_falsely_checked(self):
        for source in [table(), 'class AgentCfu { override protected def instructionTable: Seq[Instruction] = Seq.empty }']:
            result = self.source(source)
            self.assertTrue(any('encoding_mismatch' in p for p in result['problems']))
        source = table(instruction()).replace('Seq(\nInstruction', 'Seq(\nexternalEntry, Instruction')
        result = self.source(source)
        self.assertFalse(result['checked'])
        self.assertTrue(any('nonliteral member' in u for u in result['unchecked']))
        source = table(instruction()).replace('Seq(\nInstruction', '{ if (p.flag) Seq(Instruction').replace('\n) }', ') else Seq.empty } }')
        self.assertFalse(self.source(source)['checked'])
        source = table(instruction()).replace('\n) }', '\n).filter(enabledInMode) }')
        self.assertFalse(self.source(source)['checked'])
        source = table(instruction()).replace('\n) }', '.copy(subop = operationId)\n) }')
        self.assertFalse(self.source(source)['checked'])

    def test_fiber_width_and_constructor_path_are_explicit_coverage(self):
        fiber = '''class AgentCfuFiber {
          val params = CfuBusParameter(CFU_FUNCTION_ID_W = 3, CFU_RAW_INSN_W = 32)
          val cfu = new AgentCfu(cfuParam, busParam, p)
          cfu.io.bus <> cfuBus
        }'''
        result = enc.validate_sources({PATH: table(instruction()), 'AgentCfuFiber.scala': fiber}, spec())
        self.assertTrue(result['fiber_path']['checked'])
        result = enc.validate_sources({PATH: table(instruction()), 'AgentCfuFiber.scala': fiber.replace('ID_W = 3', 'ID_W = 4')}, spec())
        self.assertTrue(any('bus_encoding_width' in p for p in result['problems']))
        result = enc.validate_sources({PATH: table(instruction()), 'AgentCfuFiber.scala': fiber.replace('busParam, p)', 'busParam, adjust(p))')}, spec())
        self.assertFalse(result['fiber_path']['checked'])
        result = enc.validate_sources({PATH: table(instruction()), 'AgentCfuFiber.scala': fiber.replace('val cfu = new', 'val cfu = if (false) new')}, spec())
        self.assertFalse(result['fiber_path']['checked'])

    def test_unsupported_contract_fields_explicitly_outside_encoding_scope(self):
        contract = spec({'name': 'op', 'function_id': 1, 'subfunction_id': 1, 'kind': 'config',
                         'result': 'status', 'unsupported_behavior': 'trap'})
        result = self.source(table(instruction()), contract)
        self.assertEqual(result['coverage'][0]['kind'], 'config')
        self.assertTrue(any('unsupported_behavior' in u for u in result['unchecked']))

    def test_generated_rtl_actual_decode_catches_aha_default_zero(self):
        result = enc.validate_rtl(rtl(((1, 0), (4, 0))), spec())
        self.assertEqual(result['actual_pairs'], [[1, 0], [4, 0]])
        self.assertEqual(result['unsupported_guard'], 'checked')
        self.assertIn('rtl_encoding_mismatch', result['problems'][0])
        result = enc.validate_rtl(rtl(), spec())
        self.assertTrue(result['checked'])

    def test_rtl_does_not_search_other_modules_comments_or_unknown_decoders(self):
        # Comments and other modules cannot provide an actual decoder.
        self.assertFalse(enc.validate_rtl('module Cpu; assign x = 1; endmodule', spec())['checked'])
        self.assertFalse(enc.validate_rtl('module AgentCfu; // compare funct7=1\nendmodule', spec())['checked'])
        self.assertFalse(enc.validate_rtl(rtl(((1, 0),), complete=False), spec())['checked'])
        self.assertEqual(enc.validate_rtl(rtl(((1, 0),), complete=False), spec())['problems'], [])
        self.assertFalse(enc.validate_rtl(rtl() + '\n' + rtl(), spec())['checked'])

    def test_compiled_machine_words_not_pretty_disassembly_comments(self):
        # custom0 funct3=1 funct7=1, and little-endian byte-rendered funct7=0.
        result = enc.validate_compiled_assembly('100: 0200100b .word 0x0200100b\n', spec(), rtl_evidence=enc.validate_rtl(rtl(), spec()))
        self.assertTrue(result['checked'])
        self.assertEqual(result['observed_pairs'], [[1, 1]])
        result = enc.validate_compiled_assembly('100: 0b 10 00 00 unknown\n', spec(), rtl_evidence=enc.validate_rtl(rtl(), spec()))
        self.assertEqual(result['observed_pairs'], [[1, 0]])
        self.assertTrue(any('compiled_contract_mismatch' in p for p in result['problems']))
        self.assertFalse(enc.validate_compiled_assembly('// .word 0x0200100b', spec())['checked'])

    def test_compiled_data_sections_are_not_instruction_words(self):
        evidence = enc.validate_rtl(rtl(), spec())
        for section in ('.data', '.sdata', '.rodata', '.debug_info'):
            with self.subTest(section=section):
                assembly = ('Disassembly of section .text:\n100: 0200100b unknown\n'
                            f'Disassembly of section {section}:\n200: 0000100b .word 0x100b\n')
                result = enc.validate_compiled_assembly(assembly, spec(), rtl_evidence=evidence)
                self.assertTrue(result['checked'])
                self.assertEqual(result['observed_pairs'], [[1, 1]])
                self.assertEqual(result['problems'], [])
        assembly = ('Disassembly of section .data:\n100: 0200100b .word 0x0200100b\n'
                    'Disassembly of section .text.startup:\n200: 0000100b unknown\n')
        result = enc.validate_compiled_assembly(assembly, spec(), rtl_evidence=evidence)
        self.assertEqual(result['observed_pairs'], [[1, 0]])
        self.assertTrue(any('compiled_contract_mismatch' in p for p in result['problems']))
        self.assertFalse(enc.validate_compiled_assembly('Disassembly of section .data:\n100: 0200100b .word 0x0200100b\n', spec(), rtl_evidence=evidence)['checked'])

    def test_live_helper_evidence_and_no_conformance_do_not_disable_static_gate(self):
        box = mock.Mock()
        box.read_text.side_effect = lambda p, **kw: table(instruction(subop=None)) if p.endswith('AgentCfu.scala') else '// fiber'
        state = {'run_root': 'unit', 'isa_spec': spec(), 'require_conformance': False}
        evidence = {}
        problems = stage_services.hardware_integrity_problems(state, box, evidence=evidence)
        self.assertTrue(any('encoding_mismatch' in p for p in problems))
        self.assertTrue(evidence['candidate_hash'])
        self.assertEqual(evidence['isa_hash'], enc.digest(spec()))
        self.assertEqual(evidence['config_hash'], enc.digest(evidence['config']))

    def test_soc_post_generation_uses_actual_rtl_and_binds_candidate(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'agents/generated/runs/unit/workspace/soc/MiCoSoc.v'
            target.parent.mkdir(parents=True)
            target.write_text(rtl(((1, 0),)))
            box = mock.Mock()
            box.workdir = Path(tmp)
            box.read_text.side_effect = lambda p, **kw: (table(instruction()) if p.endswith('AgentCfu.scala')
                                                       else target.read_text() if p.endswith('MiCoSoc.v') else '// fiber')
            state = {'workdir': tmp, 'run_root': 'agents/generated/runs/unit', 'isa_spec': spec(), 'max_iters': 1,
                     'gates': {'implementation_hw': {'ok': True}}, 'require_conformance': False}
            with mock.patch.object(stage_services, '_profile_reason', return_value=''), \
                 mock.patch.object(stage_services, '_toolbox_for_stage', return_value=box), \
                 mock.patch.object(stage_services, '_write_ladder_artifact'), \
                 mock.patch.object(soc.commands_mod, 'agent_cfu_overlay_commands', return_value=[['fake']]), \
                 mock.patch.object(soc.commands_mod, '_missing_design_files', return_value=[]), \
                 mock.patch.object(soc.commands_mod, '_soc_rtl_path', return_value=target), \
                 mock.patch.object(soc.effects, 'run_command', return_value={'returncode': 0}):
                result = soc.generate_soc(state)
        gate = result['gates']['hw_build']
        box.read_text.assert_any_call('agents/generated/runs/unit/workspace/soc/MiCoSoc.v',
                                      limit=20_000_000)
        self.assertFalse(gate['ok'])
        self.assertEqual(gate['terminal_reason'], 'hardware_integrity')
        self.assertEqual(gate['instruction_encoding']['rtl']['actual_pairs'], [[1, 0]])
        self.assertEqual(gate['candidate_hash'], gate['instruction_encoding']['candidate_hash'])
