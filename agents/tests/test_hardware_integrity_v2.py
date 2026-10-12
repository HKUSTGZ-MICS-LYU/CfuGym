"""Minimal proposal fixtures; no live LLM, compiler, sweep, or real hardware."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from agents.cfu_design_agent.runtime import coding_tools, failures, hardware_integrity, stage_services, tools, policy, llm_step
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.stages.implement import intrinsics, repair_scope
from agents.cfu_design_agent.stages.implement import unified
from agents.tests.test_instruction_encoding import instruction, rtl, spec, table

PATH = 'workspace/design/AgentCfu.scala'


def compute(body, definitions='', resources='Some(Seq(CpuRead(RS1), CpuWriteRD))'):
    return definitions + f'''\nInstruction("op", AgentCfuFunction.compute,
      resources = {resources}, start = ctx => {{ {body} }})'''


def memory_contract():
    return {'commands': [{'name': 'op', 'function_id': 1, 'kind': 'compute',
                         'response': 'value', 'result': 'Signed 32-bit operation status, zero on success',
                         'semantics': 'Compute the operation and write all observable output elements before returning status.'}],
            'memory': {'load': True, 'store': True}}


def ud_writeback(payload='matrix(row).asBits', slot='0', store_slot='0'):
    return f'''when(phase === WRITE_ISSUE) {{
      when(lsu.cmd.ready) {{
        vectorRegs({slot}) := 0
        for(row <- 0 until 6) vectorRegs({slot})(32 * row, 32 bits) := {payload}
        storeSource := {store_slot}
        lsu.cmd.address := outputBase
        lsu.cmd.store := True
        phase := WRITE_WAIT
      }}
    }} elsewhen(phase === WRITE_WAIT) {{
      when(lsu.done) {{ when(lsu.error) {{ ctx.fail() }} otherwise {{ ctx.complete(B(0, 32 bits)) }} }}
    }}'''


class FixedResultV2Test(TestCase):
    def check(self, source, contract=None):
        return hardware_integrity.validate_sources({PATH: source}, contract)

    def test_tarfind_literal_boolean_branches_read_actual_data(self):
        source = compute('''when(sb =/= cb) { ctx.complete(B(0, 32 bits)) }
            elsewhen(sb === B(0, 8 bits)) { ctx.complete(B(1, 32 bits)) }
            elsewhen(cursor === U(99, 7 bits)) { ctx.complete(B(0, 32 bits)) }
            otherwise { cursor := cursor + 1 }''', '''
          val searchWords = Vec(Reg(Bits(32 bits)) init(0), 26)
          val candidateWord = Reg(Bits(32 bits)) init(0)
          when(lsu.done) {
            searchWords(searchWordIndex) := lsu.load.data
            candidateWord := lsu.load.data
          }
          val sb = searchWords(searchWordIndex)(searchByteOffset, 8 bits)
          val cb = candidateWord(candidateByteOffset, 8 bits)
          val cursor = Reg(UInt(7 bits)) init(0)
        ''')
        self.assertEqual(self.check(source), [])

    def test_when_real_lsu_predicate_otherwise_distinct_complete_is_not_fixed(self):
        self.assertEqual(self.check(compute('''when(lsu.load.data =/= ctx.read(RS1)) {
          ctx.complete(B(0, 32 bits))
        } otherwise { ctx.complete(B(1, 32 bits)) }''')), [])

    def test_dummy_load_or_done_guard_and_same_value_branches_remain_rejected(self):
        for body in ['''lsu.cmd.load := True
                     when(lsu.done) { ctx.complete(B(102, 32 bits)) }''',
                     '''when(lsu.done) { ctx.complete(B(0, 32 bits)) }
                        otherwise { ctx.complete(B(1, 32 bits)) }''',
                     '''when(lsu.load.data === ctx.read(RS1)) { ctx.complete(B(102, 32 bits)) }
                        otherwise { ctx.complete(B(102, 32 bits)) }''']:
            with self.subTest(body=body): self.assertTrue(self.check(compute(body)))

    def test_same_constant_spinal_spellings_and_annihilated_guards_are_not_boolean(self):
        for body in ['''when(lsu.load.data === ctx.read(RS1)) { ctx.complete(B(0, 32 bits)) }
                     otherwise { ctx.complete(B"32'h0") }''',
                     '''when(lsu.load.data * 0 === B(0, 32 bits)) { ctx.complete(B(0, 32 bits)) }
                     otherwise { ctx.complete(B(1, 32 bits)) }''']:
            with self.subTest(body=body): self.assertTrue(self.check(compute(body)))

    def test_ud_contract_status_with_real_local_matrix_writeback_is_legal(self):
        source = compute(ud_writeback(), '''val matrix = Vec(Reg(SInt(32 bits)) init(0), 36)
          matrix(row) := matrix(row) / matrix(pivot)
          val outputBase = ctx.read(RS1).asUInt''')
        self.assertEqual(self.check(source, memory_contract()), [])
        self.assertTrue(self.check(source))  # no explicit memory-output/status contract

    def test_edn_xor_self_status_with_output_tile_write_route_is_legal(self):
        source = compute('''vectorWrite(U(0, 1 bits), outputTile)
          storeSource := U(0, 1 bits)
          lsu.cmd.address := outputAddress
          lsu.cmd.store := True
          when(lsu.done) { when(lsu.error) { ctx.fail() } otherwise {
            ctx.complete(outputTile(31 downto 0) ^ outputTile(31 downto 0))
          }}''', '''val outputTile = Reg(Bits(256 bits)) init(0)
          outputTile(offset, 32 bits) := (lsu.load.data.asSInt * coefficient).asBits''')
        contract = memory_contract()
        contract['commands'][0]['result'] = 'rd = 0 on successful completion; nonzero on TileLink/error completion'
        self.assertEqual(self.check(source, contract), [])
        self.assertTrue(self.check(source))

    def test_store_flag_pointer_read_or_dummy_fixed_payload_is_insufficient(self):
        for body in ['''val pointer = ctx.read(RS1)
                       lsu.cmd.store := True
                       when(lsu.done) { ctx.complete(B(0, 32 bits)) }''',
                     ud_writeback(payload='B(123, 32 bits)'),
                     ud_writeback(payload='fixed()', slot='0', store_slot='0'),
                     ud_writeback(slot='1', store_slot='0')]:
            with self.subTest(body=body):
                self.assertTrue(self.check(compute(body, 'def fixed() = B(123, 32 bits)'), memory_contract()))

    def test_memory_result_not_status_or_store_false_cannot_get_exception(self):
        source = compute(ud_writeback())
        for field, value in [('result', 'Computed numerical solution in rd'), ('semantics', 'Only reads an input pointer')]:
            contract = memory_contract()
            contract['commands'][0][field] = value
            self.assertTrue(self.check(source, contract))
        contract = memory_contract()
        contract['memory']['store'] = False
        self.assertTrue(self.check(source, contract))

    def test_literal_alias_and_indexed_xor_self_computed_rd_still_rejected(self):
        for expr in ['B(17, 32 bits)', 'fixed()', 'v(31 downto 0) ^ v(31 downto 0)']:
            with self.subTest(expr=expr):
                self.assertTrue(self.check(compute('ctx.complete(' + expr + ')', 'def fixed() = B(17, 32 bits)')))

    def test_ack_config_load_store_existing_rules_survive(self):
        self.assertEqual(self.check(compute('ctx.complete(B(0, 32 bits))', resources='Some(Seq(CpuRead(RS1)))')), [])
        for function in ['config', 'load', 'store']:
            self.assertEqual(self.check(compute('ctx.complete(B(0, 32 bits))').replace('AgentCfuFunction.compute', 'AgentCfuFunction.' + function)), [])

    def test_live_table_only_not_old_template_comments_or_strings(self):
        source = table(instruction()) + '\nval template = """Instruction("bad", AgentCfuFunction.compute, start=ctx=>ctx.complete(B(102,32 bits)))"""'
        self.assertEqual(self.check(source), [])


class HardwareGateConsistencyTest(TestCase):
    def test_development_soc_tool_refuses_same_static_mismatch_without_commands(self):
        box = mock.Mock()
        box.read_text.side_effect = lambda p, **kw: table(instruction(subop=None)) if p.endswith('AgentCfu.scala') else '// fiber'
        state = {'run_root': 'unit', 'isa_spec': spec(), 'require_conformance': False}
        with mock.patch('agents.cfu_design_agent.runtime.effects.run_command', side_effect=AssertionError('no toolchain')):
            result = coding_tools.build_soc_check(state, box)
        self.assertEqual(result['returncode'], 1)
        self.assertIn('encoding_mismatch', result['stderr_tail'])

    def test_development_rtl_audit_uses_same_post_generation_crosscheck(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'AgentCfu.v'
            path.write_text(rtl(((1, 0),)))
            box = mock.Mock()
            box.read_text.side_effect = lambda p, **kw: table(instruction()) if p.endswith('AgentCfu.scala') else path.read_text() if p.endswith('.v') else '// fiber'
            result = coding_tools.audit_hardware_result({'run_root': 'unit', 'isa_spec': spec()}, box, {'returncode': 0}, path)
        self.assertEqual(result['returncode'], 1)
        self.assertIn('rtl_encoding_mismatch', result['stderr_tail'])

    def test_native_submit_and_hw_build_share_static_gate_and_rollback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = 'agents/generated/runs/fixture-native-integrity'
            hw = next(iter(tools.design_workspace_paths(run).values()))
            sw, header = run + '/workspace/kernel.c', run + '/workspace/cfu_intrinsics.h'
            sources = {hw: table(instruction()), sw: 'int kernel(void) { return 0; }', header: 'old header'}
            for path, text in sources.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            access = policy.AccessPolicy(version='1', deny=(), write_roots=(run,), read=(run + '/**',),
                        stages={'implementation': policy.StagePolicy(write=(run + '/**',), commands=())}, commands={})
            box = tools.RepoToolbox(root, policy=access, stage='implementation', run_root=run, apply_patch=True)
            state = {'workdir': str(root), 'run_root': run, 'task': 'fixture only', 'isa_spec': spec(),
                     'require_conformance': False, 'agent_tools': True, 'apply_patch': True, 'run_commands': True}
            def write_header(*args):
                (root / header).write_text('new generated header')
                return header
            def scripted_step(_state, box, **kwargs):
                tool = kwargs['toolset_factory'](box, **kwargs['toolset_kwargs'])
                hw_result = tool.run('edit_hardware', {'edits': [
                    {'path': hw, 'search': instruction(), 'replace': instruction(subop='0')}]})
                sw_result = tool.run('edit_software', {'path': sw, 'search': sources[sw], 'replace': 'int kernel(void) { return cfu_op(1); }'})
                self.assertTrue(tool.hardware_session.touched_paths, hw_result)
                self.assertTrue(tool.software_session.touched_paths, sw_result)
                accepted, reason = kwargs['submit_stage']('both domains changed')
                self.assertFalse(accepted)
                self.assertIn('encoding_mismatch', reason)
                with mock.patch('agents.cfu_design_agent.runtime.effects.run_command', side_effect=AssertionError('no toolchain')):
                    check = coding_tools.build_soc_check(state, box)
                self.assertIn('encoding_mismatch', check['stderr_tail'])
                raise failures.StageDeliveryFailure(reason, stage='implement')
            with mock.patch.object(stage_services, '_toolbox_for_stage', return_value=box), \
                 mock.patch.object(stage_services, '_software_targets', return_value=[(sw, sources[sw])]), \
                 mock.patch.object(intrinsics, 'intrinsics_header_path', return_value=header), \
                 mock.patch.object(intrinsics, '_write_intrinsics_header', side_effect=write_header), \
                 mock.patch.object(repair_scope, 'required_domains', return_value=(True, True)), \
                 mock.patch.object(llm_step, '_run_agent_step', side_effect=scripted_step):
                update = unified.implement_with_tools(state, client=mock.Mock(available=True))
            self.assertFalse(update['gates']['implementation_hw']['ok'])
            self.assertFalse(update['gates']['implementation_sw']['ok'])
            for path, text in sources.items(): self.assertEqual((root / path).read_text(), text)

    def test_typed_rejection_is_failed_gate_not_success_or_infrastructure(self):
        exc = failures.StageDeliveryFailure('hardware_integrity/fixed_compute_result: op', stage='implement')
        result = coding_tools.stage_delivery_failure_update({'run_root': 'unit'}, exc)
        self.assertFalse(result['gates']['implementation_hw']['ok'])
        self.assertFalse(result['gates']['implementation_sw']['ok'])
        self.assertEqual(result['gates']['implementation_hw']['terminal_reason'], 'hardware_integrity')
        self.assertEqual(result['gates']['implementation_hw']['failure_kind'], 'model_delivery_failure')
        self.assertNotIn('hardware_plan', result)
        self.assertNotIn('llm_raw_outputs', result)
