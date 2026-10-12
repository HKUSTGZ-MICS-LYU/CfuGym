"""Fast fixture-only fixed-result and authoritative-gate regressions."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from agents.cfu_design_agent.runtime import hardware_integrity, stage_services
from agents.cfu_design_agent.stages.verify import soc

PATH = "agents/generated/runs/unit/workspace/design/AgentCfu.scala"


def compute(body, definitions="", resources="Some(Seq(CpuRead(RS1), CpuRead(RS2), CpuWriteRD))"):
    return f'''{definitions}
Instruction("match", AgentCfuFunction.compute,
  resources = {resources},
  start = ctx => {{
    {body}
  }})
'''


class HardwareIntegrityTest(TestCase):
    def problems(self, source):
        return hardware_integrity.validate_sources({PATH: source})

    def test_literal_reply_rejected_despite_dummy_load_and_status(self):
        source = compute('''lsu.cmd.address := ctx.read(RS1).asUInt
lsu.cmd.load := True
ctx.defer()
when(lsu.done) {
  val result = B(102, cfuParam.CFU_OUTPUT_DATA_W bits)
  ctx.complete(result, lsu.error.mux(B"001", B"000"))
}''')
        self.assertIn("fixed_compute_result", self.problems(source)[0])

    def test_alias_chain_and_no_argument_helper_are_not_escape_hatches(self):
        for defs, payload in [
            ('val expected = 102\nval result = B(expected, 32 bits)', 'result'),
            ('def expected() = B(102, 32 bits)', 'expected()'),
            ('val result = Reg(Bits(32 bits)) init(102)', 'result'),
        ]:
            with self.subTest(payload=payload):
                self.assertTrue(self.problems(compute(f'ctx.complete({payload})', defs)))

    def test_multiline_and_inline_constant_wrappers_are_checked(self):
        for defs in ('val expected = B(\n102,\n32 bits)',
                     'def expected() = { B(102, 32 bits) }'):
            with self.subTest(defs=defs):
                self.assertTrue(self.problems(compute('ctx.complete(expected())', defs)))
        source = compute('val result = B(102, 32 bits); ctx.complete(result)')
        self.assertTrue(self.problems(source))

    def test_string_pseudocode_is_not_a_real_fixed_reply(self):
        source = compute('''val message = "ctx.complete(B(102, 32 bits))"
ctx.complete(ctx.read(RS1))''')
        self.assertEqual(self.problems(source), [])

    def test_gate_version_change_stales_build_and_measurement(self):
        from agents.cfu_design_agent.runtime import dependencies
        with TemporaryDirectory() as tmp:
            state = {'workdir': tmp}
            before = dependencies.input_fingerprints(state)
            with mock.patch.object(hardware_integrity, 'VERSION', hardware_integrity.VERSION + 1):
                after = dependencies.input_fingerprints(state)
            gates = dependencies.invalidate_changed_gates(
                {name: {'ok': True} for name in before}, before, after)
        self.assertTrue(gates['isa']['ok'])
        for name in ('hw_build', 'hw_conformance', 'sim', 'cost'):
            self.assertTrue(gates[name]['stale'])

    def test_annihilated_operand_expression_is_fixed(self):
        for expr in ('B(input * 0 + 102, 32 bits)', 'B(input - input, 32 bits)'):
            with self.subTest(expr=expr):
                self.assertTrue(self.problems(compute(f'ctx.complete({expr})',
                                                      'val input = ctx.read(RS1)')))

    def test_live_operand_memory_payload_and_accumulator_are_not_constant(self):
        for defs, payload in [
            ('val result = ctx.read(RS1) + ctx.read(RS2)', 'result'),
            ('val result = lsu.load.data', 'result'),
            ('val acc = Reg(UInt(32 bits)) init(0)\nacc := acc + ctx.read(RS1).asUInt', 'acc.asBits'),
            ('val result = Bits(256 bits)\nresult := 0\nresult(i * 8, 8 bits) := ctx.read(RS1)', 'result'),
        ]:
            with self.subTest(payload=payload):
                self.assertEqual(self.problems(compute(f'ctx.complete({payload})', defs)), [])

    def test_ack_config_and_comments_are_not_computational_replies(self):
        self.assertEqual(self.problems(compute('ctx.complete()')), [])
        self.assertEqual(self.problems(compute('ctx.complete(B(0, 32 bits))',
                                               resources='Some(Seq(CpuRead(RS1)))')), [])
        self.assertEqual(self.problems(compute('ctx.complete(B(0, 32 bits))').replace(
            'AgentCfuFunction.compute', 'AgentCfuFunction.config')), [])
        source = compute('''// ctx.complete(B(102, 32 bits))
/* val result = B(102, 32 bits) */
val result = ctx.read(RS1)
ctx.complete(result)''')
        self.assertEqual(self.problems(source), [])

    def test_callback_parameter_rename_does_not_evade_guard(self):
        self.assertTrue(self.problems(compute('ctx.complete(B(102, 32 bits))').replace('ctx', 'request')))

    def test_legacy_reply_and_numeric_contract_compute_are_checked(self):
        source = compute('ctx.reply(B(102, 32 bits))').replace(
            'resources = Some(Seq(CpuRead(RS1), CpuRead(RS2), CpuWriteRD)),', '')
        self.assertTrue(self.problems(source))
        source = compute('ctx.complete(B(102, 32 bits))').replace('AgentCfuFunction.compute', '1')
        spec = {'commands': [{'name': 'match', 'kind': 'compute'}]}
        self.assertTrue(hardware_integrity.validate_sources({PATH: source}, spec))

    def test_guard_has_no_model_compiler_or_simulator_dependency(self):
        with mock.patch('subprocess.run', side_effect=AssertionError('no commands')):
            self.assertTrue(self.problems(compute('ctx.complete(B(102, 32 bits))')))

    def test_host_helper_reads_live_overlay_and_independent_build_blocks_commands(self):
        box = mock.Mock()
        box.read_text.side_effect = lambda path, **kw: (
            compute('ctx.complete(B(102, 32 bits))') if path.endswith('AgentCfu.scala') else '// fiber')
        state = {'run_root': 'agents/generated/runs/unit', 'isa_spec': {}, 'max_iters': 1,
                 'gates': {'implementation_hw': {'ok': True}}}
        self.assertTrue(stage_services.hardware_integrity_problems(state, box))
        with mock.patch.object(stage_services, '_profile_reason', return_value=''), \
             mock.patch.object(stage_services, '_toolbox_for_stage', return_value=box), \
             mock.patch.object(stage_services, '_write_ladder_artifact'), \
             mock.patch.object(soc.commands_mod, 'agent_cfu_overlay_commands', return_value=[['sbt']]), \
             mock.patch.object(soc.commands_mod, '_missing_design_files', return_value=[]), \
             mock.patch.object(soc.commands_mod, '_soc_rtl_path', return_value=None), \
             mock.patch.object(soc.effects, 'run_command', side_effect=AssertionError('no sbt')):
            result = soc.generate_soc(state)
        self.assertFalse(result['gates']['hw_build']['ok'])
        self.assertEqual(result['gates']['hw_build']['terminal_reason'], 'hardware_integrity')
        self.assertEqual(result['gates']['hw_build']['commands'], 0)
        evidence = result['validation_results'][0]
        self.assertEqual(evidence['kind'], 'host_gate_check')
        self.assertEqual(evidence['failure_observation']['category'], 'hardware_integrity')
        from agents.cfu_design_agent.runtime import observations
        from agents.cfu_design_agent.stages import feedback
        current = {**state, **result}
        self.assertTrue(observations.failure_index(current)['failures'])
        self.assertIn('fixed_compute_result', feedback.all_failure_feedback(current))


class NoRealLlmTest(TestCase):
    def test_even_local_real_sdk_clients_are_blocked_without_env_escape(self):
        import os
        import openai
        with mock.patch.dict(os.environ, {'CFU_AGENT_TESTS_ALLOW_NETWORK': '1'}):
            for name in ('OpenAI', 'AsyncOpenAI', 'AzureOpenAI', 'AsyncAzureOpenAI'):
                with self.subTest(client=name), self.assertRaisesRegex(AssertionError, 'real LLM SDK'):
                    getattr(openai, name)(api_key='test', base_url='http://127.0.0.1:1/v1')

    def test_fake_transport_is_still_allowed(self):
        import openai
        fake = object()
        with mock.patch('openai.OpenAI', return_value=fake):
            self.assertIs(openai.OpenAI(api_key='test'), fake)
