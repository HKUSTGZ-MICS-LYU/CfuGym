"""Offline execution/PROMPT guards: no model, compiler or simulator calls."""
from __future__ import annotations

import unittest

from agents.cfu_design_agent import prompts
from agents.cfu_design_agent.benchmark import cfu_execution, embench
from agents.cfu_design_agent.runtime import tools
from agents.cfu_design_agent.stages.verify import software

MAIN = """int main(void) {
  initialise_board ();
  initialise_benchmark ();
  start_trigger ();
  result = benchmark ();
  stop_trigger ();
  correct = verify_benchmark (result);
  return !correct;
}
"""


class HostExecutionGuardsTest(unittest.TestCase):
    def test_driver_brackets_initialization_and_timing_without_editing_verifier(self):
        got = embench.instrument_embench_main(MAIN)
        self.assertIn('csrw 0x801, zero', got)
        self.assertIn('csrr %0, 0x801', got)
        self.assertLess(got.index('embench_host_cfu_clear_status ();'), got.index('initialise_benchmark ();'))
        self.assertLess(got.index('stop_trigger ();'), got.index('unsigned int embench_host_cfu_flags ='))
        self.assertLess(got.index('correct = verify_benchmark (result);'), got.index('if (embench_host_cfu_flags'))
        self.assertLess(got.index('if (embench_host_cfu_flags'), got.index('printf ("EMBENCH_VERIFY_'))
        self.assertNotIn('int verify_benchmark', got)
        self.assertIn('#ifdef USE_AGENT_CFU', got)

    def test_status_instrumentation_is_idempotent(self):
        once = cfu_execution.instrument_main(MAIN, embench.VERIFY_CALL)
        self.assertEqual(cfu_execution.instrument_main(once, embench.VERIFY_CALL), once)

    def test_full_output_parser_retains_status(self):
        text = ('EMBENCH_CFU_STATUS benchmark=aha-mont64 flags=1 protocol=sticky_csr_801_v1\n'
                'EMBENCH_VERIFY_PASS benchmark=aha-mont64 verify=1 result=0\n' + 'epilogue\n' * 5000)
        parsed = tools.parse_command_output(text)
        self.assertEqual(parsed['cfu_status']['flags'], 1)
        self.assertEqual(parsed['cfu_status']['benchmark'], 'aha-mont64')

    def evaluate(self, status=None, *, expected=True, category='', rc=0):
        result = {'returncode': rc, 'failure_observation': {'category': category},
                  'parsed': {'verify': {'status': 'pass', 'benchmark': 'aha-mont64'}}}
        if status is not None:
            result['parsed']['cfu_status'] = status
        return software._simulation_outcome(results=[result], name='aha-mont64',
            reference={'cycles': 19183, 'result': 0, 'verify': 1},
            profile={'cycles': 184, 'result': 0, 'verify': 1}, verification='bench_verify',
            changed=True, integrity_errors=[], cfu_status_expected=expected)

    def record(self, flags=0, **extra):
        return {'benchmark': 'aha-mont64', 'flags': flags, 'protocol': cfu_execution.PROTOCOL, **extra}

    def test_unsupported_zero_cannot_pass_original_zero_oracle(self):
        for flags in (1, 2, 4, 8, 16, 32, 63):
            with self.subTest(flags=flags):
                got = self.evaluate(self.record(flags))
                self.assertEqual(got['terminal_reason'], 'cfu_execution_error')
                self.assertFalse(got['correctness_ok'])
                self.assertFalse(got['performance_valid'])
                self.assertIsNone(got['speedup_x100'])

    def test_later_zero_status_cannot_hide_an_earlier_execution_error(self):
        text = ('EMBENCH_CFU_STATUS benchmark=aha-mont64 flags=1 protocol=sticky_csr_801_v1\n'
                'EMBENCH_CFU_STATUS benchmark=aha-mont64 flags=0 protocol=sticky_csr_801_v1\n')
        got = self.evaluate(cfu_execution.parse_status(text))
        self.assertEqual(got['terminal_reason'], 'cfu_execution_error')
        self.assertIsNone(got['speedup_x100'])
        self.assertEqual(len(got['cfu_execution']['records']), 2)

    def test_host_status_missing_does_not_mint_valid_speedup(self):
        for status in (None, self.record(benchmark='other')):
            got = self.evaluate(status)
            self.assertEqual(got['terminal_reason'], 'cfu_status_missing')
            self.assertFalse(got['cfu_execution']['checked'])
            self.assertIsNone(got['speedup_x100'])

    def test_valid_status_and_legacy_unknown_remain_distinct(self):
        got = self.evaluate(self.record())
        self.assertEqual(got['terminal_reason'], 'passed')
        self.assertTrue(got['cfu_execution']['checked'])
        legacy = self.evaluate(expected=False)
        self.assertFalse(legacy['cfu_execution']['checked'])
        self.assertIsNone(legacy['cfu_execution']['ok'])

    def test_actual_harness_crash_is_not_replaced_by_missing_diagnostic(self):
        got = self.evaluate(category='sim_harness_crash', rc=1)
        self.assertEqual(got['terminal_reason'], 'sim_harness_crash')

    def test_system_prompt_covers_semantics_encoding_and_legitimate_returns(self):
        for rule in ('valid input domain', 'decorative operand/LSU reads', 'fixed-shape specialization',
                     'Conditional Boolean', 'success status', 'unchecked default subop',
                     'clear error status', 'analysis/contract repair', 'command coverage table'):
            with self.subTest(rule=rule):
                self.assertIn(rule, prompts.SYSTEM_PROMPT)


if __name__ == '__main__':
    unittest.main()
