"""Offline contracts for diagnostic-only draft verification."""
import unittest
from unittest import mock

from agents.cfu_design_agent import prompts
from agents.cfu_design_agent.flow import gates, ladder
from agents.cfu_design_agent.runtime import dependencies, observations
from agents.cfu_design_agent.stages.verify import cost, parallel, soc, software


class DraftValidationTest(unittest.TestCase):
    def test_coordinator_never_starts_cost_in_either_mode(self):
        for mode in ('serial', 'parallel'):
            with self.subTest(mode=mode), \
                 mock.patch.object(software, 'simulate', return_value={
                     'gates': {'sim': {'ok': True}}, 'validation_results': [], 'errors': []}), \
                 mock.patch.object(cost, 'estimate_cost', side_effect=AssertionError('cost called')):
                update = parallel.validate_candidate({'validation_mode': mode, 'max_iters': 2,
                    'verify_submissions': 1, 'implementation_design_type': 'draft'})
            self.assertTrue(update['gates']['sim']['ok'])
            self.assertFalse(update['gates']['validation_pair']['ok'])
            self.assertTrue(update['gates']['validation_pair']['diagnostic_only'])
            self.assertEqual(update['gates']['validation_pair']['next'], 'implement')
            self.assertEqual(update['evaluation']['lanes']['cost'], 'not_run_draft')
            self.assertTrue(update['evaluation_id'])
            self.assertTrue(update['evaluation']['input_fingerprints']['sim'])
            self.assertNotIn('attempts', update)

    def test_draft_budget_and_nonacceptance(self):
        extra, target = ladder.draft_sim_transition({'max_iters': 1, 'verify_submissions': 1})
        self.assertEqual(target, 'review_and_route')
        self.assertEqual(extra['stopped_at'], 'draft_completion')
        self.assertFalse(gates._is_complete({'implementation_design_type': 'draft'}))

    def test_soc_submission_count_and_exhaustion(self):
        state = {'run_commands': True, 'max_iters': 2, 'verify_submissions': 0,
                 'implementation_design_type': 'draft'}
        with mock.patch.object(soc.stage_services, '_toolbox_for_stage'), \
             mock.patch.object(soc.stage_services, '_write_ladder_artifact'), \
             mock.patch.object(soc.stage_services, 'hardware_integrity_problems', side_effect=lambda *a, **k: []), \
             mock.patch.object(soc.commands_mod, 'agent_cfu_overlay_commands', return_value=[['sbt']]), \
             mock.patch.object(soc.commands_mod, '_missing_design_files', return_value=[]), \
             mock.patch.object(soc.commands_mod, '_soc_rtl_path', return_value=None), \
             mock.patch.object(soc.effects, 'run_command', return_value={'returncode': 1}) as command:
            first = soc.generate_soc(state)
            self.assertEqual(first['verify_submissions'], 1)
            self.assertEqual(first['gates']['hw_build']['next'], 'implement')
            state.update(first)
            state['implementation_design_type'] = 'complete'
            second = soc.generate_soc(state)
            self.assertEqual(second['verify_submissions'], 2)
            self.assertEqual(second['gates']['hw_build']['next'], 'review_and_route')
            state.update(second)
            exhausted = soc.generate_soc(state)
            self.assertEqual(exhausted['gates']['hw_build']['next'], 'review_and_route')
            self.assertEqual(command.call_count, 2, (first, second, exhausted))

    def test_fingerprints_preserve_legacy_and_invalidate_type(self):
        base = {'workdir': '.'}
        old = dependencies.input_fingerprints(base)
        draft = dependencies.input_fingerprints({**base, 'implementation_design_type': 'draft'})
        complete = dependencies.input_fingerprints({**base, 'implementation_design_type': 'complete'})
        self.assertNotEqual(old['sim'], draft['sim'])
        self.assertNotEqual(draft['sim'], complete['sim'])
        self.assertNotEqual(draft['cost'], complete['cost'])
        self.assertEqual(old['hw_build'], draft['hw_build'])
        stale = dependencies.invalidate_changed_gates(
            {'sim': {'ok': True}, 'cost': {'ok': True}}, draft, complete)
        self.assertFalse(stale['sim']['ok'])
        self.assertFalse(stale['cost']['ok'])
        self.assertTrue(stale['sim']['stale'])
        self.assertTrue(stale['cost']['stale'])

    def test_draft_performance_does_not_gate_sim(self):
        evidence = software._simulation_outcome(
            results=[{'returncode': 0, 'parsed': {'verify': {'status': 'pass'}}}],
            name='test', reference={'cycles': 100, 'verify': 1},
            profile={'cycles': 200, 'verify': 1}, verification='bench_verify',
            changed=True, integrity_errors=[], draft=True)
        self.assertEqual(evidence['errors'], [])
        self.assertFalse(evidence['performance_valid'])
        self.assertIsNone(evidence['speedup_x100'])

    def test_direct_cost_is_guarded(self):
        for backend in ('yosys', 'vivado'):
            with mock.patch.object(cost, '_estimate_vivado', side_effect=AssertionError), \
                 mock.patch.object(cost.stage_services, '_toolbox_for_stage', side_effect=AssertionError):
                result = cost.estimate_cost({'implementation_design_type': 'draft', 'cost_backend': backend})
            self.assertTrue(result['gates']['cost']['skipped'])

    def test_invalid_explicit_type_fails_closed_before_cost(self):
        for value in (None, 'unknown', 'bogus'):
            state = {'implementation_design_type': value}
            self.assertFalse(gates._is_complete(state))
            with mock.patch.object(cost, '_estimate_vivado', side_effect=AssertionError), \
                 mock.patch.object(cost.stage_services, '_toolbox_for_stage', side_effect=AssertionError), \
                 mock.patch.object(software, 'simulate', side_effect=AssertionError):
                with self.assertRaises(ValueError):
                    cost.estimate_cost(state)
                with self.assertRaises(ValueError):
                    parallel.validate_candidate(state)

    def test_real_sim_records_diagnostic_on_success_and_failure(self):
        state = {'run_commands': True, 'implementation_design_type': 'draft',
                 'implementation_unimplemented': ['missing op'], 'max_iters': 2,
                 'verify_submissions': 1}
        for rc in (0, 1):
            with self.subTest(rc=rc), \
                 mock.patch.object(software.stage_services, '_toolbox_for_stage'), \
                 mock.patch.object(software.stage_services, 'benchmark_integrity_problems', side_effect=lambda *a: []), \
                 mock.patch.object(software.stage_services, '_write_ladder_artifact') as writer, \
                 mock.patch.object(software.intrinsics, '_candidate_change_domains', return_value={'hw': True}), \
                 mock.patch.object(software.commands_mod, '_software_sim_commands', return_value=[['sim']]), \
                 mock.patch.object(software.effects, 'run_command', return_value={'rung': 'sim', 'returncode': rc}):
                update = software.simulate(state)
            self.assertEqual(update['gates']['sim']['ok'], rc == 0)
            self.assertEqual(update['gates']['sim']['implementation_design_type'], 'draft')
            self.assertEqual(writer.call_args.kwargs['evidence']['implementation_design_type'], 'draft')
            failures = observations.failure_index(update)['failures']
            diagnostic = [f for f in failures.values() if f['rung'] == 'draft_completion']
            self.assertEqual(len(diagnostic), 1)
            self.assertIn('missing op', diagnostic[0]['summary'])
            resolved = observations.failure_index({'validation_results': update['validation_results'] + [
                {'rung': 'draft_completion', 'returncode': 0}]})['failures']
            self.assertEqual(resolved[diagnostic[0]['id']]['status'], 'resolved')

    def test_verification_budget_warns_before_the_last_slot(self):
        first = prompts.verification_budget_block({'max_iters': 5, 'verify_submissions': 0})
        self.assertIn("0/5", first)
        self.assertIn("leaves 4 slot(s)", first)
        self.assertNotIn("final slot", first)
        last = prompts.verification_budget_block({'max_iters': 5, 'verify_submissions': 4})
        self.assertIn("4/5", last)
        self.assertIn("leaves 0 slot(s)", last)
        self.assertIn("final slot", last)
        self.assertIn("end the run incomplete", last)
        self.assertEqual(prompts.verification_budget_block({'max_iters': 0}), "")

    def test_new_candidate_supersedes_old_failures(self):
        old_hw = {'rung': 'implementation_hw', 'returncode': 1,
                  'failure_observation': {'id': 'fail_hw', 'rung': 'implementation_hw'}}
        old_draft = {'rung': 'draft_completion', 'returncode': 1,
                     'failure_observation': {'id': 'fail_draft', 'rung': 'draft_completion'}}
        new_sim = {'rung': 'sim', 'returncode': 1,
                   'failure_observation': {'id': 'fail_sim', 'rung': 'sim'}}
        old_cost = {'rung': 'cost', 'returncode': 1,
                    'failure_observation': {'id': 'fail_cost', 'rung': 'cost'}}
        state = {'validation_results': [old_hw, old_draft, new_sim],
                 'cost_results': [old_cost],
                 'implementation_candidate_epoch': 2,
                 'implementation_cost_candidate_epoch': 1,
                 'repair_evidence_epoch': {'validation_results': 0, 'cost_results': 0}}
        failures = observations.failure_index(state)['failures']
        self.assertEqual(failures['fail_hw']['status'], 'superseded')
        self.assertEqual(failures['fail_draft']['status'], 'superseded')
        self.assertEqual(failures['fail_cost']['status'], 'superseded')
        self.assertEqual(failures['fail_sim']['status'], 'current')

    def test_skipped_sim_records_design_metadata(self):
        state = {'implementation_design_type': 'draft', 'implementation_unimplemented': ['foo'],
                 'dry_run': True}
        with mock.patch.object(software.stage_services, '_write_ladder_artifact') as writer:
            update = software.simulate(state)
        self.assertEqual(writer.call_args.kwargs['evidence']['implementation_design_type'], 'draft')
        self.assertEqual(update['gates']['sim']['implementation_unimplemented'], ['foo'])
        self.assertTrue(update['gates']['sim']['skipped'])

    def test_exception_propagates(self):
        with mock.patch.object(software, 'simulate', side_effect=RuntimeError('boom')), \
             mock.patch.object(cost, 'estimate_cost', side_effect=AssertionError('cost called')):
            with self.assertRaisesRegex(RuntimeError, 'boom'):
                parallel.validate_candidate({'implementation_design_type': 'draft', 'validation_mode': 'parallel'})


if __name__ == '__main__':
    unittest.main()
