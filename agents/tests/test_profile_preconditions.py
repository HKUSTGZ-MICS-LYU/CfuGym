"""Router/unit precondition tests only; no new full-ladder scenario."""
from __future__ import annotations

import unittest
from unittest import mock

from agents.tests import test_profile_cache as profile_fixtures
from agents.cfu_design_agent.runtime.tools import RepoToolbox

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.flow import gates, ladder, wiring
from agents.cfu_design_agent.stages import profile


class ProfilePreconditionsTest(unittest.TestCase):
    def test_failed_reference_or_profile_goes_directly_to_review(self):
        update = profile._profile_verdict({}, ok=False, skipped=False,
            errors=['multiple definition of time'], history='reference build failed')
        self.assertEqual(update['stopped_at'], 'profile')
        self.assertEqual(update['stage_jobs'], 'review')
        self.assertEqual(update['gates']['profile']['terminal_reason'], 'profile_precondition')
        self.assertEqual(update['gates']['profile']['failure_kind'], 'precondition_failure')
        self.assertEqual(ladder._route_after_profile(update), 'review_and_route')
        self.assertEqual(gates._review_decision(update), ('final_report', 'failed'))
        self.assertFalse(gates._is_complete(update))
        self.assertNotIn('repairs_used', update)

    def test_reference_metrics_and_explicit_verification_must_be_valid(self):
        for cycles in (None, 0, -1, True, '100'):
            self.assertTrue(profile._baseline_problems({'demo': {'cycles': cycles}}, 'demo'))
        self.assertTrue(profile._baseline_problems({'demo': {'cycles': 100, 'verify': 0}}, 'demo'))
        self.assertEqual(profile._baseline_problems({'demo': {'cycles': 100, 'verify': 1}}, 'demo'), [])

    def test_failed_reference_build_does_not_run_profile_configuration(self):
        fixture = profile_fixtures.ProfileCacheTest('test_cache_is_reused_across_run_roots')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        scripted = profile_fixtures._ScriptedProfile()
        def failed(box, args, **kwargs):
            result = scripted(box, args, **kwargs)
            if kwargs.get('command_id') == 'embench_build':
                result.update(returncode=1, stderr_tail='multiple definition of time')
            return result
        with mock.patch.object(RepoToolbox, 'run_command', failed):
            update = profile.profile_embench(fixture._state('off'))
        self.assertEqual(scripted.calls, ['embench_build'])
        self.assertEqual(update['stopped_at'], 'profile')
        self.assertEqual(ladder._route_after_profile(update), 'review_and_route')

    def test_zero_cycle_reference_stops_before_instrumented_configuration(self):
        fixture = profile_fixtures.ProfileCacheTest('test_cache_is_reused_across_run_roots')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        scripted = profile_fixtures._ScriptedProfile()
        def invalid(box, args, **kwargs):
            result = scripted(box, args, **kwargs)
            result['stdout_tail'] = result['stdout_tail'].replace('depthconv=1200', 'depthconv=0')
            return result
        with mock.patch.object(RepoToolbox, 'run_command', invalid):
            update = profile.profile_embench(fixture._state('off'))
        self.assertEqual(scripted.calls, ['embench_build', 'soc_sim'])
        self.assertEqual(update['stopped_at'], 'profile')
        self.assertFalse(update['gates']['profile']['ok'])

    def test_success_or_explicit_skips_can_use_static_analysis(self):
        for ok, skipped in ((True, False), (False, True)):
            update = profile._profile_verdict({}, ok=ok, skipped=skipped)
            self.assertEqual(ladder._route_after_profile(update), 'extract_hotspot')
            self.assertNotIn('stopped_at', update)
            self.assertEqual(update['gates']['profile']['terminal_reason'], '')

    def test_missing_legacy_gate_retains_existing_route(self):
        self.assertEqual(ladder._route_after_profile({}), 'extract_hotspot')

    def test_profile_failure_cannot_be_overwritten_by_later_abstain(self):
        state = {'gates': {'profile': {'ok': False, 'skipped': False}, 'isa': {'abstain': True}}}
        self.assertEqual(gates._review_decision(state), ('final_report', 'failed'))

    def test_both_profile_nodes_use_native_conditional_edge(self):
        for node in ('profile_embench', 'instrument_and_profile'):
            self.assertEqual(wiring.ROUTER_NAMES[node], '_route_after_profile')
            self.assertEqual(set(wiring.ROUTE_TARGETS[node]), {'extract_hotspot', 'review_and_route'})
        self.assertIn('_route_after_profile', graph.__dict__)


if __name__ == '__main__':
    unittest.main()
