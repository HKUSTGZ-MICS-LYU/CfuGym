"""Fast concurrency contracts: no EDA, real model, or full-ladder runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import yaml

from agents.cfu_design_agent.runtime import artifacts, checks, effects, persistence, processes, validation_parallel
from agents.cfu_design_agent.stages.verify import parallel, software


def result(rung, ok=True, **fields):
    return {'gates': {rung: {'ok': ok, **fields}},
            'validation_results' if rung == 'sim' else 'cost_results': [{'rung': rung, 'returncode': 0 if ok else 1}],
            'iteration_history': [rung], 'errors': [] if ok else [rung + ' failed']}


class ParallelValidationTest(unittest.TestCase):
    def call(self, sim, cost, **state):
        with mock.patch.object(parallel.software, 'simulate', side_effect=sim), \
             mock.patch.object(parallel.cost, 'estimate_cost', side_effect=cost):
            return parallel.validate_candidate({'validation_mode': 'parallel', 'max_iters': 2, **state})

    def test_both_workers_overlap_have_independent_contexts_and_plain_commands(self):
        barrier = threading.Barrier(2)
        context = ContextVar('test_lane_context', default='outer')
        seen = []
        def worker(rung):
            def run(state, *, transition):
                self.assertFalse(transition)
                self.assertFalse(effects.inside_run())
                self.assertEqual(context.get(), 'outer')
                context.set(rung)
                state['gates'] = {'mutated_by_worker': {}}
                barrier.wait(timeout=2)
                self.assertEqual(context.get(), rung)
                seen.append((rung, state['evaluation_id']))
                return result(rung)
            return run
        state = {'validation_mode': 'parallel', 'gates': {'before': {'ok': True}}}
        with effects.use_state(state):
            update = self.call(worker('sim'), worker('cost'), gates=state['gates'])
        self.assertEqual(context.get(), 'outer')
        self.assertEqual(state['gates'], {'before': {'ok': True}})
        self.assertEqual(len({entry[1] for entry in seen}), 1)
        self.assertTrue(update['gates']['validation_pair']['ok'])
        self.assertEqual(update['iteration_history'], ['sim', 'cost'])
        self.assertNotIn('mutated_by_worker', update['gates'])

    def _failure_race(self, failing, command=False, unexpected=False):
        started = threading.Event()
        cancelled = threading.Event()
        def waiting(state, *, transition):
            token = processes.current_token()
            started.set()
            if not token.event.wait(2):
                self.fail('sibling did not cancel the waiting lane')
            cancelled.set()
            token.raise_if_cancelled()
        def fail(state, *, transition):
            self.assertTrue(started.wait(2))
            if unexpected:
                raise LookupError('unexpected worker fault')
            if command:
                # The notification must happen before the worker returns its gate.
                validation_parallel.command_failed()
                self.assertTrue(cancelled.wait(2))
            return result(failing, False)
        callbacks = {'sim': waiting, 'cost': waiting}
        callbacks[failing] = fail
        return self.call(callbacks['sim'], callbacks['cost'])

    def test_sim_failure_cancels_cost_and_spends_only_one_repair(self):
        update = self._failure_race('sim')
        self.assertEqual(update['repairs_used'], 1)
        self.assertEqual(update['attempts'], {'sim': 1})
        self.assertEqual(update['gates']['validation_pair']['next'], 'implement')
        self.assertTrue(update['gates']['cost']['cancelled'])
        self.assertNotIn('cost failed', update['errors'])
        self.assertEqual(update['evaluation']['cancelled'], ['cost'])

    def test_cost_semantic_failure_cancels_sim(self):
        update = self._failure_race('cost')
        self.assertEqual(update['attempts'], {'cost': 1})
        self.assertTrue(update['gates']['sim']['cancelled'])

    def test_command_failure_signals_before_worker_finishes(self):
        update = self._failure_race('sim', command=True)
        self.assertEqual(update['repairs_used'], 1)

    def test_unexpected_exception_cancels_sibling_and_reraises(self):
        with self.assertRaisesRegex(LookupError, 'unexpected worker fault'):
            self._failure_race('cost', unexpected=True)

    def test_serial_fallback_uses_worker_protocol_without_overlapping(self):
        calls = []
        def worker(rung):
            def run(state, *, transition):
                self.assertFalse(transition)
                calls.append(rung)
                return result(rung)
            return run
        update = self.call(worker('sim'), worker('cost'), validation_mode='serial')
        self.assertEqual(calls, ['sim', 'cost'])
        self.assertTrue(update['gates']['validation_pair']['ok'])

    def test_serial_failure_never_executes_cost(self):
        cost = mock.Mock(side_effect=AssertionError('cost must not start'))
        update = self.call(lambda *a, **k: result('sim', False), cost, validation_mode='serial')
        cost.assert_not_called()
        self.assertTrue(update['gates']['cost']['cancelled'])

    def test_skipped_lane_is_not_a_failure_and_does_not_cancel_other_lane(self):
        update = self.call(lambda *a, **k: result('sim', False, skipped=True),
                           lambda *a, **k: result('cost'))
        self.assertTrue(update['gates']['validation_pair']['skipped'])
        self.assertNotIn('repairs_used', update)
        self.assertEqual(update['gates']['validation_pair']['next'], 'review_and_route')

    def test_worker_budget_and_inherited_gate_updates_are_not_published(self):
        def sim(*a, **k):
            return {**result('sim'), 'repairs_used': 100, 'iteration': 900,
                    'gates': {'sim': {'ok': True}, 'cost': {'ok': False}, 'old': {'ok': False}}}
        update = self.call(sim, lambda *a, **k: result('cost'), gates={'old': {'ok': True}})
        self.assertNotIn('repairs_used', update)
        self.assertNotIn('iteration', update)
        self.assertTrue(update['gates']['old']['ok'])
        self.assertTrue(update['gates']['cost']['ok'])

    def test_source_change_during_work_refuses_stale_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'candidate.c'
            source.write_text('before')
            def sim(*a, **k):
                source.write_text('after')
                return result('sim')
            with self.assertRaisesRegex(RuntimeError, 'changed during validation'):
                self.call(sim, lambda *a, **k: result('cost'), validation_mode='serial',
                          workdir=directory, changed_sw_files=['candidate.c'])

    def test_native_group_task_replay_does_not_repeat_workers(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import Command, interrupt
        from agents.cfu_design_agent.state import CfuDesignState
        calls = []
        ids = []
        def worker(rung):
            def run(*a, **k):
                calls.append(rung)
                self.assertFalse(effects.inside_run())
                return result(rung)
            return run
        def node(state):
            with effects.use_state(state):
                update = parallel.validate_candidate(state)
            ids.append(update['evaluation_id'])
            interrupt('pause after pair task')
            return update
        builder = StateGraph(CfuDesignState)
        builder.add_node('pair', node)
        builder.add_edge(START, 'pair')
        builder.add_edge('pair', END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'unit-pair-replay'}}
        with mock.patch.object(parallel.software, 'simulate', side_effect=worker('sim')), \
             mock.patch.object(parallel.cost, 'estimate_cost', side_effect=worker('cost')), \
             mock.patch.object(effects, '_command_task', side_effect=AssertionError('nested task')):
            graph.invoke({'validation_mode': 'parallel'}, config)
            update = graph.invoke(Command(resume=True), config)
        self.assertCountEqual(calls, ['sim', 'cost'])
        self.assertEqual(ids[0], ids[1])
        self.assertTrue(update['gates']['validation_pair']['ok'])

    def test_checks_nonzero_notifies_group_before_persisting_log(self):
        order = []
        writer = mock.Mock()
        writer.write_text.side_effect = lambda *a, **k: (order.append('log') or {'path': 'fake.log'})
        toolbox = SimpleNamespace(workdir='.', run_command=lambda *a, **k: {
            'returncode': 1, 'command': 'false', 'stdout_tail': '', 'stderr_tail': 'failed'})
        with mock.patch.object(validation_parallel, 'command_failed', side_effect=lambda: order.append('cancel')), \
             mock.patch.object(checks.stage_services, '_artifacts_for_stage', return_value=writer):
            checks._run_check({}, toolbox, ['false'], rung='sim', command_id='fake')
        self.assertEqual(order[:2], ['cancel', 'log'])

    def test_cancelled_command_persists_full_output_without_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = SimpleNamespace(version='unit', check_write=lambda *a: None)
            writer = artifacts.RunArtifacts.create(root, policy, run_id='cancel-log')
            token = processes.CancellationToken()
            error = processes.ValidationCancelled()
            error.started = True
            error.stdout = 'complete prefix\\n' + 'output\\n' * 500
            error.stderr = 'cancelled tail'
            error.returncode = -15
            def command(*a, **k):
                token.cancel()
                raise error
            box = SimpleNamespace(workdir=root, run_command=command)
            with mock.patch.object(checks.stage_services, '_artifacts_for_stage', return_value=writer), \
                 mock.patch.object(validation_parallel, 'command_failed') as notify:
                with processes.use_token(token):
                    with self.assertRaises(processes.ValidationCancelled) as caught:
                        checks._run_check({'evaluation_id': 'eval-test'}, box, ['fake'], rung='cost', command_id='fake')
            notify.assert_not_called()
            entry = writer.manifest['artifacts'][0]
            self.assertEqual(entry['status'], 'cancelled')
            text = (root / entry['path']).read_text()
            self.assertIn(error.stdout, text)
            self.assertIn(error.stderr, text)
            self.assertEqual(caught.exception.result['returncode'], -15)
            self.assertNotIn('failure_observation', caught.exception.result)
            self.assertIn('/evaluations/eval-test/cost/logs/', entry['path'])

    def test_simulate_worker_does_not_transition(self):
        box = SimpleNamespace(workdir=Path('.'))
        with mock.patch.object(software.stage_services, '_profile_reason', return_value=''), \
             mock.patch.object(software.stage_services, '_toolbox_for_stage', return_value=box), \
             mock.patch.object(software.intrinsics, '_candidate_change_domains', return_value={'hw': True}), \
             mock.patch.object(software.commands_mod, '_software_sim_commands', return_value=[['fake']]), \
             mock.patch.object(software.stage_services, 'benchmark_integrity_problems', return_value=[]), \
             mock.patch.object(software.effects, 'run_command', return_value={'returncode': 0}), \
             mock.patch.object(software.stage_services, '_write_ladder_artifact'), \
             mock.patch.object(software.ladder, '_ladder_transition', side_effect=AssertionError('worker budget')):
            update = software.simulate({}, transition=False)
        self.assertTrue(update['gates']['sim']['ok'])
        self.assertNotIn('next', update['gates']['sim'])


class ConcurrentMetadataTest(unittest.TestCase):
    def test_journal_begin_and_commit_reload_without_serializing_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'journal.json'
            barrier = threading.Barrier(2)
            def operation(name):
                journal = persistence.OperationJournal(path, 'test')
                with journal.external_operation('command', name):
                    barrier.wait(timeout=2)
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(operation, ['one', 'two']))
            journal = persistence.OperationJournal(path, 'test')
            self.assertEqual({key: value['status'] for key, value in journal._load()['operations'].items()},
                             {'one': 'committed', 'two': 'committed'})

    def test_confirmed_cancellation_commits_journal_but_uncertain_termination_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = persistence.OperationJournal(Path(directory) / 'journal.json', 'test')
            with self.assertRaises(processes.ValidationCancelled):
                with journal.external_operation('command', 'cancelled'):
                    raise processes.ValidationCancelled()
            journal.assert_no_pending()
            self.assertEqual(journal._load()['operations']['cancelled']['outcome'], 'cancelled')
            with self.assertRaises(processes.ProcessTerminationError):
                with journal.external_operation('command', 'uncertain'):
                    raise processes.ProcessTerminationError('not reaped')
            with self.assertRaises(persistence.PendingOperationError):
                journal.assert_no_pending()

    def test_simultaneous_artifact_refs_and_metadata_survive_stale_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = SimpleNamespace(version='unit', check_write=lambda *a: None)
            instances = [artifacts.RunArtifacts.create(root, policy, run_id='metadata-unit') for _ in range(2)]
            barrier = threading.Barrier(2)
            def write(index):
                barrier.wait(timeout=2)
                return instances[index].write_text('validation', f'10_validation/lane-{index}.log', str(index))
            with ThreadPoolExecutor(max_workers=2) as executor:
                refs = list(executor.map(write, range(2)))
            instances[0].set_manifest_metadata('validation', 'one', 1)
            instances[1].set_manifest_metadata('validation', 'two', 2)
            manifest = yaml.safe_load((instances[0].root / 'run.yaml').read_text())
            self.assertCountEqual([ref['path'] for ref in manifest['artifacts']], [ref['path'] for ref in refs])
            self.assertEqual((manifest['one'], manifest['two']), (1, 2))


class ProcessCancellationTest(unittest.TestCase):
    def test_cancel_before_spawn_never_starts_process(self):
        token = processes.CancellationToken()
        token.cancel()
        with mock.patch.object(processes.subprocess, 'Popen') as spawn:
            with self.assertRaises(processes.ValidationCancelled):
                with processes.use_token(token):
                    processes.run(['fake'])
        spawn.assert_not_called()

    def test_cancel_between_spawn_and_registration_kills_and_reaps(self):
        token = processes.CancellationToken()
        process = mock.Mock(pid=12345, returncode=-15)
        process.communicate.return_value = ('partial before registration', '')
        def spawn(*args, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            token.cancel()
            return process
        with mock.patch.object(processes.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(processes, 'terminate_tree') as terminate:
            with processes.use_token(token):
                with self.assertRaises(processes.ValidationCancelled):
                    processes.run(['fake'])
        terminate.assert_called_once_with(process)

    def test_cancellation_releases_waiting_command_and_not_just_future(self):
        started = threading.Event()
        released = threading.Event()
        process = mock.Mock(pid=12345, returncode=-15)
        def communicate(**kwargs):
            started.set()
            self.assertTrue(released.wait(2))
            return ('partial', '')
        process.communicate.side_effect = communicate
        token = processes.CancellationToken()
        def run():
            with processes.use_token(token):
                return processes.run(['fake'])
        with mock.patch.object(processes.subprocess, 'Popen', return_value=process) as spawn, \
             mock.patch.object(processes, 'terminate_tree', side_effect=lambda p: released.set()) as terminate:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run)
                self.assertTrue(started.wait(2))
                token.cancel()
                with self.assertRaises(processes.ValidationCancelled) as cancelled:
                    future.result(timeout=2)
                self.assertEqual(cancelled.exception.stdout, 'partial')
                self.assertEqual(cancelled.exception.returncode, -15)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        terminate.assert_called_once_with(process)

    def test_term_grace_then_kill_targets_group_and_reaps(self):
        process = mock.Mock(pid=12345)
        with mock.patch.object(processes, 'TERM_GRACE_SECONDS', 0), \
             mock.patch.object(processes, '_group_alive', side_effect=[True, True, False, False]), \
             mock.patch.object(processes.os, 'killpg') as kill:
            processes.terminate_tree(process)
        self.assertEqual(kill.call_args_list, [mock.call(12345, processes.signal.SIGTERM),
                                               mock.call(12345, processes.signal.SIGKILL)])
        process.wait.assert_called_once_with(timeout=processes.KILL_REAP_SECONDS)

    def test_unconfirmed_termination_is_harness_error(self):
        process = mock.Mock(pid=12345)
        with mock.patch.object(processes, 'TERM_GRACE_SECONDS', 0), \
             mock.patch.object(processes, 'KILL_REAP_SECONDS', 0), \
             mock.patch.object(processes, '_group_alive', return_value=True), \
             mock.patch.object(processes.os, 'killpg'):
            with self.assertRaises(processes.ProcessTerminationError):
                processes.terminate_tree(process)

    def test_timeout_kills_tree_preserves_full_output(self):
        process = mock.Mock(pid=12345)
        process.communicate.side_effect = [subprocess.TimeoutExpired(['fake'], 1), ('all stdout', 'all stderr')]
        with mock.patch.object(processes.subprocess, 'Popen', return_value=process), \
             mock.patch.object(processes, 'terminate_tree') as terminate:
            with self.assertRaises(subprocess.TimeoutExpired) as error:
                processes.run(['fake'], timeout=1)
        self.assertEqual(error.exception.stdout, 'all stdout')
        self.assertEqual(error.exception.stderr, 'all stderr')
        terminate.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
