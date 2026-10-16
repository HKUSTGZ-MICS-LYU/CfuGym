"""One coordinator node for candidate simulation and cost validation."""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from langgraph.func import task

from ...flow import ladder, outcome
from ...runtime import dependencies, effects, stage_services, validation_parallel
from ...state import CfuDesignState
from . import cost, software

_WORKER_CHANNELS = frozenset({'validation_results', 'cost_results', 'errors', 'iteration_history'})


def _evaluate(state: CfuDesignState, evaluation_id: str) -> dict:
    mode = state.get('validation_mode', 'serial')
    if mode not in ('parallel', 'serial'):
        raise ValueError(f'unknown validation_mode: {mode!r}')
    candidate = deepcopy(state)
    candidate['evaluation_id'] = evaluation_id
    input_fingerprints = dependencies.input_fingerprints(candidate)

    def worker(rung, execute):
        # No worker may mutate shared graph state or submit its inherited gates.
        update = execute(deepcopy(candidate), transition=False)
        clean = {key: value for key, value in update.items() if key in _WORKER_CHANNELS}
        gate = (update.get('gates') or {}).get(rung)
        if not isinstance(gate, dict):
            raise RuntimeError(f'{rung} validation worker returned no rung gate')
        clean['gates'] = {rung: {**gate, 'evaluation_id': evaluation_id,
                                 'input_sha256': input_fingerprints.get(rung, '')}}
        return clean

    pair = validation_parallel.run_pair({
        'sim': lambda: worker('sim', software.simulate),
        'cost': lambda: worker('cost', cost.estimate_cost),
    }, parallel=mode == 'parallel')
    if dependencies.input_fingerprints(candidate) != input_fingerprints:
        raise RuntimeError('candidate inputs changed during validation; refusing stale evidence')
    update: dict = {}
    for rung in ('sim', 'cost'):
        if rung in pair.updates:
            update = outcome.merge_updates(update, pair.updates[rung])
        else:
            update.setdefault('gates', {})[rung] = {
                'ok': False, 'skipped': True, 'cancelled': True,
                'reason': 'sibling_failed', 'evaluation_id': evaluation_id,
                **pair.cancelled_logs.get(rung, {}),
            }
    gate_map = update['gates']
    evaluation = {
        'id': evaluation_id,
        'mode': mode,
        'iteration': int(state.get('iteration', 0) or 0),
        'round': int(state.get('optimization_round', 0) or 0),
        'rtl_snapshot_sha256': (state.get('rtl_snapshot') or {}).get('sha256', ''),
        'input_fingerprints': input_fingerprints,
        'failure': pair.failure,
        'cancelled': pair.cancelled,
        'cancelled_logs': pair.cancelled_logs,
        'lanes': {rung: ('cancelled' if gate.get('cancelled') else
                         'skipped' if gate.get('skipped') else
                         'passed' if gate.get('ok') else 'failed')
                  for rung, gate in gate_map.items()},
    }
    update.update(evaluation_id=evaluation_id, evaluation=evaluation)
    # Write inside the group task: a replay restores evidence without rerunning
    # workers. A standalone helper remains usable with minimal mocked state.
    if state.get('run_id'):
        stage_services._artifacts_for_stage(candidate, 'validation').write_yaml(
            'validation', f'10_validation/evaluations/{evaluation_id}/pair.yaml',
            {'evaluation': evaluation, 'gates': gate_map},
            status='failed' if pair.failure else 'produced')
    return update


@task
def _validation_pair_task(evaluation_id: str) -> dict:
    state = effects.current_state()
    if state is None:
        raise RuntimeError('validation group task ran without node state')
    return _evaluate(state, evaluation_id)


def validate_candidate(state: CfuDesignState) -> CfuDesignState:
    """Validate one immutable input/round, then make exactly one ladder decision.

    The native group task owns both command streams. Worker contexts force plain
    checks so nested @task futures cannot keep a failed sibling alive. Outside a
    graph, the same coordinator executes directly; legacy simulate/cost entry
    points retain their independent default transitions.
    """
    evaluation_id = (f"eval-{int(state.get('iteration', 0) or 0)}-"
                     f"{int(state.get('optimization_round', 0) or 0)}-{uuid4().hex}")
    update = deepcopy(_validation_pair_task(evaluation_id).result() if effects.inside_run()
                      else _evaluate(state, evaluation_id))
    evaluation = update['evaluation']
    if dependencies.input_fingerprints(state) != evaluation['input_fingerprints']:
        raise RuntimeError('candidate inputs changed before validation commit; refusing stale evidence')
    failed_rung = evaluation['failure']
    pair_ok = all(gate.get('ok') for gate in update['gates'].values())
    skipped = not failed_rung and not pair_ok
    extra, next_node = ladder._ladder_transition(
        {**state, 'gates': {**(state.get('gates') or {}), **update['gates']}},
        failed_rung or 'validation_pair', ok=pair_ok,
        default='review_and_route', repair_target='implement', skipped=skipped)
    update['gates'] = outcome.gates(state, {
        **update['gates'],
        'validation_pair': {
            'ok': pair_ok, 'skipped': skipped, 'next': next_node,
            'evaluation_id': update['evaluation_id'],
            'failed_rung': failed_rung, 'cancelled': evaluation['cancelled'],
            'round': evaluation['round'],
        },
    })
    update['stage_jobs'] = 'review'
    return outcome.merge_updates(update, extra)
