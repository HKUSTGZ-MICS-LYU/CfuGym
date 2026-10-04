"""Oracle authoring inside Implement for the no-tool delivery mode."""
from ... import prompts
from ...runtime import commands as commands_mod, conformance_delivery, effects, llm_step, stage_services
from ...state import CfuDesignState
from . import intrinsics


def _delivered_test_errors(state: CfuDesignState, toolbox, header_path: str) -> list[str]:
    """Host-owned compile preflight of the pair just written, before commit.

    A delivered test that cannot compile must not reach ``hw_conformance`` (run
    20261003-130005 spent its last answer-repair round on a `redefinition` in the
    generated C). Only a real compiler error (positive return code) is reported:
    a policy denial, a missing toolchain or a timeout is left to the final gate,
    because the model cannot fix it.
    """
    if not str(header_path or "").strip():
        return []
    try:
        argv = commands_mod._conformance_syntax_command(state, header_path)
    except ValueError:
        return []
    result = effects.run_command(state, toolbox, argv, stage="implementation",
                                 rung="conformance_syntax", command_id="conformance_syntax",
                                 timeout=120, mutate=True)
    returncode = result.get("returncode")
    if not isinstance(returncode, int) or returncode <= 0:
        return []
    text = " ".join(str(result.get("stderr_tail") or result.get("stdout_tail") or "").split())
    return [f"the delivered test does not compile: {text[-1200:]}"]


def write_conformance(state: CfuDesignState) -> dict:
    if state.get("dry_run") or not state.get("apply_patch", False):
        return {"iteration_history": ["Conformance oracle delivery planned; no files written or simulation run."]}
    toolbox = stage_services._toolbox_for_stage(state, "implementation")
    previous, problems = conformance_delivery.inspect_delivery(state, toolbox)
    # An approved oracle is reused only when nothing is being repaired. Every
    # repair round re-offers the oracle so the model can strengthen a contract
    # that could not see the failure; without tools this is the only way to
    # revise it (run 20261003-090854-136683-d98e5be9 could never do so).
    if not problems and not state.get("repair_scope"):
        return {"conformance_delivery": previous}
    from .. import feedback
    user = prompts.conformance_prompt(state["task"], state.get("isa_spec") or {},
        state.get("workload_spec") or {}, feedback=feedback.conformance_feedback(state),
        agent_tools=False, workdir=state.get("workdir", ""), run_root=state.get("run_root", ""))
    paths = dict(conformance_delivery.delivery_paths(state))
    header = intrinsics.intrinsics_header_path(state)
    if header:
        paths["header_path"] = header
    snapshot = conformance_delivery._snapshot(toolbox, paths, write=True)
    session = None
    try:
        result, raw = llm_step._complete_llm(state, "implement", "conformance_author", prompts.SYSTEM_PROMPT,
            user, "commands: []", response_format={"type": "json_object"})
        header = intrinsics._write_intrinsics_header(state, toolbox)
        session = conformance_delivery.ConformanceDelivery(state, toolbox, header)
        delivery, errors = session.write(result.text)
        if not errors:
            errors = _delivered_test_errors(state, toolbox, header)
        if errors:
            session.rollback()
            conformance_delivery._restore(toolbox, snapshot)
            if previous and not problems and state.get("repair_scope") != "conformance":
                # A rejected *revision* of an already valid oracle must not fail
                # the design repair it was offered for: keep the approved pair
                # and record the attempt instead of rolling the round back.
                return {"conformance_delivery": previous, "llm_raw_outputs": [raw],
                        "iteration_history": [
                            "Conformance oracle revision rejected (" + "; ".join(errors)
                            + "); kept the approved oracle."]}
            return {"conformance_delivery": state.get("conformance_delivery") or {},
                    "errors": errors, "llm_raw_outputs": [raw]}
        session.commit()
        return {"conformance_delivery": delivery, "repair_scope": "", "llm_raw_outputs": [raw]}
    except Exception:
        if session is not None:
            session.rollback()
        conformance_delivery._restore(toolbox, snapshot)
        raise
