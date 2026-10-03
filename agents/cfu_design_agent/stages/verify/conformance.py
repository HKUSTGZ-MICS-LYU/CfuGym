"""Deterministic formal conformance verification of implementation deliveries."""

from __future__ import annotations

from ... import conformance, isa, state as state_mod
from ...flow import ladder, outcome
from ...runtime import conformance_delivery, effects, commands, stage_services
from ...runtime.commands import CONFORMANCE_SUBDIR
from ...runtime.policy import PolicyError
from ...stages import feedback
from ...state import CfuDesignState


def check_conformance(state: CfuDesignState) -> CfuDesignState:
    """Compile and simulate the delivered test without requesting or writing source.

    Implement owns the oracle, generated C and intrinsics header. Verify only
    consumes a fresh delivery and records host-owned command/verdict artifacts;
    a missing or stale answer returns to Implement under the answer budget.
    """
    rung = "hw_conformance"
    reason = stage_services._profile_reason(state)
    if reason:
        result = stage_services._skipped_result("make cfu_conformance + MiCoSocSim", state, reason)
        result["rung"] = rung
        stage_services._write_ladder_artifact(state, rung, [result], "planned", reason)
        return {
            "validation_results": [result],
            "stage_jobs": "validation",
            **outcome.stage_result(state, rung=rung, ok=False, skipped=True, reason=reason,
                                   history=f"Skipped hardware conformance ({reason})."),
        }

    isa_spec = state.get("isa_spec", {}) or {}
    artifacts = stage_services._artifacts_for_stage(state, "implementation")
    toolbox = stage_services._toolbox_for_stage(state, "conformance")
    errors: list[str] = []
    oracle_attempts: list[dict] = []
    oracle_failure = False
    environment_failure = False
    delivery: dict = {}
    try:
        delivery, problems = conformance_delivery.inspect_delivery(state, toolbox)
        errors.extend(problems)
        oracle_failure = bool(problems)
    except (PolicyError, OSError) as exc:
        environment_failure = True
        errors.append(f"conformance delivery cannot be inspected: {type(exc).__name__}: {exc}")

    # The implementation gate owns this header. Never generate it in Verify or
    # substitute a newly rendered header for the delivered implementation bytes.
    approved = state.get("conformance_delivery")
    approved_header = approved.get("header_path") if isinstance(approved, dict) else ""
    header_path = str(((state.get("gates") or {}).get("implementation_sw") or {}).get(
        "intrinsics_header") or approved_header or "")
    if approved_header and header_path != approved_header:
        oracle_failure = True
        errors.append("the implementation header path differs from the approved conformance delivery")
    if not environment_failure:
        if not header_path:
            oracle_failure = True
            errors.append("the delivered CFU intrinsics header path is unavailable")
        else:
            try:
                header_text = toolbox.read_text(header_path, limit=1_000_000)
            except (PolicyError, OSError) as exc:
                environment_failure = True
                errors.append(f"CFU intrinsics header cannot be read: {type(exc).__name__}: {exc}")
            else:
                if not header_text:
                    oracle_failure = True
                    errors.append(f"the delivered CFU intrinsics header is missing or empty: {header_path}")
                elif header_text != isa.render_cfu_intrinsics(isa_spec):
                    oracle_failure = True
                    errors.append("the delivered CFU intrinsics header differs from the current ISA; regenerate in Implement")

    results: list[dict] = []
    verdict: dict = {"status": "missing", "ok": False, "commands": 0, "total": 0,
                     "cases": 0, "failed": [], "details": []}
    if not errors:
        build_argv, elf = commands._conformance_build_command(state, header_path)
        build = effects.run_command(
            state, toolbox, build_argv, stage="conformance", rung=rung,
            command_id="conformance_build", mutate=True,
        )
        results.append(build)
        if build.get("returncode") not in (0, None):
            # The test embeds oracle C, so compiler/linker rejection is an answer
            # problem unless typed evidence identifies a host/policy failure.
            oracle_failure = True
            errors.append(feedback.conformance_build_error(build, build_argv))
        else:
            for argv in commands.agent_cfu_sim_commands(state, isa_spec, elf=elf):
                sim = effects.run_command(
                    state, toolbox, argv, stage="conformance", rung=rung,
                    command_id="soc_sim", timeout=3600, mutate=True,
                )
                results.append(sim)
                verdict = (
                    (sim.get("parsed") or {}).get("conformance")
                    or conformance.parse_conformance(
                        "\n".join([str(sim.get("stdout_tail") or ""), str(sim.get("stderr_tail") or "")])
                    )
                )
                if sim.get("returncode") not in (0, None):
                    errors.append(feedback.conformance_sim_error(sim, feedback._tool_diagnostics(sim)))
                if not verdict["ok"]:
                    detail = (
                        f"commands={verdict['commands']}/{verdict['total']} "
                        f"cases={verdict['cases']}"
                    )
                    if verdict["failed"]:
                        detail += f" failed={','.join(verdict['failed'])}"
                    errors.append(f"conformance verdict {verdict['status']}: {detail}")
                if errors:
                    break

    ok = (
        not errors
        and bool(results)
        and all(r.get("returncode") in (0, None) for r in results)
        and bool(verdict["ok"])
    )
    # Host/policy failures do not request another oracle or spend repair budget.
    repair_target = (
        None if environment_failure
        else feedback.conformance_repair_target(oracle_failure, results)
    )
    if repair_target is not None:
        repair_target = "implement"
    if oracle_failure and repair_target is not None:
        attempt = feedback._rejection_attempt(errors, "")
        if attempt:
            oracle_attempts.append(attempt)

    artifacts.write_yaml(
        "implementation", f"{CONFORMANCE_SUBDIR}/result.yaml",
        {"ok": ok, "verdict": verdict, "errors": errors,
         "commands": [r.get("command") for r in results],
         "delivery": delivery, "oracle_failure": oracle_failure},
        status="passed" if ok else "failed",
    )
    stage_services._write_ladder_artifact(state, rung, results, "passed" if ok else "failed", "")
    if repair_target is None and not ok:
        note = (
            f"{rung} failed with a simulation-harness/environment/policy failure; "
            "the failure is not attributable to the design, so no repair budget is spent."
        )
        extra = {
            "stopped_at": rung,
            "stage_jobs": "review",
            "errors": [note],
            "iteration_history": state_mod.append_history(state, note),
        }
        next_node = "review_and_route"
    else:
        extra, next_node = ladder._ladder_transition(
            state, rung, ok=ok, default="build_sw", repair_target=repair_target,
            repair_kind="answer" if oracle_failure else None,
        )
        if not ok and next_node == "implement":
            extra["repair_scope"] = "conformance" if oracle_failure else "design"
    update = {
        "validation_results": results,
        "stage_jobs": "validation" if ok else "implementation",
        "errors": errors,
        "conformance_verdict": verdict,
        "conformance_attempts": oracle_attempts,
        **outcome.stage_result(
            state, rung=rung, ok=ok, next=next_node, errors=errors,
            details=verdict.get("details") or [], verdict=verdict.get("status"),
            cases=verdict.get("cases"), repair_target=repair_target,
            oracle_failure=oracle_failure, repairable=repair_target is not None,
            history=f"Hardware conformance {'passed' if ok else 'failed'} "
            f"({verdict.get('commands')}/{verdict.get('total')} commands, "
            f"{verdict.get('cases')} cases).",
        ),
    }
    return ladder._apply_transition(update, extra)


def validate_render(text: dict) -> str:
    import yaml as _yaml
    return _yaml.safe_dump(text, sort_keys=False)
