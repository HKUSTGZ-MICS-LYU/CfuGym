"""Separate a measured snapshot from the final candidate's performance.

A historical passing sim gate is evidence about its input bytes, not about a
later repair, rewind, or failed build. Missing provenance never proves a match.
"""
from __future__ import annotations

from typing import Any

from . import commands, dependencies

METRIC_FIELDS = ("reference_cycles", "cfu_cycles", "speedup_x100", "reference_result", "cfu_result",
                 "min_speedup_x100", "correctness_ok", "timing_ok", "performance_valid",
                 "measurement_protocol", "measurement", "terminal_reason")


def compiled_products(state: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Hash known ELF/RTL/ASM products in the workdir without executing tools."""
    import hashlib
    from pathlib import Path

    if (state.get("benchmark_project") or {}).get("kind") == "embench_benchmark" and not state.get("run_root"):
        return {}  # No run binding: do not invent /workspace products.
    root = Path(state.get("workdir", ".")).resolve()
    products = {}
    elf = commands._software_elf(state)
    rtl = commands._soc_rtl_path(state)
    paths = [("elf", elf), ("rtl", rtl)]
    if elf:
        paths.append(("asm", Path(elf).with_suffix(".asm")))
    for kind, raw in paths:
        if not raw:
            continue
        path = (root / raw).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"compiled product escapes workdir: {kind}")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        products[kind] = {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest()}
    return products


def candidate_provenance(state: dict[str, Any]) -> dict[str, Any]:
    """Stamp a simulation at execution time using authoritative input bytes.

    Put this dict in ``sim.performance_provenance`` before persisting sim YAML.
    Callers may append ``snapshot``, ``version``, ``sha256`` and ``log_paths``
    from real artifact references; no synthetic snapshot identifiers are needed.
    """
    gates = state.get("gates") or {}
    hw, sw = gates.get("implementation_hw") or {}, gates.get("implementation_sw") or {}
    hashes = {key: value for key, value in {
        "hardware": hw.get("candidate_hash"), "software": sw.get("candidate_hash"),
        "isa": sw.get("isa_hash"),
    }.items() if value}
    return {"input_sha256": dependencies.input_fingerprints(state)["sim"], "candidate_hashes": hashes,
            "evidence_epoch": dict(state.get("repair_evidence_epoch") or {}),
            "compiled_products": compiled_products(state)}


def observation_is_valid(sim: dict[str, Any]) -> bool:
    """Validity of the observation itself, independent of the final workspace."""
    if any(sim.get(key) is False for key in ("correctness_ok", "timing_ok", "performance_valid")):
        return False
    if (sim.get("benchmark_integrity") or {}).get("ok") is False:
        return False
    reason = sim.get("terminal_reason")
    if reason and reason not in {"passed", "speedup_floor"}:
        return False
    if not reason and not (sim.get("ok") is True or sim.get("performance_valid") is True):
        return False
    return all(isinstance(sim.get(key), (int, float)) and not isinstance(sim.get(key), bool)
               and sim[key] > 0 for key in ("reference_cycles", "cfu_cycles", "speedup_x100"))


def observed_performance(sim: dict[str, Any], *, evidence: dict[str, Any] | None = None) -> dict:
    """Copy real measurement/provenance fields, never fabricate unknown metrics."""
    if not observation_is_valid(sim):
        return {}
    result = {key: sim[key] for key in METRIC_FIELDS if key in sim}
    result["provenance"] = dict(sim.get("performance_provenance") or {})
    if sim.get("input_sha256") and not result["provenance"].get("input_sha256"):
        result["provenance"]["input_sha256"] = sim["input_sha256"]
    if evidence:
        result["evidence"] = dict(evidence)
    result["scope"] = "last_observed_valid"
    return result


def current_performance(sim: dict[str, Any], candidate: dict[str, Any], *,
                        status: str = "", stopped_at: str = "") -> dict:
    proof = sim.get("performance_provenance") or {}
    recorded_hash = proof.get("input_sha256") or sim.get("input_sha256")
    current_hash = candidate.get("input_sha256")
    matches = bool(proof and recorded_hash and current_hash and recorded_hash == current_hash)
    if proof.get("candidate_hashes"):
        matches = matches and proof["candidate_hashes"] == candidate.get("candidate_hashes")
    if "evidence_epoch" in proof:
        matches = matches and proof["evidence_epoch"] == candidate.get("evidence_epoch")
    # Legacy source-only proof cannot certify newly known binary products.
    if proof.get("compiled_products") or candidate.get("compiled_products"):
        matches = matches and proof.get("compiled_products") == candidate.get("compiled_products")
    terminal = sim.get("terminal_reason")
    upstream_failure = status in {"failed", "blocked", "needs_iteration"} and stopped_at not in {"", "sim", "cost", "review", "final_report"}
    valid = bool(observation_is_valid(sim) and matches and terminal in {"passed", "speedup_floor"}
                 and not sim.get("stale") and not upstream_failure)
    reason = ("matching candidate and valid terminal observation" if valid else
              "final candidate stopped before simulation" if upstream_failure else
              "simulation inputs are stale" if sim.get("stale") else
              "candidate provenance missing or mismatched" if not matches else
              "no valid passed/speedup_floor terminal observation")
    return {**({key: sim[key] for key in METRIC_FIELDS if key in sim} if valid else {}),
            "performance_valid": valid, "candidate_matches": matches, "reason": reason,
            "candidate_provenance": candidate, "scope": "current_candidate"}


def load_last_observed_performance(run_dir, warnings: list[str] | None = None) -> dict:
    """Read immutable measurement versions, retaining their exact content hash."""
    import hashlib
    from pathlib import Path
    import yaml

    warnings = warnings if warnings is not None else []
    root = Path(run_dir)
    validation = root / "10_validation"
    paths = [*validation.glob("versions/sim.yaml.*"), *validation.glob("versions/embench.yaml.*")]
    paths.extend(path for path in (validation / "sim.yaml", validation / "embench.yaml") if path.is_file())
    try:
        paths.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError as exc:
        warnings.append(f"performance history unavailable: {exc}")
        return {}
    for path in paths:
        try:
            data = path.read_bytes()
            sim = yaml.safe_load(data)
        except (OSError, yaml.YAMLError) as exc:
            warnings.append(f"performance snapshot {path.name} unparsed: {exc}")
            continue
        if not isinstance(sim, dict):
            continue
        # Legacy embench.yaml uses status=passed, not ok=True.
        if sim.get("status") == "passed" and not sim.get("terminal_reason"):
            sim = {**sim, "ok": True}
        observed = observed_performance(sim, evidence={"snapshot": str(path),
            "sha256": hashlib.sha256(data).hexdigest(), "version": path.name})
        if observed:
            return observed
    return {}


def archived_performance(record: dict, *, evidence: dict | None = None) -> dict:
    """Recover an archived observation, never attest to its final candidate."""
    existing = record.get("last_observed_valid_performance")
    if isinstance(existing, dict) and existing:
        return {**existing, **({"archive_evidence": evidence} if evidence else {})}
    sim = {key: record[key] for key in METRIC_FIELDS if key in record}
    if record.get("sim_terminal_reason"):
        sim["terminal_reason"] = record["sim_terminal_reason"]
    sim.update((record.get("gates") or {}).get("sim") or {})
    return observed_performance(sim, evidence=evidence)


def archive_evidence(path) -> dict:
    """Hash the actual archived record, not an unavailable original simulation."""
    import hashlib
    from pathlib import Path

    path = Path(path)
    return {"snapshot": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": path.name, "source": "archived_record"}


def performance_view(state: dict[str, Any]) -> dict:
    sim = (state.get("gates") or {}).get("sim") or {}
    candidate = state.get("current_candidate_provenance") or {}
    if not candidate and state.get("workdir") and state.get("run_root"):
        try:
            candidate = candidate_provenance(state)
        except (OSError, ValueError):
            candidate = {}
    history = state.get("last_observed_valid_performance") or {}
    if not history and state.get("workdir") and state.get("run_root"):
        from pathlib import Path
        history = load_last_observed_performance(Path(state["workdir"]) / state["run_root"])
    history = history or observed_performance(sim)
    return {"current_performance": current_performance(sim, candidate,
            status=str(state.get("status") or ""), stopped_at=str(state.get("stopped_at") or "")),
            "last_observed_valid_performance": history}
