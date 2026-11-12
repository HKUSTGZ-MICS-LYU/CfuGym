"""Conservative, report-only selection of a failed run's best simulation."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import yaml

_HASH = re.compile(r"^[0-9a-f]{64}$")


def is_floor_failure(record: dict) -> bool:
    return bool(record.get("status") in {"failed", "needs_iteration", "blocked"}
                and record.get("stopped_at") == "sim"
                and (record.get("failure") or {}).get("failure_class") == "sim_speedup_floor"
                and record.get("sim_terminal_reason") == "speedup_floor")


def best_speedup(record: dict, run_dir: Path) -> dict:
    """Use only content-addressed sim versions written during this attempt.

    This is evidence of the historical candidate, never the final candidate.
    Missing identity, protocol, timing, correctness or attempt boundary fails closed.
    """
    if not is_floor_failure(record) or record.get("stale_framework_artifacts"):
        return {}
    protocol = record.get("measurement_protocol")
    if not isinstance(protocol, str) or not protocol or protocol == "legacy":
        return {}
    try:
        start = datetime.fromisoformat(str(record.get("started_at") or "").replace("Z", "+00:00"))
        if start.tzinfo is None:
            return {}
        finish = datetime.fromisoformat(str(record.get("finished_at") or "").replace("Z", "+00:00"))
        if finish.tzinfo is None or finish < start:
            return {}
        start_time, finish_time = start.timestamp(), finish.timestamp()
    except (ValueError, OverflowError):
        return {}
    best = {}
    for path in sorted((run_dir / "10_validation/versions").glob("sim.yaml.*")):
        try:
            if not start_time <= path.stat().st_mtime <= finish_time:
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if path.name != "sim.yaml." + digest:
                continue
            sim = yaml.safe_load(raw)
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(sim, dict):
            continue
        # Missing means a pre-schema snapshot, not an implicit complete design.
        # Explicit invalid values must not be promoted to historical evidence.
        if "implementation_design_type" in sim and sim["implementation_design_type"] not in ("complete", "legacy"):
            continue
        integrity = sim.get("benchmark_integrity")
        commands = sim.get("commands")
        if not isinstance(commands, list) or not any(
                isinstance(cmd, dict) and cmd.get("kind") not in {"host_gate_check", "semantic_check"}
                and cmd.get("returncode") == 0 and not cmd.get("skipped") and not cmd.get("cancelled")
                for cmd in commands):
            continue
        if (sim.get("measurement_protocol") != protocol
                or any(sim.get(k) is not True for k in ("correctness_ok", "timing_ok", "performance_valid", "design_changed"))
                or not isinstance(integrity, dict) or integrity.get("ok") is not True
                or sim.get("terminal_reason") not in {"passed", "speedup_floor"}
                or any(sim.get(k) for k in ("stale", "skipped", "cancelled"))):
            continue
        ref, cfu, speed = (sim.get(k) for k in ("reference_cycles", "cfu_cycles", "speedup_x100"))
        if (any(type(v) is not int or v <= 0 for v in (ref, cfu))
                or type(speed) is not int or speed < 0
                or type(record.get("reference_cycles")) is not int
                or ref != record["reference_cycles"] or speed != ref * 100 // cfu):
            continue
        proof = sim.get("performance_provenance")
        if not isinstance(proof, dict):
            continue
        hashes, products = proof.get("candidate_hashes"), proof.get("compiled_products")
        if (not _HASH.fullmatch(str(proof.get("input_sha256") or ""))
                or not isinstance(hashes, dict)
                or any(not _HASH.fullmatch(str(hashes.get(k) or "")) for k in ("hardware", "software"))
                or not re.fullmatch(r"[0-9a-f]{16}", str(hashes.get("isa") or ""))
                or not isinstance(proof.get("evidence_epoch"), dict)
                or not isinstance(products, dict)
                or any(not isinstance(products.get(k), dict)
                       or not products[k].get("path")
                       or not _HASH.fullmatch(str(products[k].get("sha256") or ""))
                       for k in ("elf", "rtl", "asm"))):
            continue
        if not best or ref * best["cfu_cycles"] > best["reference_cycles"] * cfu:
            best = {"speedup_x100": speed, "reference_cycles": ref, "cfu_cycles": cfu,
                    "measurement_protocol": protocol, "terminal_reason": sim["terminal_reason"],
                    "scope": "historical_candidate", "provenance": proof,
                    "evidence": {"snapshot": str(path), "sha256": digest}}
    return best
