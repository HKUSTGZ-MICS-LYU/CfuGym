"""Experiment LLM configs (temperature control)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def materialize_configs(
    exp_id: str, models, temperature: float, workdir: Path
) -> dict[str, dict]:
    """Copy each base LLM config with an explicit ``temperature``.

    The stock configs leave temperature unset (= 0), which would collapse the R
    repeats into the same trajectory and make pass@k meaningless.  Generated
    files land beside the bases (agents/llm_configs is gitignored) and are never
    written back to the tracked/edited base config.
    """
    import yaml  # noqa: F401  (kept out of the hot path)

    cfg_dir = workdir / "agents" / "llm_configs"
    out: dict[str, dict] = {}
    for model in models:
        base_path = cfg_dir / f"{model}.json"
        if not base_path.exists():
            raise SystemExit(
                f"LLM config not found: {base_path} "
                f"(available: {', '.join(sorted(p.stem for p in cfg_dir.glob('*.json'))) or 'none'})"
            )
        data = json.loads(base_path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"LLM config must be a JSON object: {base_path}")
        name = f"{exp_id}-{model}"
        generated = dict(data)
        generated["name"] = name
        generated["temperature"] = float(temperature)
        generated["description"] = (
            f"{data.get('description', data.get('name', model))} "
            f"[experiment {exp_id}: temperature={temperature}]"
        )
        out_path = cfg_dir / f"{name}.json"
        out_path.write_text(json.dumps(generated, indent=2, ensure_ascii=False) + "\n")
        out[model] = {
            "config": name,
            "path": str(out_path.relative_to(workdir)),
            "sha256": _sha256(out_path.read_bytes()),
            "base": str(base_path.relative_to(workdir)),
            "base_sha256": _sha256(base_path.read_bytes()),
        }
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
