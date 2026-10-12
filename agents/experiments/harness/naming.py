"""Run naming: slugs and the deterministic ``<exp>-<model>-<bench>-r<RR>-mi<K>`` scheme."""

from __future__ import annotations

import re

MODEL_SLUGS = {
    "deepseek-pro": "deepseekpro",
    "deepseek-flash": "deepseekflash",
    "glm-5.2": "glm52",
    "gpt-5.6-terra": "gpt56terra",
    "gpt-5.6-luna": "gpt56luna",
    "kimi-k3": "kimik3",
}
BENCHMARK_SLUGS = {
    "matmult-int": "matmultint",
    "md5sum": "md5sum",
    "crc32": "crc32",
}


def slugify(value: str) -> str:
    """Run-id-safe slug ([A-Za-z0-9_-], matching artifacts._safe_run_id)."""
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return text or "x"


def model_slug(config_name: str) -> str:
    return MODEL_SLUGS.get(config_name, slugify(config_name))


def benchmark_slug(name: str) -> str:
    return BENCHMARK_SLUGS.get(name, slugify(name))


def layer_slug(layer: str) -> str:
    """Run-name token for an ablation layer; ``full`` (and legacy '') add none.

    The un-ablated reference flow keeps the canonical run name so existing
    experiment names stay comparable; every ablated layer carries its token.
    """
    layer = str(layer or "")
    return "" if layer in ("", "full") else slugify(layer)


def make_run_name(
    exp_id: str,
    model_config: str,
    benchmark: str,
    repeat: int,
    max_iters: int,
    infra_retry: int = 0,
    layer: str = "",
) -> str:
    name = (
        f"{exp_id}-{model_slug(model_config)}-{benchmark_slug(benchmark)}"
        f"-r{int(repeat):02d}-mi{int(max_iters)}"
    )
    suffix = layer_slug(layer)
    if suffix:
        name += f"-{suffix}"
    if infra_retry:
        name += f"-infra{int(infra_retry)}"
    return name


def job_run_name(job: dict, exp_id: str, max_iters: int, infra_retry: int = 0) -> str:
    """The run name of a scheduled job.

    ``exp_id``/``max_iters`` belong to the experiment as a whole and are passed
    by the caller (the job dicts carry them only after ``main`` stamps them on).
    """
    return make_run_name(
        exp_id, job["model"], job["benchmark"], job["repeat"], max_iters,
        infra_retry, layer=job.get("layer", ""),
    )
