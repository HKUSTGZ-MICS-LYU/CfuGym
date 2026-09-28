"""Cross-run cache of the embench reference/profile measurement."""

from __future__ import annotations

import re
import shutil
import unittest
from unittest import mock

from agents.cfu_design_agent import graph, profile_cache
from agents.cfu_design_agent.benchmark.embench_loops import loop_id
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT


def _sim_stdout(args: list[str]) -> str:
    joined = " ".join(args)
    match = re.search(r"/workspace/embench_([^/]+)/(reference|profile|cfu)/", joined)
    if match:
        name, config = match.group(1), match.group(2)
        cycles = 1200 if config == "reference" else 900
        return (
            f"CFU_AGENT_PROFILE {name}={cycles}\n"
            f"CFU_AGENT_PROFILE scalar_kernel={cycles}\n"
            "CFU_AGENT_PASS\n"
        )
    return "CFU_AGENT_PROFILE scalar_kernel=100\nCFU_AGENT_PASS\n"


class _ScriptedProfile:
    """Stand-in for ``RepoToolbox.run_command`` during profiling."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, toolbox, args, **kwargs):
        command_id = str(kwargs.get("command_id") or "")
        self.calls.append(command_id)
        stdout = _sim_stdout(args) if command_id == "soc_sim" else ""
        if command_id == "soc_sim" and "/profile/" in " ".join(args):
            ident = loop_id("depthconv.c", "hot", 1)
            stdout += f"CFU_AGENT_PROFILE {ident}=710 calls=1\n"
        return {
            "command": " ".join(args),
            "cwd": str(toolbox.workdir),
            "returncode": 0,
            "stdout_tail": stdout,
            "stderr_tail": "",
            "parsed": {},
        }


class ProfileCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"profile-cache-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = REPO_ROOT / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

        cache_dir = REPO_ROOT / profile_cache.CACHE_SUBDIR
        pre_existing = set(cache_dir.glob("*.json")) if cache_dir.is_dir() else set()
        self.addCleanup(self._remove_new_cache_files, pre_existing)

        self._seed_workspace()
        self._pin_fingerprints()

    def _remove_new_cache_files(self, pre_existing: set) -> None:
        cache_dir = REPO_ROOT / profile_cache.CACHE_SUBDIR
        if cache_dir.is_dir():
            for path in set(cache_dir.glob("*.json")) - pre_existing:
                path.unlink(missing_ok=True)

    def _pin_fingerprints(self) -> None:
        # Tool versions and the git revision are environment-specific; pin them
        # so the key is deterministic inside the test.
        patcher = mock.patch.object(
            profile_cache,
            "toolchain_fingerprint",
            lambda: {"gcc": "test-gcc", "verilator": "test-verilator"},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            profile_cache,
            "repo_fingerprint",
            lambda workdir: {"head": "test-head", "submodules": "test-subs"},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _seed_run(self, run_root: str) -> None:
        root = REPO_ROOT / run_root / "workspace" / "embench_depthconv"
        for config in ("reference", "profile"):
            (root / config / "src").mkdir(parents=True, exist_ok=True)
            (root / config / "src" / "depthconv.c").write_text(
                "int hot(int x){for(int i=0;i<x;i++){x++;break;} return x+1;}\n"
            )
        design = REPO_ROOT / run_root / "workspace" / "design"
        design.mkdir(parents=True, exist_ok=True)
        (design / "AgentCfu.scala").write_text("// scaffold\n")
        (design / "AgentCfuFiber.scala").write_text("// fiber\n")

    def _seed_workspace(self) -> None:
        self._seed_run(self.run_root)

    def _state(self, mode: str = "on", run_root: str = "") -> dict:
        run_root = run_root or self.run_root
        return {
            "workdir": str(REPO_ROOT),
            "run_root": run_root,
            "run_id": run_root.rsplit("/", 1)[-1],
            "benchmark_name": "depthconv",
            "agent_cfu": True,
            "workload_spec": {"benchmark_name": "depthconv", "operation": "embench_workload"},
            "benchmark_project": {"benchmark": "depthconv", "kind": "embench_benchmark"},
            "run_commands": True,
            "dry_run": False,
            "skip_sim": False,
            "profile_cache": mode,
        }

    def _run(self, mode: str = "on", run_root: str = ""):
        scripted = _ScriptedProfile()
        with mock.patch.object(
            RepoToolbox, "run_command", lambda self, args, **kw: scripted(self, args, **kw)
        ):
            out = graph.profile_embench(self._state(mode, run_root))
        return out, scripted

    def test_cache_is_reused_across_run_roots(self) -> None:
        # The whole point of the cache: a different run of the same benchmark on
        # the same toolchain/CPU must reuse the measurement.
        first, _ = self._run("on")
        self.assertEqual(first["gates"]["profile"]["profile_cache"]["status"], "miss")

        other_root = f"agents/generated/runs/{self.run_id}-other"
        shutil.rmtree(REPO_ROOT / other_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, REPO_ROOT / other_root, True)
        self._seed_run(other_root)

        second, scripted = self._run("on", run_root=other_root)
        gate = second["gates"]["profile"]
        self.assertEqual(gate["profile_cache"]["status"], "hit")
        self.assertEqual(scripted.calls, [], "a cross-run cache hit must run no command")
        self.assertEqual(
            second["embench_profiles"], first["embench_profiles"]
        )

    def test_miss_stores_then_hit_skips_builds_and_simulations(self) -> None:
        first, scripted_first = self._run("on")
        gate_first = first["gates"]["profile"]
        self.assertTrue(gate_first["ok"])
        self.assertEqual(gate_first["profile_cache"]["status"], "miss")
        self.assertTrue(gate_first["profile_cache"]["saved"])
        self.assertTrue(scripted_first.calls, "the first run must build and simulate")
        self.assertEqual(
            first["embench_profiles"]["reference"]["depthconv"]["cycles"], 1200
        )

        second, scripted_second = self._run("on")
        gate_second = second["gates"]["profile"]
        self.assertTrue(gate_second["ok"])
        self.assertEqual(gate_second["profile_cache"]["status"], "hit")
        self.assertEqual(scripted_second.calls, [], "a cache hit must run no command")
        self.assertEqual(second["embench_profiles"], first["embench_profiles"])
        self.assertIn("cached reference/profile", second["iteration_history"][0])

        # The cache marker is part of the run artifacts, not only the gate.
        artifact = self.run_dir / "05_profile/embench.yaml"
        self.assertIn("profile_cache:", artifact.read_text())

    def test_off_never_reads_or_writes_the_cache(self) -> None:
        cache_dir = REPO_ROOT / profile_cache.CACHE_SUBDIR
        before = set(cache_dir.glob("*.json")) if cache_dir.is_dir() else set()
        out, scripted = self._run("off")
        report = out["gates"]["profile"]["profile_cache"]
        self.assertEqual(report["status"], "disabled")
        self.assertFalse(report["saved"])
        self.assertTrue(scripted.calls)
        after = set(cache_dir.glob("*.json")) if cache_dir.is_dir() else set()
        self.assertEqual(after, before)

    def test_refresh_recomputes_and_overwrites(self) -> None:
        self._run("on")  # populate the cache
        out, scripted = self._run("refresh")
        report = out["gates"]["profile"]["profile_cache"]
        self.assertEqual(report["status"], "refresh")
        self.assertTrue(report["saved"])
        self.assertTrue(scripted.calls, "refresh must recompute instead of reusing")

    def test_key_changes_with_cpu_flags_and_with_the_design_overlay(self) -> None:
        def key() -> str:
            return profile_cache.build_key(
                REPO_ROOT,
                run_root=self.run_root,
                benchmark="depthconv",
                configs=["reference", "profile"],
                build_commands={"reference": ["make"], "profile": ["make"]},
                cpu_flags=["--mico-agent-cfu"],
                observation_plan=None,
            )["key"]

        base = key()
        self.assertEqual(base, key(), "identical inputs must hash identically")
        changed_flags = profile_cache.build_key(
            REPO_ROOT,
            run_root=self.run_root,
            benchmark="depthconv",
            configs=["reference", "profile"],
            build_commands={"reference": ["make"], "profile": ["make"]},
            cpu_flags=["--mico-agent-cfu", "--agent-cfu-vlen", "128"],
            observation_plan=None,
        )["key"]
        self.assertNotEqual(base, changed_flags)

        (self.run_dir / "workspace" / "design" / "AgentCfu.scala").write_text("// changed\n")
        self.assertNotEqual(base, key(), "a different design overlay must invalidate the key")

    def test_parse_rejects_foreign_or_incomplete_entries(self) -> None:
        text = profile_cache.serialize(
            "k",
            benchmark="b",
            configs=["reference"],
            profiles={"reference": {"b": {"cycles": 1}}},
            observation={},
            inputs={},
        )
        self.assertIsNotNone(profile_cache.parse(text, "k"))
        self.assertIsNone(profile_cache.parse(text, "other-key"))
        self.assertIsNone(profile_cache.parse("{not json", "k"))
        self.assertIsNone(profile_cache.parse('{"key": "k", "profiles": {}}', "k"))


if __name__ == "__main__":
    unittest.main()
