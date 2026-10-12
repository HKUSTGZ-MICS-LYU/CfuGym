"""Cross-run cache of the embench reference/profile measurement."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import graph, profile_cache
from agents.cfu_design_agent.benchmark import embench_integrity
from agents.cfu_design_agent.benchmark.embench_loops import loop_id
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT
from agents.cfu_design_agent.stages import profile


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
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name)
        policy_source = REPO_ROOT.joinpath("agents/specs/agent_access_policy.yaml")
        policy = self.workdir / "agents/specs/agent_access_policy.yaml"
        policy.parent.mkdir(parents=True)
        policy.write_bytes(policy_source.read_bytes())
        self.run_id = f"profile-cache-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = self.workdir / self.run_root

        # These cache/transport tests seed tiny synthetic kernels, not the real
        # adapter's guarded Embench scaffold. Keep the checker mock local to this
        # fixture; real adapter/fairness tests exercise fail-closed enforcement.
        integrity = mock.patch.object(embench_integrity, "validate_candidate", return_value=[])
        integrity.start()
        self.addCleanup(integrity.stop)
        self._seed_workspace()
        self._pin_fingerprints()

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
        root = self.workdir / run_root / "workspace" / "embench_depthconv"
        for config in ("reference", "profile"):
            (root / config / "src").mkdir(parents=True, exist_ok=True)
            (root / config / "src" / "depthconv.c").write_text(
                "int hot(int x){for(int i=0;i<x;i++){x++;break;} return x+1;}\n"
            )
        design = self.workdir / run_root / "workspace" / "design"
        design.mkdir(parents=True, exist_ok=True)
        (design / "AgentCfu.scala").write_text("// scaffold\n")
        (design / "AgentCfuFiber.scala").write_text("// fiber\n")

    def _seed_workspace(self) -> None:
        self._seed_run(self.run_root)

    def _state(self, mode: str = "on", run_root: str = "") -> dict:
        run_root = run_root or self.run_root
        return {
            "workdir": str(self.workdir),
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
            out = profile.profile_embench(self._state(mode, run_root))
        return out, scripted

    def test_cache_is_reused_across_run_roots(self) -> None:
        # The whole point of the cache: a different run of the same benchmark on
        # the same toolchain/CPU must reuse the measurement.
        first, _ = self._run("on")
        self.assertEqual(first["gates"]["profile"]["profile_cache"]["status"], "miss")

        other_root = f"agents/generated/runs/{self.run_id}-other"
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
        cache_dir = self.workdir / profile_cache.CACHE_SUBDIR
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

    def test_profile_builds_and_simulations_leave_readable_logs(self) -> None:
        """Every profile-stage build/simulation keeps a log the agent can read.

        Regression class (run 20261003-090854-136683-d98e5be9): the profile
        stage ran its embench builds and SoC simulations through the raw
        ``RepoToolbox.run_command``, so those commands left no trace at all.
        """
        _, scripted = self._run("off")
        self.assertIn("embench_build", scripted.calls)
        self.assertIn("soc_sim", scripted.calls)
        log_dir = self.run_dir / "05_profile/logs"
        build_logs = sorted(log_dir.glob("embench_build_*.log"))
        sim_logs = sorted(log_dir.glob("soc_sim_*.log"))
        self.assertTrue(build_logs, sorted(path.name for path in log_dir.glob("*")))
        self.assertTrue(sim_logs)
        text = build_logs[0].read_text()
        self.assertIn("command_id: embench_build", text)
        self.assertIn("=== STDOUT ===", text)
        # The immutable copy travels with the run too.
        self.assertTrue((log_dir / "versions").is_dir())

    def test_key_changes_with_cpu_flags_and_with_the_design_overlay(self) -> None:
        def key() -> str:
            return profile_cache.build_key(
                self.workdir,
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
            self.workdir,
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

    def test_the_reference_disassembly_travels_with_the_cache_entry(self) -> None:
        # The disassembly is a build output, so a reused measurement has none and
        # the analysis could not read the ASM candidate (observed 2026-09-30).
        toolbox = RepoToolbox(self.workdir, dry_run=False, stage="profile",
                              run_root=self.run_root)
        asm = profile._reference_disassembly_rel(self.run_root, "depthconv")
        cached = self.workdir / profile_cache.artifact_relpath(
            "a" * 64, profile_cache.DISASSEMBLY_NAME
        )
        self.addCleanup(cached.unlink, missing_ok=True)
        (self.workdir / asm).parent.mkdir(parents=True, exist_ok=True)
        (self.workdir / asm).write_text("0000 <hot>:\n  ret\n")
        profile._store_reference_disassembly(toolbox, self.run_root, "depthconv", "a" * 64, "on")
        self.assertIn("hot", cached.read_text())
        # A cache hit restores it into the workspace...
        (self.workdir / asm).unlink()
        profile._restore_reference_disassembly(toolbox, self.run_root, "depthconv", "a" * 64, "on")
        self.assertIn("hot", (self.workdir / asm).read_text())
        # ...never overwriting a build that did happen...
        (self.workdir / asm).write_text("newer\n")
        profile._restore_reference_disassembly(toolbox, self.run_root, "depthconv", "a" * 64, "on")
        self.assertEqual((self.workdir / asm).read_text(), "newer\n")
        # ...and staying out of the way when the cache is off.
        (self.workdir / asm).unlink()
        profile._restore_reference_disassembly(toolbox, self.run_root, "depthconv", "a" * 64, "off")
        self.assertFalse((self.workdir / asm).exists())
        profile._store_reference_disassembly(toolbox, self.run_root, "depthconv", "b" * 64, "off")
        self.assertFalse(
            (self.workdir / profile_cache.artifact_relpath(
                "b" * 64, profile_cache.DISASSEMBLY_NAME)).exists()
        )

    def test_artifact_relpath_requires_a_safe_key_and_name(self) -> None:
        with self.assertRaises(ValueError):
            profile_cache.artifact_relpath("../escape", "x.asm")
        with self.assertRaises(ValueError):
            profile_cache.artifact_relpath("a" * 64, "../../etc/passwd")
        self.assertEqual(
            profile_cache.artifact_relpath("a" * 64, "embench_main.asm"),
            f"{profile_cache.CACHE_SUBDIR}/{'a' * 64}.embench_main.asm",
        )

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


class ProfileCacheProtocolTest(unittest.TestCase):
    """Protocol key regressions using only disposable source trees."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name)
        self.run_root = "runs/fixture"
        reference = self.workdir / self.run_root / "workspace/embench_crc32/reference"
        (reference / "src").mkdir(parents=True)
        self.kernel = reference / "src/crc32.c"
        self.kernel.write_text("int benchmark_body(void) { return 1; }\n")
        (reference / "support").mkdir()
        self.preparation = reference / "support/beebsc.c"
        self.preparation.write_text("void initialise_benchmark(void) {}\n")
        for name in ("toolchain_fingerprint", "repo_fingerprint"):
            patcher = mock.patch.object(profile_cache, name, return_value={"fixture": "fixed"})
            patcher.start()
            self.addCleanup(patcher.stop)

    def _key(self, **kwargs) -> dict:
        return profile_cache.build_key(
            self.workdir, run_root=self.run_root, benchmark="crc32",
            configs=["reference", "profile"],
            build_commands={"reference": ["make"], "profile": ["make"]},
            cpu_flags=["--mico-agent-cfu"], observation_plan=None, **kwargs,
        )

    def test_protocol_is_explicit_and_default_keeps_existing_api(self) -> None:
        from agents.cfu_design_agent.benchmark import embench

        with mock.patch.object(embench, "MEASUREMENT_PROTOCOL", "cold_single_invocation_v1", create=True):
            default = self._key()
        explicit = self._key(measurement_protocol="cold_single_invocation_v1")
        self.assertEqual(default, explicit)
        self.assertEqual(default["inputs"]["key_version"], "7")
        self.assertEqual(default["inputs"]["measurement_protocol"], "cold_single_invocation_v1")
        self.assertNotEqual(default["key"], self._key(measurement_protocol="legacy")["key"])
        self.assertNotEqual(default["key"], self._key(measurement_protocol="future_v2")["key"])

    def test_pristine_kernel_and_preparation_sources_still_invalidate_key(self) -> None:
        base = self._key(measurement_protocol="cold_single_invocation_v1")
        self.kernel.write_text("int benchmark_body(void) { return 2; }\n")
        changed = self._key(measurement_protocol="cold_single_invocation_v1")
        self.assertNotEqual(base["key"], changed["key"])
        self.preparation.write_text("void initialise_benchmark(void) { prepare(); }\n")
        self.assertNotEqual(changed["key"], self._key(measurement_protocol="cold_single_invocation_v1")["key"])

    def test_version_six_entry_cannot_match_current_key(self) -> None:
        import hashlib
        import json

        current = self._key(measurement_protocol="cold_single_invocation_v1")
        old_inputs = dict(current["inputs"], key_version="6")
        old_inputs.pop("measurement_protocol")
        canonical = json.dumps(old_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        old_key = hashlib.sha256(canonical.encode()).hexdigest()
        old_entry = profile_cache.serialize(
            old_key, benchmark="crc32", configs=["reference"],
            profiles={"reference": {"crc32": {"cycles": 100}}}, observation={}, inputs=old_inputs,
        )
        self.assertIsNone(profile_cache.parse(old_entry, current["key"]))


if __name__ == "__main__":
    unittest.main()
