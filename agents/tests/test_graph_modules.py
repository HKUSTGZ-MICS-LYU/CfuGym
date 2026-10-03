"""Structural invariants of the graph.py decomposition.

These are cheap, deterministic guards on the *shape* of the package rather than
on behaviour:

* ``graph.py`` is a composition root, not a dumping ground again.
* The helpers that moved out stay out of ``graph``: a stale
  ``mock.patch.object(graph, "<helper>")`` must raise AttributeError instead of
  silently patching a binding nobody reads.
* Cross-module functions are reached through their owning module, so mocking
  that module is actually observed by the caller.
* No single module grows back into a monolith.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from agents.cfu_design_agent import graph
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)

PACKAGE = Path("agents/cfu_design_agent")

# Per-module line caps for the modules this decomposition created. graph.py is
# deliberately tiny; a stage module may be substantial but must not become the
# next 6k-line file.
LIMITS = {
    "graph.py": 400,
    "flow/gates.py": 500,
    "flow/instrument.py": 500,
    "flow/ladder.py": 500,
    "flow/optimize.py": 500,
    "runtime/checks.py": 500,
    "runtime/commands.py": 600,
    "runtime/llm_step.py": 600,
    "runtime/parsing.py": 200,
    "runtime/stage_services.py": 500,
    "stages/context.py": 500,
    "stages/profile.py": 700,
    "stages/analysis.py": 400,
    "stages/contract.py": 500,
    "stages/feedback.py": 800,
    "stages/fallbacks.py": 300,
    "stages/review/__init__.py": 500,
    "stages/review/optimize.py": 500,
    "stages/review/report.py": 600,
    "stages/implement/hw.py": 500,
    "stages/implement/sw.py": 500,
    "stages/implement/unified.py": 400,
    "stages/implement/delivery.py": 400,
    "stages/implement/intrinsics.py": 600,
    "stages/verify/soc.py": 500,
    "stages/verify/conformance.py": 600,
    "stages/verify/software.py": 600,
    "stages/verify/cost.py": 400,
}

# Modules whose functions must be reached module-qualified by their callers.
SEAM_MODULES = {
    "agents.cfu_design_agent.flow.gates",
    "agents.cfu_design_agent.flow.ladder",
    "agents.cfu_design_agent.flow.optimize",
    "agents.cfu_design_agent.runtime.checks",
    "agents.cfu_design_agent.runtime.commands",
    "agents.cfu_design_agent.runtime.llm_step",
    "agents.cfu_design_agent.runtime.parsing",
    "agents.cfu_design_agent.runtime.stage_services",
}

# Names the composition root binds directly: python_nodes() resolves node
# functions from this module's globals, and the routers are looked up there too.
GRAPH_BY_NAME = set(graph.NODE_NAMES) | set(graph.ROUTER_NAMES.values()) | {
    "STAGE_ORDER", "enable_progress", "disable_progress", "progress",
    "LlmResult", "PolicyError", "RepoToolbox",
}

# Helpers that moved out of graph.py. Re-exporting any of them would let a stale
# patch succeed while changing nothing.
MOVED_SEAMS = (
    "_llm", "_complete_llm", "_run_check", "_toolbox_for_stage", "_artifacts_for_stage",
    "_write_hardware_artifact", "_write_isa_artifact", "_write_hotspot_artifact",
    "_software_targets", "_software_exercises_cfu", "_software_candidate_hash",
    "_choose_optimization", "_snapshot_candidate", "_missing_design_files",
    "agent_cfu_overlay_commands", "agent_cfu_check_commands", "_soc_rtl_path",
)


def _module_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "generated" not in str(p))


class GraphIsACompositionRootTest(unittest.TestCase):
    def test_graph_py_stays_small(self) -> None:
        lines = len((PACKAGE / "graph.py").read_text().splitlines())
        self.assertLessEqual(lines, LIMITS["graph.py"], f"graph.py grew to {lines} lines")

    def test_no_module_exceeds_its_budget(self) -> None:
        for rel, limit in sorted(LIMITS.items()):
            path = PACKAGE / rel
            with self.subTest(module=rel):
                self.assertTrue(path.is_file(), f"{rel} disappeared")
                lines = len(path.read_text().splitlines())
                self.assertLessEqual(lines, limit, f"{rel} grew to {lines} lines")

    def test_moved_helpers_are_not_re_exported(self) -> None:
        for name in MOVED_SEAMS:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(graph, name),
                    f"{name} is re-exported by graph.py; a stale patch would silently no-op",
                )

    def test_the_public_surface_and_node_functions_remain(self) -> None:
        for name in ("build_graph", "run_agent", "python_nodes", "enable_progress",
                     "disable_progress", "export_graph_visualization", "STAGE_ORDER"):
            self.assertTrue(hasattr(graph, name), name)
        self.assertEqual(set(graph.python_nodes()), set(graph.NODE_NAMES))
        for symbol in graph.ROUTER_NAMES.values():
            self.assertTrue(callable(getattr(graph, symbol, None)), symbol)


class SeamDisciplineTest(unittest.TestCase):
    """A function imported by name is a binding the owning module cannot patch."""

    @staticmethod
    def _module_name(path: Path) -> str:
        rel = path.relative_to(PACKAGE).with_suffix("").as_posix().replace("/", ".")
        return "agents.cfu_design_agent." + rel.removesuffix(".__init__")

    @staticmethod
    def _resolve(path: Path, node: ast.ImportFrom) -> str:
        """Absolute dotted module for a relative ImportFrom in ``path``."""
        parts = SeamDisciplineTest._module_name(path).split(".")
        package = parts[:-1]
        keep = len(package) - (node.level - 1)
        base = package[:keep] if keep > 0 else []
        return ".".join(base + ([node.module] if node.module else []))

    def test_seam_functions_are_not_imported_by_name(self) -> None:
        offenders: list[str] = []
        for path in _module_files():
            if path.name == "graph.py":
                continue                      # the composition root resolves by name on purpose
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = self._resolve(path, node)
                if module not in SEAM_MODULES:
                    continue
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name[:1].islower() or alias.name.startswith("_"):
                        offenders.append(f"{path.as_posix()}: from {module} import {alias.name}"
                                         + (f" as {local}" if local != alias.name else ""))
        self.assertEqual(offenders, [], "reach these functions through their module instead")

    def test_every_seam_module_is_importable(self) -> None:
        import importlib

        for module in sorted(SEAM_MODULES):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


class NodeWrapperContractTest(unittest.TestCase):
    """The wrapper's state-key contract.

    A scripted code move once renamed the recorded key to
    ``"dependencies.input_fingerprints"``: every read then saw an empty mapping,
    so gate invalidation silently stopped happening while the suite stayed green.
    """

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from agents.cfu_design_agent.flow import instrument

        self.instrument = instrument
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name).resolve()

    def _state(self, **overrides) -> dict:
        state = {"workdir": str(self.workdir), "gates": {}, "errors": []}
        state.update(overrides)
        return state

    def test_the_wrapper_records_the_documented_state_key(self) -> None:
        def step(state: dict) -> dict:
            return {"errors": [], "gates": {"sim": {"ok": True}}}

        update = self.instrument._wrap_node(step)(self._state())
        self.assertIn("input_fingerprints", update)
        self.assertIsInstance(update["input_fingerprints"], dict)
        for key in update:
            self.assertNotIn(".", key, f"namespaced state key written: {key}")

    def test_an_upstream_input_change_marks_the_downstream_gate_stale(self) -> None:
        def step(state: dict) -> dict:
            return {"errors": [], "gates": {"sim": {"ok": True}}}

        first = self.instrument._wrap_node(step)(self._state())
        self.assertTrue(first["gates"]["sim"]["ok"])

        second = self.instrument._wrap_node(step)(
            self._state(workload_analysis="a different analysis",
                        input_fingerprints=first["input_fingerprints"])
        )
        self.assertTrue(second["gates"]["sim"]["stale"])
        self.assertFalse(second["gates"]["sim"]["ok"])
        # The reason names whichever upstream fingerprint changed first.
        self.assertIn("upstream input changed", second["gates"]["sim"]["reason"])

    def test_an_unchanged_state_never_invalidates(self) -> None:
        def step(state: dict) -> dict:
            return {"errors": [], "gates": {"sim": {"ok": True}}}

        first = self.instrument._wrap_node(step)(self._state())
        again = self.instrument._wrap_node(step)(
            self._state(input_fingerprints=first["input_fingerprints"])
        )
        self.assertNotIn("stale", again["gates"]["sim"])


if __name__ == "__main__":
    unittest.main()
