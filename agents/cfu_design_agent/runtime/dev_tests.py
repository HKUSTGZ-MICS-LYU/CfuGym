"""Agent-authored development tests (dev tests) for the implementation stage.

Conformance used to be a host-rendered C test driven by a model-authored YAML
oracle.  Requiring that oracle to satisfy a schema made the agent spend its
rounds on *validity* rather than on observing the design, and the schema still
could not express a multi-command sequence (``data.roles`` describes the shape
of one call, not the order of many).

A dev test is plain C in the run workspace:

* the model owns two functions, ``cfu_conformance_dut()`` (calls the generated
  ``cfu_*`` intrinsics, in any order and any number of times) and
  ``cfu_conformance_ref()`` (an independent scalar reference that must not call
  any ``cfu_*`` intrinsic);
* the host owns the file header, the data block, ``main()``, the comparison and
  the verdict line, so a dev test can never certify itself.

Ownership boundary (do not weaken)
----------------------------------
* A dev test is **evidence**, never a Gate. ``run_conformance`` records a
  result; the authoritative rungs (``sim``/``cost``) still decide the run.
* The model may not edit outside the marked region, may not touch the host
  runner, and cannot predict the per-run nonce carried by the verdict line, so
  a hand-written "PASS" line is ignored by the parser.
* Dev tests live under ``workspace/dev_tests/`` — a path
  ``agent_access_policy.yaml`` already reserves for exactly this purpose, kept
  separate from the repository's own ``sw/tests``.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .tools import RepoToolbox

DEV_TESTS_SUBDIR = "workspace/dev_tests"
HOST_SUBDIR = "_host"
RUNNER_SOURCE = "cfu_dev_runner.c"
HOST_HEADER = "cfu_conf.h"
MAX_TESTS = 8
MAX_EDIT_CHARS = 60_000

#: A test name is a file stem the tools concatenate into a path, so it is a
#: strict allowlist rather than a sanitised string: no separators, no dots, no
#: traversal, and nothing that could name the host-owned ``_host`` directory.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")

MODEL_REGION_BEGIN = "/* ==== MODEL REGION BEGIN (you may edit below this line) ==== */\n"
MODEL_REGION_END = "/* ==== MODEL REGION END (do not edit this line) ==== */\n"

#: Everything up to and including the two prototypes is host-owned. An edit
#: that changes it is rejected: the header is what binds the model's code to the
#: host runner and the harness data, and a model-owned header could redefine the
#: observation the runner compares.
TEMPLATE_HEADER = """/* CfuGym dev test: a development check, never the formal conformance verdict.
 *
 * You own ONLY the two functions inside the marked region:
 *
 *   uint32_t cfu_conformance_dut(void);   call the generated cfu_* intrinsics,
 *                                         in any order and any number of times
 *   uint32_t cfu_conformance_ref(void);   an independent scalar reference, which
 *                                         must NOT call any cfu_* intrinsic
 *
 * The host owns this header, the test data, cfu_conf.h and main(): it sweeps
 * cfu_conf_len / cfu_conf_index, compares both return values and prints the
 * verdict. To check memory or the register file, fold it into the returned
 * value (for example a checksum) on BOTH sides.
 *
 * Each invocation must start from a defined state: reset any design state your
 * dut() depends on, because the host calls dut() once per case in one run.
 */
#include <stdint.h>

#include "cfu_intrinsics.h"
#include "cfu_conf.h"

uint32_t cfu_conformance_dut(void);
uint32_t cfu_conformance_ref(void);

"""

REGION_STUB = """uint32_t cfu_conformance_dut(void) {
  /* TODO: drive the intrinsics for this scenario; see the ISA contract. */
  return 0xFFFFFFFFu;
}

uint32_t cfu_conformance_ref(void) {
  /* TODO: compute the same value with scalar C, transcribed from the
   * workload's own source (include it instead of copying tables). */
  return 0u;
}
"""


def template_text() -> str:
    """The file a new dev test starts as.

    The two stubs deliberately return different constants, so an untouched
    template FAILS instead of passing on two zeros.
    """
    return TEMPLATE_HEADER + MODEL_REGION_BEGIN + REGION_STUB + MODEL_REGION_END


def validate_name(name: Any) -> str:
    """Return the normalised test name, or raise ``ValueError`` with the rule."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("test name must be a non-empty string")
    value = name.strip()
    if not _NAME_RE.fullmatch(value):
        raise ValueError(
            f"invalid test name {value!r}: use 1-32 characters starting with a "
            "letter, then letters/digits/_/- only (it names a file under "
            f"{DEV_TESTS_SUBDIR}/)"
        )
    return value


def root_for(state: Any) -> str:
    """Repo-relative dev-test directory for this run, or "" without a run root."""
    run_root = str((state or {}).get("run_root") or "").strip().rstrip("/")
    if not run_root:
        return ""
    return f"{run_root}/{DEV_TESTS_SUBDIR}"


def _function_body(region: str, name: str) -> str:
    """Body of one ``name(...) { ... }`` definition inside the model region."""
    match = re.search(rf"\b{name}\s*\([^;{{)]*\)\s*\{{", region)
    if match is None:
        return ""
    start = match.end()
    depth = 1
    index = start
    while index < len(region) and depth:
        char = region[index]
        depth += (char == "{") - (char == "}")
        index += 1
    return region[start:index - 1] if depth == 0 else ""


_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _without_comments(text: str) -> str:
    """Blank C comments so prose like ``main()`` in a header cannot be mistaken
    for code, while keeping offsets irrelevant (checks are structural)."""
    return _COMMENT_RE.sub(" ", text)


def region_problems(text: str, *, allow_stub: bool = False) -> list[str]:
    """Why this file is not a usable dev test, or ``[]`` when it is.

    Structural only: the model's C is never parsed. These checks enforce the
    ownership boundary (host header intact, code inside the region) and the one
    property the runner depends on (the reference is independent of the DUT).
    ``allow_stub`` accepts the pristine template, which is a valid *starting*
    point but must be filled before it can be run.
    """
    value = str(text or "")
    problems: list[str] = []
    if not value.startswith(TEMPLATE_HEADER):
        problems.append(
            "the host-owned header was modified: call add_conformance_test again "
            "for a fresh template and put your code inside the model region"
        )
    begin = value.find(MODEL_REGION_BEGIN)
    end = value.find(MODEL_REGION_END)
    if begin < 0 or end < 0:
        problems.append(
            "the MODEL REGION markers are missing; the host cannot tell your code "
            "from its own header"
        )
        return problems
    if end < begin:
        problems.append("the MODEL REGION end marker appears before the begin marker")
        return problems
    if value[end + len(MODEL_REGION_END):].strip():
        problems.append("content after the MODEL REGION end marker is not allowed")
    region = _without_comments(value[begin + len(MODEL_REGION_BEGIN):end])
    if allow_stub and region.strip() == _without_comments(REGION_STUB).strip():
        return problems
    for name in ("cfu_conformance_dut", "cfu_conformance_ref"):
        if not re.search(rf"\b{name}\s*\([^;{{)]*\)\s*\{{", region):
            problems.append(f"the model region must define function {name}(void)")
    if re.search(r"(?<![A-Za-z0-9_])main\s*\(", _without_comments(value)):
        problems.append(
            "the test must not define main(): the host owns the entry point and "
            "the data sweep"
        )
    ref_body = _function_body(region, "cfu_conformance_ref")
    device_calls = sorted(set(re.findall(r"\b(cfu_[A-Za-z0-9_]+)\s*\(", ref_body)))
    if device_calls:
        problems.append(
            "cfu_conformance_ref calls the device intrinsic(s) "
            + ", ".join(device_calls)
            + ": the reference must be independent scalar C, otherwise it only "
            "re-states the hardware and proves nothing"
        )
    dut_body = _function_body(region, "cfu_conformance_dut")
    if dut_body and not re.search(r"\bcfu_[A-Za-z0-9_]+\s*\(", dut_body):
        problems.append(
            "cfu_conformance_dut calls no cfu_* intrinsic: a test that never "
            "exercises the design cannot observe it"
        )
    if dut_body and re.search(r"\bcfu_conformance_ref\s*\(", dut_body):
        problems.append(
            "cfu_conformance_dut calls cfu_conformance_ref(): returning the "
            "reference's own value makes the comparison vacuous, so the design "
            "would never be exercised"
        )
    return problems


def _snapshot(toolbox: RepoToolbox, path: str) -> bytes | None:
    full = toolbox.workdir / path
    return full.read_bytes() if full.exists() else None


def run_one(
    state: Any,
    workspace: "DevTestWorkspace",
    test_name: str,
    header_path: str,
    run_toolbox: RepoToolbox,
    *,
    build_timeout: int = 1800,
    sim_timeout: int = 3600,
) -> dict[str, Any]:
    """Build and simulate one dev test; return the last command result + verdict.

    Two toolboxes are involved on purpose: the model-adjacent writes (template,
    host files) belong to the ``implementation`` stage, while the compile and
    simulation commands are the ``conformance`` stage's authority. Running them
    under the stage that already owns them means introducing dev tests does not
    widen what Implement may execute.

    A build failure never falls through to a simulation: that would simulate the
    previous ELF and report a stale verdict as if it were this candidate's.
    """
    from .. import conformance_dev_runner
    from . import commands as commands_mod, dev_commands, effects

    name = validate_name(test_name)
    if not workspace.exists(name):
        return {
            "returncode": 1,
            "stderr_tail": f"dev test {name!r} does not exist",
            "parsed": {"dev_conformance": {"status": "not_run", "ok": False,
                                           "test": name, "cases": 0, "failed": 0,
                                           "details": [], "reason": "missing test"}},
        }
    # Refuse an unusable test before spending a compile and a simulation on it:
    # an untouched template or a DUT with no intrinsic call cannot observe
    # anything, so its "result" would only cost a budget slot.
    problems = region_problems(workspace.read(name), allow_stub=False)
    if problems:
        return {
            "returncode": 1,
            "stderr_tail": "dev test is not runnable: " + "; ".join(problems),
            "parsed": {"dev_conformance": {
                "status": "not_run", "ok": False, "test": name, "cases": 0,
                "failed": 0, "details": problems,
                "reason": "the test was rejected before it was compiled",
            }},
        }
    workspace.ensure_host_files(name)
    nonce = conformance_dev_runner.new_nonce()
    build_argv, elf = dev_commands.dev_conformance_build_command(
        state, name, header_path, nonce,
    )
    try:
        os.unlink(workspace.toolbox.workdir / elf)
    except FileNotFoundError:
        pass
    build = effects.run_command(
        state, run_toolbox, build_argv, stage="conformance", rung="hw_conformance",
        # A distinct id: the legacy `conformance_build` declaration requires the
        # fixed `cfu_conformance` bin name, which would reject every test whose
        # name the agent chose differently.
        command_id="dev_conformance_build", timeout=build_timeout, mutate=True,
    )
    results = [build]
    verdict: dict[str, Any] = {"status": "not_run", "ok": False, "test": name,
                               "cases": 0, "failed": 0, "details": [], "reason": ""}
    if build.get("returncode") not in (0, None):
        verdict["reason"] = "the dev test did not compile"
        # A link failure on the workload's bump allocator is the most common
        # first-build miss once a test includes the pristine scalar source: the
        # host deliberately does not link `beebsc.c` (a test may include it).
        # Naming the fix here saves a whole repair round.
        build = dict(build)
        build["stderr_tail"] = str(build.get("stderr_tail") or "") + (
            "\nHINT: undefined *_beebs symbols mean the included scalar source "
            "allocates; add `#include \"beebsc.c\"` and call "
            "`init_heap_beebs(heap, sizeof heap)` with a static heap. The host "
            "does not link the allocator automatically."
            if "beebs" in str(build.get("stderr_tail") or "") else ""
        )
    else:
        sim_commands = commands_mod.agent_cfu_sim_commands(
            state, state.get("isa_spec") or {}, elf=elf,
        )
        if not sim_commands:
            verdict["reason"] = "no simulation command for the agent CFU"
        for argv in sim_commands:
            sim = effects.run_command(
                state, run_toolbox, argv, stage="conformance", rung="hw_conformance",
                command_id="soc_sim", timeout=sim_timeout, mutate=True,
            )
            results.append(sim)
            text = "\n".join([
                str(sim.get("stdout_tail") or ""),
                str(sim.get("stderr_tail") or ""),
            ])
            verdict = conformance_dev_runner.parse_verdict(text, nonce, test=name)
            if sim.get("returncode") not in (0, None) and verdict["status"] == "missing":
                verdict["reason"] = "the simulation exited before printing a verdict"
            if not verdict["ok"]:
                break
    last = dict(results[-1])
    # The host decides the check's status from the verdict line, not from the
    # simulator's exit code (which is always 0 now). A failed verdict is a
    # readable result the model can act on, not a `fail_symbol` crash.
    if verdict["status"] != "not_run":
        last["returncode"] = 0 if verdict["ok"] else 1
        if not verdict["ok"]:
            head = (
                f"dev test {name}: {verdict['status'].upper()} "
                f"cases={verdict['cases']} failed={verdict['failed']}"
            )
            detail_lines = [str(item) for item in (verdict.get("details") or [])][:4]
            last["stderr_tail"] = "\n".join([head, *detail_lines])
    parsed = dict(last.get("parsed") or {})
    parsed["dev_conformance"] = verdict
    last["parsed"] = parsed
    workspace.record_run(name, last)
    return last


@dataclass
class DevTestWorkspace:
    """One stage's dev-test files, with the same commit/rollback contract.

    The set of files is not known when the stage starts (the model creates
    them), so this is not a :class:`CodingSession` over a fixed target list.
    It snapshots each file at first touch instead, and restores it on rollback,
    which keeps a rejected implementation attempt from leaving a test behind
    that the next candidate would inherit.
    """

    toolbox: RepoToolbox
    root: str
    max_tests: int = MAX_TESTS
    max_edit_chars: int = MAX_EDIT_CHARS
    snapshots: dict[str, bytes | None] = field(default_factory=dict)
    touched_paths: list[str] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False

    # ---- paths ---------------------------------------------------------
    @property
    def host_dir(self) -> str:
        return f"{self.root}/{HOST_SUBDIR}"

    def path(self, name: str) -> str:
        return f"{self.root}/{validate_name(name)}.c"

    def runner_path(self) -> str:
        return f"{self.host_dir}/{RUNNER_SOURCE}"

    def runner_stub_path(self, name: str) -> str:
        """Per-test ``MAIN`` translation unit.

        ``sw/Makefile`` links exactly one ``$(MAIN).c`` and writes
        ``$(MAIN).elf``, so a shared ``MAIN`` would make two tests overwrite one
        ELF and could let a stale binary be simulated after a failed build. A
        one-line stub that includes the host runner keeps one ELF per test
        without duplicating the runner source.
        """
        return f"{self.host_dir}/{validate_name(name)}_runner.c"

    def header_path(self) -> str:
        return f"{self.host_dir}/{HOST_HEADER}"

    def names(self) -> list[str]:
        directory = self.toolbox.workdir / self.root
        if not directory.is_dir():
            return []
        return sorted(
            path.stem for path in directory.glob("*.c")
            if path.is_file() and _NAME_RE.fullmatch(path.stem)
        )

    def exists(self, name: str) -> bool:
        return (self.toolbox.workdir / self.path(name)).is_file()

    def read(self, name: str) -> str:
        return self.toolbox.read_text(self.path(name), limit=1_000_000)

    # ---- host-owned companion files ------------------------------------
    def ensure_host_files(self, name: str | None = None) -> None:
        """(Re)write the runner, header and (for one test) its MAIN stub.

        Idempotent and never snapshotted: these are harness assets, not model
        work. A rollback must not delete them, and a stale copy left by an older
        runner version must not be linked into a new test.
        """
        from .. import conformance_dev_runner

        expected = {
            self.runner_path(): conformance_dev_runner.render_runner(),
            self.header_path(): conformance_dev_runner.render_header(),
        }
        if name is not None:
            expected[self.runner_stub_path(name)] = (
                '/* Host-owned: includes the CfuGym dev-test runner. */\n'
                '#include "cfu_dev_runner.c"\n'
            )
        for path, text in expected.items():
            if _snapshot(self.toolbox, path) == text.encode("utf-8"):
                continue
            self.toolbox.write_text(path, text)

    # ---- model operations ----------------------------------------------
    def create(self, name: str) -> str:
        name = validate_name(name)
        if self.committed or self.rolled_back:
            return "error: this stage's dev-test transaction is closed"
        existing = self.names()
        if self.exists(name):
            return (
                f"error: dev test {name!r} already exists; edit it with "
                f"edit_conformance_test(name={name!r}, ...) instead of recreating it"
            )
        if len(existing) >= self.max_tests:
            return (
                f"error: dev-test budget exhausted ({self.max_tests}); reuse "
                f"one of: {', '.join(existing)}"
            )
        self.ensure_host_files(name)
        path = self.path(name)
        self._snapshot_once(path)
        self.toolbox.write_text(path, template_text())
        return (
            f"created {path} from the host template; fill cfu_conformance_dut() "
            "and cfu_conformance_ref() inside the model region, then call "
            f"run_conformance(test={name!r})"
        )

    def edit(self, name: str, search: str, replace: str) -> str:
        name = validate_name(name)
        if self.committed or self.rolled_back:
            return "error: this stage's dev-test transaction is closed"
        if not self.exists(name):
            return (
                f"error: dev test {name!r} does not exist; create it with "
                f"add_conformance_test(name={name!r}) first"
            )
        if not isinstance(search, str) or not search:
            return "usage error: search must be a non-empty string"
        if not isinstance(replace, str):
            return "usage error: replace must be a string"
        if len(search) + len(replace) > self.max_edit_chars:
            return "usage error: edit character budget exceeded"
        path = self.path(name)
        before = self.toolbox.read_text(path, limit=1_000_000)
        self._snapshot_once(path)
        result = self.toolbox.apply_edit_blocks([(path, [(search, replace)])])
        if result.get("returncode") != 0:
            return "edit rejected: " + str(result.get("stderr_tail", ""))[:2000]
        if not self.toolbox.is_write_allowed(path):
            self.toolbox.write_text(path, before)
            return f"denied by policy: {path} is not a writable dev-test target"
        problems = region_problems(
            self.toolbox.read_text(path, limit=1_000_000), allow_stub=True,
        )
        if problems:
            # Restore the pre-edit bytes so a rejected edit cannot leave a file
            # that is neither the template nor valid model code.
            self.toolbox.write_text(path, before)
            return "edit rejected: " + "; ".join(problems)
        return (
            f"edited {path} (search={len(search)} chars, replace={len(replace)} "
            "chars); run_conformance checks it against the harness data"
        )

    # ---- transaction ---------------------------------------------------
    def _snapshot_once(self, path: str) -> None:
        if path in self.snapshots:
            return
        self.snapshots[path] = _snapshot(self.toolbox, path)
        if path not in self.touched_paths:
            self.touched_paths.append(path)

    def record_run(self, name: str, result: dict[str, Any]) -> None:
        entry = {
            "test": name,
            "path": self.path(name),
            "returncode": result.get("returncode"),
            "verdict": (result.get("parsed") or {}).get("dev_conformance") or {},
            "failure_id": (result.get("failure_observation") or {}).get("id", ""),
            "log_path": result.get("full_log_path", result.get("log_path", "")),
        }
        self.runs.append(entry)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        if self.committed or self.rolled_back:
            return
        restored: list[str] = []
        try:
            for path, data in self.snapshots.items():
                full = self.toolbox.workdir / path
                if data is None:
                    if full.exists():
                        full.unlink()
                else:
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_bytes(data)
                restored.append(path)
        except OSError as exc:
            raise OSError(
                f"dev-test rollback failed after restoring {restored}: {exc}"
            ) from exc
        self.rolled_back = True
        self.touched_paths.clear()


def run_from_state(
    state: Any, workspace: "DevTestWorkspace", header_path: str, args: dict[str, Any],
) -> dict[str, Any]:
    """``run_conformance``'s body: validate the named test, then build + simulate.

    Kept here rather than in the stage so the tool callback stays a one-liner and
    the "which toolbox may run which command" rule lives with the files it
    governs: writes use the implementation stage, commands use conformance.
    """
    from . import stage_services

    if workspace is None:
        return {"returncode": 1, "stderr_tail": "dev tests are disabled in this run"}
    name = args.get("test")
    try:
        validate_name(name)
    except ValueError as exc:
        return {"returncode": 1, "stderr_tail": str(exc)}
    if not workspace.exists(str(name)):
        return {
            "returncode": 1,
            "stderr_tail": (
                f"dev test {name!r} does not exist; create it with "
                f"add_conformance_test(name={name!r}). Existing: "
                + (", ".join(workspace.names()) or "none")
            ),
        }
    run_toolbox = stage_services._toolbox_for_stage(state, "conformance")
    return run_one(state, workspace, str(name), header_path, run_toolbox)


def evidence_block(runs: list[dict[str, Any]], tests: list[dict[str, Any]]) -> str:
    """The dev-test section of a repair prompt.

    A ``sim`` failure is a whole-program verdict: it says the answer is wrong,
    not which instruction or input is wrong. The dev tests are the localizer, so
    their last verdict travels with the failure. Absence is stated plainly: a run
    with no dev test has no localizing evidence at all.
    """
    if not runs and not tests:
        return ""
    lines = ["## Dev-test evidence (localizing signal, not the run verdict)"]
    if not runs:
        lines.append(
            "- no `run_conformance` result is recorded: write a dev test that "
            "reproduces this failure before changing the design blindly"
        )
    for run in runs[-4:]:
        verdict = run.get("verdict") or {}
        status = str(verdict.get("status") or "not_run")
        lines.append(
            f"- {run.get('test', '')}: {status} "
            f"cases={verdict.get('cases', 0)} failed={verdict.get('failed', 0)}"
            + (f" ({verdict['reason']})" if verdict.get("reason") else "")
        )
        lines.extend(
            f"  - {item}" for item in [str(entry) for entry in (verdict.get("details") or [])][:3]
        )
        if run.get("log_path"):
            lines.append(f"  - full log: {run['log_path']}")
    if runs and all((run.get("verdict") or {}).get("ok") for run in runs):
        lines.append(
            "- every recorded dev test passes while the workload fails: the tests "
            "do not yet cover the defect, or the defect is in the software path "
            "rather than in one instruction. Strengthen a test or inspect the kernel."
        )
    return "\n".join(lines)


def tree_digest(workspace: "DevTestWorkspace") -> dict[str, str]:
    """``{name: sha256}`` for every dev test, for gate evidence and reports.

    A promote verdict is only meaningful for the test bytes it ran, so the gate
    records them: a re-run that edits a test cannot inherit an older pass.
    """
    digests: dict[str, str] = {}
    for name in workspace.names():
        try:
            text = workspace.read(name)
        except OSError:
            continue
        digests[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digests


#: Calls that belong to the harness/template rather than to the ISA contract.
_NON_ISA_CALL_PREFIXES = ("cfu_conf", "cfu_conformance", "cfu_intrinsics")


def stale_tests_from_state(state: Any) -> dict[str, list[str]]:
    """``{test: [missing intrinsic, ...]}`` for tests the delivered ISA cannot run.

    A design may be rewritten (or its ISA revised) after a test was written, and
    the test then references intrinsics that no longer exist. The test is still
    the run's best localizing evidence *while it compiles*, but once it does not,
    the report must say so instead of showing an old PASS/FAIL as if it still
    described the delivered candidate.
    """
    from .. import conformance
    from pathlib import Path

    root = root_for(state)
    if not root:
        return {}
    isa = state.get("isa_spec") or {}
    if not isinstance(isa, dict) or not isa.get("commands"):
        # No contract in this run: there is no authoritative intrinsic list to
        # compare against, so no test can be called stale.
        return {}
    allowed = conformance.allowed_intrinsics(isa)
    directory = Path(str(state.get("workdir") or ".")) / root
    if not directory.is_dir():
        return {}
    stale: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.c")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        # Identifiers, not calls: a test may reference an intrinsic as a
        # function pointer (`{cfu_step_r7, cfu_step_r12, ...}`), which has no
        # parentheses and would otherwise look absent.
        calls = {
            name for name in re.findall(r"\b(cfu_[A-Za-z0-9_]+)\b", text)
            if not name.startswith(_NON_ISA_CALL_PREFIXES)
        }
        missing = sorted(calls - allowed)
        if missing:
            stale[path.stem] = missing
    return stale
