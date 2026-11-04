"""Command construction for agent-authored dev tests.

Kept out of ``runtime/commands.py`` because these are not ladder commands: they
build and run a test the model wrote, and they are invoked from the
implementation stage (``run_conformance``) and from a promoting
``hw_conformance`` rung. The ladder's own commands stay in ``runtime/commands.py``
and are reached module-qualified, unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..benchmark import embench
from ..benchmark.embench import DEFAULT_MARCH
from ..state import CfuDesignState
from . import commands as ladder_commands
from . import dev_tests


def dev_conformance_paths(state: CfuDesignState, test_name: str) -> dict[str, str]:
    """Repo-relative paths of one dev test and its host-owned companions.

    Mirrors the formal rung's layout, but under ``workspace/dev_tests/`` and with
    one ``MAIN`` stub per test so two tests cannot share (and overwrite) an ELF.
    ``sw/Makefile`` links ``$(MAIN).c`` and writes ``$(MAIN).elf``; the stub
    includes the shared host runner, so the runner source exists exactly once.
    """
    root = dev_tests.root_for(state)
    name = dev_tests.validate_name(test_name)
    return {
        "root": root,
        "source": f"{root}/{name}.c",
        "main": f"{root}/{dev_tests.HOST_SUBDIR}/{name}_runner",
        "elf": f"{root}/{dev_tests.HOST_SUBDIR}/{name}_runner.elf",
    }


def dev_conformance_build_command(
    state: CfuDesignState, test_name: str, header_path: str, nonce: int,
) -> tuple[list[str], str]:
    """Build one dev test against the host runner and return ``(argv, elf)``.

    The model's file is ``EXTRA_SOURCES`` and the host stub is ``MAIN``: the
    verdict comes from the host runner, never from the model's translation unit.
    The per-run nonce arrives as an integer macro, so a PASS line written into
    the test cannot reproduce it.
    """
    from .. import conformance_dev_runner
    if not str(header_path or "").strip():
        raise ValueError("building a dev test requires an intrinsics header path")
    paths = dev_conformance_paths(state, test_name)
    if not paths["root"]:
        raise ValueError("building a dev test requires a run-scoped workspace")
    rel = os.path.relpath(paths["root"], "sw")
    # MAIN must be the stub's FULL repo-relative path (it lives in `_host/`), not
    # `root/<name>_runner`: `$(MAIN).c` is resolved from `sw/`, so dropping the
    # `_host/` segment makes make look for a file that does not exist and every
    # dev test dies with "No rule to make target".
    main_rel = os.path.relpath(paths["main"], "sw")
    header_dir = os.path.relpath(os.path.dirname(header_path) or ".", "sw")
    include_dirs = [
        header_dir,
        f"{rel}/{dev_tests.HOST_SUBDIR}",
        *(
            os.path.relpath(directory, "sw")
            for directory in ladder_commands.conformance_scalar_includes(state)
        ),
    ]
    scalar_defines = ladder_commands.conformance_scalar_defines(state)
    flags = [conformance_dev_runner.build_macros(nonce), *scalar_defines]
    if scalar_defines or ladder_commands.conformance_scalar_includes(state):
        flags.insert(0, embench.DEFAULT_C_STANDARD)
    return (
        [
            "make",
            "-C",
            "sw",
            "TARGET=vexii_soc",
            f"MARCH={DEFAULT_MARCH}",
            f"MAIN={main_rel}",
            f"BUILD={rel}/build",
            f"EXTRA_SOURCES={rel}/{os.path.basename(paths['source'])}",
            f"EXTRA_INCLUDES={' '.join(include_dirs)}",
            f"EXTRA_CFLAGS={' '.join(flags)}",
            "compile",
        ],
        paths["elf"],
    )


def dev_conformance_syntax_command(
    state: CfuDesignState, test_name: str, header_path: str,
) -> list[str]:
    """Syntax-only preflight of one dev test; never links or simulates.

    ``-fsyntax-only`` catches the class of mistake a full build would report
    only after assembling the whole SoC test, and it cannot be used to observe
    the design, so it stays a cheap pre-check rather than a second verdict.
    """
    if not str(header_path or "").strip():
        raise ValueError("dev-test syntax check requires an intrinsics header path")
    paths = dev_conformance_paths(state, test_name)
    includes = [
        "-I", str(Path(header_path).parent),
        "-I", str(Path(paths["root"]) / dev_tests.HOST_SUBDIR),
        "-I", "sw/MiCo-Lib/targets/vexii_soc",
    ]
    scalar_dirs = ladder_commands.conformance_scalar_includes(state)
    for directory in scalar_dirs:
        includes.extend(["-I", directory])
    includes.extend(ladder_commands.conformance_scalar_defines(state))
    c_standard = embench.DEFAULT_C_STANDARD if scalar_dirs else "-std=gnu99"
    return [
        "riscv64-unknown-elf-gcc",
        "-fsyntax-only",
        c_standard,
        "-Werror=implicit-function-declaration",
        "-march=rv32imc_zicsr_zifencei",
        "-mabi=ilp32",
        *includes,
        paths["source"],
    ]

