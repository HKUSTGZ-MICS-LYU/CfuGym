"""Host-owned runner, data and verdict parsing for agent-authored dev tests.

The model writes only ``cfu_conformance_dut()`` and ``cfu_conformance_ref()``
(see :mod:`cfu_design_agent.runtime.dev_tests`). This module owns everything the
verdict depends on:

* the entry point, so the model cannot print its own PASS line;
* the data block and the length/index sweep, so a test cannot pass on a
  hand-picked input;
* the comparison, so "the expected value is what I wrote next to the check" is
  not expressible;
* a per-run nonce embedded in the verdict line, so a literal PASS written into
  the test file is ignored by the parser.

The runner is plain text, regenerated before every run from this module, which
is what makes it authoritative: no run can link an older runner.
"""

from __future__ import annotations

import re
import secrets
from typing import Any

RUNNER_NONCE_MACRO = "CFU_CONF_NONCE"

#: How many boundary lengths and selector values the host sweep covers. The
#: product is the ``cases=`` number the verdict reports.
BOUNDARY_LENGTHS = (0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
SELECTOR_VALUES = (0, 1, 2)

#: sbt prefixes every line of the forked simulator's output
#: ("[info] CFU_CONFORMANCE_..."), so the parser must strip the level marker.
#: A mock that feeds bare lines hides this, which is why it is explicit here.
_SBT_PREFIX_RE = re.compile(r"^\s*\[(?:info|error|warn|success)\]\s?", re.MULTILINE)

_VERDICT_RE = re.compile(
    r"^CFU_CONFORMANCE_(?P<nonce>[0-9a-fA-F]{8})\s+"
    r"(?P<status>PASS|FAIL)\s+"
    r"cases=(?P<cases>\d+)\s+failed=(?P<failed>\d+)\s*$",
    re.MULTILINE,
)
_DETAIL_RE = re.compile(r"^CFU_CONFORMANCE_CASE\b.*$", re.MULTILINE)

#: Maximum bytes the harness data block may hold. Kept small so a full sweep
#: still finishes quickly on the simulator.
DATA_BYTES = 256


def render_header() -> str:
    """The host-owned ``cfu_conf.h`` a dev test includes."""
    return f"""/* CfuGym dev-test harness data (host-owned; never a model write target).
 *
 * The runner defines these. Read `cfu_conf_data[0 .. cfu_conf_len-1]` in your
 * dut/reference, and use `cfu_conf_index` as the selector/accumulator slot the
 * scenario needs.
 */
#ifndef CFU_CONF_H
#define CFU_CONF_H

#include <stdint.h>

#define CFU_CONF_DATA_BYTES {DATA_BYTES}u

extern uint8_t cfu_conf_data[CFU_CONF_DATA_BYTES];
extern uint32_t cfu_conf_len;
extern uint32_t cfu_conf_index;

#endif /* CFU_CONF_H */
"""


def render_runner() -> str:
    """The host-owned ``main()``: sweep, compare, print one nonce-tagged verdict."""
    lengths = ", ".join(f"{value}u" for value in BOUNDARY_LENGTHS)
    selectors = ", ".join(f"{value}u" for value in SELECTOR_VALUES)
    return f"""/* CfuGym dev-test runner (host-owned; never a model write target).
 *
 * Ownership: the model owns only the two functions it defines in its test file.
 * This file owns the entry point, the harness data, the length/selector sweep,
 * the comparison and the verdict line. Regenerated from the framework before
 * every run, so a stale runner can never be linked.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfu_intrinsics.h"
#include "cfu_conf.h"

#ifndef {RUNNER_NONCE_MACRO}
#define {RUNNER_NONCE_MACRO} 0u
#endif

/* ---- harness-owned data (declared in cfu_conf.h) ---- */
uint8_t cfu_conf_data[CFU_CONF_DATA_BYTES];
uint32_t cfu_conf_len = 0u;
uint32_t cfu_conf_index = 0u;

/* ---- the model's two functions ---- */
uint32_t cfu_conformance_dut(void);
uint32_t cfu_conformance_ref(void);

static const uint32_t k_lengths[] = {{ {lengths} }};
static const uint32_t k_selectors[] = {{ {selectors} }};

static uint32_t g_master[CFU_CONF_DATA_BYTES / 4u];
static uint32_t g_rng = 0x9e3779b9u;

static uint32_t next_u32(void) {{
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng;
}}

static void fill_data(uint32_t seed) {{
  uint32_t i;
  g_rng = seed ? seed : 1u;
  for (i = 0u; i < CFU_CONF_DATA_BYTES; i++) {{
    cfu_conf_data[i] = (uint8_t)(next_u32() >> 24);
  }}
}}

static void save_master(void) {{
  memcpy(g_master, cfu_conf_data, CFU_CONF_DATA_BYTES);
}}

static void restore_master(void) {{
  memcpy(cfu_conf_data, g_master, CFU_CONF_DATA_BYTES);
}}

int main(void) {{
  uint32_t total = 0u;
  uint32_t failed = 0u;
  uint32_t shown = 0u;
  uint32_t li;
  uint32_t si;

  cfu_enable();
  fill_data(0x5eedu);
  save_master();

  for (li = 0u; li < (uint32_t)(sizeof(k_lengths) / sizeof(k_lengths[0])); li++) {{
    for (si = 0u; si < (uint32_t)(sizeof(k_selectors) / sizeof(k_selectors[0])); si++) {{
      uint32_t len = k_lengths[li];
      uint32_t index = k_selectors[si];
      uint32_t got;
      uint32_t want;

      if (len > CFU_CONF_DATA_BYTES) {{
        len = CFU_CONF_DATA_BYTES;
      }}

      /* Reference first, on a pristine copy of the same data the DUT will see. */
      cfu_conf_len = len;
      cfu_conf_index = index;
      restore_master();
      want = cfu_conformance_ref();

      cfu_conf_len = len;
      cfu_conf_index = index;
      restore_master();
      cfu_memory_fence();
      got = cfu_conformance_dut();

      total++;
      if (got != want) {{
        failed++;
        if (shown < 3u) {{
          shown++;
          printf("CFU_CONFORMANCE_CASE nonce=%08x len=%u index=%u got=%08x want=%08x\\n",
                 (unsigned){RUNNER_NONCE_MACRO}, (unsigned)len, (unsigned)index,
                 (unsigned)got, (unsigned)want);
        }}
      }}
    }}
  }}

  printf("CFU_CONFORMANCE_%08x %s cases=%u failed=%u\\n",
         (unsigned){RUNNER_NONCE_MACRO}, failed ? "FAIL" : "PASS",
         (unsigned)total, (unsigned)failed);
  /* Always exit 0: a mismatch is a *verdict*, not a crash. Returning non-zero
   * traps the simulator on the `fail` symbol, which turns a readable
   * "got/want" report into a SpinalHDL stack trace and misclassifies the run
   * as `fail_symbol`. The host decides pass/fail from the verdict line. */
  return 0;
}}
"""


def new_nonce() -> int:
    """A per-run value the model cannot predict when it writes the test.

    Using ``secrets`` rather than a counter matters: a counter is guessable from
    the test file's own context, and the whole point is that a hand-written
    ``CFU_CONFORMANCE_... PASS`` line must not be accepted.
    """
    return secrets.randbits(32)


def build_macros(nonce: int) -> str:
    """``EXTRA_CFLAGS`` entries carrying the run nonce (integer, so no quoting)."""
    return f"-D{RUNNER_NONCE_MACRO}=0x{int(nonce) & 0xFFFFFFFF:08x}u"


def parse_verdict(text: str, nonce: int, *, test: str = "") -> dict[str, Any]:
    """Parse this run's verdict line; silence is ``missing``, never a pass.

    A line whose nonce does not match this run is ignored entirely: it can only
    come from the test file, not from the runner.
    """
    body = _SBT_PREFIX_RE.sub("", str(text or ""))
    details = [line.strip() for line in _DETAIL_RE.findall(body)]
    expected = f"{int(nonce) & 0xFFFFFFFF:08x}"
    matches = [
        match for match in _VERDICT_RE.finditer(body)
        if match.group("nonce").lower() == expected
    ]
    if not matches:
        return {
            "status": "missing",
            "ok": False,
            "test": test,
            "cases": 0,
            "failed": 0,
            "details": details,
            "nonce": expected,
            "reason": (
                "no verdict line carrying this run's nonce was printed: either the "
                "test did not run to completion (crash/hang) or its output was lost"
            ),
        }
    match = matches[-1]
    status = match.group("status").lower()
    cases = int(match.group("cases"))
    failed = int(match.group("failed"))
    # A PASS that ran no case, or that reports failures, is not a pass.
    ok = status == "pass" and failures_consistent(cases, failed)
    return {
        "status": status,
        "ok": ok,
        "test": test,
        "cases": cases,
        "failed": failed,
        "details": details,
        "nonce": expected,
        "reason": "" if ok else (
            f"{failed} of {cases} cases disagreed with the reference"
            if cases else "the runner executed no case"
        ),
    }


def failures_consistent(cases: int, failed: int) -> bool:
    """A PASS line must cover at least one case and report zero failures."""
    return cases > 0 and failed == 0


def expected_cases() -> int:
    """How many cases the runner sweeps (mirrors :func:`render_runner`)."""
    return len(BOUNDARY_LENGTHS) * len(SELECTOR_VALUES)
