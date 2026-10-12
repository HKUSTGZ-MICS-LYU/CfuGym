"""Host-owned Embench CFU status diagnostics, not a semantic-equivalence test."""
from __future__ import annotations

import re

PROTOCOL = "sticky_csr_801_v1"
PREAMBLE = """/* Host-owned CFU execution status; model kernels must not clear this CSR. */
#ifdef USE_AGENT_CFU
static unsigned int embench_host_cfu_status(void)
{
  unsigned int flags;
  __asm__ volatile ("csrr %0, 0x801" : "=r" (flags) : : "memory");
  return flags & 63u;
}
static void embench_host_cfu_clear_status(void)
{
  __asm__ volatile ("csrw 0x801, zero" : : : "memory");
}
#endif
"""
BEGIN = """#ifdef USE_AGENT_CFU
  embench_host_cfu_clear_status ();
#endif
  """
SNAPSHOT = """\n#ifdef USE_AGENT_CFU
  unsigned int embench_host_cfu_flags = embench_host_cfu_status ();
  printf ("EMBENCH_CFU_STATUS benchmark=%s flags=%u protocol=sticky_csr_801_v1\\n",
          EMBENCH_STRINGIFY (EMBENCH_NAME), embench_host_cfu_flags);
#endif
"""
VERDICT = """\n#ifdef USE_AGENT_CFU
  if (embench_host_cfu_flags != 0u) correct = 0;
#endif
"""
_STATUS = re.compile(r"EMBENCH_CFU_STATUS\s+benchmark=(\S+)\s+flags=(\d+)\s+protocol=(\S+)")


def instrument_main(text: str, verify_call: str) -> str:
    """Bracket only the trusted driver's real benchmark, outside cycle timing."""
    if verify_call not in text or "embench_host_cfu_clear_status ();" in text:
        return text
    start, stop = "start_trigger ();", "stop_trigger ();"
    if start not in text or stop not in text:
        return text
    # Clear before initialization too: CFU setup errors must not be erased.
    initialize = "initialise_benchmark ();"
    begin = initialize if initialize in text else start
    text = text.replace(begin, BEGIN + begin, 1)
    text = text.replace(stop, stop + SNAPSHOT, 1)
    text = text.replace(verify_call, verify_call + VERDICT, 1)
    return PREAMBLE + text


def parse_status(text: str) -> dict | None:
    matches = list(_STATUS.finditer(text or ""))
    if not matches:
        return None
    records = [{"benchmark": m[1], "flags": int(m[2]), "protocol": m[3],
                "ok": int(m[2]) == 0 and m[3] == PROTOCOL} for m in matches]
    result = dict(records[-1])
    if len(records) > 1:
        result["records"] = records
    return result


def status_evidence(results: list[dict], name: str, *, expected: bool = False) -> dict:
    records = []
    for result in results:
        parsed = result.get("parsed") or {}
        record = parsed.get("cfu_status") or parse_status("\n".join(
            str(result.get(k) or "") for k in ("stdout_tail", "stderr_tail")))
        if isinstance(record, dict):
            records.extend(r for r in (record.get("records") or [record]) if isinstance(r, dict))
    matching = [r for r in records if r.get("benchmark") == name]
    failures = [r for r in matching if r.get("flags") != 0 or r.get("protocol") != PROTOCOL]
    if failures:
        return {"checked": True, "ok": False, "protocol": PROTOCOL,
                "records": matching, "reason": "cfu_execution_error",
                "error": "CFU execution reported unsupported/error status: " + str(failures)}
    if expected and not matching:
        return {"checked": False, "ok": False, "protocol": PROTOCOL, "records": [],
                "reason": "cfu_status_missing", "error": "Host-owned CFU status diagnostic is missing"}
    return {"checked": bool(matching), "ok": True if matching else None,
            "protocol": PROTOCOL, "records": matching, "reason": "", "error": ""}
