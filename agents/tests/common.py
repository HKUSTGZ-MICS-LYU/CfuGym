"""Shared helpers for the CFU agent tests.

Tests must never call a real model endpoint: that spends the operator's
tokens and makes results depend on the environment. Two protections live
here: an explicit offline client for tests that run the real graph, and a
backstop that turns any accidental outbound connection into a loud error
instead of a silent API call.
"""

from __future__ import annotations

import contextlib
import os
import socket
from pathlib import Path
from typing import Iterator
from unittest import mock

from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime import llm_step


REPO_ROOT = Path(__file__).resolve().parents[2]

# Escape hatch for a test that genuinely needs the network (none should).
ALLOW_NETWORK_ENV = "CFU_AGENT_TESTS_ALLOW_NETWORK"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


def repo_files(path: str) -> Path:
    return REPO_ROOT / path


# A minimal contract that passes the ISA linter; shared by tests that need a
# valid contract without building one from scratch.
VALID_ISA = """spec_version: "1"
identity:
  name: u8_wavg_agent
  status: draft
  opcode: custom0 (0x0B)
  function_id_width: 3
  reference_workload: uint8 weighted average
commands:
  - function_id: 1
    name: compute
    operand: rs1, rs2
    result: sum
    semantics: multiply and accumulate uint8 lanes
    response: status
datapath:
  vlen: 256
  xlen: 32
  implementation:
    result_width_bits: 32
  regDepth: 2
software:
  intrinsics_header: cfu_intrinsics.h
  scalar_fallback: true
integration:
  hardware_component: AgentCfu
  hardware_fiber: AgentCfuFiber
  bus_style: CfuBus
  single_cpu_bus_owner: true
validation:
  status: draft
  requirements: [scalar reference equals CFU output]
assumptions: {}
unknowns: []"""


class OfflineLlm:
    """An unavailable client: every call degrades to the deterministic fallback."""

    available = False
    config_name = "offline-test"
    model = "offline-test"
    base_url = ""
    api_mode = "chat"
    thinking = False
    reasoning_effort = None
    thinking_body = None
    stream = False

    def complete(self, system: str, user: str, *, fallback: str,
                 response_format=None) -> LlmResult:
        return LlmResult(fallback, used_llm=False, model=self.model,
                         fallback_reason="offline test client")

    def complete_with_tools(self, system: str, user: str, *, tools=None,
                            tool_runner=None, fallback: str = "", **kwargs) -> LlmResult:
        return LlmResult(fallback, used_llm=False, model=self.model,
                         fallback_reason="offline test client")


@contextlib.contextmanager
def offline_llm() -> Iterator[OfflineLlm]:
    """Force the graph's LLM factory to the offline client for this test.

    Use around any test that runs ``run_agent`` without its own fake model,
    so a real config in the environment cannot be called.
    """
    from agents.cfu_design_agent import graph

    client = OfflineLlm()
    with mock.patch.object(llm_step, "_llm", lambda state: client):
        yield client


def _network_allowed() -> bool:
    return os.environ.get(ALLOW_NETWORK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def block_outbound_network() -> None:
    """Refuse non-local TCP connects so a stray real API call fails loudly.

    Applied at import of the shared test helpers. Local sockets (and child
    processes, which get their own interpreter) are unaffected.
    """
    if _network_allowed() or getattr(block_outbound_network, "_installed", False):
        return
    block_outbound_network._installed = True  # type: ignore[attr-defined]
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def _local(address) -> bool:
        host = address[0] if isinstance(address, tuple) and address else ""
        return isinstance(host, str) and host in _LOCAL_HOSTS

    def guarded_connect(self, address):
        if not _local(address):
            raise RuntimeError(
                f"outbound network blocked in tests (tried {address!r}); "
                f"patch the LLM client or set {ALLOW_NETWORK_ENV}=1 intentionally"
            )
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _local(address):
            raise RuntimeError(f"outbound network blocked in tests (tried {address!r})")
        return real_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _local(address):
            raise RuntimeError(f"outbound network blocked in tests (tried {address!r})")
        return real_create(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]


block_outbound_network()
