"""Tests for the CfuGym CFU agent framework.

Importing any test module installs the outbound-network guard so a stray real
model call fails loudly instead of silently spending the operator's tokens.
Legitimate network use requires ``CFU_AGENT_TESTS_ALLOW_NETWORK=1``.
"""

from .common import block_outbound_network

block_outbound_network()
