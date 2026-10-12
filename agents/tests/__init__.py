"""Tests for the CfuGym CFU agent framework.

Importing any test module installs the outbound-network guard so a stray real
model call fails loudly instead of silently spending the operator's tokens.
Real LLM SDK clients are also blocked, including localhost endpoints. There is
no network opt-in in the regression suite; use fixtures or injected fake clients.
"""

from .common import block_outbound_network

block_outbound_network()
