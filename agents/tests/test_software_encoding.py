"""Compiled word coverage, pure fixtures and in-memory toolbox."""
from __future__ import annotations

import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import instruction_encoding, software_encoding

SPEC = {'commands': [{'name': 'run', 'function_id': 1, 'subfunction_id': 1}]}
ASM = '80000628: 0200150b .word 0x0200150b\n'


class CompiledEvidenceTest(unittest.TestCase):
    def state(self, actual=((1, 1),)):
        binding = {'isa_hash': instruction_encoding.digest(SPEC), 'candidate_hash': 'hw',
                   'rtl': {'actual_pairs': [list(p) for p in actual], 'unsupported_guard': 'checked'}}
        return {'isa_spec': SPEC, 'benchmark_project': {'elf_path': 'fake/main.elf'},
                'gates': {'hw_build': {'instruction_encoding': binding},
                          'implementation_hw': {'candidate_hash': 'hw'}}}

    def test_compiled_word_must_select_actual_decoder(self):
        box = mock.Mock()
        box.read_text.return_value = ASM
        evidence = software_encoding.compiled_evidence(self.state(), box)
        self.assertTrue(evidence['checked'])
        self.assertTrue(evidence['ok'])
        self.assertEqual(evidence['observed_pairs'], [[1, 1]])
        self.assertEqual(evidence['assembly_path'], 'fake/main.asm')
        bad = software_encoding.compiled_evidence(self.state(((1, 0),)), box)
        self.assertFalse(bad['ok'])
        self.assertIn('compiled_rtl_mismatch', '\n'.join(bad['problems']))

    def test_missing_words_or_missing_disassembly_with_bound_hardware_fail(self):
        box = mock.Mock()
        for text in ('', '80000628: 00000013 nop\n'):
            box.read_text.return_value = text
            evidence = software_encoding.compiled_evidence(self.state(), box)
            self.assertFalse(evidence['ok'])
            self.assertFalse(evidence['checked'])

    def test_changed_rtl_product_cannot_reuse_decoder_snapshot(self):
        state = self.state()
        state.update(workdir='/tmp', run_root='fixture-run')
        state['gates']['hw_build']['instruction_encoding']['rtl']['rtl_hash'] = 'old-rtl'
        box = mock.Mock()
        box.read_text.side_effect = ['module AgentCfu(); endmodule', ASM]
        evidence = software_encoding.compiled_evidence(state, box)
        self.assertFalse(evidence['ok'])
        self.assertFalse(evidence['checked'])
        self.assertIn('stale_rtl', '\n'.join(evidence['problems']))

    def test_isa_or_hardware_binding_cannot_be_reused(self):
        box = mock.Mock()
        box.read_text.return_value = ASM
        for field, value in (('isa_hash', 'other-isa'), ('candidate_hash', 'other-hardware')):
            state = self.state()
            state['gates']['hw_build']['instruction_encoding'][field] = value
            evidence = software_encoding.compiled_evidence(state, box)
            self.assertFalse(evidence['ok'])
            self.assertFalse(evidence['checked'])

    def test_ablation_and_old_standalone_are_unknown_not_proven(self):
        box = mock.Mock()
        box.read_text.return_value = ''
        evidence = software_encoding.compiled_evidence({}, box)
        self.assertFalse(evidence['checked'])
        self.assertTrue(evidence['ok'])
        old = {'isa_spec': SPEC, 'benchmark_project': {'elf_path': 'fake/main.elf'}}
        evidence = software_encoding.compiled_evidence(old, box)
        self.assertFalse(evidence['checked'])
        self.assertTrue(evidence['ok'])
        self.assertTrue(evidence['unchecked'])


if __name__ == '__main__':
    unittest.main()
