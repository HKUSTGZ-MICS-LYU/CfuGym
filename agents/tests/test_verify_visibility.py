"""Agent-only verifier redaction: pure fixtures/fake SDKs, no model or toolchain."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml
from agents.cfu_design_agent.ir import observation
from agents.cfu_design_agent.runtime import agent_tools, llm, llm_step, policy, source_visibility, tools
from agents.cfu_design_agent.runtime import stage_executor

SECRET = "PRIVATE_VERIFIER_SENTINEL"
ANSWER = "31415927"
SOURCE = ('int kernel(int x) { return x + 1; }\n'
          'int\nverify_benchmark(int r)\n{\n'
          f'  /* {SECRET} }} {{ */\n'
          f'  if (r) {{ const char *s = "}} {{"; return r == {ANSWER}; }}\n'
          '  return 0;\n}\nint helper(void) { return 9; }\n')


class MaskSourceTest(unittest.TestCase):
    def assert_hidden(self, text):
        self.assertNotIn(SECRET, text)
        self.assertNotIn(ANSWER, text)

    def test_nested_body_signature_other_functions_and_line_count(self):
        visible = source_visibility.mask_verifier_bodies(SOURCE)
        self.assert_hidden(visible)
        self.assertIn('verify_benchmark(int r)', visible)
        self.assertIn('int kernel(int x) { return x + 1; }', visible)
        self.assertIn('int helper(void) { return 9; }', visible)
        self.assertEqual(visible.count('\n'), SOURCE.count('\n'))
        self.assertEqual(source_visibility.mask_verifier_bodies(visible), visible)

    def test_inline_attribute_and_multiple_definitions(self):
        for code in (f'int verify_benchmark(int r) {{ return {ANSWER}; }}',
                     f'int verify_benchmark(int r) __attribute__((unused)) {{ return {ANSWER}; }}'):
            visible = source_visibility.mask_verifier_bodies(code + '\n' + code)
            self.assert_hidden(visible)
            self.assertEqual(visible.count(source_visibility.MARKER), 2)

    def test_comments_literals_prototypes_calls_are_not_definitions(self):
        for code in ('int verify_benchmark(int r);\nint kernel() {return 1;}\n',
                     'int kernel() {return verify_benchmark(3);}\n',
                     '// int verify_benchmark(int r) { return 1; }\n',
                     'char *s = "verify_benchmark(int r) { return 1; }";\n'):
            self.assertEqual(source_visibility.mask_verifier_bodies(code), code)

    def test_truncated_body_is_hidden_through_eof(self):
        visible = source_visibility.mask_verifier_bodies(SOURCE[:SOURCE.index('  return 0;')])
        self.assert_hidden(visible)
        self.assertIn(source_visibility.MARKER, visible)

    def test_numbered_prompt_source(self):
        numbered = '\n'.join(f'{i:6}\t{line}' for i, line in enumerate(SOURCE.splitlines(), 1))
        visible = source_visibility.mask_verifier_bodies(numbered)
        self.assert_hidden(visible)
        self.assertIn('     6\t', visible)
        self.assertEqual(visible.count('\n'), numbered.count('\n'))
        self.assertEqual(source_visibility.mask_verifier_bodies(visible), visible)

    def test_llvm_assembler_and_disassembly(self):
        examples = [
            f'define i32 @verify_benchmark(i32 %r) #0 {{\n  ret i32 {ANSWER}\n}}\n'
            'define i32 @kernel() { ret i32 9 }\n',
            f'verify_benchmark:\n  li a0,{ANSWER}\n  ret\n.size verify_benchmark,.-verify_benchmark\n'
            '.type kernel,@function\nkernel:\n  li a0,9\n',
            f'80001000 <verify_benchmark.constprop.0>:\n  li a0,{ANSWER}\n  ret\n'
            '80001010 <kernel>:\n  li a0,9\n']
        for code in examples:
            visible = source_visibility.mask_verifier_bodies(code)
            self.assert_hidden(visible)
            self.assertIn('kernel', visible)
            self.assertEqual(visible.count('\n'), code.count('\n'))

    def test_features_xml_masks_only_verifier_function(self):
        code = (f'<ir_features><function name="verify_benchmark"><constant>{ANSWER}</constant>'
                '</function><function name="kernel"><constant>9</constant></function></ir_features>')
        visible = source_visibility.mask_verifier_bodies(code)
        self.assert_hidden(visible)
        self.assertIn('<constant>9</constant>', visible)

    def test_nested_request_json_code_is_masked_without_mutating_input(self):
        payload = {'messages': [{'role': 'tool', 'content': json.dumps({'source': SOURCE})}],
                   'measured_result': int(ANSWER)}
        masked = source_visibility.mask_model_payload(payload)
        self.assertIn(SECRET, payload['messages'][0]['content'])
        self.assert_hidden(masked['messages'][0]['content'])
        self.assertEqual(masked['measured_result'], int(ANSWER))

    def test_escaped_code_inside_a_legacy_prompt_is_hidden(self):
        prompt = 'Run memory:\nsource fact: ' + json.dumps({'source': SOURCE})
        visible = source_visibility.mask_model_payload(prompt)
        self.assert_hidden(visible)
        self.assertIn('Run memory:', visible)

    def test_escaped_json_yaml_and_versioned_audit_views(self):
        for path, serialized in [('events.json', json.dumps({'source': SOURCE, 'value': 102})),
                                 ('events.jsonl', json.dumps({'source': SOURCE}) + '\n'),
                                 ('calls.yaml', yaml.safe_dump({'source': SOURCE, 'value': 102})),
                                 ('calls.yaml.' + 'a' * 64, yaml.safe_dump({'source': SOURCE}))]:
            visible = source_visibility.mask_file_view(serialized, path)
            self.assert_hidden(visible)
            self.assertEqual(visible.count('\n'), serialized.count('\n'))
            decoded = yaml.safe_load(visible) if '.yaml' in path else json.loads(visible)
            self.assertIn(source_visibility.MARKER, decoded['source'])
            if 'value' in decoded:
                self.assertEqual(decoded['value'], 102)


class SourceToolTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'bench').mkdir()
        (self.root / 'bench/test.c').write_text(SOURCE)
        access = policy.AccessPolicy(version='1', deny=(), write_roots=('bench',),
            stages={name: policy.StagePolicy(write=('bench/**',), commands=())
                    for name in ('context', 'analysis', 'isa', 'implementation')},
            commands={}, read=('bench/**',))
        self.toolbox = tools.RepoToolbox(self.root, policy=access, stage='analysis')

    def test_all_stages_read_body_only_ranges_without_disclosure(self):
        for stage in ('context', 'analysis', 'isa', 'implementation'):
            self.toolbox.stage = stage
            toolset = agent_tools.ReadOnlyToolset(self.toolbox)
            for selection in ({}, {'start_line': 5, 'end_line': 6}):
                text = toolset.run('read_file', {'path': 'bench/test.c', **selection})
                self.assertNotIn(SECRET, text)
                self.assertNotIn(ANSWER, text)
            self.assertEqual(self.toolbox.read_text('bench/test.c'), SOURCE)

    def test_search_cannot_match_hidden_body_but_can_find_interface(self):
        toolset = agent_tools.ReadOnlyToolset(self.toolbox)
        for pattern in (SECRET, ANSWER):
            self.assertEqual(toolset.run('search_text', {'path': 'bench/test.c', 'pattern': pattern}),
                             '(no matches)')
        self.assertIn('verify_benchmark', toolset.run('search_text', {
            'path': 'bench/test.c', 'pattern': 'verify_benchmark'}))

    def test_diff_and_mismatch_context_do_not_disclose_verifier(self):
        after = SOURCE.replace('return 0;', 'return 1;')
        diff = tools.render_change_diff(SOURCE, after, 'bench/test.c')
        self.assertNotIn(ANSWER, diff)
        self.assertNotIn(SECRET, diff)
        error = tools._describe_mismatch(SOURCE.splitlines(),
            ['verify_benchmark(int r)', '{', '  WRONG'])
        self.assertNotIn(SECRET, error)
        self.assertIn('First difference', error)
        self.assertEqual((self.root / 'bench/test.c').read_text(), SOURCE)

    def test_observation_views_refuse_verifier_internals_without_reading_artifacts(self):
        run = 'agents/generated/runs/view-test'
        ident = 'obs_verify'
        index = observation.build_observation_index(run_root=run, benchmark='slre',
            observation={'points': [{'id': ident, 'kind': 'function', 'file': 'slre.c',
                                     'function': 'verify_benchmark', 'line': 3}], 'applied': [ident]},
            profiles={}, ir_result={})
        toolbox = SimpleNamespace(run_root=run, stage='analysis', workdir=self.root,
                                  read_text=mock.Mock(side_effect=AssertionError('must not read verifier')))
        for view in ('summary', 'asm', 'ir', 'features'):
            self.assertIn('hidden', observation.inspect_observation(index, toolbox, ident, view))
        toolbox.read_text.assert_not_called()


class RequestVisibilityTest(unittest.TestCase):
    def test_plain_completion_masks_before_fake_client(self):
        client = SimpleNamespace(complete=mock.Mock(return_value=llm.LlmResult('ok', used_llm=True)))
        with mock.patch.object(llm_step, '_record_llm_output', return_value='audit.txt'):
            llm_step._complete_llm({}, 'analysis', 'test', SOURCE, SOURCE, '', client=client)
        sent = str(client.complete.call_args)
        self.assertNotIn(SECRET, sent)
        self.assertNotIn(ANSWER, sent)

    def test_sdk_chat_and_responses_mask_real_request_body_using_fake_transports(self):
        response = SimpleNamespace(id='fake', usage=None)
        target = mock.Mock(return_value=response)
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=target)),
                              responses=SimpleNamespace(create=target))
        client = llm.OpenAiTextClient(config={'api_key': 'fake', 'model': 'fake', 'stream': False})
        for endpoint, body in [('chat', {'messages': [{'role': 'user', 'content': SOURCE}]}),
                               ('responses', {'instructions': SOURCE, 'input': SOURCE})]:
            with mock.patch.object(llm.request_audit, 'record_request', return_value='req-test') as audit:
                client._sdk_request(sdk, endpoint, body)
            self.assertNotIn(SECRET, str(target.call_args))
            self.assertNotIn(ANSWER, str(target.call_args))
            self.assertEqual(audit.call_args.args[0], target.call_args.kwargs)
            self.assertIn(SECRET, str(body))  # Original host payload remains unchanged.

    def test_tool_output_wrapper_masks_and_keeps_run_path_virtualization(self):
        inner = SimpleNamespace(run=lambda *_: SOURCE + 'agents/generated/runs/unit/source.c')
        view = stage_executor._RunDirView(inner, 'agents/generated/runs/unit')
        output = view.run('read_file', {})
        self.assertNotIn(SECRET, output)
        self.assertNotIn(ANSWER, output)
        self.assertIn('{{run-dir}}/source.c', output)
